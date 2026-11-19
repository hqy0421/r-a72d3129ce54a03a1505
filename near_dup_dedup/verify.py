"""Integrity verification of a committed dedup database.

The checks re-derive what the database claims rather than trusting it:

* meta fingerprint and input content identity still match the file;
* line indices are contiguous from 0 and agree with progress;
* every ``duplicate_of`` points at an accepted earlier record whose exact
  Jaccard with the duplicate meets the threshold and whose LSH candidate
  relationship was recorded as evidence;
* stored ``best_jaccard`` equals the recomputed value;
* the rebuilt LSH contains exactly the accepted record keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import DedupConfig, ParameterMismatchError
from .dedup import exact_jaccard, scan_input
from .storage import SQLiteStore


@dataclass
class VerificationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    checks_run: int = 0
    accepted: int = 0
    duplicate: int = 0

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)


def verify(store: SQLiteStore, config=None) -> VerificationReport:
    """Run all integrity checks against ``store``."""
    report = VerificationReport()
    meta = store.meta()
    if not meta:
        report.fail("database has no meta table rows")
        return report

    # 1. Input identity + parameters.
    report.checks_run += 1
    identity = scan_input(meta["input_abs_path"])
    for key, identity_key in (
        ("input_size", "size"),
        ("input_sha256", "sha256"),
        ("input_total_lines", "total_lines"),
    ):
        if meta.get(key) != str(identity[identity_key]):
            report.fail(f"input identity drift on {key}: stored={meta.get(key)!r} current={identity[identity_key]!r}")
    if config is not None and meta.get("fingerprint") != config.fingerprint:
        report.fail("requested config fingerprint does not match stored run")

    records = list(store.iter_records())
    counts = store.counts()
    report.accepted = counts["accepted"]
    report.duplicate = counts["duplicate"]

    # 2. Contiguous line indices matching progress.
    report.checks_run += 1
    indices = [r["line_index"] for r in records]
    if indices != list(range(len(indices))):
        report.fail("records.line_index is not contiguous 0..N-1")
    prog = store.progress()
    if prog["lines_committed"] != len(indices):
        report.fail(f"progress.lines_committed={prog['lines_committed']} but records has {len(indices)} rows")
    total = int(meta["input_total_lines"])
    if len(indices) != total:
        report.fail(f"committed rows {len(indices)} != input lines {total}")

    # 3. Decision shape and duplicate references.
    report.checks_run += 1
    tokens_by_index: dict[int, list[str]] = {}
    accepted_keys: set[int] = set()
    for r in records:
        tokens_by_index[r["line_index"]] = json.loads(r["tokens_json"])
        if r["decision"] == "accepted":
            accepted_keys.add(r["line_index"])
            if r["duplicate_of"] is not None:
                report.fail(f"line {r['line_index']} accepted but has duplicate_of")
        elif r["decision"] == "duplicate":
            ref = r["duplicate_of"]
            if ref is None:
                report.fail(f"line {r['line_index']} duplicate without target")
                continue
            if ref not in accepted_keys:
                report.fail(f"line {r['line_index']} -> {ref}: target is not an earlier accepted record")
            if ref >= r["line_index"]:
                report.fail(f"line {r['line_index']} -> {ref}: duplicate must point to an earlier source line")
        else:
            report.fail(f"line {r['line_index']}: unknown decision {r['decision']!r}")

    # 4. Recompute exact Jaccard for every duplicate and its stored evidence.
    report.checks_run += 1
    threshold = config.jaccard_threshold if config is not None else DedupConfig.from_meta(meta).jaccard_threshold
    for r in records:
        if r["decision"] != "duplicate":
            continue
        idx = r["line_index"]
        ref = r["duplicate_of"]
        jacc = exact_jaccard(tokens_by_index[idx], tokens_by_index[ref])
        if jacc < threshold:
            report.fail(f"line {idx} -> {ref}: recomputed Jaccard {jacc:.4f} < threshold {threshold}")
        if r["best_jaccard"] is None or abs(jacc - r["best_jaccard"]) > 1e-12:
            report.fail(f"line {idx}: stored best_jaccard {r['best_jaccard']} != recomputed {jacc}")
        evidence = store.evidence_for(idx)
        qualified = [e for e in evidence if e["qualified"]]
        if not qualified:
            report.fail(f"line {idx}: duplicate without qualified evidence")
            continue
        ev_targets = sorted(e["candidate_line_index"] for e in qualified)
        if ref not in ev_targets:
            report.fail(f"line {idx}: duplicate_of {ref} absent from qualified evidence {ev_targets}")
        earliest = min(ev_targets)
        if ref != earliest:
            report.fail(f"line {idx}: duplicate_of {ref} is not the earliest qualified candidate {earliest}")
        # Stored Jaccard values in evidence must match exact recomputation.
        for e in evidence:
            recomputed = exact_jaccard(
                tokens_by_index[idx],
                tokens_by_index[e["candidate_line_index"]],
            )
            if abs(recomputed - e["jaccard"]) > 1e-12:
                report.fail(
                    f"line {idx} evidence -> {e['candidate_line_index']}: stored Jaccard {e['jaccard']} != {recomputed}"
                )

    # 5. Rebuild LSH and compare its key set with the accepted records.
    report.checks_run += 1
    if config is not None:
        lsh = config.new_lsh()
        for idx in accepted_keys:
            lsh.insert(idx, config.minhash_for_tokens(tokens_by_index[idx]))
        # OrderedStorage exposes .keys(); not every backend is directly
        # iterable, so keep the explicit call (SIM118 is a false positive).
        lsh_keys = {int(k) for k in lsh.keys.keys()}  # noqa: SIM118
        if lsh_keys != accepted_keys:
            report.fail(f"rebuilt LSH keys {sorted(lsh_keys)} disagree with accepted records {sorted(accepted_keys)}")
    if store.lsh_key_count() != len(accepted_keys):
        report.fail("accepted count disagrees with store.lsh_key_count()")

    return report


def assert_resumable(store: SQLiteStore, config) -> None:
    """Convenience: raise :class:`ParameterMismatchError` on any drift."""
    report = verify(store, config)
    if not report.ok:
        raise ParameterMismatchError("; ".join(report.errors))
