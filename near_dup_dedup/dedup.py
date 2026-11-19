"""Resumable near-duplicate cleaning engine.

Order of operations per batch (the basis of all crash guarantees):

  1. Open one SQLite transaction (``BEGIN`` via :meth:`SQLiteStore.begin_batch`).
  2. Evaluate every record in the batch against candidates from
     a. the persistent LSH index (only keys of records committed in earlier
        batches) and
     b. an ephemeral LSH holding records *accepted earlier in this same
        batch*, so in-batch decisions are no less strict than cross-batch
     ones.  The persistent index itself is never touched before commit.
  3. Confirm each LSH candidate with exact Jaccard on the original token
     sets.  The first accepted source record among qualifying candidates
     (lowest source line index) becomes ``duplicate_of``; every other record
     is accepted.
  4. One durable ``COMMIT`` publishes decisions, evidence and progress.
  5. Only then insert the batch's accepted keys into the persistent LSH.

On reopen the LSH is rebuilt entirely from committed ``records`` rows, so
the index can never contain a record whose decision was not committed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import DedupConfig, ParameterMismatchError
from .storage import SQLiteStore

# -- pure helpers ---------------------------------------------------------


def exact_jaccard(a: list[str], b: list[str]) -> float:
    """Exact Jaccard similarity of two token sequences' set representations."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        # Two empty sets are conventionally identical; LSH cannot represent
        # this case anyway (empty MinHash never collides), kept for symmetry.
        return 1.0
    return len(sa & sb) / len(sa | sb)


def scan_input(path: str) -> dict:
    """Return size, SHA-256 and total line count of the JSONL input.

    The file is streamed in binary mode so the hash covers exactly the bytes
    used for offset-based resumes.
    """
    h = hashlib.sha256()
    size = 0
    total_lines = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
            total_lines += chunk.count(b"\n")
        # A final line without a trailing newline is still a line.
        if size > 0:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                total_lines += 1
    return {
        "path": path,
        "abs_path": os.path.abspath(path),
        "size": size,
        "sha256": h.hexdigest(),
        "total_lines": total_lines,
    }


@dataclass
class DuplicateVerdict:
    """Decision for one input line."""

    line_index: int
    byte_offset: int
    decision: str  # "accepted" | "duplicate"
    duplicate_of: Optional[int]
    best_jaccard: Optional[float]
    tokens: list[str]
    evidence: list[dict] = field(default_factory=list)


@dataclass
class RunStats:
    """Counters returned by :meth:`Deduplicator.run`."""

    lines_processed: int = 0
    accepted: int = 0
    duplicate: int = 0
    batches_committed: int = 0
    resumed_from_line: int = -1


class Deduplicator:
    """Owns the SQLite state and the MinHashLSH index for one dedup run."""

    def __init__(
        self,
        config: DedupConfig,
        db_path: str,
        crash_hook: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Args:
        config: Run parameters; must match on every resume.
        db_path: SQLite database file (created if absent).
        crash_hook: Test hook ``hook(line_index)`` invoked before each
            batch commit; raising from it simulates a mid-run crash.
        """
        self.config = config
        self.db_path = db_path
        self.crash_hook = crash_hook
        self.store = SQLiteStore(db_path)
        identity = scan_input(config.input_path)
        if self.store.is_fresh:
            probe = config.new_lsh()
            self.store.initialize(
                params_json=config.to_json(),
                fingerprint=config.fingerprint,
                input_identity=identity,
                lsh_b=probe.b,
                lsh_r=probe.r,
            )
            self.lsh = probe
        else:
            self._verify_resume(identity)
            self.lsh = self._rebuild_lsh()
        self.identity = identity
        self.stats = RunStats(resumed_from_line=self.store.progress()["last_committed_line"])

    # -- resume verification ---------------------------------------------

    def _verify_resume(self, identity: dict) -> None:
        meta = self.store.meta()
        if meta.get("fingerprint") != self.config.fingerprint:
            raise ParameterMismatchError(
                "Stored run parameters differ from the requested ones. "
                f"stored_fingerprint={meta.get('fingerprint')} "
                f"requested_fingerprint={self.config.fingerprint}. "
                "Use a new database for different parameters or input."
            )
        # Fingerprint covers params + input abs path; additionally verify the
        # file's content identity so an overwritten same-path file is caught.
        for key, identity_key in (
            ("input_size", "size"),
            ("input_sha256", "sha256"),
            ("input_total_lines", "total_lines"),
        ):
            if meta.get(key) != str(identity[identity_key]):
                raise ParameterMismatchError(
                    f"Input identity field {key} changed: stored={meta.get(key)!r} current={identity[identity_key]!r}"
                )
        probe = self.config.new_lsh()
        if (int(meta["lsh_b"]), int(meta["lsh_r"])) != (probe.b, probe.r):
            raise ParameterMismatchError("Effective LSH (b, r) layout differs from the stored run")

    def _rebuild_lsh(self) -> "object":
        """Reconstruct the persistent index solely from committed accepted
        records — no uncommitted key can survive a restart.
        """
        lsh = self.config.new_lsh()
        n = 0
        for line_index, tokens in self.store.iter_accepted():
            lsh.insert(line_index, self.config.minhash_for_tokens(tokens))
            n += 1
        if n != self.store.lsh_key_count():
            raise RuntimeError(f"LSH rebuild count disagrees with records table ({n} vs {self.store.lsh_key_count()})")
        return lsh

    # -- per-batch processing --------------------------------------------

    def _parse_line(self, raw: bytes, line_index: int) -> list[str]:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"line {line_index} is not valid UTF-8") from exc
        stripped = text.strip()
        if not stripped:
            raise ValueError(f"line {line_index} is blank")
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_index} is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"line {line_index}: top-level JSON must be an object")
        tokens = obj.get(self.config.tokens_key)
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError(f"line {line_index}: field {self.config.tokens_key!r} must be an array of strings")
        return tokens

    def _judge_batch(self, raw_items: list[tuple[int, int, bytes]]) -> list[DuplicateVerdict]:
        """Evaluate a batch without touching the persistent LSH."""
        cfg = self.config
        # Ephemeral index for keys accepted earlier inside this batch.
        batch_lsh = cfg.new_lsh()
        batch_tokens: dict[int, list[str]] = {}
        verdicts: list[DuplicateVerdict] = []

        for line_index, byte_offset, raw in raw_items:
            tokens = self._parse_line(raw, line_index)
            # Jaccard is a set measure; keep the canonical set representation
            # (order-preserving) for hashing, verification and storage.
            unique = list(dict.fromkeys(tokens))
            mh = cfg.minhash_for_tokens(unique)
            evidence: list[dict] = []
            if unique:
                committed_keys = [int(k) for k in self.lsh.query(mh) if int(k) != line_index]
                batch_keys = [int(k) for k in batch_lsh.query(mh) if int(k) != line_index]
                candidate_map: dict[int, str] = {}
                for k in committed_keys:
                    candidate_map[k] = "committed"
                for k in batch_keys:
                    candidate_map.setdefault(k, "batch")
                token_map = self.store.get_tokens_for_keys(
                    [k for k, src in candidate_map.items() if src == "committed"]
                )
                token_map.update(batch_tokens)
                scored = []
                for cand, source in candidate_map.items():
                    jacc = exact_jaccard(unique, token_map[cand])
                    scored.append((cand, source, jacc))
                # Rank: strongest evidence first, earliest source line as
                # deterministic tiebreak.
                scored.sort(key=lambda x: (-x[2], x[0]))
                qualified = [s for s in scored if s[2] >= cfg.jaccard_threshold]
                evidence = [
                    {
                        "candidate_line_index": cand,
                        "source": source,
                        "jaccard": jacc,
                        "qualified": jacc >= cfg.jaccard_threshold,
                        "rank": rank,
                    }
                    for rank, (cand, source, jacc) in enumerate(scored)
                ]
                if qualified:
                    # Every qualified candidate is accepted by construction;
                    # earliest source line wins as the canonical duplicate.
                    dup_of = min(cand for cand, _, _ in qualified)
                    best = max(j for _, _, j in qualified)
                    verdicts.append(
                        DuplicateVerdict(
                            line_index=line_index,
                            byte_offset=byte_offset,
                            decision="duplicate",
                            duplicate_of=dup_of,
                            best_jaccard=best,
                            tokens=unique,
                            evidence=evidence,
                        )
                    )
                    continue
            # Accepted: empty token sets can't collide in LSH and are kept.
            batch_lsh.insert(line_index, mh)
            batch_tokens[line_index] = unique
            verdicts.append(
                DuplicateVerdict(
                    line_index=line_index,
                    byte_offset=byte_offset,
                    decision="accepted",
                    duplicate_of=None,
                    best_jaccard=None,
                    tokens=unique,
                    evidence=evidence,
                )
            )
        return verdicts

    def _commit_verdict(self, verdicts: list[DuplicateVerdict], raw_items: list[tuple[int, int, bytes]]) -> None:
        store = self.store
        first_index = verdicts[0].line_index
        store.begin_batch(first_index)
        try:
            raw_by_index = {idx: raw for idx, _, raw in raw_items}
            for v in verdicts:
                store.add_record(
                    line_index=v.line_index,
                    byte_offset=v.byte_offset,
                    raw_line=raw_by_index[v.line_index].decode("utf-8"),
                    tokens_json=json.dumps(v.tokens, ensure_ascii=False),
                    decision=v.decision,
                    duplicate_of=v.duplicate_of,
                    best_jaccard=v.best_jaccard,
                )
                for e in v.evidence:
                    store.add_evidence(
                        line_index=v.line_index,
                        candidate_line_index=e["candidate_line_index"],
                        source=e["source"],
                        jaccard=e["jaccard"],
                        qualified=e["qualified"],
                        rank=e["rank"],
                    )
            last_index = verdicts[-1].line_index
            last_offset = verdicts[-1].byte_offset + len(raw_by_index[last_index])
            lines_committed = last_index + 1
            if self.crash_hook is not None:
                self.crash_hook(last_index)
            store.commit_batch(last_index, last_offset, lines_committed)
        except BaseException:
            store.rollback_batch()
            raise
        # Durable commit succeeded: now the index may learn the new keys.
        for v in verdicts:
            if v.decision == "accepted":
                if v.line_index in self.lsh:
                    raise RuntimeError(
                        f"key {v.line_index} already present in LSH; aborting to protect index integrity"
                    )
                tokens = self.store.get_tokens(v.line_index)
                self.lsh.insert(v.line_index, self.config.minhash_for_tokens(tokens))

    # -- main loop --------------------------------------------------------

    def run(self) -> RunStats:
        """Process until the input is exhausted. Safe to call again after a
        crash with a freshly opened :class:`Deduplicator`.
        """
        if self.store.is_finished():
            return self.stats
        cfg = self.config
        prog = self.store.progress()
        skip_lines = prog["lines_committed"]
        skip_offset = prog["last_committed_offset"]

        batch: list[tuple[int, int, bytes]] = []
        with open(cfg.input_path, "rb") as fh:
            if skip_offset:
                fh.seek(skip_offset)
            line_index = skip_lines
            # Defensive offset check: the committed offset must land on a
            # line boundary, proving the input's byte layout still matches
            # the run that produced the progress row.
            if skip_offset > 0:
                fh.seek(skip_offset - 1)
                preceding = fh.read(1)
                fh.seek(skip_offset)
                if preceding != b"\n":
                    raise ParameterMismatchError(
                        f"Committed byte offset {skip_offset} is not at a line boundary of the current input file"
                    )
            while True:
                offset = fh.tell()
                raw = fh.readline()
                if not raw:
                    break
                batch.append((line_index, offset, raw))
                line_index += 1
                if len(batch) >= cfg.batch_size:
                    self._process_batch(batch)
                    batch.clear()
        if batch:
            self._process_batch(batch)
            batch.clear()

        # Sanity: every input line must now have a durable decision.
        final = self.store.progress()
        if final["lines_committed"] != self.identity["total_lines"]:
            raise RuntimeError(
                f"Committed {final['lines_committed']} lines but input has {self.identity['total_lines']}"
            )
        self.store.mark_finished()
        self._refresh_stats()
        return self.stats

    def _process_batch(self, raw_items: list[tuple[int, int, bytes]]) -> None:
        verdicts = self._judge_batch(raw_items)
        self._commit_verdict(verdicts, raw_items)
        self.stats.batches_committed += 1
        self.stats.lines_processed += len(verdicts)
        self.stats.accepted += sum(v.decision == "accepted" for v in verdicts)
        self.stats.duplicate += sum(v.decision == "duplicate" for v in verdicts)

    def _refresh_stats(self) -> None:
        counts = self.store.counts()
        self.stats.accepted = counts["accepted"]
        self.stats.duplicate = counts["duplicate"]
        self.stats.lines_processed = counts["total"]

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Deduplicator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
