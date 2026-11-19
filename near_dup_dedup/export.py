"""Export committed dedup state back to JSONL.

The original records are reconstructed from the durable ``records.raw_line``
column rather than re-reading the input file, so an export is a faithful view
of what was actually committed.  Each line gains a small annotation block:

    {"...original fields...", "<annotation_key>": {
        "line_index": 12,
        "decision": "duplicate",
        "duplicate_of": 3,
        "best_jaccard": 0.923,
        "qualified_candidates": [3, 7]
    }}

All records are retained by default (the cleaning decision is metadata);
``mode="accepted"`` reproduces the deduped stream and ``mode="duplicate"``
isolates the removed records.
"""

from __future__ import annotations

import json
from typing import Iterator, TextIO

from .storage import SQLiteStore

MODES = ("all", "accepted", "duplicate")
DEFAULT_ANNOTATION_KEY = "_dedup"


def _annotated_lines(
    store: SQLiteStore,
    mode: str,
    annotation_key: str,
) -> Iterator[str]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    for rec in store.iter_records():
        if mode == "accepted" and rec["decision"] != "accepted":
            continue
        if mode == "duplicate" and rec["decision"] != "duplicate":
            continue
        obj = json.loads(rec["raw_line"])
        if annotation_key in obj:
            raise ValueError(
                f"line {rec['line_index']}: annotation key "
                f"{annotation_key!r} already exists on the source record; "
                "choose a different --annotation-key"
            )
        qualified = [e["candidate_line_index"] for e in store.evidence_for(rec["line_index"]) if e["qualified"]]
        obj[annotation_key] = {
            "line_index": rec["line_index"],
            "decision": rec["decision"],
            "duplicate_of": rec["duplicate_of"],
            "best_jaccard": rec["best_jaccard"],
            "qualified_candidates": qualified,
        }
        yield json.dumps(obj, ensure_ascii=False)


def export_jsonl(
    store: SQLiteStore,
    out_path: str,
    mode: str = "all",
    annotation_key: str = DEFAULT_ANNOTATION_KEY,
) -> int:
    """Write annotated JSONL to ``out_path``; return the number of lines."""
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for line in _annotated_lines(store, mode, annotation_key):
            fh.write(line)
            fh.write("\n")
            n += 1
    return n


def export_evidence(store: SQLiteStore, out: TextIO) -> int:
    """Write the full candidate/verification trail as JSONL to ``out``."""
    n = 0
    for rec in store.iter_records():
        if rec["decision"] != "duplicate":
            continue
        out.write(
            json.dumps(
                {
                    "line_index": rec["line_index"],
                    "duplicate_of": rec["duplicate_of"],
                    "best_jaccard": rec["best_jaccard"],
                    "evidence": store.evidence_for(rec["line_index"]),
                },
                ensure_ascii=False,
            )
        )
        out.write("\n")
        n += 1
    return n
