"""Command-line interface for resumable near-duplicate cleaning.

Subcommands:
  run       Process (or resume) a JSONL input into a SQLite state database.
  status    Show committed progress and decision counts.
  verify    Re-derive every decision from the input and stored evidence.
  export    Emit annotated JSONL (all / accepted / duplicate records).
  evidence  Emit the full LSH-candidate + exact-Jaccard audit trail.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_LSH_THRESHOLD,
    DEFAULT_NUM_PERM,
    DEFAULT_SEED,
    DEFAULT_TOKENS_KEY,
    DedupConfig,
    ParameterMismatchError,
)
from .dedup import Deduplicator
from .export import DEFAULT_ANNOTATION_KEY, MODES, export_evidence, export_jsonl
from .storage import SQLiteStore
from .verify import verify


def _build_config(args: argparse.Namespace) -> DedupConfig:
    lsh_params = None
    if args.bands is not None:
        if args.rows is None:
            raise SystemExit("--bands requires --rows")
        lsh_params = (args.bands, args.rows)
    return DedupConfig(
        input_path=args.input,
        tokens_key=args.tokens_key,
        jaccard_threshold=args.jaccard_threshold,
        num_perm=args.num_perm,
        lsh_threshold=args.lsh_threshold,
        seed=args.seed,
        batch_size=args.batch_size,
        lsh_params=lsh_params,
    )


def _add_common_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True, help="JSONL input file")
    p.add_argument("--db", required=True, help="SQLite state database path")
    p.add_argument("--tokens-key", default=DEFAULT_TOKENS_KEY, help="name of the string-array field (default: tokens)")
    p.add_argument(
        "--jaccard-threshold",
        type=float,
        default=DEFAULT_JACCARD_THRESHOLD,
        help="exact Jaccard confirm threshold (default: 0.9)",
    )
    p.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM, help="MinHash permutations (default: 128)")
    p.add_argument(
        "--lsh-threshold",
        type=float,
        default=DEFAULT_LSH_THRESHOLD,
        help="MinHashLSH candidate threshold (default: 0.9)",
    )
    p.add_argument("--bands", type=int, default=None, help="optional explicit LSH band count (with --rows)")
    p.add_argument("--rows", type=int, default=None, help="optional explicit LSH rows per band (with --bands)")
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="MinHash seed, must stay fixed on resume (default: 1)"
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="lines per committed transaction (default: 1000)"
    )


def _cmd_run(args: argparse.Namespace) -> int:
    config = _build_config(args)
    try:
        with Deduplicator(config, args.db) as dedup:
            stats = dedup.run()
            payload = {
                "db": args.db,
                "input": args.input,
                "lines_processed": stats.lines_processed,
                "accepted": stats.accepted,
                "duplicate": stats.duplicate,
                "batches_committed": stats.batches_committed,
                "resumed_from_line": stats.resumed_from_line,
                "finished": dedup.store.is_finished(),
            }
    except ParameterMismatchError as exc:
        print(f"error: refusing to resume: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        meta = store.meta()
        if not meta:
            print(f"error: {args.db} is not an initialized dedup database", file=sys.stderr)
            return 2
        counts = store.counts()
        prog = store.progress()
        payload = {
            "db": args.db,
            "input": meta.get("input_abs_path"),
            "input_total_lines": int(meta["input_total_lines"]),
            "lines_committed": prog["lines_committed"],
            "last_committed_line": prog["last_committed_line"],
            "accepted": counts["accepted"],
            "duplicate": counts["duplicate"],
            "lsh_b": int(meta["lsh_b"]),
            "lsh_r": int(meta["lsh_r"]),
            "fingerprint": meta["fingerprint"],
            "finished": store.is_finished(),
        }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        config = None
        if args.input:
            config = _build_config(args)
        report = verify(store, config)
    print(
        json.dumps(
            {
                "ok": report.ok,
                "checks_run": report.checks_run,
                "accepted": report.accepted,
                "duplicate": report.duplicate,
                "errors": report.errors,
            },
            indent=2,
        )
    )
    return 0 if report.ok else 1


def _cmd_export(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        n = export_jsonl(store, args.output, mode=args.mode, annotation_key=args.annotation_key)
    print(json.dumps({"output": args.output, "mode": args.mode, "lines": n}, indent=2))
    return 0


def _cmd_evidence(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        n = export_evidence(store, sys.stdout)
    print(json.dumps({"duplicate_records": n}), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="near-dup-dedup",
        description="Resumable LSH-candidate + exact-Jaccard near-duplicate cleaning with SQLite-backed state.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="process or resume a run")
    _add_common_input_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="show committed state")
    p_status.add_argument("--db", required=True)
    p_status.set_defaults(func=_cmd_status)

    p_verify = sub.add_parser("verify", help="verify committed decisions against the input")
    p_verify.add_argument("--db", required=True)
    # Input/parameter args are optional for verify: the stored params are
    # used when the CLI flags are omitted.
    p_verify.add_argument("--input", default=None)
    p_verify.add_argument("--tokens-key", default=DEFAULT_TOKENS_KEY)
    p_verify.add_argument("--jaccard-threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD)
    p_verify.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    p_verify.add_argument("--lsh-threshold", type=float, default=DEFAULT_LSH_THRESHOLD)
    p_verify.add_argument("--bands", type=int, default=None)
    p_verify.add_argument("--rows", type=int, default=None)
    p_verify.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_verify.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_verify.set_defaults(func=_cmd_verify)

    p_export = sub.add_parser("export", help="export annotated JSONL")
    p_export.add_argument("--db", required=True)
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--mode", choices=MODES, default="all")
    p_export.add_argument("--annotation-key", default=DEFAULT_ANNOTATION_KEY)
    p_export.set_defaults(func=_cmd_export)

    p_ev = sub.add_parser("evidence", help="dump duplicate evidence trail as JSONL")
    p_ev.add_argument("--db", required=True)
    p_ev.set_defaults(func=_cmd_evidence)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
