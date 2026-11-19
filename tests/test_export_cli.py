"""Tests for JSONL export, evidence dump and the command-line interface."""

from __future__ import annotations

import io
import json
import subprocess
import sys

from near_dup_dedup.cli import main
from near_dup_dedup.dedup import Deduplicator
from near_dup_dedup.export import DEFAULT_ANNOTATION_KEY, export_evidence
from near_dup_dedup.storage import SQLiteStore


def _run_once(input_path, db, batch_size=2):
    from near_dup_dedup.config import DedupConfig

    cfg = DedupConfig(
        input_path=input_path,
        num_perm=64,
        lsh_threshold=0.8,
        jaccard_threshold=0.9,
        batch_size=batch_size,
    )
    with Deduplicator(cfg, db) as dedup:
        dedup.run()
    return cfg


def test_export_roundtrip_preserves_records(tmp_paths, write_jsonl, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    _run_once(tmp_paths["input"], tmp_paths["db"])
    with SQLiteStore(tmp_paths["db"]) as store:
        from near_dup_dedup.export import export_jsonl

        n_all = export_jsonl(store, tmp_paths["out"], mode="all")
        accepted_path = tmp_paths["out"] + ".kept"
        n_kept = export_jsonl(store, accepted_path, mode="accepted")
        dup_path = tmp_paths["out"] + ".dup"
        n_dup = export_jsonl(store, dup_path, mode="duplicate")

    assert n_all == len(corpus)
    with open(tmp_paths["out"], encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    # Original payload survives untouched alongside the annotation.
    assert lines[0]["tokens"] == corpus[0]
    assert lines[0][DEFAULT_ANNOTATION_KEY]["decision"] == "accepted"
    assert lines[2][DEFAULT_ANNOTATION_KEY]["duplicate_of"] == 0
    assert lines[2][DEFAULT_ANNOTATION_KEY]["best_jaccard"] == 1.0
    assert n_kept + n_dup == n_all

    # The "accepted" export is the cleaned stream: no duplicate records.
    with open(accepted_path, encoding="utf-8") as fh:
        kept = [json.loads(line) for line in fh]
    assert all(row[DEFAULT_ANNOTATION_KEY]["decision"] == "accepted" for row in kept)


def test_evidence_dump(tmp_paths, write_jsonl, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    _run_once(tmp_paths["input"], tmp_paths["db"])
    buf = io.StringIO()
    with SQLiteStore(tmp_paths["db"]) as store:
        n = export_evidence(store, buf)
    rows = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert n == len(rows)
    assert n > 0
    for row in rows:
        assert row["duplicate_of"] is not None
        assert row["evidence"]
        assert all("jaccard" in e and "source" in e for e in row["evidence"])


def test_cli_run_status_verify_export(tmp_paths, write_jsonl, corpus, capsys):
    write_jsonl(tmp_paths["input"], corpus)
    common = [
        "--input",
        tmp_paths["input"],
        "--db",
        tmp_paths["db"],
        "--num-perm",
        "64",
        "--lsh-threshold",
        "0.8",
        "--jaccard-threshold",
        "0.9",
        "--batch-size",
        "2",
    ]
    assert main(["run", *common]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["finished"] is True
    assert out["accepted"] + out["duplicate"] == len(corpus)

    assert main(["status", "--db", tmp_paths["db"]]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["lines_committed"] == len(corpus)
    assert status["finished"] is True

    assert (
        main(
            [
                "verify",
                "--db",
                tmp_paths["db"],
                "--input",
                tmp_paths["input"],
                "--num-perm",
                "64",
                "--lsh-threshold",
                "0.8",
                "--jaccard-threshold",
                "0.9",
                "--batch-size",
                "2",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True

    assert main(["export", "--db", tmp_paths["db"], "--output", tmp_paths["out"], "--mode", "all"]) == 0
    with open(tmp_paths["out"], encoding="utf-8") as fh:
        assert sum(1 for _ in fh) == len(corpus)


def test_cli_rejects_changed_parameters(tmp_paths, write_jsonl, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    base = [
        "--input",
        tmp_paths["input"],
        "--db",
        tmp_paths["db"],
        "--num-perm",
        "64",
        "--lsh-threshold",
        "0.8",
        "--jaccard-threshold",
        "0.9",
        "--batch-size",
        "2",
    ]
    assert main(["run", *base]) == 0
    rc = main(["run", *base, "--jaccard-threshold", "0.5"])
    assert rc == 2


def test_module_invocation(tmp_paths, write_jsonl, corpus):
    """``python -m near_dup_dedup`` works with the repo root as cwd.

    ``-s`` keeps user-site isolation without dropping cwd from sys.path
    (``-I`` would also ignore PYTHONPATH).
    """
    write_jsonl(tmp_paths["input"], corpus)
    proc = subprocess.run(
        [sys.executable, "-s", "-m", "near_dup_dedup", "status", "--db", tmp_paths["db"]],
        capture_output=True,
        text=True,
        cwd="/workspace",
    )
    # No database tables yet -> the CLI fails cleanly with exit 2.
    assert proc.returncode == 2
    assert "not an initialized" in proc.stderr
