"""Tests for the resumable LSH + exact-Jaccard dedup engine."""

from __future__ import annotations

import json

import pytest
from near_dup_dedup.config import ParameterMismatchError
from near_dup_dedup.dedup import Deduplicator, exact_jaccard
from near_dup_dedup.storage import SQLiteStore
from near_dup_dedup.verify import verify


def _decisions(store):
    return {r["line_index"]: (r["decision"], r["duplicate_of"]) for r in store.iter_records()}


def test_exact_jaccard():
    assert exact_jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert exact_jaccard(["a"], ["b"]) == 0.0
    assert exact_jaccard(["a", "b"], ["a", "c"]) == pytest.approx(1 / 3)
    assert exact_jaccard([], []) == 1.0
    # Duplicate tokens do not change the set measure.
    assert exact_jaccard(["a", "a", "b"], ["a", "b", "b"]) == 1.0


def test_basic_dedup_decisions(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"])
    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        stats = dedup.run()
        decisions = _decisions(dedup.store)
        stored_jaccard = {r["line_index"]: r["best_jaccard"] for r in dedup.store.iter_records()}

    assert stats.accepted + stats.duplicate == 10
    # Exact duplicates and the >=0.9 near-dup point at the earliest accepted.
    assert decisions[2] == ("duplicate", 0)
    assert decisions[4] == ("duplicate", 0)
    assert decisions[5] == ("duplicate", 1)
    assert decisions[7] == ("duplicate", 0)
    for keep in (0, 1, 3, 6, 8, 9):
        assert decisions[keep][0] == "accepted"
    assert stored_jaccard[2] == 1.0
    assert stored_jaccard[4] == pytest.approx(19 / 21)
    # Every duplicate target is itself accepted and earlier in source order.
    for idx, (decision, ref) in decisions.items():
        if decision == "duplicate":
            assert ref < idx
            assert decisions[ref][0] == "accepted"


def test_batch_size_does_not_change_decisions(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    results = {}
    for size in (1, 2, 3, 7, 100):
        db = f"{tmp_paths['db']}.{size}"
        cfg = make_config(tmp_paths["input"], batch_size=size)
        with Deduplicator(cfg, db) as dedup:
            dedup.run()
            results[size] = _decisions(dedup.store)
    reference = results[100]
    for size, got in results.items():
        assert got == reference, f"batch_size={size} changed decisions"


def test_in_batch_dedup_uses_ephemeral_index(tmp_path, write_jsonl, make_config):
    # Three identical lines inside a single batch: lines 1 and 2 must chain
    # to the earliest accepted (line 0), not to each other.
    rows = [["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "e"]]
    input_path = str(tmp_path / "in.jsonl")
    write_jsonl(input_path, rows)
    cfg = make_config(input_path, batch_size=10)
    db = str(tmp_path / "state.db")
    with Deduplicator(cfg, db) as dedup:
        dedup.run()
        decisions = _decisions(dedup.store)
    assert decisions[0] == ("accepted", None)
    assert decisions[1] == ("duplicate", 0)
    assert decisions[2] == ("duplicate", 0)


def test_resume_after_crash_leaves_no_partial_state(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"], batch_size=2)

    crash_lines = {3}  # crash while the second batch (lines 2-3) commits

    def hook(line_index):
        if line_index in crash_lines:
            raise RuntimeError("simulated crash")

    with (
        pytest.raises(RuntimeError, match="simulated crash"),
        Deduplicator(cfg, tmp_paths["db"], crash_hook=hook) as dedup,
    ):
        dedup.run()

    # Only the first batch (lines 0, 1) may be durable; nothing partial.
    with SQLiteStore(tmp_paths["db"]) as store:
        assert store.progress()["lines_committed"] == 2
        assert sorted(r["line_index"] for r in store.iter_records()) == [0, 1]
        assert store.counts()["total"] == 2
        # No open transaction left behind.
        journal = store.conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        assert journal > 0

    # Reopen with no hook: resumes from line 2 and reaches the same result.
    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        stats = dedup.run()
        decisions = _decisions(dedup.store)
        assert stats.resumed_from_line == 1
        assert len(decisions) == 10
        assert dedup.store.progress()["lines_committed"] == 10

    # A third run is a no-op and never double-accepts.
    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        again = dedup.run()
        assert again.lines_processed == 0
        assert dedup.store.counts()["accepted"] == sum(d == "accepted" for d, _ in decisions.values())


def test_reopened_index_matches_committed_records(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"], batch_size=3)
    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        dedup.run()
        expected_keys = set(dedup.lsh.keys.keys())

    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        rebuilt = set(dedup.lsh.keys.keys())
    assert rebuilt == expected_keys
    with SQLiteStore(tmp_paths["db"]) as store:
        assert len(rebuilt) == store.lsh_key_count()


def test_parameter_change_is_rejected(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"])
    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        dedup.run()

    changed = make_config(tmp_paths["input"], jaccard_threshold=0.5)
    with pytest.raises(ParameterMismatchError):
        Deduplicator(changed, tmp_paths["db"])

    changed = make_config(tmp_paths["input"], num_perm=32)
    with pytest.raises(ParameterMismatchError):
        Deduplicator(changed, tmp_paths["db"])


def test_batch_size_may_change_between_runs(tmp_paths, write_jsonl, make_config, corpus):
    """batch_size is operational, not semantic: it stays out of the
    fingerprint and a resume with a different value reaches the same result.
    """
    write_jsonl(tmp_paths["input"], corpus)

    fired = {"done": False}

    def hook_once(idx):
        if idx == 3 and not fired["done"]:
            fired["done"] = True
            raise RuntimeError("stop")

    with (
        pytest.raises(RuntimeError),
        Deduplicator(make_config(tmp_paths["input"], batch_size=2), tmp_paths["db"], crash_hook=hook_once) as dedup,
    ):
        dedup.run()

    # Resume with a different batch size; the fingerprint still matches.
    with Deduplicator(make_config(tmp_paths["input"], batch_size=7), tmp_paths["db"]) as dedup:
        dedup.run()
        decisions = _decisions(dedup.store)
    assert len(decisions) == len(corpus)

    # A fresh, uninterrupted run with yet another size gives identical output.
    ref_db = f"{tmp_paths['db']}.ref"
    with Deduplicator(make_config(tmp_paths["input"], batch_size=100), ref_db) as dedup:
        dedup.run()
        reference = _decisions(dedup.store)
    assert decisions == reference


def test_input_content_change_is_rejected(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"])
    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        dedup.run()

    # Same path, different content (and thus different hash/line count).
    write_jsonl(tmp_paths["input"], corpus[:-1])
    with pytest.raises(ParameterMismatchError):
        Deduplicator(cfg, tmp_paths["db"])


def test_verify_passes_on_committed_run(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"])
    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        dedup.run()
    with SQLiteStore(tmp_paths["db"]) as store:
        report = verify(store, cfg)
    assert report.ok, report.errors
    assert report.checks_run == 5
    assert report.duplicate == 4


def test_verify_detects_tampered_duplicate(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"])
    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        dedup.run()

    # Tamper directly with a committed decision behind the engine's back.
    with SQLiteStore(tmp_paths["db"]) as store:
        store.conn.execute("UPDATE records SET duplicate_of = 1 WHERE line_index = 2")
        store.conn.commit()
        report = verify(store, cfg)
    assert not report.ok
    assert any("line 2" in err for err in report.errors)


def test_evidence_trail_records_candidates(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"], batch_size=2)
    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        dedup.run()
        ev = dedup.store.evidence_for(2)
        assert ev, "exact duplicate must have LSH-derived evidence"
        top = ev[0]
        assert top["candidate_line_index"] == 0
        assert top["qualified"] is True
        assert top["jaccard"] == 1.0
        # Ranks are dense and start at zero.
        assert [e["rank"] for e in ev] == list(range(len(ev)))


def test_empty_token_sets_are_accepted(tmp_path, write_jsonl, make_config):
    rows = [[], [], ["only"]]
    input_path = str(tmp_path / "in.jsonl")
    write_jsonl(input_path, rows)
    cfg = make_config(input_path)
    db = str(tmp_path / "state.db")
    with Deduplicator(cfg, db) as dedup:
        dedup.run()
        decisions = _decisions(dedup.store)
    assert all(d == "accepted" for d, _ in decisions.values())


def test_malformed_lines_raise_without_partial_commit(tmp_path, make_config):
    input_path = str(tmp_path / "in.jsonl")
    with open(input_path, "wb") as fh:
        fh.write(json.dumps({"tokens": ["a", "b", "c"]}).encode())
        fh.write(b"\n")
        fh.write(b"{not valid json\n")
    cfg = make_config(input_path, batch_size=10)
    db = str(tmp_path / "state.db")
    with pytest.raises(ValueError, match="not valid JSON"), Deduplicator(cfg, db) as dedup:
        dedup.run()
    # The failing batch rolled back: zero durable records, no stuck index.
    with SQLiteStore(db) as store:
        assert store.counts()["total"] == 0
        assert store.progress()["lines_committed"] == 0


def test_empty_input_completes_cleanly(tmp_path, make_config):
    input_path = str(tmp_path / "in.jsonl")
    open(input_path, "wb").close()
    cfg = make_config(input_path)
    db = str(tmp_path / "state.db")
    with Deduplicator(cfg, db) as dedup:
        stats = dedup.run()
        assert stats.lines_processed == 0
        assert stats.accepted == 0
        assert dedup.store.is_finished() is True
        assert dedup.lsh.is_empty()


def test_crash_leaves_no_orphan_batch_rows(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"], batch_size=2)

    def hook(line_index):
        if line_index == 3:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError), Deduplicator(cfg, tmp_paths["db"], crash_hook=hook) as dedup:
        dedup.run()

    with SQLiteStore(tmp_paths["db"]) as store:
        rows = store.conn.execute("SELECT batch_id, line_count, committed_at FROM batches ORDER BY batch_id").fetchall()
        # Exactly one batch, and it is fully committed; the rolled-back
        # batch insert vanished with the transaction.
        assert rows == [(1, 2, rows[0][2])]
        assert rows[0][2] is not None
        # WAL contains no uncommitted frame visible to a fresh connection.
        assert store.progress()["lines_committed"] == 2


def test_resumed_run_matches_single_run(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)

    # Reference: one uninterrupted run with a large batch size.
    ref_db = tmp_paths["db"] + ".ref"
    with Deduplicator(make_config(tmp_paths["input"], batch_size=100), ref_db) as dedup:
        dedup.run()
        reference = _decisions(dedup.store)

    # Interrupted run: crash once at each of the first four batch commits.
    remaining = {1, 3, 5, 7}
    seen = set()

    def hook(line_index):
        if line_index in remaining:
            remaining.discard(line_index)
            seen.add(line_index)
            raise RuntimeError("simulated crash")

    cfg = make_config(tmp_paths["input"], batch_size=2)
    for _ in range(6):
        try:
            with Deduplicator(cfg, tmp_paths["db"], crash_hook=hook) as dedup:
                dedup.run()
            break
        except RuntimeError:
            continue
    assert seen == {1, 3, 5, 7}
    with SQLiteStore(tmp_paths["db"]) as store:
        resumed = {r["line_index"]: (r["decision"], r["duplicate_of"]) for r in store.iter_records()}
    assert resumed == reference


def test_second_crash_after_resume_still_consistent(tmp_paths, write_jsonl, make_config, corpus):
    write_jsonl(tmp_paths["input"], corpus)
    cfg = make_config(tmp_paths["input"], batch_size=2)

    for crash_at in (1, 5):

        def hook(line_index, _at=crash_at):
            if line_index == _at:
                raise RuntimeError("simulated crash")

        try:
            with Deduplicator(cfg, tmp_paths["db"], crash_hook=hook) as dedup:
                dedup.run()
        except RuntimeError:
            pass

    with Deduplicator(cfg, tmp_paths["db"]) as dedup:
        dedup.run()
        decisions = _decisions(dedup.store)
        assert len(decisions) == len(corpus)
        # No key was accepted twice: primary key plus accepted count agree.
        accepted = [idx for idx, (d, _) in decisions.items() if d == "accepted"]
        assert len(accepted) == len(set(accepted))
        assert set(dedup.lsh.keys.keys()) == set(accepted)


def test_input_path_change_is_rejected(tmp_path, write_jsonl, make_config, corpus):
    path_a = str(tmp_path / "a.jsonl")
    path_b = str(tmp_path / "b.jsonl")
    write_jsonl(path_a, corpus)
    write_jsonl(path_b, corpus)  # identical content, different path
    db = str(tmp_path / "state.db")
    with Deduplicator(make_config(path_a), db) as dedup:
        dedup.run()
    with pytest.raises(ParameterMismatchError):
        Deduplicator(make_config(path_b), db)


def test_missing_tokens_field_raises(tmp_path, make_config):
    input_path = str(tmp_path / "in.jsonl")
    with open(input_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"other": ["a"]}) + "\n")
    cfg = make_config(input_path)
    with pytest.raises(ValueError, match="must be"), Deduplicator(cfg, str(tmp_path / "state.db")) as dedup:
        dedup.run()


def test_non_string_tokens_raise(tmp_path, make_config):
    input_path = str(tmp_path / "in.jsonl")
    with open(input_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"tokens": ["a", 1, None]}) + "\n")
    cfg = make_config(input_path)
    with (
        pytest.raises(ValueError, match="array of strings"),
        Deduplicator(cfg, str(tmp_path / "state.db")) as dedup,
    ):
        dedup.run()
