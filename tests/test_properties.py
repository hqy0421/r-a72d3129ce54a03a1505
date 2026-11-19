"""Property-level tests: recall guarantees and scale/stability."""

from __future__ import annotations

import json
import random

from near_dup_dedup.dedup import Deduplicator, exact_jaccard
from near_dup_dedup.storage import SQLiteStore
from near_dup_dedup.verify import verify


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for tokens in rows:
            fh.write(json.dumps({"tokens": tokens}))
            fh.write("\n")


def _cluster_corpus(seed=7):
    """Build clusters with exact copies and near-duplicate mutations."""
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(400)]
    rows = []
    exact_groups = []  # (canonical_index, [exact_copy_indices])
    for _ in range(12):
        canonical = rng.sample(vocab, 40)
        base_idx = len(rows)
        rows.append(list(canonical))
        copies = []
        for _ in range(rng.randint(1, 3)):
            copies.append(len(rows))
            rows.append(list(canonical))  # exact copy, shuffled later
        # Near duplicates: replace 1-3 tokens (~92-97% Jaccard).
        for _ in range(2):
            mutated = list(canonical)
            for pos in rng.sample(range(40), rng.randint(1, 3)):
                mutated[pos] = rng.choice([w for w in vocab if w not in canonical])
            rows.append(mutated)
        # Unrelated record between clusters.
        rows.append(rng.sample(vocab, 40))
        exact_groups.append((base_idx, copies))
    rng.shuffle(rows)  # order shuffling is handled by line indices below
    return rows, exact_groups


def test_exact_duplicates_always_recalled(tmp_path, make_config):
    """Identical token sets have identical MinHash signatures and collide in
    every band, so exact-duplicate recall must be 1.0 regardless of the LSH
    threshold's approximate behavior on near duplicates.
    """
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(600)]
    input_path = str(tmp_path / "in.jsonl")
    db = str(tmp_path / "state.db")

    # Emit each cluster as: canonical first, then its exact copies (which
    # must chain to the canonical), then near-duplicate mutants, then an
    # unrelated spacer.
    rows = []
    expected = {}
    for _ in range(12):
        canonical = rng.sample(vocab, 40)
        canonical_idx = len(rows)
        rows.append(list(canonical))
        expected[canonical_idx] = "accepted"
        for _ in range(rng.randint(1, 3)):
            copy_idx = len(rows)
            shuffled = list(canonical)
            rng.shuffle(shuffled)  # order must not matter, only the set
            rows.append(shuffled)
            expected[copy_idx] = canonical_idx
        for _ in range(2):
            mutated = list(canonical)
            for pos in rng.sample(range(40), rng.randint(1, 3)):
                mutated[pos] = rng.choice([w for w in vocab if w not in canonical])
            rows.append(mutated)  # verdict depends on exact Jaccard
        rows.append(rng.sample(vocab, 40))
    _write(input_path, rows)

    cfg = make_config(input_path, num_perm=32, lsh_threshold=0.9, jaccard_threshold=0.9, batch_size=16)
    with Deduplicator(cfg, db) as dedup:
        dedup.run()
        records = {r["line_index"]: r for r in dedup.store.iter_records()}

    for idx, want in expected.items():
        rec = records[idx]
        if want == "accepted":
            assert rec["decision"] == "accepted"
        else:
            assert rec["decision"] == "duplicate"
            assert rec["duplicate_of"] == want
            assert rec["best_jaccard"] == 1.0


def test_no_false_positives_under_jaccard_threshold(tmp_path, make_config):
    """Every reported duplicate must clear the exact Jaccard threshold;
    LSH only supplies candidates and can never cause a false verdict.
    """
    input_path = str(tmp_path / "in.jsonl")
    db = str(tmp_path / "state.db")
    rows, _ = _cluster_corpus(seed=11)
    _write(input_path, rows)
    cfg = make_config(input_path, num_perm=48, lsh_threshold=0.7, jaccard_threshold=0.9, batch_size=10)
    with Deduplicator(cfg, db) as dedup:
        dedup.run()
        records = list(dedup.store.iter_records())
        token_sets = {r["line_index"]: set(json.loads(r["tokens_json"])) for r in records}
    for r in records:
        if r["decision"] == "duplicate":
            jacc = exact_jaccard(
                list(token_sets[r["line_index"]]),
                list(token_sets[r["duplicate_of"]]),
            )
            assert jacc >= 0.9
            assert r["best_jaccard"] >= 0.9


def test_larger_run_with_intermittent_crashes(tmp_path, make_config):
    rng = random.Random(3)
    vocab = [f"tok{i}" for i in range(300)]
    rows = []
    for _ in range(600):
        n = rng.randint(5, 60)
        rows.append(rng.sample(vocab, n))
        if rng.random() < 0.25:  # sprinkle exact duplicates
            rows.append(list(rows[-1]))
    input_path = str(tmp_path / "in.jsonl")
    db = str(tmp_path / "state.db")
    _write(input_path, rows)
    cfg = make_config(input_path, num_perm=64, lsh_threshold=0.8, jaccard_threshold=0.9, batch_size=64)

    crash_budget = {500}  # crash once mid-run

    def hook(line_index):
        if line_index in crash_budget:
            crash_budget.discard(line_index)
            raise RuntimeError("simulated crash")

    stats = None
    try:
        with Deduplicator(cfg, db, crash_hook=hook) as dedup:
            stats = dedup.run()
    except RuntimeError:
        with Deduplicator(cfg, db) as dedup:
            stats = dedup.run()
    assert stats is not None

    with SQLiteStore(db) as store:
        assert store.progress()["lines_committed"] == len(rows)
        report = verify(store, cfg)
        assert report.ok, report.errors
        counts = store.counts()
        assert counts["total"] == len(rows)
        # Rebuilt index keys equal the accepted set exactly.
        accepted = {r["line_index"] for r in store.iter_records() if r["decision"] == "accepted"}
    with Deduplicator(cfg, db) as dedup:
        assert set(dedup.lsh.keys.keys()) == accepted
        assert stats.accepted == len(accepted)
