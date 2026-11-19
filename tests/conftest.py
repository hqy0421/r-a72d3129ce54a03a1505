"""Shared pytest fixtures for the resumable near-duplicate cleaner."""

from __future__ import annotations

import json
import os
import sys

import pytest

# Make the in-repo package importable without installing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from near_dup_dedup.config import DedupConfig


@pytest.fixture()
def tmp_paths(tmp_path):
    return {
        "input": str(tmp_path / "in.jsonl"),
        "db": str(tmp_path / "state.db"),
        "out": str(tmp_path / "out.jsonl"),
    }


@pytest.fixture()
def write_jsonl():
    def _write(path, rows):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for i, tokens in enumerate(rows):
                fh.write(json.dumps({"id": i, "tokens": tokens}, ensure_ascii=False))
                fh.write("\n")

    return _write


@pytest.fixture()
def make_config():
    def _make(input_path, **over):
        defaults = dict(
            num_perm=64,
            lsh_threshold=0.8,
            jaccard_threshold=0.9,
            batch_size=2,
            seed=1,
        )
        defaults.update(over)
        return DedupConfig(input_path=input_path, **defaults)

    return _make


@pytest.fixture()
def corpus():
    """10 lines with known exact-Jaccard relationships."""
    base = [f"t{i}" for i in range(20)]
    return [
        list(base),  # 0: canonical A
        ["x", "y", "z", "q"],  # 1: unrelated B
        list(base),  # 2: exact dup of 0
        [*base[:18], "new1", "new2"],  # 3: 18/22 = 0.818 keep
        [*base[:19], "k"],  # 4: 19/21 = 0.905 dup of 0
        ["x", "y", "z", "q"],  # 5: exact dup of 1
        ["x", "y", "z", "q", "r"],  # 6: 4/5 = 0.8 keep
        list(dict.fromkeys(base + base[:5])),  # 7: same set as 0
        ["p", "q", "r", "s", "t", "u"],  # 8: unrelated C
        ["p", "q", "r", "s", "t", "u", "v"],  # 9: 6/7 = 0.857 keep
    ]
