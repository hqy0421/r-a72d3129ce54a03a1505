"""Resumable near-duplicate cleaning of JSONL token arrays.

Pipeline:
  1. datasketch :class:`~datasketch.MinHashLSH` produces candidate pairs
     (approximate recall, inherited from LSH).
  2. Exact Jaccard similarity over the *original* token sets confirms a
     candidate before anything is marked duplicate.
  3. Every decision, accepted record and source progress is durably written
     to SQLite *before* the corresponding key enters the in-memory LSH index,
     so a crash can never leave an uncommitted record in the index.

Only the public MinHashLSH interface (``insert`` / ``query`` /
``__contains__`` / ``is_empty``) is used; datasketch itself is not modified.
"""

from .config import DedupConfig, ParameterMismatchError
from .dedup import Deduplicator, DuplicateVerdict, RunStats
from .storage import (
    SCHEMA_VERSION,
    SQLiteStore,
    get_meta,
    list_committed_records,
)

__all__ = [
    "SCHEMA_VERSION",
    "DedupConfig",
    "Deduplicator",
    "DuplicateVerdict",
    "ParameterMismatchError",
    "RunStats",
    "SQLiteStore",
    "get_meta",
    "list_committed_records",
]
