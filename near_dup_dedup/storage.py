"""SQLite-backed durable state for the resumable dedup pipeline.

Crash-safety contract
----------------------
Decisions for a batch are accumulated on one connection inside a single
transaction and made visible with one ``COMMIT``. The in-memory MinHashLSH
index is only updated *after* that commit succeeds. Therefore:

* a crash before commit leaves the database exactly at the previous
  batch's committed state (the transaction is rolled back by SQLite);
* a crash after commit is recoverable by rebuilding the LSH index from the
  committed ``records`` table;
* no uncommitted record can ever be observable through the index, because
  index mutation strictly follows the durable commit.

WAL plus ``synchronous=FULL`` is used so the commit is fsynced before the
call returns.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterator, Optional

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE batches (
    batch_id        INTEGER PRIMARY KEY,
    started_at      TEXT NOT NULL,
    committed_at    TEXT,
    first_line_index INTEGER NOT NULL,
    last_line_index  INTEGER,
    line_count      INTEGER NOT NULL,
    accepted_count  INTEGER NOT NULL,
    duplicate_count INTEGER NOT NULL
);

CREATE TABLE records (
    line_index     INTEGER PRIMARY KEY,
    batch_id       INTEGER NOT NULL REFERENCES batches(batch_id),
    byte_offset    INTEGER NOT NULL,
    raw_line       TEXT NOT NULL,
    tokens_json    TEXT NOT NULL,
    decision       TEXT NOT NULL CHECK (decision IN ('accepted', 'duplicate')),
    duplicate_of   INTEGER REFERENCES records(line_index),
    best_jaccard   REAL,
    created_at     TEXT NOT NULL
);

CREATE INDEX idx_records_decision ON records(decision);
CREATE INDEX idx_records_duplicate_of ON records(duplicate_of);

CREATE TABLE evidence (
    line_index           INTEGER NOT NULL REFERENCES records(line_index),
    candidate_line_index INTEGER NOT NULL REFERENCES records(line_index),
    source               TEXT NOT NULL CHECK (source IN ('committed', 'batch')),
    jaccard              REAL NOT NULL,
    qualified            INTEGER NOT NULL,
    rank                 INTEGER NOT NULL,
    PRIMARY KEY (line_index, candidate_line_index)
);

CREATE INDEX idx_evidence_candidate ON evidence(candidate_line_index);

CREATE TABLE progress (
    id                     INTEGER PRIMARY KEY CHECK (id = 1),
    last_committed_line    INTEGER NOT NULL,
    last_committed_offset  INTEGER NOT NULL,
    lines_committed        INTEGER NOT NULL,
    updated_at             TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_meta(conn: sqlite3.Connection) -> dict:
    """Return the full meta table as a dict (empty dict on a fresh db)."""
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
    if not exists:
        return {}
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return dict(rows)


class SQLiteStore:
    """Thin, transaction-oriented wrapper over the dedup SQLite database."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys=ON")
        # FULL fsyncs the WAL on every commit; durability over throughput.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        stored = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if stored == 0:
            self._fresh = True
        elif stored != SCHEMA_VERSION:
            raise RuntimeError(f"Database {db_path} has schema version {stored}, this code supports {SCHEMA_VERSION}")
        else:
            self._fresh = False
        self._open_batch: Optional[int] = None

    # -- lifecycle --------------------------------------------------------

    @property
    def is_fresh(self) -> bool:
        return self._fresh

    def initialize(
        self,
        params_json: str,
        fingerprint: str,
        input_identity: dict,
        lsh_b: int,
        lsh_r: int,
    ) -> None:
        """Create the schema and seed meta/progress for a brand-new run."""
        if not self._fresh:
            raise RuntimeError("initialize() called on an existing database")
        now = _utcnow()
        with self.conn:
            self.conn.executescript(_DDL)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            meta = {
                "schema_version": str(SCHEMA_VERSION),
                "params_json": params_json,
                "fingerprint": fingerprint,
                "input_path": input_identity["path"],
                "input_abs_path": input_identity["abs_path"],
                "input_size": str(input_identity["size"]),
                "input_sha256": input_identity["sha256"],
                "input_total_lines": str(input_identity["total_lines"]),
                "lsh_b": str(lsh_b),
                "lsh_r": str(lsh_r),
                "created_at": now,
                "updated_at": now,
            }
            self.conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", meta.items())
            self.conn.execute(
                "INSERT INTO progress(id, last_committed_line, "
                "last_committed_offset, lines_committed, updated_at) "
                "VALUES (1, -1, 0, 0, ?)",
                (now,),
            )
        self._fresh = False

    def close(self) -> None:
        if self._open_batch is not None:
            # Defensive: normal flow always commits or rolls back explicitly.
            self.conn.rollback()
            self._open_batch = None
        # Fold the WAL back into the main database file.
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- meta / progress reads -------------------------------------------

    def meta(self) -> dict:
        return get_meta(self.conn)

    def progress(self) -> dict:
        row = self.conn.execute(
            "SELECT last_committed_line, last_committed_offset, lines_committed FROM progress WHERE id = 1"
        ).fetchone()
        if row is None:
            return {"last_committed_line": -1, "last_committed_offset": 0, "lines_committed": 0}
        return {
            "last_committed_line": row[0],
            "last_committed_offset": row[1],
            "lines_committed": row[2],
        }

    def counts(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*), SUM(decision = 'accepted'), SUM(decision = 'duplicate') FROM records"
        ).fetchone()
        return {
            "total": row[0] or 0,
            "accepted": row[1] or 0,
            "duplicate": row[2] or 0,
        }

    def is_finished(self) -> bool:
        m = self.meta()
        return m.get("finished_at") is not None

    # -- batch transaction ------------------------------------------------

    def begin_batch(self, first_line_index: int) -> int:
        """Open the batch transaction. All writes until
        :meth:`commit_batch` share it and are invisible to other readers.
        """
        if self._open_batch is not None:
            raise RuntimeError("a batch is already open")
        cur = self.conn.execute(
            "INSERT INTO batches(started_at, first_line_index, line_count, "
            "accepted_count, duplicate_count) VALUES (?, ?, 0, 0, 0)",
            (_utcnow(), first_line_index),
        )
        self._open_batch = cur.lastrowid
        return self._open_batch

    def add_record(
        self,
        line_index: int,
        byte_offset: int,
        raw_line: str,
        tokens_json: str,
        decision: str,
        duplicate_of: Optional[int],
        best_jaccard: Optional[float],
    ) -> None:
        if self._open_batch is None:
            raise RuntimeError("add_record() outside of a batch transaction")
        self.conn.execute(
            "INSERT INTO records(line_index, batch_id, byte_offset, raw_line, "
            "tokens_json, decision, duplicate_of, best_jaccard, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                line_index,
                self._open_batch,
                byte_offset,
                raw_line,
                tokens_json,
                decision,
                duplicate_of,
                best_jaccard,
                _utcnow(),
            ),
        )
        if decision == "accepted":
            self.conn.execute(
                "UPDATE batches SET accepted_count = accepted_count + 1 WHERE batch_id = ?",
                (self._open_batch,),
            )
        else:
            self.conn.execute(
                "UPDATE batches SET duplicate_count = duplicate_count + 1 WHERE batch_id = ?",
                (self._open_batch,),
            )
        self.conn.execute(
            "UPDATE batches SET line_count = line_count + 1 WHERE batch_id = ?",
            (self._open_batch,),
        )

    def add_evidence(
        self,
        line_index: int,
        candidate_line_index: int,
        source: str,
        jaccard: float,
        qualified: bool,
        rank: int,
    ) -> None:
        if self._open_batch is None:
            raise RuntimeError("add_evidence() outside of a batch transaction")
        self.conn.execute(
            "INSERT INTO evidence(line_index, candidate_line_index, source, "
            "jaccard, qualified, rank) VALUES (?, ?, ?, ?, ?, ?)",
            (line_index, candidate_line_index, source, jaccard, 1 if qualified else 0, rank),
        )

    def commit_batch(
        self,
        last_line_index: int,
        last_byte_offset: int,
        lines_committed: int,
    ) -> None:
        """Durably publish everything written in the open batch."""
        if self._open_batch is None:
            raise RuntimeError("commit_batch() without an open batch")
        batch_id = self._open_batch
        now = _utcnow()
        self.conn.execute(
            "UPDATE batches SET committed_at = ?, last_line_index = ? WHERE batch_id = ?",
            (now, last_line_index, batch_id),
        )
        self.conn.execute(
            "UPDATE progress SET last_committed_line = ?, "
            "last_committed_offset = ?, lines_committed = ?, updated_at = ? "
            "WHERE id = 1",
            (last_line_index, last_byte_offset, lines_committed, now),
        )
        self.conn.execute("UPDATE meta SET value = ? WHERE key = 'updated_at'", (now,))
        self.conn.commit()
        self._open_batch = None

    def rollback_batch(self) -> None:
        """Discard the open batch (error path / crash simulation)."""
        if self._open_batch is not None:
            self.conn.rollback()
            self._open_batch = None

    def mark_finished(self) -> None:
        now = _utcnow()
        prog = self.progress()
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (
                    ("final_lines_committed", str(prog["lines_committed"])),
                    ("finished_at", now),
                    ("updated_at", now),
                ),
            )

    # -- reads used by the engine and exporters ---------------------------

    def get_tokens(self, line_index: int) -> list[str]:
        row = self.conn.execute(
            "SELECT tokens_json FROM records WHERE line_index = ?",
            (line_index,),
        ).fetchone()
        if row is None:
            raise KeyError(line_index)
        return json.loads(row[0])

    def get_tokens_for_keys(self, line_indices: list[int]) -> dict[int, list[str]]:
        if not line_indices:
            return {}
        # Only parameter placeholders ("?") are interpolated; all values are
        # bound, so S608 (SQL injection) does not apply.
        placeholders = ",".join("?" for _ in line_indices)
        rows = self.conn.execute(
            f"SELECT line_index, tokens_json FROM records "  # noqa: S608
            f"WHERE line_index IN ({placeholders})",
            line_indices,
        ).fetchall()
        return {idx: json.loads(tj) for idx, tj in rows}

    def get_record(self, line_index: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT line_index, batch_id, byte_offset, raw_line, tokens_json, "
            "decision, duplicate_of, best_jaccard FROM records "
            "WHERE line_index = ?",
            (line_index,),
        ).fetchone()
        if row is None:
            return None
        return {
            "line_index": row[0],
            "batch_id": row[1],
            "byte_offset": row[2],
            "raw_line": row[3],
            "tokens_json": row[4],
            "decision": row[5],
            "duplicate_of": row[6],
            "best_jaccard": row[7],
        }

    def iter_records(self) -> Iterator[dict]:
        """All committed records in source order."""
        for row in self.conn.execute(
            "SELECT line_index, byte_offset, raw_line, tokens_json, decision, "
            "duplicate_of, best_jaccard FROM records ORDER BY line_index"
        ):
            yield {
                "line_index": row[0],
                "byte_offset": row[1],
                "raw_line": row[2],
                "tokens_json": row[3],
                "decision": row[4],
                "duplicate_of": row[5],
                "best_jaccard": row[6],
            }

    def iter_accepted(self) -> Iterator[tuple[int, list[str]]]:
        """Accepted records as (line_index, tokens) — LSH rebuild source."""
        for idx, tj in self.conn.execute(
            "SELECT line_index, tokens_json FROM records WHERE decision = 'accepted' ORDER BY line_index"
        ):
            yield idx, json.loads(tj)

    def evidence_for(self, line_index: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT candidate_line_index, source, jaccard, qualified, rank "
            "FROM evidence WHERE line_index = ? ORDER BY rank",
            (line_index,),
        ).fetchall()
        return [
            {
                "candidate_line_index": r[0],
                "source": r[1],
                "jaccard": r[2],
                "qualified": bool(r[3]),
                "rank": r[4],
            }
            for r in rows
        ]

    def lsh_key_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM records WHERE decision = 'accepted'").fetchone()
        return row[0]


def list_committed_records(store: SQLiteStore) -> list[dict]:
    """Convenience wrapper used by exporters and tests."""
    return list(store.iter_records())
