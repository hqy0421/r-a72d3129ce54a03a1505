"""Configuration of a near-duplicate cleaning run.

All parameters that affect the result are captured in :class:`DedupConfig`
and serialized (via :attr:`DedupConfig.fingerprint`) into the SQLite ``meta``
table.  When an existing database is reopened, the fingerprint of the
requested run must match the stored one byte for byte, otherwise
:class:`ParameterMismatchError` is raised and nothing is processed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

# Bumped when the SQLite schema or on-disk semantics change in an
# incompatible way.  Stored in sqlite ``PRAGMA user_version`` and in meta.
SCHEMA_VERSION = 1

# Fixed so MinHash permutations are deterministic across processes and
# resumes; recorded in the fingerprint as well.
DEFAULT_SEED = 1
DEFAULT_NUM_PERM = 128
DEFAULT_LSH_THRESHOLD = 0.9
DEFAULT_JACCARD_THRESHOLD = 0.9
DEFAULT_BATCH_SIZE = 1000
# Keys used inside LSH are integers (source line index); they must stay
# JSON/serialization friendly for any storage backend.
DEFAULT_TOKENS_KEY = "tokens"


class ParameterMismatchError(RuntimeError):
    """Raised when an existing database was built with different inputs or
    parameters than the ones requested.
    """


@dataclass(frozen=True)
class DedupConfig:
    """All inputs that determine a run's decisions.

    Args:
        input_path: JSONL file whose lines are JSON objects containing a
            string array of tokens.
        tokens_key: Name of the tokens array field on each line.
        jaccard_threshold: Exact Jaccard threshold (on the original token
            sets) at or above which a confirmed candidate is a duplicate.
        num_perm: Number of MinHash permutation functions.
        lsh_threshold: Threshold passed to :class:`MinHashLSH`, used to pick
            the optimal band/row layout.
        lsh_weights: False-positive / false-negative weights for the LSH
            parameter optimization.
        lsh_params: Optional explicit ``(b, r)`` layout; when given,
            ``lsh_threshold`` / ``lsh_weights`` are ignored for layout
            selection but still recorded.
        seed: MinHash random seed (must stay fixed across resumes).
        batch_size: Number of input lines evaluated, then committed in one
            SQLite transaction.

    """

    input_path: str
    tokens_key: str = DEFAULT_TOKENS_KEY
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD
    num_perm: int = DEFAULT_NUM_PERM
    lsh_threshold: float = DEFAULT_LSH_THRESHOLD
    lsh_weights: tuple[float, float] = (0.5, 0.5)
    lsh_params: Optional[tuple[int, int]] = None
    seed: int = DEFAULT_SEED
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        if not 0.0 <= self.jaccard_threshold <= 1.0:
            raise ValueError("jaccard_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.lsh_threshold <= 1.0:
            raise ValueError("lsh_threshold must be in [0.0, 1.0]")
        if self.num_perm < 2:
            raise ValueError("num_perm must be >= 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if len(self.lsh_weights) != 2 or sum(self.lsh_weights) != 1.0:
            raise ValueError("lsh_weights must be two floats summing to 1.0")
        if self.lsh_params is not None:
            b, r = self.lsh_params
            if b < 2 or r < 1 or b * r > self.num_perm:
                raise ValueError("lsh_params (b, r) need b >= 2, r >= 1 and b*r <= num_perm")

    # -- canonical (de)serialization -------------------------------------

    # Parameters that determine the *decisions* of a run. ``batch_size`` is
    # deliberately absent: it only sizes the commit transactions and the
    # in-batch ephemeral index makes decisions batch-size independent.
    _SEMANTIC_FIELDS = (
        "input_path",
        "tokens_key",
        "jaccard_threshold",
        "num_perm",
        "lsh_threshold",
        "lsh_weights",
        "lsh_params",
        "seed",
    )

    def _canonical_data(self, include_abs_path: bool) -> dict:
        data = asdict(self)
        data["lsh_weights"] = list(self.lsh_weights)
        data["lsh_params"] = list(self.lsh_params) if self.lsh_params is not None else None
        if include_abs_path:
            data["input_abs_path"] = os.path.abspath(self.input_path)
        return data

    def to_json(self) -> str:
        """Full canonical JSON, stored verbatim in the meta table."""
        return json.dumps(self._canonical_data(True), sort_keys=True, separators=(",", ":"))

    def semantic_json(self) -> str:
        """Canonical JSON of decision-affecting parameters only.

        Used for the resume fingerprint so that an operational parameter
        such as ``batch_size`` can change between invocations without
        invalidating a run.
        """
        data = self._canonical_data(True)
        data = {key: data[key] for key in self._SEMANTIC_FIELDS}
        data["input_abs_path"] = os.path.abspath(self.input_path)
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """SHA-256 over the semantic parameter JSON plus schema version.

        A resume only proceeds when the stored fingerprint equals this one,
        guaranteeing identical input identity and algorithm parameters.
        """
        h = hashlib.sha256()
        h.update(f"schema={SCHEMA_VERSION}|".encode("ascii"))
        h.update(self.semantic_json().encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def from_meta(meta: dict) -> "DedupConfig":
        data = json.loads(meta["params_json"])
        data.pop("input_abs_path", None)
        if data.get("lsh_params") is not None:
            data["lsh_params"] = tuple(data["lsh_params"])
        data["lsh_weights"] = tuple(data["lsh_weights"])
        return DedupConfig(**data)

    def new_lsh(self):
        """Build a fresh :class:`datasketch.MinHashLSH` with this config.

        Uses only the public constructor; the index interface is untouched.
        """
        from datasketch import MinHashLSH

        kwargs: dict = {
            "threshold": self.lsh_threshold,
            "num_perm": self.num_perm,
            "weights": tuple(self.lsh_weights),
        }
        if self.lsh_params is not None:
            kwargs["params"] = tuple(self.lsh_params)
        return MinHashLSH(**kwargs)

    def new_minhash(self):
        """Build a fresh empty :class:`datasketch.MinHash` for this config."""
        from datasketch import MinHash

        return MinHash(num_perm=self.num_perm, seed=self.seed)

    def minhash_for_tokens(self, tokens: list[str]):
        """Deterministically build the MinHash of one token sequence."""
        mh = self.new_minhash()
        if tokens:
            mh.update_batch(t.encode("utf-8") for t in tokens)
        return mh
