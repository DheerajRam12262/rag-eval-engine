"""Dense vector store.

The offline :class:`InMemoryVectorStore` does *exact* nearest-neighbor over normalized vectors
(cosine == dot product), which is correct and reproducible for a portfolio-scale corpus. The
"what changes at 10M docs" ANN/sharding story lives in docs/DECISIONS.md; a Qdrant adapter is on
the roadmap behind the ``qdrant`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from rag_eval.config import VectorStoreConfig

FloatArray = npt.NDArray[np.float32]


class VectorStore(Protocol):
    """Add normalized vectors, then search by cosine similarity."""

    def add(self, ids: list[str], vectors: FloatArray) -> None: ...

    def search(self, query: FloatArray, top_k: int) -> list[tuple[str, float]]: ...


class InMemoryVectorStore:
    """Exact cosine nearest-neighbor over an in-memory matrix."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._matrix: FloatArray = np.zeros((0, 0), dtype=np.float32)

    def add(self, ids: list[str], vectors: FloatArray) -> None:
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] != len(ids):
            raise ValueError("vectors must be 2-D with one row per id")
        if self._matrix.size == 0:
            self._matrix = mat
        else:
            if mat.shape[1] != self._matrix.shape[1]:
                raise ValueError("vector dimension mismatch")
            self._matrix = np.vstack([self._matrix, mat]).astype(np.float32)
        self._ids.extend(ids)

    def search(self, query: FloatArray, top_k: int) -> list[tuple[str, float]]:
        n = self._matrix.shape[0]
        if n == 0 or top_k <= 0:
            return []
        scores = self._matrix @ np.asarray(query, dtype=np.float32)
        k = min(top_k, n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._ids)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._matrix)
        (path / "ids.json").write_text(json.dumps(self._ids))

    @classmethod
    def load(cls, path: Path) -> InMemoryVectorStore:
        store = cls()
        store._matrix = np.load(path / "vectors.npy").astype(np.float32)
        store._ids = list(json.loads((path / "ids.json").read_text()))
        return store


def get_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Build the vector store named by the config."""
    if config.backend == "memory":
        return InMemoryVectorStore()
    # Honest, labeled gap: the Qdrant adapter is on the roadmap (see docs/DECISIONS.md).
    raise NotImplementedError(
        "vector_store.backend 'qdrant' is not yet implemented; use 'memory'. "
        "The Qdrant adapter (behind the 'qdrant' extra) is on the roadmap."
    )
