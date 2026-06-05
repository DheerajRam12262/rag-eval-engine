"""Embedders: a deterministic offline default plus a sentence-transformers adapter.

The offline :class:`HashingEmbedder` uses the *signed feature-hashing* trick: each token is
hashed to a bucket with a +/- sign, accumulated, then L2-normalized. Cosine similarity between
two hashed vectors tracks (sign-cancelled) shared-token overlap, so dense retrieval genuinely
discriminates between texts -- with no model download and full determinism. Swap to real
semantic embeddings (e.g. bge) by setting ``embedder.backend: sentence-transformers``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, assert_never

import numpy as np
import numpy.typing as npt

from rag_eval.config import EmbedderConfig
from rag_eval.text import tokenize

FloatArray = npt.NDArray[np.float32]


class Embedder(Protocol):
    """Maps texts to L2-normalized dense vectors of a fixed dimension."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> FloatArray:
        """Return a ``(len(texts), dim)`` float32 array of unit vectors."""
        ...


class HashingEmbedder:
    """Deterministic signed feature-hashing embedder (offline default)."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> FloatArray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            bucket = h % self._dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows = [self._embed_one(t) for t in texts]
        return np.vstack(rows).astype(np.float32)


class SentenceTransformerEmbedder:
    """Adapter for real semantic embeddings. Requires the ``embeddings`` extra.

    Not exercised in CI (needs torch); it is a thin, real wrapper, not a stub.
    """

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers not installed; run: pip install -e '.[embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:  # pragma: no cover - optional dependency
        return self._dim

    def embed(self, texts: Sequence[str]) -> FloatArray:  # pragma: no cover - optional dependency
        emb = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(emb, dtype=np.float32)


def get_embedder(config: EmbedderConfig) -> Embedder:
    """Build the embedder named by the config."""
    backend = config.backend
    if backend == "hashing":
        return HashingEmbedder(config.dim)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(config.model)
    assert_never(backend)
