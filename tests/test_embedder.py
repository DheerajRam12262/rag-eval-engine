"""HashingEmbedder: shape, normalization, determinism, and discrimination."""

from __future__ import annotations

import numpy as np

from rag_eval.retrieve.embedder import HashingEmbedder


def test_shape_and_unit_norm() -> None:
    emb = HashingEmbedder(dim=64)
    vecs = emb.embed(["the quick brown fox", "lorem ipsum dolor"])
    assert vecs.shape == (2, 64)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_deterministic() -> None:
    a = HashingEmbedder(128).embed(["reproducible embedding"])
    b = HashingEmbedder(128).embed(["reproducible embedding"])
    assert np.array_equal(a, b)


def test_similar_text_scores_higher_than_dissimilar() -> None:
    emb = HashingEmbedder(dim=512)
    q = emb.embed(["the planet mars has two moons"])[0]
    related = emb.embed(["mars is a planet with moons named phobos and deimos"])[0]
    unrelated = emb.embed(["structured query language manages relational databases"])[0]
    assert float(q @ related) > float(q @ unrelated)


def test_empty_input_returns_empty_matrix() -> None:
    emb = HashingEmbedder(dim=32)
    out = emb.embed([])
    assert out.shape == (0, 32)
