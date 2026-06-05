"""Reranking: a deterministic lexical reranker plus a cross-encoder adapter.

A cross-encoder scores (query, passage) jointly and is the quality bar. The offline
:class:`LexicalReranker` re-scores candidates by query content-term *coverage* (with raw overlap
as a tiebreak) -- a precision-oriented signal distinct from BM25/RRF, fully deterministic. Swap
to a real cross-encoder via ``retrieval.rerank.backend: cross-encoder``.
"""

from __future__ import annotations

from typing import Protocol, assert_never

from rag_eval.config import RerankConfig
from rag_eval.text import content_tokens
from rag_eval.types import ScoredChunk


class Reranker(Protocol):
    """Re-orders candidate chunks for a query and returns the top_k."""

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class LexicalReranker:
    """Re-score by query content-term coverage; deterministic id tie-break."""

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        qterms = set(content_tokens(query))
        if not qterms:
            return candidates[:top_k]
        rescored: list[ScoredChunk] = []
        for cand in candidates:
            ctoks = content_tokens(cand.chunk.text)
            coverage = len(qterms & set(ctoks)) / len(qterms)
            overlap = sum(1 for t in ctoks if t in qterms)
            score = coverage + 1e-4 * overlap  # coverage dominates; overlap breaks ties
            rescored.append(ScoredChunk(cand.chunk, score, "rerank"))
        rescored.sort(key=lambda s: (-s.score, s.chunk.id))
        return rescored[:top_k]


class CrossEncoderReranker:
    """Adapter for a real cross-encoder. Requires the ``embeddings`` extra (sentence-transformers)."""

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "cross-encoder needs sentence-transformers; run: pip install -e '.[embeddings]'"
            ) from exc
        self._model = CrossEncoder(model)

    def rerank(  # pragma: no cover - optional dependency
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda p: float(p[1]), reverse=True
        )
        return [ScoredChunk(c.chunk, float(s), "rerank") for c, s in ranked[:top_k]]


def get_reranker(config: RerankConfig) -> Reranker:
    """Build the reranker named by the config."""
    backend = config.backend
    if backend == "lexical":
        return LexicalReranker()
    if backend == "cross-encoder":
        return CrossEncoderReranker(config.model)
    assert_never(backend)
