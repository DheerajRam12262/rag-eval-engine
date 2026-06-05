"""Retrievers: dense, BM25, and a config-driven hybrid that fuses them with RRF.

Each enabled retriever fetches ``candidate_k`` results; the hybrid fuses them and returns the
requested ``top_k``. With a single retriever enabled, fusion is skipped (the list passes through
with its original source tag). With none enabled (closed-book), retrieval returns nothing.
"""

from __future__ import annotations

from typing import Protocol

from rag_eval.config import RetrievalConfig
from rag_eval.ingest.indexer import IndexBundle
from rag_eval.retrieve.fusion import reciprocal_rank_fusion
from rag_eval.types import ScoredChunk


class Retriever(Protocol):
    """Returns chunks ranked by relevance to a query."""

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]: ...


class DenseRetriever:
    """Embeds the query and searches the dense vector store."""

    def __init__(self, bundle: IndexBundle) -> None:
        self._bundle = bundle

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        qvec = self._bundle.embedder.embed([query])[0]
        hits = self._bundle.vector_store.search(qvec, top_k)
        return [ScoredChunk(self._bundle.chunks_by_id[cid], score, "dense") for cid, score in hits]


class BM25Retriever:
    """Sparse BM25 ranking over chunk text."""

    def __init__(self, bundle: IndexBundle) -> None:
        self._bundle = bundle

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        hits = self._bundle.bm25.search(query, top_k)
        return [ScoredChunk(self._bundle.chunks_by_id[cid], score, "bm25") for cid, score in hits]


class HybridRetriever:
    """BM25 + dense, fused with RRF; toggled entirely by config."""

    def __init__(self, bundle: IndexBundle, config: RetrievalConfig) -> None:
        self._config = config
        self._retrievers: list[Retriever] = []
        if config.bm25.enabled:
            self._retrievers.append(BM25Retriever(bundle))
        if config.dense.enabled:
            self._retrievers.append(DenseRetriever(bundle))

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not self._retrievers:  # closed-book
            return []
        candidate_k = max(self._config.candidate_k, top_k)
        rankings = [r.retrieve(query, candidate_k) for r in self._retrievers]
        if len(rankings) == 1:
            return rankings[0][:top_k]
        return reciprocal_rank_fusion(rankings, self._config.fusion.rrf_k, top_k)


def get_retriever(bundle: IndexBundle, config: RetrievalConfig) -> Retriever:
    """Build the hybrid retriever for a config."""
    return HybridRetriever(bundle, config)
