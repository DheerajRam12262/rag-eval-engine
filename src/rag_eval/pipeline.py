"""The configurable end-to-end RAG pipeline: retrieve -> rerank -> generate.

Every stage is timed (per-stage latency telemetry) and the final ranking is exposed separately
from the generation context so the eval harness can score retrieval depth (recall@k for k up to
10) while the generator only sees ``top_k`` chunks. ``retrieval.mode: oracle`` short-circuits
retrieval and feeds the gold-relevant chunks directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rag_eval.config import Config
from rag_eval.generate.llm import Generator, TokenUsage, get_generator
from rag_eval.ingest.indexer import IndexBundle, load_index
from rag_eval.retrieve.rerank import Reranker, get_reranker
from rag_eval.retrieve.retrievers import Retriever, get_retriever
from rag_eval.types import ScoredChunk


@dataclass(slots=True)
class StageLatencies:
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class PipelineResult:
    query: str
    answer: str
    citations: list[str]
    abstained: bool
    ranked: list[ScoredChunk]  # full final ranking, for retrieval metrics
    contexts: list[ScoredChunk]  # the top_k chunks shown to the generator
    usage: TokenUsage
    latencies: StageLatencies


class RagPipeline:
    """Wires an :class:`IndexBundle` and a :class:`Config` into a queryable pipeline."""

    def __init__(self, bundle: IndexBundle, config: Config) -> None:
        self._bundle = bundle
        self._config = config
        self._retriever: Retriever = get_retriever(bundle, config.retrieval)
        self._reranker: Reranker | None = (
            get_reranker(config.retrieval.rerank) if config.retrieval.rerank.enabled else None
        )
        self._generator: Generator = get_generator(config.generation)

    def _retrieve(self, query: str, oracle_chunk_ids: list[str] | None) -> list[ScoredChunk]:
        if self._config.retrieval.mode == "oracle":
            ids = oracle_chunk_ids or []
            return [
                ScoredChunk(self._bundle.chunks_by_id[cid], 1.0, "oracle")
                for cid in ids
                if cid in self._bundle.chunks_by_id
            ]
        return self._retriever.retrieve(query, self._config.retrieval.candidate_k)

    def query(self, text: str, oracle_chunk_ids: list[str] | None = None) -> PipelineResult:
        rcfg = self._config.retrieval
        t_start = time.perf_counter()

        t = time.perf_counter()
        ranked = self._retrieve(text, oracle_chunk_ids)
        retrieve_ms = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        if self._reranker is not None and ranked:
            ranked = self._reranker.rerank(text, ranked, len(ranked))
        rerank_ms = (time.perf_counter() - t) * 1000.0

        contexts = ranked[: rcfg.top_k]

        t = time.perf_counter()
        gen = self._generator.generate(text, contexts, self._config.generation.abstain_threshold)
        generate_ms = (time.perf_counter() - t) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return PipelineResult(
            query=text,
            answer=gen.answer,
            citations=gen.citations,
            abstained=gen.abstained,
            ranked=ranked,
            contexts=contexts,
            usage=gen.usage,
            latencies=StageLatencies(retrieve_ms, rerank_ms, generate_ms, total_ms),
        )


def build_pipeline(config: Config, bundle: IndexBundle | None = None) -> RagPipeline:
    """Build a pipeline, loading the persisted index if a bundle is not supplied."""
    if bundle is None:
        bundle = load_index(config)
    return RagPipeline(bundle, config)
