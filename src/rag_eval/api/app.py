"""FastAPI serving layer: POST /query and GET /health.

The pipeline is built once at startup (offline by default) and reused per request. Every
response carries per-stage latency and token/cost telemetry straight from the pipeline.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_eval import __version__
from rag_eval.config import load_config, seed_everything
from rag_eval.ingest.indexer import IndexBundle, build_index
from rag_eval.pipeline import PipelineResult, RagPipeline

_SNIPPET_CHARS = 240


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="The user question.")
    top_k: int | None = Field(default=None, gt=0, description="Override the number of contexts.")


class ContextItem(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    score: float
    snippet: str


class Latencies(BaseModel):
    retrieve_ms: float
    rerank_ms: float
    generate_ms: float
    total_ms: float


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[str]
    abstained: bool
    contexts: list[ContextItem]
    latencies: Latencies
    usage: Usage


class HealthResponse(BaseModel):
    status: str
    config: str
    num_chunks: int
    version: str


def _to_response(result: PipelineResult) -> QueryResponse:
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        citations=result.citations,
        abstained=result.abstained,
        contexts=[
            ContextItem(
                chunk_id=c.chunk.id,
                doc_id=c.chunk.doc_id,
                title=c.chunk.title,
                score=round(c.score, 6),
                snippet=c.chunk.text[:_SNIPPET_CHARS],
            )
            for c in result.contexts
        ],
        latencies=Latencies(
            retrieve_ms=result.latencies.retrieve_ms,
            rerank_ms=result.latencies.rerank_ms,
            generate_ms=result.latencies.generate_ms,
            total_ms=result.latencies.total_ms,
        ),
        usage=Usage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            cost_usd=result.usage.cost_usd,
        ),
    )


def create_app(config_path: str | None = None, bundle: IndexBundle | None = None) -> FastAPI:
    """Build a FastAPI app around a pipeline. Pass ``bundle`` in tests to avoid a rebuild."""
    config = load_config(config_path or os.environ.get("RAG_CONFIG", "config/base.yaml"))
    seed_everything(config.seed)
    if bundle is None:
        bundle = build_index(config, persist=False)
    pipeline = RagPipeline(bundle, config)
    num_chunks = len(bundle.chunks)

    app = FastAPI(title="rag-eval-engine", version=__version__)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok", config=config.name, num_chunks=num_chunks, version=__version__
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        result = pipeline.query(request.query, top_k=request.top_k)
        return _to_response(result)

    return app


app = create_app()
