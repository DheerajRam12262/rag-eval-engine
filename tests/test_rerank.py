"""Lexical reranker: reorders by query content-coverage, deterministically."""

from __future__ import annotations

from rag_eval.config import RerankConfig
from rag_eval.retrieve.rerank import LexicalReranker, get_reranker
from rag_eval.types import Chunk, ScoredChunk


def _cand(cid: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(Chunk(id=cid, doc_id=cid, text=text, ordinal=0), score, "rrf")


def test_reranker_promotes_higher_coverage() -> None:
    # B has the lower initial score but covers more query terms; it should win after rerank.
    candidates = [
        _cand("A", "the sun is a star at the center of the solar system", 0.9),
        _cand("B", "mars is the red planet and has two small moons", 0.1),
    ]
    out = LexicalReranker().rerank("mars red planet moons", candidates, top_k=2)
    assert [s.chunk.id for s in out] == ["B", "A"]
    assert out[0].source == "rerank"
    assert out[0].score > out[1].score


def test_reranker_respects_top_k() -> None:
    cands = [_cand(str(i), f"doc number {i} about planets", 0.0) for i in range(5)]
    assert len(LexicalReranker().rerank("planets", cands, top_k=2)) == 2


def test_reranker_empty_query_terms_passthrough() -> None:
    cands = [_cand("A", "alpha", 0.0), _cand("B", "beta", 0.0)]
    # a query of only stopwords has no content terms
    out = LexicalReranker().rerank("the of and", cands, top_k=2)
    assert [s.chunk.id for s in out] == ["A", "B"]


def test_get_reranker_lexical() -> None:
    rr = get_reranker(RerankConfig(enabled=True, backend="lexical", model="x"))
    assert isinstance(rr, LexicalReranker)
