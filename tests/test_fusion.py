"""RRF correctness on hand-built rankings."""

from __future__ import annotations

from rag_eval.retrieve.fusion import reciprocal_rank_fusion
from rag_eval.types import Chunk, ScoredChunk


def _sc(cid: str, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(Chunk(id=cid, doc_id=cid, text=cid, ordinal=0), score)


def test_rrf_rewards_agreement_across_lists() -> None:
    # A is rank 1 in both lists; B and C appear once each at rank 2.
    list1 = [_sc("A"), _sc("B")]
    list2 = [_sc("A"), _sc("C")]
    fused = reciprocal_rank_fusion([list1, list2], rrf_k=60, top_k=3)
    assert [s.chunk.id for s in fused] == ["A", "B", "C"]  # B<C tie-break by id
    assert fused[0].source == "rrf"
    assert fused[0].score > fused[1].score


def test_rrf_score_formula() -> None:
    fused = reciprocal_rank_fusion([[_sc("A"), _sc("B")]], rrf_k=10, top_k=2)
    a = next(s for s in fused if s.chunk.id == "A")
    b = next(s for s in fused if s.chunk.id == "B")
    assert a.score == 1.0 / (10 + 1)  # rank 1
    assert b.score == 1.0 / (10 + 2)  # rank 2


def test_rrf_respects_top_k() -> None:
    fused = reciprocal_rank_fusion([[_sc("A"), _sc("B"), _sc("C")]], rrf_k=60, top_k=2)
    assert len(fused) == 2


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([], rrf_k=60, top_k=5) == []
