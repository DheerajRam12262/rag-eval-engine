"""IR + decision metrics with closed-form expected values."""

from __future__ import annotations

import math

from rag_eval.eval.metrics import (
    abstention_correct,
    ndcg_at_k,
    precision_at_k,
    ranked_doc_ids,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval.types import Chunk, ScoredChunk


def _sc(doc_id: str, ordinal: int = 0) -> ScoredChunk:
    cid = f"{doc_id}#{ordinal}"
    return ScoredChunk(Chunk(id=cid, doc_id=doc_id, text="", ordinal=ordinal), 0.0)


def test_ranked_doc_ids_dedups_preserving_best_rank() -> None:
    ranked = [_sc("a", 0), _sc("a", 1), _sc("b", 0)]
    assert ranked_doc_ids(ranked) == ["a", "b"]


def test_recall_and_precision() -> None:
    docs = ["a", "b", "c", "d"]
    rel = ["a", "c", "e"]
    assert recall_at_k(docs, rel, 4) == 2 / 3  # found a,c of {a,c,e}
    assert precision_at_k(docs, rel, 2) == 1 / 2  # a relevant, b not


def test_recall_nan_when_no_relevant() -> None:
    assert math.isnan(recall_at_k(["a"], [], 1))
    assert math.isnan(ndcg_at_k(["a"], [], 1))


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "a", "b"], ["a"]) == 1 / 2
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_perfect_and_imperfect() -> None:
    # one relevant doc at rank 1 => perfect
    assert ndcg_at_k(["a", "b"], ["a"], 2) == 1.0
    # relevant doc at rank 2 => 1/log2(3) normalized by ideal 1.0
    assert math.isclose(ndcg_at_k(["b", "a"], ["a"], 2), 1.0 / math.log2(3))


def test_abstention_correct() -> None:
    assert abstention_correct(no_answer=True, abstained=True) == 1.0
    assert abstention_correct(no_answer=False, abstained=False) == 1.0
    assert abstention_correct(no_answer=True, abstained=False) == 0.0
    assert abstention_correct(no_answer=False, abstained=True) == 0.0
