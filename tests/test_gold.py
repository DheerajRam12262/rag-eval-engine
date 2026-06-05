"""Gold set loading, split tagging, validation, and pooling."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.eval.gold import filter_split, load_gold, pool_candidates
from rag_eval.types import Chunk, ScoredChunk

GOLD_DIR = Path(__file__).resolve().parents[1] / "eval" / "gold"


def test_load_committed_gold_set() -> None:
    gold = load_gold(GOLD_DIR)
    assert len(gold) >= 30
    dev, test = filter_split(gold, "dev"), filter_split(gold, "test")
    assert dev and test
    assert len(dev) + len(test) == len(gold)


def test_no_answer_questions_have_no_relevant_docs() -> None:
    gold = load_gold(GOLD_DIR)
    no_answer = [q for q in gold if q.no_answer]
    assert no_answer, "gold set must include no-answer questions for abstention eval"
    assert all(q.relevant_doc_ids == () for q in no_answer)
    assert all(q.relevant_doc_ids for q in gold if not q.no_answer)


def test_invalid_no_answer_with_relevant_docs_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "test.jsonl"
    bad.write_text('{"id": "x", "question": "q", "no_answer": true, "relevant_doc_ids": ["d"]}\n')
    with pytest.raises(ValueError):
        load_gold(bad)


def test_pooling_unions_top_depth_across_systems() -> None:
    def sc(doc: str) -> ScoredChunk:
        return ScoredChunk(Chunk(id=f"{doc}#0", doc_id=doc, text="", ordinal=0), 0.0)

    system_a = [sc("d1"), sc("d2"), sc("d3")]
    system_b = [sc("d2"), sc("d4"), sc("d5")]
    # depth=2 => {d1,d2} from A, {d2,d4} from B
    assert pool_candidates([system_a, system_b], depth=2) == ["d1", "d2", "d4"]
