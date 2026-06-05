"""End-to-end pipeline behavior across variants (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.config import load_config
from rag_eval.ingest.indexer import IndexBundle
from rag_eval.pipeline import RagPipeline

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.mark.integration
def test_full_pipeline_answers_and_times_stages(base_index: IndexBundle) -> None:
    cfg = load_config(CONFIG_DIR / "variants" / "hybrid_rerank.yaml")
    result = RagPipeline(base_index, cfg).query("what process do plants use to make glucose")
    assert not result.abstained
    assert result.citations
    assert result.ranked  # retrieval produced a ranking
    assert len(result.contexts) <= cfg.retrieval.top_k
    assert result.latencies.total_ms >= 0.0
    # the cited chunk is one the generator actually saw
    assert result.citations[0] in {c.chunk.id for c in result.contexts}


@pytest.mark.integration
def test_closed_book_abstains(base_index: IndexBundle) -> None:
    cfg = load_config(CONFIG_DIR / "variants" / "closed_book.yaml")
    result = RagPipeline(base_index, cfg).query("what is the boiling point of water")
    assert result.ranked == []
    assert result.abstained  # offline extractive generator has no parametric knowledge


@pytest.mark.integration
def test_oracle_mode_uses_supplied_chunks(base_index: IndexBundle) -> None:
    cfg = load_config(CONFIG_DIR / "variants" / "oracle.yaml")
    gold = next(c.id for c in base_index.chunks if c.doc_id == "doc-015")  # Mars
    result = RagPipeline(base_index, cfg).query(
        "what are the moons of mars called", oracle_chunk_ids=[gold]
    )
    assert [c.chunk.id for c in result.contexts] == [gold]
    assert not result.abstained


@pytest.mark.integration
def test_pipeline_abstains_on_adversarial_no_answer(base_index: IndexBundle) -> None:
    cfg = load_config(CONFIG_DIR / "variants" / "hybrid_rerank.yaml")
    result = RagPipeline(base_index, cfg).query(
        "who won the fictional intergalactic quidditch championship in 3024"
    )
    assert result.abstained
