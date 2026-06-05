"""Retriever behavior on the sample corpus: dense, BM25, hybrid, and closed-book."""

from __future__ import annotations

from pathlib import Path

from rag_eval.config import load_config
from rag_eval.ingest.indexer import IndexBundle
from rag_eval.retrieve.retrievers import BM25Retriever, DenseRetriever, get_retriever

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_dense_retriever(base_index: IndexBundle) -> None:
    top = DenseRetriever(base_index).retrieve(
        "largest gas giant planet made of hydrogen and helium", top_k=1
    )
    assert top[0].chunk.doc_id == "doc-016"  # Jupiter
    assert top[0].source == "dense"


def test_bm25_retriever(base_index: IndexBundle) -> None:
    top = BM25Retriever(base_index).retrieve(
        "git distributed version control linus torvalds", top_k=1
    )
    assert top[0].chunk.doc_id == "doc-005"  # Git
    assert top[0].source == "bm25"


def test_hybrid_returns_top_k_and_is_relevant(base_index: IndexBundle) -> None:
    cfg = load_config(CONFIG_DIR / "base.yaml")
    res = get_retriever(base_index, cfg.retrieval).retrieve(
        "how do vaccines train the immune system to fight pathogens", top_k=3
    )
    assert len(res) == 3
    assert res[0].source == "rrf"
    assert any(s.chunk.doc_id == "doc-025" for s in res)  # Vaccines


def test_closed_book_retrieves_nothing(base_index: IndexBundle) -> None:
    cfg = load_config(CONFIG_DIR / "variants" / "closed_book.yaml")
    assert get_retriever(base_index, cfg.retrieval).retrieve("anything at all", top_k=5) == []


def test_dense_only_variant_keeps_source_tag(base_index: IndexBundle) -> None:
    cfg = load_config(CONFIG_DIR / "variants" / "dense_only.yaml")
    res = get_retriever(base_index, cfg.retrieval).retrieve("mars the red planet", top_k=3)
    assert res
    assert all(s.source == "dense" for s in res)  # no fusion with a single retriever
