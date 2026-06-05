"""Chunking correctness: budgets, overlap, ids, and edge cases."""

from __future__ import annotations

import re

import pytest

from rag_eval.config import ChunkingConfig
from rag_eval.ingest.chunking import FixedChunker, RecursiveChunker, get_chunker
from rag_eval.types import Document


def _doc(text: str) -> Document:
    return Document(id="d1", text=text, title="t")


def test_fixed_chunker_window_and_overlap() -> None:
    tokens = " ".join(str(i) for i in range(10))
    chunks = FixedChunker(chunk_size=4, overlap=1).chunk(_doc(tokens))
    # step = 3 -> windows start at 0,3,6; the window at 6 (tokens 6..9) reaches the end,
    # so we stop rather than emit a redundant trailing chunk.
    assert [c.text.split()[0] for c in chunks] == ["0", "3", "6"]
    assert all(len(c.text.split()) <= 4 for c in chunks)
    assert chunks[-1].text.split()[-1] == "9"  # last token is covered
    # consecutive windows share the overlap token
    assert chunks[0].text.split()[-1] == chunks[1].text.split()[0]


def test_fixed_chunker_ids_unique_and_ordered() -> None:
    chunks = FixedChunker(4, 1).chunk(_doc(" ".join(str(i) for i in range(10))))
    ids = [c.id for c in chunks]
    assert ids == sorted(ids, key=lambda s: int(s.split("#")[1]))
    assert len(set(ids)) == len(ids)
    assert all(c.doc_id == "d1" for c in chunks)


def test_recursive_respects_budget_and_covers_text() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
    chunks = RecursiveChunker(chunk_size=6, overlap=2).chunk(_doc(text))
    assert len(chunks) >= 2
    assert all(len(c.text.split()) <= 6 for c in chunks)
    # every source word appears somewhere (normalize away retained punctuation/case)
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())  # noqa: E731
    produced = {norm(w) for c in chunks for w in c.text.split()}
    expected = {norm(w) for w in text.split()}
    assert expected.issubset(produced)


def test_recursive_hard_splits_oversized_sentence() -> None:
    long_sentence = " ".join(str(i) for i in range(20))  # one "sentence", 20 tokens
    chunks = RecursiveChunker(chunk_size=5, overlap=0).chunk(_doc(long_sentence))
    assert all(len(c.text.split()) <= 5 for c in chunks)
    assert len(chunks) == 4


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_empty_document_yields_no_chunks(strategy: str) -> None:
    cfg = ChunkingConfig(strategy=strategy, chunk_size=8, chunk_overlap=2)  # type: ignore[arg-type]
    assert get_chunker(cfg).chunk(_doc("   ")) == []


def test_semantic_is_labeled_unimplemented() -> None:
    cfg = ChunkingConfig(strategy="semantic", chunk_size=8, chunk_overlap=2)
    with pytest.raises(NotImplementedError):
        get_chunker(cfg)
