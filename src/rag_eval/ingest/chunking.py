"""Chunking strategies. Chunk size/strategy are ablation variables, not constants.

Token counts are approximate (whitespace-delimited words). Two strategies ship:

* ``fixed``     -- sliding window of N tokens with overlap; ignores structure.
* ``recursive`` -- pack whole sentences greedily up to the budget, carrying an overlap of
  trailing tokens between chunks; respects sentence boundaries.

``semantic`` is declared in the config schema but not yet implemented (see :func:`get_chunker`).
"""

from __future__ import annotations

from typing import Protocol

from rag_eval.config import ChunkingConfig
from rag_eval.text import split_sentences
from rag_eval.types import Chunk, Document


class Chunker(Protocol):
    """Turns a document into an ordered list of chunks."""

    def chunk(self, doc: Document) -> list[Chunk]: ...


def _make_chunk(doc: Document, ordinal: int, tokens: list[str]) -> Chunk:
    return Chunk(
        id=f"{doc.id}#{ordinal}",
        doc_id=doc.id,
        text=" ".join(tokens),
        ordinal=ordinal,
        title=doc.title,
        metadata=dict(doc.metadata),
    )


class FixedChunker:
    """Sliding window of ``chunk_size`` tokens advancing by ``chunk_size - overlap``."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        tokens = doc.text.split()
        if not tokens:
            return []
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for ordinal, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + self.chunk_size]
            chunks.append(_make_chunk(doc, ordinal, window))
            if start + self.chunk_size >= len(tokens):
                break
        return chunks


class RecursiveChunker:
    """Greedily pack sentences up to the token budget, with a trailing-token overlap.

    A sentence longer than the budget is hard-split into windows so no chunk blows the budget.
    """

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _units(self, text: str) -> list[list[str]]:
        """Sentence token-lists, hard-splitting any oversized sentence."""
        units: list[list[str]] = []
        for sentence in split_sentences(text):
            toks = sentence.split()
            if len(toks) <= self.chunk_size:
                units.append(toks)
            else:
                for i in range(0, len(toks), self.chunk_size):
                    units.append(toks[i : i + self.chunk_size])
        return units

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text.split():
            return []
        chunks: list[Chunk] = []
        ordinal = 0
        current: list[str] = []
        for unit in self._units(doc.text):
            if current and len(current) + len(unit) > self.chunk_size:
                chunks.append(_make_chunk(doc, ordinal, current))
                ordinal += 1
                carry = current[-self.overlap :] if self.overlap else []
                current = carry + unit
            else:
                current = current + unit
        if current:
            chunks.append(_make_chunk(doc, ordinal, current))
        return chunks


def get_chunker(config: ChunkingConfig) -> Chunker:
    """Build the chunker named by the config."""
    if config.strategy == "fixed":
        return FixedChunker(config.chunk_size, config.chunk_overlap)
    if config.strategy == "recursive":
        return RecursiveChunker(config.chunk_size, config.chunk_overlap)
    # Honest, labeled gap rather than a silent fallback.
    raise NotImplementedError(
        "semantic chunking is a stretch goal and not yet implemented; "
        "use strategy: fixed | recursive"
    )


def chunk_documents(docs: list[Document], config: ChunkingConfig) -> list[Chunk]:
    """Chunk a whole corpus with the configured strategy."""
    chunker = get_chunker(config)
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunker.chunk(doc))
    return out
