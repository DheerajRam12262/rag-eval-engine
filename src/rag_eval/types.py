"""Core data types shared across ingestion, retrieval, generation, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A raw corpus document, prior to chunking."""

    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit produced from a :class:`Document` by a chunking strategy."""

    id: str
    doc_id: str
    text: str
    ordinal: int
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk paired with a relevance score and the stage that produced it."""

    chunk: Chunk
    score: float
    source: str = ""  # e.g. "bm25", "dense", "rrf", "rerank"
