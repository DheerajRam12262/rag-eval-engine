"""Shared, deterministic text utilities.

A single tokenizer is reused by the hashing embedder, BM25, the lexical reranker, and the
offline judge so that "lexical overlap" means the same thing everywhere.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric word tokens. Deterministic."""
    return _TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on terminal punctuation followed by whitespace.

    Deliberately dependency-free; good enough for chunk-boundary detection.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
