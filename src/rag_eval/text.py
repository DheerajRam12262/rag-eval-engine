"""Shared, deterministic text utilities.

A single tokenizer is reused by the hashing embedder, BM25, the lexical reranker, and the
offline judge so that "lexical overlap" means the same thing everywhere.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, fixed stopword set so "content overlap" ignores function words. Deliberately
# minimal and committed (not downloaded) to keep the offline path reproducible.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "about",
        "as",
        "into",
        "over",
        "after",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "not",
        "no",
        "nor",
        "too",
        "very",
        "s",
        "t",
        "and",
        "it",
        "its",
        "it's",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "do",
        "you",
        "i",
        "we",
        "they",
        "he",
        "she",
        "them",
        "his",
        "her",
        "their",
        "our",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric word tokens. Deterministic."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed -- used for overlap/coverage signals."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on terminal punctuation followed by whitespace.

    Deliberately dependency-free; good enough for chunk-boundary detection.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
