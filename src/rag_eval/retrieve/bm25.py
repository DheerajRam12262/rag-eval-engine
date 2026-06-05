"""Sparse BM25 index over chunk text, wrapping ``rank_bm25.BM25Okapi``.

Persisted as the tokenized corpus + ids (JSON) and rebuilt on load -- portable and avoids
pickling a third-party object.
"""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_eval.text import tokenize


class BM25Index:
    """Okapi BM25 ranking over a fixed set of (id, text) chunks."""

    def __init__(self, ids: list[str], tokens: list[list[str]]) -> None:
        if len(ids) != len(tokens):
            raise ValueError("ids and tokens must be the same length")
        self._ids = ids
        self._tokens = tokens
        # BM25Okapi requires at least one document; guard the empty case.
        self._bm25 = BM25Okapi(tokens) if tokens else None

    @classmethod
    def build(cls, ids: list[str], texts: list[str]) -> BM25Index:
        return cls(ids, [tokenize(t) for t in texts])

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or top_k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:top_k]]

    def __len__(self) -> int:
        return len(self._ids)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "bm25.json").write_text(json.dumps({"ids": self._ids, "tokens": self._tokens}))

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        data = json.loads((path / "bm25.json").read_text())
        return cls(list(data["ids"]), [list(t) for t in data["tokens"]])
