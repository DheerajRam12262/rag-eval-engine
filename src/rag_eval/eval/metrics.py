"""Retrieval and decision metrics. All operate on document-level relevance.

Document-level (not chunk-level) relevance keeps metrics valid across chunk-size ablations:
ranked chunks are collapsed to their parent ``doc_id`` (best rank wins) before scoring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rag_eval.types import ScoredChunk


def ranked_doc_ids(ranked: Sequence[ScoredChunk]) -> list[str]:
    """Collapse a ranked chunk list to unique parent doc ids, preserving best rank."""
    seen: set[str] = set()
    out: list[str] = []
    for scored in ranked:
        doc = scored.chunk.doc_id
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def recall_at_k(ranked_docs: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k. NaN if there are no relevant docs."""
    rel = set(relevant)
    if not rel:
        return math.nan
    return len(rel & set(ranked_docs[:k])) / len(rel)


def precision_at_k(ranked_docs: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top-k that are relevant."""
    if k <= 0:
        return 0.0
    rel = set(relevant)
    return len(rel & set(ranked_docs[:k])) / k


def reciprocal_rank(ranked_docs: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant doc (0 if none retrieved)."""
    rel = set(relevant)
    for i, doc in enumerate(ranked_docs, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_docs: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary-gain nDCG@k. NaN if there are no relevant docs."""
    rel = set(relevant)
    if not rel:
        return math.nan
    gains = [1.0 if doc in rel else 0.0 for doc in ranked_docs[:k]]
    idcg = _dcg([1.0] * min(len(rel), k))
    if idcg == 0.0:
        return 0.0
    return _dcg(gains) / idcg


def abstention_correct(no_answer: bool, abstained: bool) -> float:
    """1.0 when the abstain decision matches the question type, else 0.0.

    No-answer questions should be abstained on; answerable questions should be answered.
    """
    return 1.0 if abstained == no_answer else 0.0
