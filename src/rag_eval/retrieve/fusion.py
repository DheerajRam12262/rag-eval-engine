"""Reciprocal Rank Fusion (RRF).

RRF combines multiple ranked lists using only ranks, not scores, so retrievers on different
score scales (BM25 vs cosine) fuse cleanly. For a document d:

    score(d) = sum_i  1 / (rrf_k + rank_i(d))

where ``rank_i`` is the 1-based rank of d in list i (absent => no contribution). Larger
``rrf_k`` flattens the contribution of top ranks. Ties break by chunk id for determinism.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_eval.types import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]],
    rrf_k: int,
    top_k: int,
) -> list[ScoredChunk]:
    """Fuse several ranked lists into one, returning the top_k fused chunks."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, scored in enumerate(ranking):
            cid = scored.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            chunks[cid] = scored.chunk

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        ScoredChunk(chunk=chunks[cid], score=score, source="rrf") for cid, score in ordered[:top_k]
    ]
