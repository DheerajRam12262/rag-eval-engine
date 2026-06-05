"""Grounded prompt assembly. Context chunks are numbered so the model can cite them as [n]."""

from __future__ import annotations

from collections.abc import Sequence

from rag_eval.types import Chunk

GROUNDED_INSTRUCTIONS = (
    "Answer the question using ONLY the numbered context below. "
    "Cite the sources you use with bracketed numbers like [1] or [2]. "
    "If the context does not contain the answer, reply exactly: "
    '"I don\'t have enough information in the provided context to answer that."'
)


def assemble_context(chunks: Sequence[Chunk]) -> str:
    """Render chunks as a numbered context block (1-based)."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        prefix = f"{chunk.title}: " if chunk.title else ""
        lines.append(f"[{i}] {prefix}{chunk.text}")
    return "\n".join(lines)


def build_prompt(query: str, chunks: Sequence[Chunk]) -> str:
    """Assemble the full grounded prompt sent to a generator."""
    context = assemble_context(chunks) if chunks else "(no context retrieved)"
    return f"{GROUNDED_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
