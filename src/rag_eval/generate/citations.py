"""Parse bracketed citation markers ([1], [2], ...) from a generated answer."""

from __future__ import annotations

import re

from rag_eval.types import Chunk

_CITE_RE = re.compile(r"\[(\d+)\]")


def parse_citation_indices(text: str, num_sources: int) -> list[int]:
    """Return the valid 1-based source indices cited in ``text``, in order of first appearance."""
    seen: list[int] = []
    for match in _CITE_RE.findall(text):
        n = int(match)
        if 1 <= n <= num_sources and n not in seen:
            seen.append(n)
    return seen


def citation_chunk_ids(text: str, context: list[Chunk]) -> list[str]:
    """Map cited indices in ``text`` to the chunk ids of the numbered ``context``."""
    return [context[i - 1].id for i in parse_citation_indices(text, len(context))]
