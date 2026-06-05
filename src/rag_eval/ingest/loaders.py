"""Corpus loaders. The sample corpus is a JSONL file of ``{id, title, text}`` records."""

from __future__ import annotations

import json
from pathlib import Path

from rag_eval.types import Document


def load_corpus(path: str | Path) -> list[Document]:
    """Load documents from a JSONL file (or a directory containing ``corpus.jsonl``)."""
    p = Path(path)
    jsonl = p / "corpus.jsonl" if p.is_dir() else p
    if not jsonl.exists():
        raise FileNotFoundError(f"corpus not found: {jsonl}")

    docs: list[Document] = []
    for raw in jsonl.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        record = json.loads(line)
        docs.append(
            Document(
                id=str(record["id"]),
                text=str(record["text"]),
                title=str(record.get("title", "")),
                metadata=dict(record.get("metadata", {})),
            )
        )
    if not docs:
        raise ValueError(f"corpus is empty: {jsonl}")
    return docs
