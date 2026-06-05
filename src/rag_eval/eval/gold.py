"""Gold question set: schema, loading, dev/test split, and pooling.

Relevance is recorded at the document level (robust to chunk-size ablations). The set includes
no-answer / adversarial questions (empty ``relevant_doc_ids``) to measure abstention. A dev/test
split lets us tune knobs on dev and report headline numbers on test exactly once.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_eval.types import ScoredChunk


@dataclass(frozen=True, slots=True)
class GoldQuestion:
    id: str
    question: str
    reference_answer: str
    relevant_doc_ids: tuple[str, ...]
    no_answer: bool = False
    split: str = "test"


def _parse(record: dict[str, Any], split: str) -> GoldQuestion:
    relevant = tuple(str(d) for d in record.get("relevant_doc_ids", []))
    no_answer = bool(record.get("no_answer", False))
    if no_answer and relevant:
        raise ValueError(f"{record['id']}: no-answer questions must have no relevant docs")
    if not no_answer and not relevant:
        raise ValueError(f"{record['id']}: answerable questions need >=1 relevant doc")
    return GoldQuestion(
        id=str(record["id"]),
        question=str(record["question"]),
        reference_answer=str(record.get("reference_answer", "")),
        relevant_doc_ids=relevant,
        no_answer=no_answer,
        split=split,
    )


def _read_jsonl(path: Path, split: str) -> list[GoldQuestion]:
    out: list[GoldQuestion] = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(_parse(json.loads(line), split))
    return out


def load_gold(path: str | Path) -> list[GoldQuestion]:
    """Load gold questions.

    If ``path`` is a directory, reads ``dev.jsonl`` and ``test.jsonl`` and tags each with its
    split. If it's a file, the split is inferred from the filename stem.
    """
    p = Path(path)
    if p.is_dir():
        questions: list[GoldQuestion] = []
        for split in ("dev", "test"):
            f = p / f"{split}.jsonl"
            if f.exists():
                questions.extend(_read_jsonl(f, split))
        if not questions:
            raise FileNotFoundError(f"no dev.jsonl/test.jsonl in {p}")
        return questions
    return _read_jsonl(p, p.stem)


def filter_split(gold: Sequence[GoldQuestion], split: str) -> list[GoldQuestion]:
    """Return only questions in the named split ('dev' or 'test')."""
    return [q for q in gold if q.split == split]


def pool_candidates(rankings: Sequence[Sequence[ScoredChunk]], depth: int) -> list[str]:
    """Union of the top-``depth`` doc ids across several systems' rankings.

    This is the unbiased way to build a relevance-judgment pool: labeling from a single
    retriever's output would bias recall toward that retriever.
    """
    pool: set[str] = set()
    for ranking in rankings:
        for scored in ranking[:depth]:
            pool.add(scored.chunk.doc_id)
    return sorted(pool)
