"""Validate the LLM judge against human labels.

An unvalidated judge is not evidence. This module runs a judge over a human-labeled subset and
reports agreement: Cohen's kappa on binarized correctness/faithfulness, plus Spearman between the
judge's continuous correctness and the human label. The same harness validates the offline proxy
*and* the Anthropic judge (swap the judge backend); only the labels are fixed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag_eval.eval.judge import Judge
from rag_eval.eval.stats import average_ranks


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters. 1.0 = perfect, 0.0 = chance-level."""
    if len(a) != len(b):
        raise ValueError("rater sequences must be equal length")
    n = len(a)
    if n == 0:
        return math.nan
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa = sum(1 for x in a if x) / n
    pb = sum(1 for x in b if x) / n
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks; scipy-free)."""
    if len(x) != len(y):
        raise ValueError("inputs must be equal length")
    if len(x) < 2:
        return math.nan
    rx = average_ranks(np.asarray(x, dtype=float))
    ry = average_ranks(np.asarray(y, dtype=float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom == 0.0:
        return math.nan
    return float((rx * ry).sum() / denom)


@dataclass(frozen=True, slots=True)
class LabeledExample:
    id: str
    question: str
    reference_answer: str
    answer: str
    context: str
    abstained: bool
    no_answer: bool
    human_correct: bool
    human_faithful: bool


@dataclass(frozen=True, slots=True)
class AgreementReport:
    n: int
    threshold: float
    kappa_correctness: float
    kappa_faithfulness: float
    spearman_correctness: float


def load_labels(path: str | Path) -> list[LabeledExample]:
    examples: list[LabeledExample] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r: dict[str, Any] = json.loads(line)
        examples.append(
            LabeledExample(
                id=str(r["id"]),
                question=str(r["question"]),
                reference_answer=str(r.get("reference_answer", "")),
                answer=str(r["answer"]),
                context=str(r.get("context", "")),
                abstained=bool(r["abstained"]),
                no_answer=bool(r["no_answer"]),
                human_correct=bool(r["human_correct"]),
                human_faithful=bool(r["human_faithful"]),
            )
        )
    return examples


def validate_judge(
    judge: Judge, examples: Sequence[LabeledExample], threshold: float = 0.5
) -> AgreementReport:
    """Score each labeled example with ``judge`` and compute agreement vs human labels."""
    judge_correct: list[bool] = []
    judge_faithful: list[bool] = []
    judge_correctness_vals: list[float] = []
    human_correct: list[bool] = []
    human_faithful: list[bool] = []

    for ex in examples:
        scores = judge.score(
            question=ex.question,
            reference_answer=ex.reference_answer,
            answer=ex.answer,
            context=ex.context,
            abstained=ex.abstained,
            no_answer=ex.no_answer,
        )
        judge_correct.append(scores.correctness >= threshold)
        judge_faithful.append(scores.faithfulness >= threshold)
        judge_correctness_vals.append(scores.correctness)
        human_correct.append(ex.human_correct)
        human_faithful.append(ex.human_faithful)

    return AgreementReport(
        n=len(examples),
        threshold=threshold,
        kappa_correctness=cohen_kappa(judge_correct, human_correct),
        kappa_faithfulness=cohen_kappa(judge_faithful, human_faithful),
        spearman_correctness=spearman(
            judge_correctness_vals, [1.0 if h else 0.0 for h in human_correct]
        ),
    )


def format_agreement(report: AgreementReport) -> str:
    return (
        f"judge validation (n={report.n}, threshold={report.threshold}):\n"
        f"  Cohen's kappa (correctness):  {report.kappa_correctness:.3f}\n"
        f"  Cohen's kappa (faithfulness): {report.kappa_faithfulness:.3f}\n"
        f"  Spearman (correctness):       {report.spearman_correctness:.3f}"
    )
