"""Agreement statistics and judge validation against the committed human labels."""

from __future__ import annotations

import math
from pathlib import Path

from rag_eval.eval.judge import OfflineJudge
from rag_eval.eval.validate import cohen_kappa, load_labels, spearman, validate_judge

LABELS = Path(__file__).resolve().parents[1] / "eval" / "judge_labels.jsonl"


def test_cohen_kappa_perfect_and_chance() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # total disagreement -> negative kappa
    assert cohen_kappa([True, True, False, False], [False, False, True, True]) < 0.0


def test_cohen_kappa_degenerate_single_category() -> None:
    # both raters always True: perfect agreement, expected agreement also 1 -> defined as 1.0
    assert cohen_kappa([True, True, True], [True, True, True]) == 1.0


def test_spearman_monotonic() -> None:
    assert math.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


def test_offline_judge_agrees_with_human_labels() -> None:
    examples = load_labels(LABELS)
    assert len(examples) >= 12
    report = validate_judge(OfflineJudge(), examples)
    assert report.n == len(examples)
    # the offline lexical proxy shows at least moderate agreement with human labels
    assert report.kappa_correctness >= 0.4
    assert report.spearman_correctness >= 0.4
    assert report.kappa_faithfulness >= 0.3


def test_labels_contain_disagreement_material() -> None:
    # the set must include both correct and incorrect human judgments (else kappa is meaningless)
    examples = load_labels(LABELS)
    assert any(e.human_correct for e in examples)
    assert any(not e.human_correct for e in examples)
