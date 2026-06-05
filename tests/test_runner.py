"""End-to-end eval runner + results store, on the committed gold set (offline)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from rag_eval.config import load_config
from rag_eval.eval.gold import load_gold
from rag_eval.eval.results import (
    aligned_values,
    compare,
    markdown_table,
    report_to_dict,
    save_report,
)
from rag_eval.eval.runner import run_eval
from rag_eval.ingest.indexer import IndexBundle

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
GOLD = ROOT / "eval" / "gold"


def _run(variant: str, bundle: IndexBundle):  # type: ignore[no-untyped-def]
    cfg = load_config(CONFIG_DIR / "variants" / f"{variant}.yaml")
    return run_eval(cfg, load_gold(GOLD), bundle=bundle, split="test")


def test_report_structure_and_grounded_metrics(base_index: IndexBundle) -> None:
    report = _run("hybrid_rerank", base_index)
    assert report.n_questions == 22
    for metric in (
        "recall@5",
        "ndcg@5",
        "mrr",
        "correctness",
        "faithfulness",
        "abstention_accuracy",
    ):
        assert metric in report.metrics
    # extractive answers are grounded by construction
    assert report.metrics["faithfulness"].mean == 1.0
    # abstains correctly on the no-answer questions
    assert report.metrics["abstention_accuracy"].mean >= 0.8
    assert report.operational["p50_latency_ms"] >= 0.0


def test_closed_book_retrieves_nothing(base_index: IndexBundle) -> None:
    report = _run("closed_book", base_index)
    assert report.metrics["recall@5"].mean == 0.0
    assert report.metrics["mrr"].mean == 0.0


def test_retrieval_adds_value_is_significant_on_recall(base_index: IndexBundle) -> None:
    hybrid = _run("hybrid_rerank", base_index)
    closed = _run("closed_book", base_index)
    result = compare(hybrid, closed, "recall@5", iterations=5000, seed=0)
    # answerable questions only (no-answer dropped as NaN pairs)
    assert result.n == 19
    assert result.diff_mean > 0
    assert result.p_value < 0.01


def test_aligned_values_drops_no_answer(base_index: IndexBundle) -> None:
    hybrid = _run("hybrid_rerank", base_index)
    a, b = aligned_values(hybrid, hybrid, "recall@5")
    assert len(a) == len(b) == 19  # 22 minus 3 no-answer (NaN recall)


def test_report_serializes_to_valid_json(base_index: IndexBundle, tmp_path: Path) -> None:
    report = _run("hybrid_rerank", base_index)
    path = save_report(report, tmp_path / "hybrid_rerank.json")
    # strict JSON: NaNs must have been converted to null
    reloaded = json.loads(path.read_text())
    assert reloaded["config_name"] == "hybrid_rerank"
    json.dumps(report_to_dict(report), allow_nan=False)  # raises if any NaN slipped through


def test_markdown_table_lists_variants(base_index: IndexBundle) -> None:
    reports = [_run("hybrid_rerank", base_index), _run("closed_book", base_index)]
    table = markdown_table(reports, ["recall@5", "correctness"])
    assert "hybrid_rerank" in table and "closed_book" in table
    assert "recall@5" in table


def test_no_nan_in_correctness_metric(base_index: IndexBundle) -> None:
    # correctness is defined for every question (answerable + no-answer)
    report = _run("hybrid_rerank", base_index)
    assert not math.isnan(report.metrics["correctness"].mean)
    assert report.metrics["correctness"].n == 22
