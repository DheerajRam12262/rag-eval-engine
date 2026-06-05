"""The committed thresholds must pass on the current (deterministic) code."""

from __future__ import annotations

from pathlib import Path

from rag_eval.config import load_config
from rag_eval.eval.gold import load_gold
from rag_eval.eval.regression import (
    check_regressions,
    format_report,
    has_failures,
    load_thresholds,
)
from rag_eval.eval.runner import run_eval
from rag_eval.ingest.indexer import IndexBundle

ROOT = Path(__file__).resolve().parents[1]


def test_current_code_meets_committed_thresholds(base_index: IndexBundle) -> None:
    config = load_config(ROOT / "config" / "variants" / "hybrid_rerank.yaml")
    report = run_eval(config, load_gold(ROOT / "eval" / "gold"), bundle=base_index, split="test")
    thresholds = load_thresholds(ROOT / "eval" / "regression_thresholds.json")
    results = check_regressions(report, thresholds)
    assert not has_failures(results), "\n" + format_report(results)
    assert {r.metric for r in results} == set(thresholds)


def test_floor_above_value_is_a_failure(base_index: IndexBundle) -> None:
    config = load_config(ROOT / "config" / "variants" / "hybrid_rerank.yaml")
    report = run_eval(config, load_gold(ROOT / "eval" / "gold"), bundle=base_index, split="test")
    results = check_regressions(report, {"correctness": 0.99})  # impossible floor
    assert has_failures(results)
