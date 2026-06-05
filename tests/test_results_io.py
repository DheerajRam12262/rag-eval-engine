"""Results serialization round-trip and plot generation (smoke)."""

from __future__ import annotations

from pathlib import Path

from rag_eval.eval.plots import plot_metric_bars, plot_recall_grouped
from rag_eval.eval.results import load_report, save_report
from rag_eval.eval.runner import EvalReport
from rag_eval.eval.stats import Estimate


def _mk(name: str) -> EvalReport:
    metrics = {f"recall@{k}": Estimate(0.8, 0.6, 0.95, 10) for k in (1, 3, 5)}
    metrics["correctness"] = Estimate(0.5, 0.3, 0.7, 12)
    return EvalReport(
        config_name=name,
        split="test",
        seed=42,
        n_questions=12,
        metrics=metrics,
        per_question={"recall@1": {"q1": 1.0, "q2": float("nan")}},
        operational={
            "p50_latency_ms": 1.0,
            "p95_latency_ms": 2.0,
            "mean_prompt_tokens": 10.0,
            "mean_completion_tokens": 3.0,
            "mean_cost_usd": 0.0,
        },
    )


def test_report_round_trip(tmp_path: Path) -> None:
    path = save_report(_mk("v"), tmp_path / "v.json")
    back = load_report(path)
    assert back.config_name == "v"
    assert back.metrics["recall@1"].mean == 0.8
    assert back.metrics["correctness"].n == 12
    # NaN preserved through none<->nan conversion
    import math

    assert math.isnan(back.per_question["recall@1"]["q2"])


def test_plots_are_written(tmp_path: Path) -> None:
    reports = [_mk("a"), _mk("b")]
    recall_png = plot_recall_grouped(reports, [1, 3, 5], tmp_path / "recall.png")
    corr_png = plot_metric_bars(reports, "correctness", tmp_path / "corr.png")
    assert recall_png.exists() and recall_png.stat().st_size > 0
    assert corr_png.exists() and corr_png.stat().st_size > 0
