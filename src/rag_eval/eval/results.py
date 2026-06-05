"""Persist eval reports, compare variants with significance tests, and render tables."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_eval.eval.runner import EvalReport
from rag_eval.eval.stats import Estimate, PairedTest, paired_bootstrap_test


def _nan_to_none(x: float) -> float | None:
    return None if math.isnan(x) else x


def _none_to_nan(x: float | None) -> float:
    return math.nan if x is None else x


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "config_name": report.config_name,
        "split": report.split,
        "seed": report.seed,
        "n_questions": report.n_questions,
        "metrics": {
            name: {
                "mean": _nan_to_none(est.mean),
                "ci_low": _nan_to_none(est.ci_low),
                "ci_high": _nan_to_none(est.ci_high),
                "n": est.n,
            }
            for name, est in report.metrics.items()
        },
        "operational": {k: _nan_to_none(v) for k, v in report.operational.items()},
        "per_question": {
            name: {qid: _nan_to_none(v) for qid, v in values.items()}
            for name, values in report.per_question.items()
        },
    }


def save_report(report: EvalReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_to_dict(report), indent=2, sort_keys=True))
    return out


def load_report(path: str | Path) -> EvalReport:
    """Reconstruct an EvalReport from a saved JSON file (records are not restored)."""
    data = json.loads(Path(path).read_text())
    metrics = {
        name: Estimate(
            _none_to_nan(m["mean"]), _none_to_nan(m["ci_low"]), _none_to_nan(m["ci_high"]), m["n"]
        )
        for name, m in data["metrics"].items()
    }
    per_question = {
        name: {qid: _none_to_nan(v) for qid, v in values.items()}
        for name, values in data["per_question"].items()
    }
    operational = {k: _none_to_nan(v) for k, v in data["operational"].items()}
    return EvalReport(
        config_name=data["config_name"],
        split=data["split"],
        seed=data["seed"],
        n_questions=data["n_questions"],
        metrics=metrics,
        per_question=per_question,
        operational=operational,
        records=[],
    )


def aligned_values(a: EvalReport, b: EvalReport, metric: str) -> tuple[list[float], list[float]]:
    """Per-question values for ``metric`` over questions both reports share (NaN pairs dropped)."""
    da = a.per_question.get(metric, {})
    db = b.per_question.get(metric, {})
    xa: list[float] = []
    xb: list[float] = []
    for qid in sorted(set(da) & set(db)):
        va, vb = da[qid], db[qid]
        if math.isnan(va) or math.isnan(vb):
            continue
        xa.append(va)
        xb.append(vb)
    return xa, xb


def compare(
    a: EvalReport, b: EvalReport, metric: str, iterations: int = 10000, seed: int = 0
) -> PairedTest:
    """Paired bootstrap test of whether ``a`` beats ``b`` on ``metric`` (per-question)."""
    xa, xb = aligned_values(a, b, metric)
    return paired_bootstrap_test(xa, xb, iterations=iterations, seed=seed)


def markdown_table(reports: Sequence[EvalReport], metric_names: Sequence[str]) -> str:
    """Render variants x metrics as a markdown table of ``mean [lo, hi]`` cells."""
    header = "| variant | n | " + " | ".join(metric_names) + " |"
    divider = "|" + "---|" * (len(metric_names) + 2)
    rows = [header, divider]
    for report in reports:
        cells = [report.config_name, str(report.n_questions)]
        for metric in metric_names:
            est = report.metrics.get(metric)
            cells.append(str(est) if est is not None else "n/a")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_summary(report: EvalReport) -> str:
    """A short human-readable summary of a single report."""
    lines = [f"=== {report.config_name} (split={report.split}, n={report.n_questions}) ==="]
    for name in sorted(report.metrics):
        lines.append(f"  {name:<20} {report.metrics[name]}")
    op = report.operational
    lines.append(
        f"  latency p50/p95 (ms): {op['p50_latency_ms']:.2f} / {op['p95_latency_ms']:.2f}"
        f"   tokens(prompt): {op['mean_prompt_tokens']:.0f}   $/q: {op['mean_cost_usd']:.4f}"
    )
    return "\n".join(lines)
