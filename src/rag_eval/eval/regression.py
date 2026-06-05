"""Regression gate: compare an eval report's metrics against committed floors.

The offline pipeline is deterministic, so floors can sit just below the current numbers and
still catch real regressions from code changes. Floors are committed in
``eval/regression_thresholds.json`` and version-controlled alongside the results.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from rag_eval.eval.runner import EvalReport


@dataclass(frozen=True, slots=True)
class RegressionResult:
    metric: str
    value: float
    floor: float
    passed: bool


def load_thresholds(path: str | Path) -> dict[str, float]:
    return {k: float(v) for k, v in json.loads(Path(path).read_text()).items()}


def check_regressions(
    report: EvalReport, thresholds: dict[str, float], epsilon: float = 1e-6
) -> list[RegressionResult]:
    """Check each thresholded metric's mean against its floor."""
    results: list[RegressionResult] = []
    for metric, floor in thresholds.items():
        est = report.metrics.get(metric)
        value = math.nan if est is None else est.mean
        passed = (not math.isnan(value)) and value >= floor - epsilon
        results.append(RegressionResult(metric, value, floor, passed))
    return results


def has_failures(results: list[RegressionResult]) -> bool:
    return any(not r.passed for r in results)


def format_report(results: list[RegressionResult]) -> str:
    lines = ["metric                value    floor    status"]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"{r.metric:<20} {r.value:>6.3f}  >= {r.floor:>5.3f}   {status}")
    return "\n".join(lines)
