"""Matplotlib plots for the ablation study (headless Agg backend)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rag_eval.eval.runner import EvalReport  # noqa: E402


def _err(
    reports: Sequence[EvalReport], metric: str
) -> tuple[list[float], list[float], list[float]]:
    means = [r.metrics[metric].mean for r in reports]
    low = [r.metrics[metric].mean - r.metrics[metric].ci_low for r in reports]
    high = [r.metrics[metric].ci_high - r.metrics[metric].mean for r in reports]
    return means, low, high


def plot_recall_grouped(reports: Sequence[EvalReport], ks: Sequence[int], out_path: Path) -> Path:
    """Grouped bar chart of recall@k per variant with 95% CI error bars."""
    names = [r.config_name for r in reports]
    x = np.arange(len(names))
    width = 0.8 / len(ks)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, k in enumerate(ks):
        means, low, high = _err(reports, f"recall@{k}")
        ax.bar(x + i * width, means, width, yerr=[low, high], capsize=3, label=f"recall@{k}")
    ax.set_xticks(x + width * (len(ks) - 1) / 2)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall")
    ax.set_title("Retrieval recall@k by variant (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_metric_bars(reports: Sequence[EvalReport], metric: str, out_path: Path) -> Path:
    """Single-metric bar chart per variant with 95% CI error bars."""
    names = [r.config_name for r in reports]
    means, low, high = _err(reports, metric)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, means, yerr=[low, high], capsize=4, color="#4C72B0")
    ax.set_ylim(0, max(1.05, max(means) * 1.2 if means else 1.05))
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by variant (95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
