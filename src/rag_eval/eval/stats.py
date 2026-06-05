"""Statistical rigor for the eval harness.

A bare mean is not evidence. Every reported metric carries a bootstrap confidence interval over
the question set, and every "A beats B" claim is backed by a paired significance test on the
per-question scores. Two tests are provided: a paired bootstrap (primary; assumption-free) and a
Wilcoxon signed-rank (secondary; classic non-parametric), implemented without scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with a bootstrap confidence interval."""

    mean: float
    ci_low: float
    ci_high: float
    n: int

    def __str__(self) -> str:
        if self.n == 0:
            return "n/a"
        return f"{self.mean:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}]"


@dataclass(frozen=True, slots=True)
class PairedTest:
    """Result of a paired comparison between two systems' per-question scores."""

    diff_mean: float  # mean(a - b)
    p_value: float
    n: int


def bootstrap_mean_ci(
    values: Sequence[float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for the mean of ``values`` (NaNs are dropped)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return Estimate(math.nan, math.nan, math.nan, 0)
    if n == 1:
        v = float(arr[0])
        return Estimate(v, v, v, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iterations, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return Estimate(
        mean=float(arr.mean()),
        ci_low=float(np.quantile(boot_means, alpha)),
        ci_high=float(np.quantile(boot_means, 1.0 - alpha)),
        n=n,
    )


def paired_bootstrap_test(
    a: Sequence[float],
    b: Sequence[float],
    iterations: int = 10000,
    seed: int = 0,
) -> PairedTest:
    """Two-sided paired bootstrap test of H0: mean(a - b) == 0.

    Resamples question indices with replacement and reports the two-sided bootstrap p-value
    (twice the smaller tail mass beyond zero).
    """
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    if da.shape != db.shape:
        raise ValueError("paired test requires equal-length inputs")
    diff = da - db
    n = int(diff.size)
    if n == 0:
        return PairedTest(math.nan, math.nan, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iterations, n))
    boot = diff[idx].mean(axis=1)
    p_low = float(np.mean(boot <= 0.0))
    p_high = float(np.mean(boot >= 0.0))
    p_value = min(1.0, 2.0 * min(p_low, p_high))
    return PairedTest(diff_mean=float(diff.mean()), p_value=p_value, n=n)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties resolved to their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _normal_sf(x: float) -> float:
    """Survival function of the standard normal (1 - CDF), via erfc."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> PairedTest:
    """Wilcoxon signed-rank test (normal approximation). Zero-differences dropped."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[diff != 0.0]
    n = int(diff.size)
    if n == 0:
        return PairedTest(0.0, 1.0, 0)
    ranks = average_ranks(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w == 0.0:
        return PairedTest(float(np.mean(diff)), 1.0, n)
    z = (w - mean_w) / math.sqrt(var_w)
    p_value = min(1.0, 2.0 * _normal_sf(abs(z)))
    return PairedTest(diff_mean=float(np.mean(diff)), p_value=p_value, n=n)
