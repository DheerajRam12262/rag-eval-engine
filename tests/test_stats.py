"""Bootstrap CIs and paired significance tests."""

from __future__ import annotations

import math

from rag_eval.eval.stats import (
    bootstrap_mean_ci,
    paired_bootstrap_test,
    wilcoxon_signed_rank,
)


def test_bootstrap_mean_is_exact_and_ci_brackets_mean() -> None:
    est = bootstrap_mean_ci([0.0, 1.0, 0.0, 1.0], iterations=2000, seed=1)
    assert math.isclose(est.mean, 0.5)
    assert est.ci_low <= est.mean <= est.ci_high
    assert est.n == 4


def test_bootstrap_constant_has_zero_width_ci() -> None:
    est = bootstrap_mean_ci([0.7, 0.7, 0.7], seed=3)
    assert math.isclose(est.ci_low, 0.7) and math.isclose(est.ci_high, 0.7)


def test_bootstrap_empty_and_singleton() -> None:
    assert bootstrap_mean_ci([]).n == 0
    one = bootstrap_mean_ci([0.42])
    assert one.mean == 0.42 == one.ci_low == one.ci_high


def test_bootstrap_drops_nans() -> None:
    est = bootstrap_mean_ci([1.0, math.nan, 1.0], seed=0)
    assert est.n == 2 and math.isclose(est.mean, 1.0)


def test_paired_bootstrap_detects_consistent_difference() -> None:
    # a is consistently better than b by 0.2 on every item => significant, positive diff
    a = [0.9, 0.8, 0.85, 0.95, 0.7, 0.88]
    b = [0.7, 0.6, 0.65, 0.75, 0.5, 0.68]
    res = paired_bootstrap_test(a, b, iterations=5000, seed=2)
    assert res.diff_mean > 0
    assert res.p_value < 0.05


def test_paired_bootstrap_no_difference_is_not_significant() -> None:
    a = [0.5, 0.6, 0.4, 0.55, 0.45]
    res = paired_bootstrap_test(a, a, iterations=5000, seed=2)
    assert math.isclose(res.diff_mean, 0.0)
    assert res.p_value > 0.05


def test_wilcoxon_matches_direction() -> None:
    a = [0.9, 0.8, 0.85, 0.95, 0.7, 0.88, 0.91]
    b = [0.7, 0.6, 0.65, 0.75, 0.5, 0.68, 0.6]
    res = wilcoxon_signed_rank(a, b)
    assert res.diff_mean > 0
    assert res.p_value < 0.05


def test_wilcoxon_all_zero_diffs() -> None:
    res = wilcoxon_signed_rank([0.5, 0.5], [0.5, 0.5])
    assert res.p_value == 1.0
