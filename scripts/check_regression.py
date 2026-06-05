#!/usr/bin/env python
"""CI regression gate: run a smoke eval and fail if key metrics regress past their floors.

Invoked by `make smoke-eval` and the CI workflow. Uses the deterministic offline backend, so it
needs no GPU or API key.
"""

from __future__ import annotations

import sys

from rag_eval.config import load_config
from rag_eval.eval.gold import load_gold
from rag_eval.eval.regression import (
    check_regressions,
    format_report,
    has_failures,
    load_thresholds,
)
from rag_eval.eval.runner import run_eval

CONFIG = "config/variants/hybrid_rerank.yaml"
GOLD = "eval/gold"
THRESHOLDS = "eval/regression_thresholds.json"


def main() -> int:
    report = run_eval(load_config(CONFIG), load_gold(GOLD), split="test")
    results = check_regressions(report, load_thresholds(THRESHOLDS))
    print(format_report(results))
    if has_failures(results):
        print("\nREGRESSION DETECTED: a key metric dropped below its floor.", file=sys.stderr)
        return 1
    print("\nOK: all gated metrics are within thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
