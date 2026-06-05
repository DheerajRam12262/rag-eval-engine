# rag-eval-engine -- developer & reproducibility entrypoints.
# Tooling runs from the project venv when present (.venv, created by `make install`),
# and falls back to the active interpreter otherwise (e.g. on CI, where deps are installed
# system-wide). So every target works both locally and in CI.

VENV := .venv
BIN  := $(shell [ -x $(VENV)/bin/python ] && echo $(VENV)/bin/ )
PY   := $(BIN)python

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test cov ci ingest serve eval smoke-eval repro plots clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install the package (editable) with dev extras.
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

lint: ## ruff + black --check (no changes).
	$(BIN)ruff check src tests scripts
	$(BIN)black --check src tests scripts

format: ## Auto-fix lint + format.
	$(BIN)ruff check --fix src tests scripts
	$(BIN)black src tests scripts

typecheck: ## mypy (strict, src only).
	$(BIN)mypy

test: ## Run the test suite.
	$(BIN)pytest

cov: ## Run tests with coverage report.
	$(BIN)pytest --cov=rag_eval --cov-report=term-missing

ci: lint typecheck test ## Everything CI runs, locally.

# ---- Pipeline ----
ingest: ## Build dense + BM25 indexes from the corpus.
	$(PY) -m rag_eval.cli ingest --config config/base.yaml

serve: ## Run the FastAPI service.
	$(BIN)uvicorn rag_eval.api.app:app --reload --port 8000

# ---- Evaluation ----
eval: ## Run the eval harness for the base config.
	$(PY) -m rag_eval.cli eval --config config/base.yaml

smoke-eval: ## Regression gate: run the smoke eval and fail if key metrics regress.
	$(PY) scripts/check_regression.py

plots: ## Regenerate result plots from eval/results/.
	$(PY) -m rag_eval.cli plots

repro: ## Rebuild indexes and reproduce the README ablation table + plots.
	$(PY) -m rag_eval.cli ingest --config config/base.yaml
	$(PY) -m rag_eval.cli ablate --suite config/variants --out eval/results

clean: ## Remove caches and build artifacts (keeps committed results).
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
