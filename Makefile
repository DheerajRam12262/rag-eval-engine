# rag-eval-engine -- developer & reproducibility entrypoints.
# Everything runs inside the project venv (.venv), created by `make install`.

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test cov ci ingest serve eval smoke-eval repro plots clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install the package (editable) with dev extras.
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint: ## ruff + black --check (no changes).
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/black --check src tests

format: ## Auto-fix lint + format.
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/black src tests

typecheck: ## mypy (strict, src only).
	$(VENV)/bin/mypy

test: ## Run the test suite.
	$(VENV)/bin/pytest

cov: ## Run tests with coverage report.
	$(VENV)/bin/pytest --cov=rag_eval --cov-report=term-missing

ci: lint typecheck test ## Everything CI runs, locally.

# ---- Pipeline (Phase 1+) ----
ingest: ## Build dense + BM25 indexes from the corpus.
	$(PY) -m rag_eval.cli ingest --config config/base.yaml

serve: ## Run the FastAPI service (Phase 6).
	$(VENV)/bin/uvicorn rag_eval.api.app:app --reload --port 8000

# ---- Evaluation (Phase 4+) ----
eval: ## Run the full eval harness for the base config.
	$(PY) -m rag_eval.cli eval --config config/base.yaml

smoke-eval: ## Regression gate: run the smoke eval and fail if key metrics regress.
	$(PY) scripts/check_regression.py

plots: ## Regenerate result plots from eval/results/.
	$(PY) -m rag_eval.cli plots

repro: install ## One command to rebuild indexes and reproduce the README metrics table.
	$(PY) -m rag_eval.cli ingest --config config/base.yaml
	$(PY) -m rag_eval.cli ablate --suite config/variants --out eval/results

clean: ## Remove caches and build artifacts (keeps committed results).
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
