# CLAUDE.md — rag-eval-engine

<!-- Loaded every session: kept lean on purpose. Full plan: docs/PROJECT_PLAN.md. -->

## Project
- **Name:** `rag-eval-engine` (repo named this exactly, everywhere).
- **Problem:** Most RAG systems are demoed, not measured. This is a hybrid-retrieval RAG service
  with a rigorous, reproducible **evaluation harness** that quantifies retrieval quality, answer
  faithfulness, latency, and cost — and **gates regressions in CI**.
- **Role signal:** GenAI / LLM Engineer.

## How we work together
- Built to show senior-level engineering AND to defend every decision in an interview.
- **Explain before you build.** State the approach + 1–2 tradeoffs, then implement. Teach the *why*.
- **Commit incrementally** with Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`,
  `refactor:`). Never dump a feature — or the project — in one commit. History is a hiring signal.
- **No fakery.** No placeholder text, no stubs presented as working. Label intentional stubs.
- **Clarity over cleverness.** A reviewer's "why?" is answered in code, README, or `docs/DECISIONS.md`.

## Engineering bar (non-negotiable)
- **Reproducibility:** pinned deps; seed every RNG; config-driven (no magic numbers);
  `make repro` reproduces every number. LLM calls are cached + model snapshots pinned (see D2).
- **Structure & typing:** `src/` layout; full type hints; `mypy --strict` clean; docstrings on
  public APIs.
- **Testing:** real unit + integration tests (no `assert True`); run in CI; pass before merge.
- **Data & secrets:** never commit raw corpora or secrets; `.gitignore` + build scripts;
  `.env.example` committed, real `.env` ignored.
- **CI/CD:** GitHub Actions = `ruff` + `black --check` + `mypy` + `pytest` on every push/PR;
  badge in README; `pre-commit` hooks.
- **Docs:** README skimmable in 60s (problem → architecture → results → run → limitations);
  `docs/DECISIONS.md` for tradeoffs + "what changes at 100×".

## Anti-patterns — do NOT
- Ship a chatbot with no eval. Report a single run / bare mean with no CI or significance.
- Use an unvalidated (or self-judging) LLM judge as proof. Skip the closed-book baseline.
- Label relevance from one retriever. Commit the corpus or API keys.
- Land the whole thing in one "initial RAG system" commit.

> Full architecture, build plan (phases 0–7), and Definition of Done: **docs/PROJECT_PLAN.md**.
