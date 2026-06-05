# rag-eval-engine

<!-- CI badge is inserted after the first push to GitHub (needs the repo slug). -->

> Most RAG systems are *demoed*, not *measured*. **rag-eval-engine** is a hybrid-retrieval RAG
> service over a non-trivial corpus with a rigorous, reproducible **evaluation harness** that
> quantifies retrieval quality, answer faithfulness, latency, and cost — and **gates regressions
> in CI**.

The differentiator is the eval harness: bootstrap confidence intervals, paired significance
tests, a validated LLM-as-judge, and baselines (closed-book + oracle) that *prove* retrieval adds
value rather than assuming it.

> **Status:** building — Phase 0 (scaffold) complete. The results table below is populated in
> Phase 5; this README is kept honest to what exists at each commit.

## Architecture

```
                          INGESTION (offline)
  corpus ──▶ chunker ──▶ embeddings ──▶ vector store (dense)
                   └───▶ BM25 index (sparse)

                          QUERY PATH (online)
  query ──▶ hybrid retrieve (BM25 + dense) ──▶ RRF fusion ──▶ cross-encoder rerank
        ──▶ context assembly ──▶ LLM generate ──▶ answer + citations
        (every stage emits latency + token/cost telemetry)

                          EVAL PATH (the centerpiece)
  gold set ──▶ run variant ──▶ metrics ──▶ results store ──▶ bootstrap CIs + significance
           ──▶ ablation table + plots ──▶ CI regression gate
```

### Runs offline by default
Every component that normally needs a GPU or an API key (embedder, vector store, generator,
judge) has a **real, deterministic offline reference implementation**, with production adapters
(sentence-transformers, Qdrant, Anthropic/OpenAI) behind optional installs. So the full pipeline
and eval harness run — and are tested in CI — with **no GPU and no API key**. See
[docs/DECISIONS.md](docs/DECISIONS.md).

## Results

_Populated in Phase 5 (`make repro`)._ Will show the ablation matrix — closed-book, oracle,
dense-only, hybrid, hybrid+rerank — with `recall@k`, `nDCG@k`, faithfulness, latency and cost,
each as `mean ± 95% CI`, plus a paired significance test for the headline comparison.

## How to run

```bash
make install        # venv + pinned deps (editable, with dev extras)
make ci             # ruff + black --check + mypy + pytest (what CI runs)
make ingest         # build dense + BM25 indexes from the sample corpus   (Phase 1+)
make eval           # run the full eval harness                            (Phase 4+)
make repro          # rebuild indexes and reproduce the results table      (Phase 5+)
make serve          # FastAPI service on :8000                             (Phase 6+)
```

Optional production backends:

```bash
pip install -e ".[embeddings]"   # sentence-transformers + torch
pip install -e ".[qdrant]"       # qdrant-client
pip install -e ".[llm]"          # anthropic + openai
```

Copy `.env.example` to `.env` to point at real backends. Nothing in `.env` is required for the
default offline path.

## Limitations
- The offline reference implementations (hashing embedder, lexical reranker/judge) are
  deterministic *proxies* chosen for reproducibility, not state-of-the-art quality. Headline
  quality numbers in a real deployment use the API/model backends; the harness is identical.
- The in-memory vector store does **exact** nearest-neighbor (O(N·d)); production uses ANN.
  "What changes at 10M docs" is documented in [docs/DECISIONS.md](docs/DECISIONS.md).

## Project docs
- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) — architecture, build plan, definition of done.
- [docs/DECISIONS.md](docs/DECISIONS.md) — design decisions, tradeoffs, and "what changes at 100×".

## License
MIT.
