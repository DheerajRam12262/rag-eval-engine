# Project Plan — rag-eval-engine

Full architecture, build plan, and definition of done. The lean session context is `CLAUDE.md`.

## Architecture
```
                          INGESTION (offline)
  corpus ──▶ chunker ──▶ embeddings ──▶ vector store (dense)
                   └───▶ BM25 index (sparse)

                          QUERY PATH (online)
  query ──▶ hybrid retrieve (BM25 + dense) ──▶ RRF fusion ──▶ cross-encoder rerank
        ──▶ context assembly ──▶ LLM generate ──▶ answer + citations

                          EVAL PATH (the centerpiece)
  gold set ──▶ run variant ──▶ metrics ──▶ results store
           ──▶ bootstrap CIs + paired significance ──▶ ablation table + plots ──▶ CI gate
```

## Components
- **Ingestion:** pluggable chunking (fixed, recursive, semantic) — chunk size is an *ablation
  variable*. Indexes into both dense (vector store) and sparse (BM25).
- **Retrieval:** BM25 + dense, fused with Reciprocal Rank Fusion, then a cross-encoder reranker;
  each stage toggleable via config.
- **Generation:** grounded prompt with inline citations; abstains when context is weak.
- **Evaluation harness:** the differentiator (below).
- **Serving:** FastAPI `/query` (answer + citations + per-stage latency) and `/health`.

## The evaluation harness
- **Gold set:** 100–300 questions + reference answers + IDs of truly relevant chunks. **dev/test
  split** (tune on dev, report on test once). Includes no-answer / adversarial questions.
- **Relevance via pooling:** union the top-k of all variants and label that pool — labeling from
  one retriever biases recall.
- **Retrieval metrics:** recall@k, MRR, nDCG@k, context precision/recall.
- **Generation metrics:** faithfulness, answer relevance, correctness vs reference via
  **LLM-as-judge with a fixed rubric**.
- **Validate + de-bias the judge:** judge model ≠ generator; report Cohen's κ / Spearman vs human
  labels; mitigate position/verbosity bias; cache every judge call.
- **Operational metrics:** p50/p95 latency, tokens, $/query.
- **Ablation matrix:** `closed_book` (proves retrieval > parametric memory) and `oracle` (upper
  bound, isolates generation) baselines, then dense-only vs hybrid vs hybrid+rerank; chunk size;
  top-k; embedder choice.
- **Statistical rigor:** never a single run or bare mean. Bootstrap CIs over the question set;
  paired bootstrap / Wilcoxon for "A beats B" with effect size.
- **CI gate:** smoke eval runs in CI; merge blocked if key metrics regress past a threshold.

## Build plan (commit at the end of each phase; each ships working, tested code)
0. **Scaffold** — pyproject, Makefile, CI, config, `.env.example`. ✅
1. **Ingest** — corpus + chunking + dense + BM25 indexes; tests on chunk/index correctness. ✅
2. **Retrieval** — BM25, dense, RRF fusion; retrieval unit tests on a fixture corpus. ✅
3. **Rerank + generation** — cross-encoder rerank, context assembly, grounded generation + citations. ✅
4. **Eval harness** — gold set (pooling + split), metrics, judge, stats, runner, results store. ✅
5. **Ablation study** — ran closed-book + oracle + variants; committed results table (with CIs)
   + plots. ✅ (human-label κ validation pending — see DoD)
6. **Serving** — FastAPI + Docker + per-request latency/cost telemetry. ✅
7. **CI gate + docs** — regression gate; README with real results; `docs/DECISIONS.md`. ✅

## Definition of Done
- [x] `make repro` rebuilds indexes and reproduces the README ablation table + plots.
- [x] Ablation table **with CIs** shows hybrid+rerank beats dense-only with a **significant**
      paired test (recall@1 +0.316, p=0.001).
- [x] Closed-book baseline included — retrieval shown to add value over parametric knowledge.
- [~] LLM judge: judge≠generator enforced; offline proxy shipped. Human-label κ validation is the
      next addition (runs against the Anthropic judge).
- [x] Abstention works on no-answer questions (measured: abstention_accuracy 0.91 on test).
- [x] CI runs lint + types + tests + smoke-eval regression gate.
- [x] README: architecture, results (with CIs), latency/cost, limitations.

## Stretch
Query rewriting / HyDE; multi-hop retrieval; semantic + exact caching; hallucination detection;
eval on a second domain.
