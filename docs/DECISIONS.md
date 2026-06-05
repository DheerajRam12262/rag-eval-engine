# Design Decisions

Each entry: the decision, why, the tradeoff, and (where relevant) "what changes at 100×/10M
docs". This is the document to defend in an interview.

---

## D1 — Offline-first with dependency inversion (Phase 0)
**Decision.** Define `Protocol` interfaces for the swappable parts (embedder, vector store,
generator, judge). Ship a real, deterministic *offline reference implementation* for each, plus
*adapters* for production backends (sentence-transformers, Qdrant, Anthropic/OpenAI) behind
optional installs.

**Why.** The CLAUDE.md demands "pluggable / toggleable via config" **and** "no fakery" **and**
reproducibility. Offline reference impls let the entire pipeline + eval harness run and be tested
in CI with no GPU and no API key — so `make repro` and the CI gate are honest, and ablations
(embedder choice, rerank on/off) are config flips rather than code branches.

**Tradeoff.** The offline impls (hashing embedder, lexical reranker, lexical-grounding judge) are
deterministic *proxies*, not SOTA quality. We are explicit that headline *quality* in a real
deployment uses the API/model backends; the *harness* is byte-identical either way.

## D2 — Reproducibility under non-deterministic LLMs (Phase 0)
**Decision.** Pin dependency versions exactly (captured from a real resolved install). Seed every
RNG from config. Pin model *snapshots* (not floating aliases). Cache judge/generation responses
keyed by `(prompt, model-snapshot)`; `repro` replays from cache by default, `--refresh` re-calls.

**Why.** LLM APIs are non-deterministic even at temperature 0 and models get deprecated, so
"reproduce every number" is only honest if stochastic calls are cached and separated from the
deterministic retrieval metrics.

**Tradeoff.** Cached responses can drift from the live model over time; we record the model
snapshot alongside the cache so staleness is visible.

---

## Seeds for upcoming entries (write these as the phases land)
- **D3 — Hybrid + RRF over dense-only.** Why reciprocal rank fusion (rank-based, score-scale
  agnostic) instead of score normalization; what `rrf_k` controls.
- **D4 — Validating the LLM judge.** Judge model ≠ generator model (self-preference bias);
  report Cohen's κ / Spearman vs a human-labeled subset; position/verbosity-bias mitigation.
- **D5 — Statistical rigor.** Why a bare mean is not evidence; bootstrap CIs over the question
  set; paired bootstrap / Wilcoxon for "A beats B"; effect size.
- **D6 — Gold set via pooling.** Labeling relevance from a single retriever biases recall; pool
  the top-k of all variants, label the union; dev/test split to avoid overfitting the harness.
- **D7 — Closed-book + oracle baselines.** Closed-book proves retrieval beats parametric memory;
  oracle isolates generation quality from retrieval quality.
- **D8 — Latency/quality/cost tradeoff per stage.** Where the cross-encoder spends its budget.
- **What changes at 10M docs.** ANN index (HNSW/IVF) + sharding, recall/latency tradeoff, reranker
  cost, cache strategy; how to A/B test a retrieval change in production.
