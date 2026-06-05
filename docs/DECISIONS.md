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

## D3 — Hybrid + RRF over dense-only
**Decision.** Fuse BM25 and dense with Reciprocal Rank Fusion (`score = Σ 1/(rrf_k + rank)`)
rather than normalizing and adding scores.

**Why.** BM25 and cosine live on different, unbounded scales; min-max/z-score normalization is
brittle and corpus-dependent. RRF uses only ranks, so it is scale-agnostic and has one
interpretable knob (`rrf_k`, which flattens the influence of top ranks).

**Tradeoff.** RRF discards score magnitude — a weakly-relevant rank-1 counts like a strongly
relevant one. The cross-encoder reranker restores precision on the fused candidates. **Evidence:**
`hybrid` lifts recall@1 over `dense_only` (0.68→0.79), and `hybrid_rerank` reaches 1.00
(+0.316 vs dense, paired bootstrap p=0.001).

## D4 — Validating and de-biasing the LLM judge
**Decision.** Judge model MUST differ from the generator (enforced by a config validator). Score
faithfulness/relevance/correctness from a fixed rubric with structured output; cache every call.
Validate against a human-labeled subset and report Cohen's κ / Spearman before trusting it.

**Why.** A model grading its own output exhibits self-preference bias; an unvalidated judge is not
evidence. Caching makes the (non-deterministic) judge reproducible.

**Status — validated.** `validate-judge` scores a 16-example human-labeled set and reports
agreement. The offline lexical proxy lands at **Cohen's κ = 0.60 (correctness), 0.46
(faithfulness), Spearman = 0.70** — substantial-but-imperfect, with disagreements concentrated on
semantic paraphrases and short numeric references (where a lexical judge is weakest). That is the
honest ceiling of the proxy and the motivation for the Anthropic judge, which the *same* harness
validates (swap the judge backend; the labels are fixed).

## D5 — Statistical rigor
**Decision.** Never report a bare mean or a single run. Every metric is a bootstrap mean CI over
the question set; every "A beats B" is a paired bootstrap test (Wilcoxon as a non-parametric
cross-check) on per-question scores.

**Why.** With n=22, "0.68 vs 1.00" could be noise. The paired test answers whether the
per-question advantage is real (recall@1: p=0.001 → yes; recall@3: p=0.24 → can't tell at this n).

## D6 — Gold set via pooling + dev/test split
**Decision.** Relevance is recorded at the document level (survives chunk-size ablations). A
`pool_candidates` utility unions the top-k of all variants for unbiased labeling. The set is split
dev/test.

**Why.** Labeling from one retriever's output biases recall toward that retriever. Tuning and
reporting on the same split overfits the harness. (On this small authored corpus relevance is
exhaustively known, so pooling is methodological rather than load-bearing — stated honestly.)

## D7 — Closed-book + oracle baselines
**Decision.** Always include `closed_book` (no retrieval) and `oracle` (gold context) rows.

**Why / evidence.** `closed_book` proves retrieval beats parametric memory (recall 0, correctness
0.14). `oracle` isolates generation from retrieval — its 0.39 correctness ceiling showed that the
*extractive generator*, not retrieval, is the bottleneck here (correctness is flat ~0.30 across all
retrieval variants). Without these baselines you cannot attribute wins to the right component.

## D8 — Latency / quality / cost per stage
Per-stage telemetry (offline, warm): retrieve ≈ 0.2–1.3 ms, rerank ≈ 0.25 ms, generate ≈ 0.14 ms.
With API backends, generation dominates latency and is the only nonzero cost; the cross-encoder
reranker is the main *local* compute. The harness records p50/p95 latency, tokens, and $/query so
the quality/latency/cost tradeoff is visible per variant.

## What changes at 10M docs
- **Vector store:** exact NN (O(N·d)) → ANN (HNSW/IVF-PQ); accept a recall/latency tradeoff and
  measure it (the harness already reports recall, so the ANN recall hit is quantifiable).
- **Sharding** the index; BM25 moves to OpenSearch/Elasticsearch.
- **Reranker cost** grows with candidate depth → cap `candidate_k`, or cascade (cheap filter →
  cross-encoder on a short list).
- **Caching:** exact + semantic query caches; precomputed embeddings.
- **Shipping a retrieval change:** offline eval gate (this harness) → online A/B with
  interleaving, watching recall proxies, latency p95, and downstream task metrics.
