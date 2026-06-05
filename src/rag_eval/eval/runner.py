"""Run a pipeline variant over the gold set and aggregate metrics with confidence intervals.

For each question we record retrieval ranking, abstention, judge scores, latency and cost, then
aggregate every metric as a bootstrap mean CI. Per-question values are retained so two variants
can be compared with a paired significance test (see :mod:`rag_eval.eval.results`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rag_eval.config import Config, seed_everything
from rag_eval.eval.gold import GoldQuestion, filter_split
from rag_eval.eval.judge import Judge, JudgeScores, get_judge
from rag_eval.eval.metrics import (
    abstention_correct,
    ndcg_at_k,
    ranked_doc_ids,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval.eval.stats import Estimate, bootstrap_mean_ci
from rag_eval.ingest.indexer import IndexBundle, build_index
from rag_eval.pipeline import RagPipeline


@dataclass(slots=True)
class QuestionRecord:
    question_id: str
    no_answer: bool
    abstained: bool
    answer: str
    citations: list[str]
    ranked_docs: list[str]
    relevant_docs: list[str]
    judge: JudgeScores
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass(slots=True)
class EvalReport:
    config_name: str
    split: str
    seed: int
    n_questions: int
    metrics: dict[str, Estimate]
    per_question: dict[str, dict[str, float]]  # metric -> {question_id: value (may be NaN)}
    operational: dict[str, float]
    records: list[QuestionRecord] = field(default_factory=list)


def _oracle_chunk_ids(bundle: IndexBundle, relevant_docs: tuple[str, ...]) -> list[str]:
    relevant = set(relevant_docs)
    return [c.id for c in bundle.chunks if c.doc_id in relevant]


def run_eval(
    config: Config,
    gold: list[GoldQuestion],
    *,
    bundle: IndexBundle | None = None,
    judge: Judge | None = None,
    split: str | None = None,
    smoke: bool = False,
) -> EvalReport:
    """Evaluate ``config`` over the gold questions, returning an aggregated report."""
    seed_everything(config.seed)
    if bundle is None:
        bundle = build_index(config, persist=False)
    if judge is None:
        judge = get_judge(config.eval.judge)

    questions = filter_split(gold, split) if split else list(gold)
    if smoke:
        questions = questions[: config.eval.smoke.max_questions]

    is_oracle = config.retrieval.mode == "oracle"
    pipeline = RagPipeline(bundle, config)
    k_values = config.eval.metrics.k_values

    records: list[QuestionRecord] = []
    samples: dict[str, dict[str, float]] = {}

    def record_metric(name: str, qid: str, value: float) -> None:
        samples.setdefault(name, {})[qid] = value

    for q in questions:
        oracle_ids = _oracle_chunk_ids(bundle, q.relevant_doc_ids) if is_oracle else None
        result = pipeline.query(q.question, oracle_chunk_ids=oracle_ids)
        ranked_docs = ranked_doc_ids(result.ranked)
        relevant = list(q.relevant_doc_ids)
        context_text = "\n".join(c.chunk.text for c in result.contexts)
        scores = judge.score(
            question=q.question,
            reference_answer=q.reference_answer,
            answer=result.answer,
            context=context_text,
            abstained=result.abstained,
            no_answer=q.no_answer,
        )

        records.append(
            QuestionRecord(
                question_id=q.id,
                no_answer=q.no_answer,
                abstained=result.abstained,
                answer=result.answer,
                citations=result.citations,
                ranked_docs=ranked_docs,
                relevant_docs=relevant,
                judge=scores,
                latency_ms=result.latencies.total_ms,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cost_usd=result.usage.cost_usd,
            )
        )

        # Retrieval metrics: answerable only (NaN otherwise, dropped in aggregation).
        for k in k_values:
            record_metric(f"recall@{k}", q.id, recall_at_k(ranked_docs, relevant, k))
            record_metric(f"ndcg@{k}", q.id, ndcg_at_k(ranked_docs, relevant, k))
        record_metric("mrr", q.id, reciprocal_rank(ranked_docs, relevant) if relevant else math.nan)

        # Generation metrics.
        record_metric("faithfulness", q.id, math.nan if result.abstained else scores.faithfulness)
        record_metric(
            "answer_relevance",
            q.id,
            scores.answer_relevance if (not q.no_answer and not result.abstained) else math.nan,
        )
        record_metric("correctness", q.id, scores.correctness)
        record_metric(
            "abstention_accuracy", q.id, abstention_correct(q.no_answer, result.abstained)
        )

    boot = config.eval.bootstrap
    metrics = {
        name: bootstrap_mean_ci(
            list(values.values()),
            iterations=boot.iterations,
            confidence=boot.confidence,
            seed=config.seed,
        )
        for name, values in samples.items()
    }

    latencies = [r.latency_ms for r in records]
    operational = {
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else math.nan,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else math.nan,
        "mean_prompt_tokens": (
            float(np.mean([r.prompt_tokens for r in records])) if records else 0.0
        ),
        "mean_completion_tokens": (
            float(np.mean([r.completion_tokens for r in records])) if records else 0.0
        ),
        "mean_cost_usd": float(np.mean([r.cost_usd for r in records])) if records else 0.0,
    }

    return EvalReport(
        config_name=config.name,
        split=split or "all",
        seed=config.seed,
        n_questions=len(questions),
        metrics=metrics,
        per_question=samples,
        operational=operational,
        records=records,
    )
