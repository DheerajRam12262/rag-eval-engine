"""LLM-as-judge for generation quality, with a deterministic offline proxy.

Three scores per answer:
  * faithfulness    -- are the answer's claims grounded in the retrieved context?
  * answer_relevance-- does the answer address the question?
  * correctness     -- does the answer match the reference?

The offline :class:`OfflineJudge` computes these with deterministic lexical proxies (grounding =
fraction of answer terms found in context; correctness = token-F1 vs reference). It is a proxy,
not a semantic judge -- so the harness ships a real :class:`LLMJudge` adapter (judge model MUST
differ from the generator, see config validation) whose calls are cached for reproducibility.
An unvalidated judge is not evidence: validate it against human labels (Phase 5) and report
agreement.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_eval.config import JudgeConfig
from rag_eval.text import content_tokens

_CITE_RE = re.compile(r"\[\d+\]")


@dataclass(frozen=True, slots=True)
class JudgeScores:
    faithfulness: float
    answer_relevance: float
    correctness: float


def _strip_citations(text: str) -> str:
    return _CITE_RE.sub(" ", text)


def _token_f1(pred: list[str], ref: list[str]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _grounding(answer: str, context: str) -> float:
    atoks = content_tokens(_strip_citations(answer))
    if not atoks:
        return 1.0
    cset = set(content_tokens(context))
    return sum(1 for t in atoks if t in cset) / len(atoks)


class Judge(Protocol):
    """Scores a generated answer for faithfulness, relevance, and correctness."""

    def score(
        self,
        *,
        question: str,
        reference_answer: str,
        answer: str,
        context: str,
        abstained: bool,
        no_answer: bool,
    ) -> JudgeScores: ...


class OfflineJudge:
    """Deterministic lexical-proxy judge (no model, no network)."""

    def score(
        self,
        *,
        question: str,
        reference_answer: str,
        answer: str,
        context: str,
        abstained: bool,
        no_answer: bool,
    ) -> JudgeScores:
        # Faithfulness: abstaining makes no unsupported claims => perfectly faithful.
        faithfulness = 1.0 if abstained else _grounding(answer, context)

        # Relevance: how much the answer overlaps the question (0 if it abstained).
        if abstained:
            relevance = 0.0
        else:
            relevance = _token_f1(
                content_tokens(_strip_citations(answer)), content_tokens(question)
            )

        # Correctness: for no-answer questions, the correct behavior is to abstain.
        if no_answer:
            correctness = 1.0 if abstained else 0.0
        elif abstained:
            correctness = 0.0
        else:
            correctness = _token_f1(
                content_tokens(_strip_citations(answer)), content_tokens(reference_answer)
            )

        return JudgeScores(faithfulness, relevance, correctness)


class JudgeCache:
    """A tiny JSON-file cache so LLM judge calls are reproducible and cheap to replay."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict[str, float]] = {}
        if path.exists():
            self._data = json.loads(path.read_text())

    @staticmethod
    def key(model: str, **parts: str) -> str:
        payload = json.dumps({"model": model, **parts}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> JudgeScores | None:
        raw = self._data.get(key)
        if raw is None:
            return None
        return JudgeScores(raw["faithfulness"], raw["answer_relevance"], raw["correctness"])

    def put(self, key: str, scores: JudgeScores) -> None:
        self._data[key] = {
            "faithfulness": scores.faithfulness,
            "answer_relevance": scores.answer_relevance,
            "correctness": scores.correctness,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, sort_keys=True))


class LLMJudge:
    """Adapter for an LLM judge (Anthropic). Requires the ``llm`` extra; calls are cached."""

    _RUBRIC = (
        "You are grading a RAG system's answer. Score each 0.0-1.0 and reply ONLY with JSON "
        '{"faithfulness": x, "answer_relevance": x, "correctness": x}. '
        "faithfulness = every claim is supported by the context; "
        "answer_relevance = the answer addresses the question; "
        "correctness = the answer agrees with the reference."
    )

    def __init__(self, model: str, cache: JudgeCache) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("anthropic not installed; run: pip install -e '.[llm]'") from exc
        self._client = anthropic.Anthropic()
        self._model = model
        self._cache = cache

    def score(  # pragma: no cover - requires network + API key
        self,
        *,
        question: str,
        reference_answer: str,
        answer: str,
        context: str,
        abstained: bool,
        no_answer: bool,
    ) -> JudgeScores:
        key = self._cache.key(
            self._model,
            question=question,
            reference=reference_answer,
            answer=answer,
            context=context,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        prompt = (
            f"{self._RUBRIC}\n\nQuestion: {question}\nReference: {reference_answer}\n"
            f"Context:\n{context}\n\nAnswer: {answer}"
        )
        message = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in message.content if b.type == "text")
        payload = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))  # type: ignore[union-attr]
        scores = JudgeScores(
            float(payload["faithfulness"]),
            float(payload["answer_relevance"]),
            float(payload["correctness"]),
        )
        self._cache.put(key, scores)
        return scores


def get_judge(config: JudgeConfig, cache_dir: Path | None = None) -> Judge:
    """Build the judge named by the config."""
    if config.provider == "offline":
        return OfflineJudge()
    if config.provider == "anthropic":
        cache_path = (cache_dir or Path("eval/cache")) / "judge.json"
        return LLMJudge(config.model, JudgeCache(cache_path))
    raise NotImplementedError(
        "eval.judge.provider 'openai' is not yet implemented; use 'offline' or 'anthropic'."
    )
