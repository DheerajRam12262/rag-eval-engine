"""Generators: a deterministic extractive default plus an Anthropic adapter.

The offline :class:`OfflineExtractiveGenerator` answers with the single highest content-overlap
sentence from the retrieved context and cites its chunk. Because the answer is copied verbatim
from context, it is faithful by construction -- a useful, honest baseline. It abstains when there
is no context or when the best evidence covers too little of the query (a scale-free signal, set
by ``generation.abstain_threshold``). Swap to a real LLM via ``generation.provider: anthropic``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag_eval.config import GenerationConfig
from rag_eval.generate.prompt import build_prompt
from rag_eval.text import content_tokens, split_sentences, tokenize
from rag_eval.types import ScoredChunk

ABSTAIN_MESSAGE = "I don't have enough information in the provided context to answer that."


@dataclass(slots=True)
class TokenUsage:
    """Token counts and dollar cost for one generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class GenerationResult:
    answer: str
    citations: list[str] = field(default_factory=list)  # chunk ids
    abstained: bool = False
    usage: TokenUsage = field(default_factory=TokenUsage)


class Generator(Protocol):
    """Produces a grounded answer (or abstains) from a query and retrieved context."""

    def generate(
        self, query: str, contexts: list[ScoredChunk], abstain_threshold: float
    ) -> GenerationResult: ...


class OfflineExtractiveGenerator:
    """Deterministic extractive QA over the retrieved context."""

    def __init__(self, max_context_chunks: int) -> None:
        self._max = max_context_chunks

    def _abstain(self, prompt_tokens: int) -> GenerationResult:
        return GenerationResult(
            answer=ABSTAIN_MESSAGE,
            citations=[],
            abstained=True,
            usage=TokenUsage(prompt_tokens, len(tokenize(ABSTAIN_MESSAGE)), 0.0),
        )

    def generate(
        self, query: str, contexts: list[ScoredChunk], abstain_threshold: float
    ) -> GenerationResult:
        ctx = contexts[: self._max]
        prompt_tokens = len(tokenize(build_prompt(query, [c.chunk for c in ctx])))
        qterms = set(content_tokens(query))
        if not ctx or not qterms:
            return self._abstain(prompt_tokens)

        best_overlap = 0
        best_idx = -1
        best_sentence = ""
        for idx, scored in enumerate(ctx):
            for sentence in split_sentences(scored.chunk.text):
                overlap = len(qterms & set(content_tokens(sentence)))
                if overlap > best_overlap:
                    best_overlap, best_idx, best_sentence = overlap, idx, sentence

        coverage = best_overlap / len(qterms)
        if best_idx < 0 or coverage < abstain_threshold:
            return self._abstain(prompt_tokens)

        answer = f"{best_sentence} [{best_idx + 1}]"
        return GenerationResult(
            answer=answer,
            citations=[ctx[best_idx].chunk.id],
            abstained=False,
            usage=TokenUsage(prompt_tokens, len(tokenize(answer)), 0.0),
        )


class AnthropicGenerator:
    """Adapter for Anthropic Claude. Requires the ``llm`` extra and ANTHROPIC_API_KEY."""

    def __init__(self, model: str, max_context_chunks: int) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("anthropic not installed; run: pip install -e '.[llm]'") from exc
        self._client = anthropic.Anthropic()
        self._model = model
        self._max = max_context_chunks

    def generate(  # pragma: no cover - requires network + API key
        self, query: str, contexts: list[ScoredChunk], abstain_threshold: float
    ) -> GenerationResult:
        from rag_eval.generate.citations import citation_chunk_ids

        ctx = contexts[: self._max]
        chunks = [c.chunk for c in ctx]
        prompt = build_prompt(query, chunks)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = "".join(block.text for block in message.content if block.type == "text").strip()
        usage = TokenUsage(
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            cost_usd=0.0,  # populate from a per-model price table when enabling billing
        )
        abstained = ABSTAIN_MESSAGE[:40] in answer
        citations = [] if abstained else citation_chunk_ids(answer, chunks)
        return GenerationResult(answer, citations, abstained, usage)


def get_generator(config: GenerationConfig) -> Generator:
    """Build the generator named by the config."""
    if config.provider == "offline":
        return OfflineExtractiveGenerator(config.max_context_chunks)
    if config.provider == "anthropic":
        return AnthropicGenerator(config.model, config.max_context_chunks)
    # Honest, labeled gap: an OpenAI adapter is on the roadmap.
    raise NotImplementedError(
        "generation.provider 'openai' is not yet implemented; use 'offline' or 'anthropic'."
    )
