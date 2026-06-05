"""Prompt assembly, citation parsing, and the offline extractive generator."""

from __future__ import annotations

from rag_eval.generate.citations import citation_chunk_ids, parse_citation_indices
from rag_eval.generate.llm import ABSTAIN_MESSAGE, OfflineExtractiveGenerator
from rag_eval.generate.prompt import assemble_context, build_prompt
from rag_eval.types import Chunk, ScoredChunk


def _ctx(cid: str, text: str) -> ScoredChunk:
    return ScoredChunk(Chunk(id=cid, doc_id=cid, text=text, ordinal=0, title="T"), 1.0, "rrf")


def test_assemble_context_numbers_sources() -> None:
    block = assemble_context([_ctx("a", "first").chunk, _ctx("b", "second").chunk])
    assert "[1] T: first" in block
    assert "[2] T: second" in block


def test_build_prompt_includes_question_and_instructions() -> None:
    prompt = build_prompt("what is x?", [_ctx("a", "x is a thing").chunk])
    assert "what is x?" in prompt
    assert "Cite the sources" in prompt


def test_parse_citation_indices_dedups_and_bounds() -> None:
    assert parse_citation_indices("uses [1] and [2] and again [1] and bogus [9]", 2) == [1, 2]


def test_citation_chunk_ids_maps_to_context() -> None:
    ctx = [_ctx("doc-a", "alpha").chunk, _ctx("doc-b", "beta").chunk]
    assert citation_chunk_ids("see [2]", ctx) == ["doc-b"]


def test_offline_generator_answers_and_cites_faithfully() -> None:
    gen = OfflineExtractiveGenerator(max_context_chunks=3)
    contexts = [
        _ctx("doc-x", "Photosynthesis converts carbon dioxide and water into glucose and oxygen."),
        _ctx("doc-y", "The Roman Empire was centered on the city of Rome."),
    ]
    result = gen.generate("how does photosynthesis convert water into oxygen", contexts, 0.2)
    assert not result.abstained
    assert result.citations == ["doc-x"]
    # extractive answer is copied verbatim from the cited chunk => faithful by construction
    answer_text = result.answer.rsplit(" [", 1)[0]
    assert answer_text in contexts[0].chunk.text
    assert result.usage.prompt_tokens > 0


def test_offline_generator_abstains_without_context() -> None:
    result = OfflineExtractiveGenerator(3).generate("anything", [], 0.2)
    assert result.abstained
    assert result.answer == ABSTAIN_MESSAGE
    assert result.citations == []


def test_offline_generator_abstains_on_low_coverage() -> None:
    contexts = [_ctx("doc-x", "The Roman Empire was centered on the city of Rome.")]
    result = OfflineExtractiveGenerator(3).generate(
        "quantum entanglement cryptography algorithm", contexts, 0.2
    )
    assert result.abstained
