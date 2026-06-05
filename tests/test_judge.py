"""Offline judge proxies, caching, and the judge!=generator config guard."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_eval.config import load_config
from rag_eval.eval.judge import JudgeCache, JudgeScores, OfflineJudge

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _score(**kw: object) -> JudgeScores:
    defaults: dict[str, object] = {
        "question": "",
        "reference_answer": "",
        "answer": "",
        "context": "",
        "abstained": False,
        "no_answer": False,
    }
    defaults.update(kw)
    return OfflineJudge().score(**defaults)  # type: ignore[arg-type]


def test_correctness_high_when_answer_matches_reference() -> None:
    good = _score(
        question="what are the moons of mars",
        reference_answer="Phobos and Deimos",
        answer="The moons are Phobos and Deimos [1]",
        context="Mars has two moons named Phobos and Deimos",
    )
    bad = _score(
        question="what are the moons of mars",
        reference_answer="Phobos and Deimos",
        answer="The Roman Empire was centered on Rome [1]",
        context="The Roman Empire was centered on Rome",
    )
    assert good.correctness > 0.5
    assert bad.correctness == 0.0


def test_faithfulness_rewards_grounded_answers() -> None:
    grounded = _score(answer="glucose and oxygen [1]", context="produces glucose and oxygen")
    ungrounded = _score(answer="unicorns and dragons", context="produces glucose and oxygen")
    assert grounded.faithfulness == 1.0
    assert ungrounded.faithfulness == 0.0


def test_abstention_scoring_on_no_answer_questions() -> None:
    correct = _score(no_answer=True, abstained=True)
    wrong = _score(no_answer=True, abstained=False, answer="some made up fact", context="")
    assert correct.correctness == 1.0
    assert correct.faithfulness == 1.0  # abstaining is faithful
    assert wrong.correctness == 0.0


def test_abstaining_on_answerable_is_incorrect() -> None:
    s = _score(no_answer=False, abstained=True, reference_answer="something")
    assert s.correctness == 0.0
    assert s.answer_relevance == 0.0


def test_judge_cache_round_trip(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "judge.json")
    key = cache.key("judge-model", question="q", answer="a", reference="r", context="c")
    assert cache.get(key) is None
    cache.put(key, JudgeScores(1.0, 0.5, 0.25))
    # reload from disk
    reloaded = JudgeCache(tmp_path / "judge.json").get(key)
    assert reloaded == JudgeScores(1.0, 0.5, 0.25)


def test_config_rejects_self_judging() -> None:
    base = (CONFIG_DIR / "base.yaml").read_text()
    # make generation use anthropic with the SAME model as the judge
    bad = base.replace("provider: offline       # offline (extractive", "provider: anthropic")
    bad = bad.replace("model: claude-sonnet-4-6", "model: claude-opus-4-8")
    with pytest.raises(ValidationError):
        # write to a temp file path via load_config requires a path; use a NamedTemp-like approach
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(bad)
            tmp_name = fh.name
        load_config(tmp_name)
