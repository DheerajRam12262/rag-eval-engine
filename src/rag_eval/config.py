"""Typed, validated configuration loaded from YAML.

Configs are the single source of truth for every knob (no magic numbers in code). Variants
in ``config/variants/`` declare ``extends: base`` and are deep-merged over ``config/base.yaml``
so ablations stay DRY and directly comparable.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator


class CorpusConfig(BaseModel):
    name: str
    path: str


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "semantic"]
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class EmbedderConfig(BaseModel):
    backend: Literal["hashing", "sentence-transformers"]
    model: str
    dim: int = Field(gt=0)


class VectorStoreConfig(BaseModel):
    backend: Literal["memory", "qdrant"]


class ToggleConfig(BaseModel):
    enabled: bool


class FusionConfig(BaseModel):
    method: Literal["rrf"]
    rrf_k: int = Field(gt=0)


class RerankConfig(BaseModel):
    enabled: bool
    backend: Literal["lexical", "cross-encoder"]
    model: str


class RetrievalConfig(BaseModel):
    candidate_k: int = Field(gt=0)
    top_k: int = Field(gt=0)
    mode: Literal["standard", "oracle"] = "standard"
    bm25: ToggleConfig
    dense: ToggleConfig
    fusion: FusionConfig
    rerank: RerankConfig


class GenerationConfig(BaseModel):
    provider: Literal["offline", "anthropic", "openai"]
    model: str
    max_context_chunks: int = Field(gt=0)
    abstain_threshold: float = Field(ge=0)


class MetricsConfig(BaseModel):
    k_values: list[int]


class JudgeConfig(BaseModel):
    provider: Literal["offline", "anthropic", "openai"]
    model: str


class BootstrapConfig(BaseModel):
    iterations: int = Field(gt=0)
    confidence: float = Field(gt=0, lt=1)


class SmokeConfig(BaseModel):
    max_questions: int = Field(gt=0)


class EvalConfig(BaseModel):
    metrics: MetricsConfig
    judge: JudgeConfig
    bootstrap: BootstrapConfig
    smoke: SmokeConfig


class Config(BaseModel):
    """The fully-resolved, validated pipeline configuration."""

    name: str
    seed: int
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedder: EmbedderConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    eval: EvalConfig

    @model_validator(mode="after")
    def _judge_is_not_the_generator(self) -> Config:
        """Guard against self-preference bias: an API judge must not be the generator model."""
        if (
            self.generation.provider != "offline"
            and self.eval.judge.provider == self.generation.provider
            and self.eval.judge.model == self.generation.model
        ):
            raise ValueError(
                "eval.judge.model must differ from generation.model to avoid self-preference bias"
            )
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    out = dict(base)
    for key, value in override.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


def _resolve_extends(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    """If ``raw`` declares ``extends: <name>``, merge it over that base config."""
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    # base lives in config/, variants live in config/variants/
    candidates = [path.parent / f"{parent}.yaml", path.parent.parent / f"{parent}.yaml"]
    base_path = next((c for c in candidates if c.exists()), None)
    if base_path is None:
        raise FileNotFoundError(f"extends: {parent!r} not found near {path}")
    base_raw = yaml.safe_load(base_path.read_text())
    base_raw.pop("extends", None)
    return _deep_merge(base_raw, raw)


def load_config(path: str | Path) -> Config:
    """Load, resolve ``extends``, and validate a config file."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    raw = _resolve_extends(raw, path)
    return Config.model_validate(raw)


def seed_everything(seed: int) -> None:
    """Seed every RNG we rely on so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
