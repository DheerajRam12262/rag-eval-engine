"""Config loading, ``extends`` deep-merge, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_eval.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def test_load_base() -> None:
    cfg = load_config(CONFIG_DIR / "base.yaml")
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert cfg.retrieval.bm25.enabled and cfg.retrieval.dense.enabled
    assert cfg.retrieval.rerank.enabled


def test_variant_extends_and_overrides_base() -> None:
    """dense_only flips bm25/rerank off but inherits everything else (e.g. top_k) from base."""
    base = load_config(CONFIG_DIR / "base.yaml")
    cfg = load_config(CONFIG_DIR / "variants" / "dense_only.yaml")
    assert cfg.name == "dense_only"
    assert cfg.retrieval.bm25.enabled is False
    assert cfg.retrieval.rerank.enabled is False
    assert cfg.retrieval.dense.enabled is True
    # inherited untouched:
    assert cfg.retrieval.top_k == base.retrieval.top_k
    assert cfg.embedder.dim == base.embedder.dim


def test_closed_book_disables_all_retrieval() -> None:
    cfg = load_config(CONFIG_DIR / "variants" / "closed_book.yaml")
    assert not cfg.retrieval.bm25.enabled
    assert not cfg.retrieval.dense.enabled
    assert not cfg.retrieval.rerank.enabled


def test_oracle_mode() -> None:
    cfg = load_config(CONFIG_DIR / "variants" / "oracle.yaml")
    assert cfg.retrieval.mode == "oracle"


def test_invalid_overlap_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    base = (CONFIG_DIR / "base.yaml").read_text()
    bad.write_text(base.replace("chunk_overlap: 32", "chunk_overlap: 999"))
    with pytest.raises(ValidationError):
        load_config(bad)
