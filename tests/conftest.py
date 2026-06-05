"""Shared fixtures. The base index is built once per session (offline, fast)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.config import Config, load_config
from rag_eval.ingest.indexer import IndexBundle, build_index

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(scope="session")
def base_config() -> Config:
    return load_config(CONFIG_DIR / "base.yaml")


@pytest.fixture(scope="session")
def base_index(base_config: Config) -> IndexBundle:
    return build_index(base_config, persist=False)
