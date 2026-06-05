"""Phase 0 sanity tests: the package is installable, versioned, and configurable.

These assert real invariants (version parity, config schema presence, CLI exit code) so the
CI gate is meaningful from the very first commit.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

import rag_eval
from rag_eval.cli import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject() -> None:
    """The package __version__ must equal the version declared in pyproject.toml."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert rag_eval.__version__ == pyproject["project"]["version"]


def test_base_config_has_required_sections() -> None:
    """The base config must declare every section the pipeline depends on."""
    config = yaml.safe_load((REPO_ROOT / "config" / "base.yaml").read_text())
    for section in (
        "corpus",
        "chunking",
        "embedder",
        "vector_store",
        "retrieval",
        "generation",
        "eval",
    ):
        assert section in config, f"missing config section: {section}"
    assert config["seed"] == 42


def test_every_variant_extends_base() -> None:
    """Ablation variants must extend base so they stay DRY and comparable."""
    variants = list((REPO_ROOT / "config" / "variants").glob("*.yaml"))
    assert variants, "expected at least one ablation variant"
    for path in variants:
        cfg = yaml.safe_load(path.read_text())
        assert cfg.get("extends") == "base", f"{path.name} must extend base"


def test_cli_version_returns_zero(capsys) -> None:  # type: ignore[no-untyped-def]
    """`rag-eval version` exits 0 and prints the version."""
    assert main(["version"]) == 0
    assert rag_eval.__version__ in capsys.readouterr().out


def test_parser_builds() -> None:
    """The argument parser constructs without error and knows the version command."""
    parser = build_parser()
    args = parser.parse_args(["version"])
    assert args.command == "version"
