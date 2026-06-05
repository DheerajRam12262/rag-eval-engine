"""Command-line entrypoint for rag-eval-engine.

Subcommands are registered as each build phase lands, so the CLI never advertises a
command it cannot perform. Phase 0 ships `version`; `ingest`, `eval`, `ablate`, `plots`
and `serve` are wired in their respective phases.
"""

from __future__ import annotations

import argparse
import sys

from rag_eval import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Hybrid-retrieval RAG with a reproducible evaluation harness.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rag-eval-engine {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.add_parser("version", help="Print the version and exit.")

    ingest = subparsers.add_parser("ingest", help="Build dense + BM25 indexes from the corpus.")
    ingest.add_argument("--config", default="config/base.yaml", help="Path to a config YAML.")

    ev = subparsers.add_parser("eval", help="Run the eval harness for one config.")
    ev.add_argument("--config", default="config/base.yaml", help="Path to a config YAML.")
    ev.add_argument("--split", default="test", help="Gold split: dev | test | all.")
    ev.add_argument("--gold", default="eval/gold", help="Path to the gold set.")
    ev.add_argument("--out", default="eval/results", help="Directory for the results JSON.")
    ev.add_argument("--smoke", action="store_true", help="Cap questions for a fast CI smoke eval.")

    return parser


def _cmd_ingest(config_path: str) -> int:
    from rag_eval.config import load_config, seed_everything
    from rag_eval.ingest.indexer import build_index, index_dir

    config = load_config(config_path)
    seed_everything(config.seed)
    bundle = build_index(config, persist=True)
    out = index_dir(config)
    print(
        f"ingested '{config.name}': {len(bundle.chunks)} chunks "
        f"(embedder={config.embedder.backend}, store={config.vector_store.backend}) -> {out}"
    )
    return 0


def _cmd_eval(config_path: str, split: str, gold_path: str, out_dir: str, smoke: bool) -> int:
    from pathlib import Path

    from rag_eval.config import load_config
    from rag_eval.eval.gold import load_gold
    from rag_eval.eval.results import render_summary, save_report
    from rag_eval.eval.runner import run_eval

    config = load_config(config_path)
    gold = load_gold(gold_path)
    report = run_eval(config, gold, split=None if split == "all" else split, smoke=smoke)
    print(render_summary(report))
    out = save_report(report, Path(out_dir) / f"{config.name}.json")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "version"):
        print(f"rag-eval-engine {__version__}")
        return 0
    if args.command == "ingest":
        return _cmd_ingest(args.config)
    if args.command == "eval":
        return _cmd_eval(args.config, args.split, args.gold, args.out, args.smoke)

    # argparse rejects unknown commands before we get here; this guards future additions.
    print(f"unknown command: {args.command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
