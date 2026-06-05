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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "version"):
        print(f"rag-eval-engine {__version__}")
        return 0

    # argparse rejects unknown commands before we get here; this guards future additions.
    print(f"unknown command: {args.command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
