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

    ab = subparsers.add_parser("ablate", help="Run an ablation suite and write a table + plots.")
    ab.add_argument("--suite", default="config/variants", help="Directory of variant configs.")
    ab.add_argument("--split", default="test", help="Gold split: dev | test | all.")
    ab.add_argument("--gold", default="eval/gold", help="Path to the gold set.")
    ab.add_argument("--out", default="eval/results", help="Output directory.")

    pl = subparsers.add_parser("plots", help="Regenerate plots from saved result JSONs.")
    pl.add_argument("--out", default="eval/results", help="Directory of result JSONs.")

    return parser


# Conventional display order for the pipeline ablation.
_ABLATION_ORDER = ["closed_book", "dense_only", "hybrid", "hybrid_rerank", "oracle"]
_ABLATION_METRICS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "ndcg@5",
    "mrr",
    "faithfulness",
    "answer_relevance",
    "correctness",
    "abstention_accuracy",
]


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


def _cmd_ablate(suite_dir: str, gold_path: str, split: str, out_dir: str) -> int:
    from pathlib import Path

    from rag_eval.config import load_config
    from rag_eval.eval.gold import load_gold
    from rag_eval.eval.plots import plot_metric_bars, plot_recall_grouped
    from rag_eval.eval.results import compare, markdown_table, save_report
    from rag_eval.eval.runner import run_eval
    from rag_eval.ingest.indexer import build_index, index_signature

    configs = [load_config(p) for p in sorted(Path(suite_dir).glob("*.yaml"))]
    configs.sort(
        key=lambda c: _ABLATION_ORDER.index(c.name) if c.name in _ABLATION_ORDER else len(configs)
    )
    gold = load_gold(gold_path)
    out = Path(out_dir)

    bundles: dict[str, object] = {}
    reports = []
    for cfg in configs:
        sig = index_signature(cfg)
        if sig not in bundles:
            bundles[sig] = build_index(cfg, persist=False)
        rep = run_eval(
            cfg, gold, bundle=bundles[sig], split=None if split == "all" else split  # type: ignore[arg-type]
        )
        reports.append(rep)
        save_report(rep, out / f"{cfg.name}.json")

    table = markdown_table(reports, _ABLATION_METRICS)
    by_name = {r.config_name: r for r in reports}
    lines = [
        "# Ablation results",
        "",
        f"Backend: offline (deterministic). Split: `{split}`. Cells: `mean [95% CI]` (bootstrap).",
        "",
        table,
        "",
        "## Significance (paired bootstrap, per-question)",
        "",
    ]
    if {"hybrid_rerank", "dense_only"} <= by_name.keys():
        for m in ("recall@1", "recall@3", "correctness"):
            c = compare(by_name["hybrid_rerank"], by_name["dense_only"], m)
            verdict = "significant" if c.p_value < 0.05 else "n.s."
            lines.append(
                f"- **hybrid_rerank vs dense_only** on `{m}`: "
                f"diff={c.diff_mean:+.3f}, p={c.p_value:.4f} (n={c.n}) — {verdict}"
            )
    if {"hybrid_rerank", "closed_book"} <= by_name.keys():
        c = compare(by_name["hybrid_rerank"], by_name["closed_book"], "recall@5")
        lines.append(
            f"- **retrieval vs closed-book** on `recall@5`: "
            f"diff={c.diff_mean:+.3f}, p={c.p_value:.4f} (n={c.n})"
        )
    report_md = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.md").write_text(report_md + "\n")
    plot_recall_grouped(reports, [1, 3, 5], out / "ablation_recall.png")
    plot_metric_bars(reports, "correctness", out / "ablation_correctness.png")

    print(report_md)
    print(f"\nwrote {out}/ablation.md, ablation_recall.png, ablation_correctness.png")
    return 0


def _cmd_plots(out_dir: str) -> int:
    from pathlib import Path

    from rag_eval.eval.plots import plot_metric_bars, plot_recall_grouped
    from rag_eval.eval.results import load_report

    out = Path(out_dir)
    paths = sorted(p for p in out.glob("*.json"))
    if not paths:
        print(f"no result JSONs in {out}; run `rag-eval ablate` first")
        return 1
    reports = [load_report(p) for p in paths]
    reports.sort(
        key=lambda r: (
            _ABLATION_ORDER.index(r.config_name)
            if r.config_name in _ABLATION_ORDER
            else len(reports)
        )
    )
    plot_recall_grouped(reports, [1, 3, 5], out / "ablation_recall.png")
    plot_metric_bars(reports, "correctness", out / "ablation_correctness.png")
    print(f"regenerated plots in {out}")
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
    if args.command == "ablate":
        return _cmd_ablate(args.suite, args.gold, args.split, args.out)
    if args.command == "plots":
        return _cmd_plots(args.out)

    # argparse rejects unknown commands before we get here; this guards future additions.
    print(f"unknown command: {args.command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
