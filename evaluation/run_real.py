"""
Run the harness on the repo's real benchmark (Phase-B collections + trigger queries).

    python -m evaluation.run_real --preflight
    python -m evaluation.run_real --d4 random_forest --heldout-only
    python -m evaluation.run_real --attacks injection --ratios 0.20 --generate
    python -m evaluation.run_real --provenance derived      # exploratory, see metadata_bridge.py

Needs ChromaDB running with the Phase-B collections (and Ollama for --generate).
Writes eval_* tables to --db and, under --out: runs.csv, queries.csv,
summary_by_split.csv, skipped_configs.json and report.md (with the caveats that
apply to the run). See evaluation/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.config import make_config                          # noqa: E402
from evaluation.real_source import (build_chroma_source, heldout_query_ids,   # noqa: E402
                                    preflight)
from evaluation.runner import run_sweep                             # noqa: E402
from evaluation.store import ResultStore                            # noqa: E402

METRICS = ["raw_success", "defended_success", "gen_success", "clean_retention",
           "latency_ms", "n_poison_candidates"]


def build_configs(with_d4: bool) -> list:
    stage_sets = [(), ("d1",), ("d2",), ("d3",), ("d1", "d2", "d3")]
    if with_d4:
        stage_sets += [("d4",), ("d1", "d4"), ("d1", "d2", "d3", "d4")]
    configs = [make_config(s) for s in stage_sets]
    if with_d4:
        configs.append(make_config(("d1", "d2", "d3", "d4"), use_router=True))
    return configs


def summarize(result, heldout: set[str]) -> pd.DataFrame:
    """Per config/attack/ratio means, split into held-out vs training-overlap queries."""
    if result.queries.empty:
        return pd.DataFrame()
    runs = result.runs[["run_id", "config_name", "attack_type", "poisoning_ratio"]]
    frame = result.queries.merge(runs, on="run_id")
    frame["split"] = frame["query_id"].str.lower().map(lambda q: "heldout" if q in heldout else "train_overlap")
    cols = [c for c in METRICS if c in frame.columns]
    out = (frame.groupby(["config_name", "attack_type", "poisoning_ratio", "split"])[cols]
           .mean(numeric_only=True).reset_index())
    out["n_queries"] = frame.groupby(["config_name", "attack_type", "poisoning_ratio", "split"]).size().values
    return out


def write_report(path: Path, args, source_name: str, d4_label: str | None, skipped: list, n_runs: int) -> None:
    lines = ["# Real-benchmark evaluation run", "",
             f"- source: `{source_name}`  |  attacks: {args.attacks}  |  ratios: {args.ratios}  |  seed: 0 (collections are static)",
             f"- D4 model: `{d4_label or 'none'}`  |  queries: {'held-out only' if args.heldout_only else 'all 20 (split column separates held-out)'}",
             f"- runs written: {n_runs}  |  configs skipped: {len(skipped)}", "", "## Read this before quoting numbers", ""]
    if "[derived]" in source_name:
        lines.append("- **Provenance is derived, not authentic.** D3's integrity check cannot fail, and chunks with an empty "
                     "timestamp fail D3's temporal check regardless of poisoning. Treat D2/D3 results as exploratory.")
    if "[thin]" in source_name:
        lines.append("- Collections hold only thin metadata, so D2, D3 and the router were skipped (see skipped_configs.json).")
    if d4_label:
        lines.append("- D4 was trained on this benchmark's controlled attack texts and shares documents with the queries. "
                     "Quote the `heldout` split, and do not present it as generalisation to unseen attackers or corpora.")
    lines += ["- Generation success exists only for the RPI attack (marker POISONBENCH_RPI_SUCCESS); it is blank for sm/obea.",
              "- Ground-truth labels come from `is_poisoned` metadata and are used only to score, never as defense inputs."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true", help="only check which collections exist")
    ap.add_argument("--attacks", nargs="+", default=["naive", "optimized", "injection"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.20])
    ap.add_argument("--provenance", default="auto", choices=["auto", "thin", "stored", "derived"])
    ap.add_argument("--d4", default="none", choices=["none", "random_forest", "xgboost"])
    ap.add_argument("--heldout-only", action="store_true")
    ap.add_argument("--generate", action="store_true", help="also run the LLM (RPI attack only has a marker)")
    ap.add_argument("--db", default="sqlite:///data/results/eval_real/poisonbench_eval.db")
    ap.add_argument("--out", default="data/results/eval_real")
    ap.add_argument("--group", default="real_phase_b")
    args = ap.parse_args(argv)

    heldout = heldout_query_ids()
    source = build_chroma_source(args.provenance, heldout if args.heldout_only else None)
    if args.preflight:
        for row in preflight(source.client, args.attacks, args.ratios):
            print(f"{'OK     ' if row['exists'] else 'MISSING'} {row['collection']:34s} "
                  f"chunks={row['chunks']:<6} poisoned={row['poisoned']}")
        return 0

    d4 = None
    if args.d4 != "none":
        from evaluation.d4_live_text import load_live_text_model
        d4 = load_live_text_model(args.d4)
    generator = None
    if args.generate:
        from evaluation.generation import ApiLLMGenerator
        generator = ApiLLMGenerator()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.db.startswith("sqlite:///"):
        Path(args.db[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    store = ResultStore(args.db)
    result = run_sweep(source, build_configs(d4 is not None), args.attacks, args.ratios, [0],
                       group=args.group, store=store, generator=generator, d4_model=d4, verbose=True)

    summary = summarize(result, heldout)
    result.runs.to_csv(out_dir / "runs.csv", index=False)
    result.queries.to_csv(out_dir / "queries.csv", index=False)
    summary.to_csv(out_dir / "summary_by_split.csv", index=False)
    (out_dir / "skipped_configs.json").write_text(json.dumps(result.skipped, indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", args, source.name, d4.label if d4 else None,
                 result.skipped, len(result.runs))
    if not summary.empty:
        print(summary[summary["split"] == "heldout"].round(3).to_string(index=False))
    print(f"\nwrote {len(result.runs)} runs to {args.db}; files in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
