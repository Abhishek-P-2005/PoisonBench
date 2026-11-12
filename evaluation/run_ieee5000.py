"""
Run the whole evaluation chain on the 5,000-record corpus (ieee_5000/).

    python -m evaluation.run_ieee5000 --preflight          # do the 4 collections exist and have the right sizes?
    python -m evaluation.ieee5000_d4 --train               # once: six-feature D4 models for this corpus
    python -m evaluation.run_ieee5000                      # everything, all three provenance tiers
    python -m evaluation.run_ieee5000 --tier forged --d4 xgboost
    python -m evaluation.run_ieee5000 --d4 transfer_random_forest    # old 316-chunk model, unchanged

Needs the Chroma server with the ieee_5000 collections (docker compose up chromadb; the
scripts/ingest_*_ieee.py scripts build them) and the MiniLM model. Ollama only for --generate.

What it does, per provenance tier (see evaluation/ieee5000_source.py):
  A. calibrate D1/D2/D3 and the router threshold on the 80 NON-held-out queries
  B. sweep every defense subset, plus the corpus-reference baselines, on the 20 HELD-OUT queries
  C. defense ordering, interaction (marginal uplift, redundancy, unique catches), Pareto front
  D. router benchmark: static pipeline vs router (latency, security, share sent to D4)
  E. extra measurements: did the legitimate CVE survive, was the context emptied, Wilson /
     bootstrap intervals, AUC per defense, prevalence-adjusted precision
Everything is written under --out (default data/results/eval_ieee5000/); the eval_* tables go
to --db (SQLite by default). Collections and existing result files are never modified.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.analysis import calibration, interaction, ordering, pareto            # noqa: E402
from evaluation.config import make_config                                              # noqa: E402
from evaluation.ieee5000_d4 import MANIFEST as D4_MANIFEST, MODEL_DIR, load_d4, train  # noqa: E402
from evaluation.ieee5000_report import (chunk_auc, prevalence_table,                   # noqa: E402
                                        query_outcomes, summarize_outcomes)
from evaluation.ieee5000_source import (ANALYSIS_TIERS, CLEAN_COLLECTION, HARNESS_ATTACKS,  # noqa: E402
                                        IEEE_RATIO, build_source, preflight, split_query_ids)
from evaluation.reference import CorpusReference, set_reference                       # noqa: E402
from evaluation.router_benchmark import (benchmark_static_vs_dynamic, calibrate_from_scores,  # noqa: E402
                                         check_static_overhead, plot_router_benchmark,
                                         plot_router_curve, summarize_benchmark)
from evaluation.runner import run_sweep, score_pass                                    # noqa: E402
from evaluation.store import ResultStore                                               # noqa: E402

DEFAULT_OUT = ROOT / "data" / "results" / "eval_ieee5000"
RATIOS = [IEEE_RATIO]
SEEDS = [0]
ROUTER_SWEEP = (0.05, 0.15, 0.25, 0.40, 0.60)


class Problems(list):
    """Sections that failed; the run keeps going and the report lists them, so nothing fails silently."""

    def section(self, tier: str, name: str, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:   # noqa: BLE001  - recorded, not hidden
            self.append({"tier": tier, "section": name, "error": f"{type(exc).__name__}: {exc}"})
            logging.getLogger("ieee5000").warning("section %s/%s failed: %s", tier, name, exc)
            traceback.print_exc()
            return None


def build_eval_configs(params: dict, has_d4: bool) -> tuple[list, list]:
    """(sweep configs, configs for the per-query outcome table)."""
    defenses = ("d1", "d2", "d3", "d4") if has_d4 else ("d1", "d2", "d3")
    configs = [make_config([])]
    for base_params, label in ((params, "cal"), ({}, "default")):
        configs.extend(interaction.subset_configs(stages=defenses, params=base_params, label=label))
    extras = [make_config(("d1c",)), make_config(("ref",)), make_config(("d1c", "d3"))]
    if has_d4:
        extras += [make_config(("d1c", "d4")), make_config(("ref", "d4")), make_config(("d1c", "d2", "d3", "d4"))]
    configs.extend(extras)
    return configs


def outcome_configs(params: dict, has_d4: bool) -> list:
    cfgs = [make_config([]), make_config(("d1",)), make_config(("d2",)), make_config(("d3",)),
            make_config(("d1c",)), make_config(("ref",)), make_config(("d1", "d2", "d3"))]
    cfgs.append(make_config(("d1", "d2", "d3"), params, label="cal"))
    if has_d4:
        cfgs += [make_config(("d4",)), make_config(("ref", "d4")), make_config(("d3", "d4")),
                 make_config(("d1", "d2", "d3", "d4")),
                 make_config(("d1", "d2", "d3", "d4"), params, label="cal"),
                 make_config(("d1", "d4")),                       # D1 may hard-block ahead of D4
                 make_config(("d1", "d4"), params, label="cal")]
    return cfgs


def run_tier(tier, base_source, d4_model, args, store, out_dir: Path, problems: Problems) -> dict:
    out = out_dir / tier
    out.mkdir(parents=True, exist_ok=True)
    attacks = HARNESS_ATTACKS
    has_d4 = d4_model is not None
    cal_ids, held_ids = split_query_ids([{"query_id": t["query_id"]} for t in base_source.trigger_queries])
    eval_ids = held_ids if args.eval_queries == "heldout" else {t["query_id"].lower() for t in base_source.trigger_queries}
    source = base_source.with_tier(tier)
    cal_source = source.with_queries(cal_ids)
    eval_source = source.with_queries(eval_ids)
    if args.max_eval_queries:
        keep = sorted(eval_ids)[: args.max_eval_queries]
        eval_source = source.with_queries(set(keep))
    print(f"[{tier}] calibration queries: {len(cal_source.trigger_queries)}  "
          f"evaluation queries: {len(eval_source.trigger_queries)} ({args.eval_queries})")

    # A. calibration (D4 keeps its predeclared 0.5: it was trained on these targets, so its scores here are optimistic)
    cal_scores = score_pass(cal_source, attacks, RATIOS, SEEDS, group=f"cal|{tier}", store=store, d4_model=None)
    reports = calibration.calibrate_all(cal_scores)
    params = calibration.thresholds_to_params(reports)
    router_scores = score_pass(cal_source, attacks, RATIOS, SEEDS, group=f"cal_router|{tier}", store=store,
                               d4_model=None, params=params)
    router_cal = calibrate_from_scores(router_scores)
    (out / "calibration.json").write_text(json.dumps({
        "params": params, "reports": {k: v.summary() for k, v in reports.items()},
        "router": router_cal.__dict__, "calibration_queries": sorted(cal_ids)[:5] + ["..."],
    }, indent=2, default=float), encoding="utf-8")
    problems.section(tier, "roc_pr_plot", calibration.plot_roc_pr, reports, str(out / "fig_roc_pr.png"),
                     f"Threshold calibration ({tier})")
    print(f"[{tier}] calibrated params: { {k: round(v, 3) for k, v in params.items()} }  "
          f"router threshold {router_cal.threshold:.3f} (recall {router_cal.recall:.2f}, attained={router_cal.attained})")

    # B. sweeps on the evaluation queries
    configs = build_eval_configs(params, has_d4)
    generator = None
    if args.generate:
        from evaluation.generation import ApiLLMGenerator, CachedGenerator
        generator = CachedGenerator(ApiLLMGenerator())
    main = run_sweep(eval_source, configs, attacks, RATIOS, SEEDS, group=f"main|{tier}", store=store,
                     d4_model=d4_model, verbose=args.verbose)
    if has_d4:
        main = main.merge(run_sweep(
            eval_source,
            [make_config(("d1", "d2", "d3", "d4"), params, use_router=True, router_threshold=router_cal.threshold, label="cal"),
             make_config(("d1", "d3", "d4"), params, use_router=True, router_threshold=router_cal.threshold, label="cal")],
            attacks, RATIOS, SEEDS, group=f"main|{tier}", store=store, d4_model=d4_model))
    if generator is not None:
        gen_configs = [make_config([])] + ([make_config(("d1", "d2", "d3", "d4"), params, label="cal")] if has_d4 else [])
        gen = problems.section(tier, "generation", run_sweep, eval_source, gen_configs, ["injection"], RATIOS, SEEDS,
                               group=f"gen|{tier}", store=store, generator=generator, d4_model=d4_model)
        if gen is not None:
            gen.runs.to_csv(out / "generation_runs.csv", index=False)

    # C. ordering, interaction, Pareto
    tail = ("d4",) if has_d4 else ()
    order_cfgs = ordering.permutation_configs(params=params, tail=tail, label="cal")
    order = run_sweep(eval_source, order_cfgs, attacks, RATIOS, SEEDS, group=f"ordering|{tier}", store=store, d4_model=d4_model)
    eval_scores = score_pass(eval_source, attacks, RATIOS, SEEDS, group=f"eval_scores|{tier}", store=store, d4_model=d4_model)

    tables = {"runs": main.runs, "ordering_runs": order.runs}
    cal_runs = main.runs[main.runs["config_name"].str.contains(r"\[cal\]") | (main.runs["config_name"] == "none")]
    no_router = cal_runs[~cal_runs["config_name"].str.contains("router")]
    tables["marginal_uplift"] = problems.section(tier, "marginal_uplift", interaction.marginal_uplift, no_router)
    tables["redundancy"] = problems.section(tier, "redundancy", interaction.redundancy_table, no_router)
    flagged = interaction.with_calibrated_flags(eval_scores, params)
    tables["pairwise_agreement"] = problems.section(tier, "pairwise_agreement", interaction.pairwise_agreement,
                                                    flagged, params.get("d4_threshold", 0.5))
    tables["unique_catches"] = problems.section(tier, "unique_catches", interaction.unique_catches,
                                                flagged, params.get("d4_threshold", 0.5))
    order_names = [c.name for c in order_cfgs]
    for metric, key in (("attack_success_defended", "ordering_attack_success"), ("security_blocked_pct", "ordering_security"),
                        ("latency_ms_mean", "ordering_latency_ms")):
        matrix = problems.section(tier, key, ordering.ordering_matrix, order.runs, metric, order_names)
        if matrix is not None:
            tables[key] = matrix.reset_index()
            if metric == "attack_success_defended":
                problems.section(tier, "ordering_heatmap", ordering.plot_permutation_heatmap, matrix,
                                 str(out / "fig_permutation_heatmap.png"), f"Attack success by defense order ({tier})",
                                 "attack success rate")
    tables["ordering_best"] = problems.section(tier, "ordering_best", ordering.best_ordering, order.runs, order_names)
    pareto_tbl = problems.section(tier, "pareto", pareto.pareto_table, main.runs)
    if pareto_tbl is not None:
        tables["pareto"] = pareto_tbl
        problems.section(tier, "pareto_plot", pareto.plot_pareto_3d, pareto_tbl,
                         str(out / "fig_pareto_3d.html"), str(out / "fig_pareto_3d.png"))

    # D. router benchmark
    bench_summary = None
    if has_d4 and not args.no_router_bench:
        bench = problems.section(tier, "router_benchmark", benchmark_static_vs_dynamic, eval_source, d4_model, params,
                                 router_cal.threshold, attacks, RATIOS, SEEDS, repeats=args.repeats,
                                 extra_thresholds=ROUTER_SWEEP)
        if bench is not None:
            bench_summary = summarize_benchmark(bench)
            tables["router_benchmark"], tables["router_benchmark_summary"] = bench, bench_summary
            problems.section(tier, "router_plots", lambda: (
                plot_router_benchmark(bench_summary, str(out / "fig_router_benchmark.png")),
                plot_router_curve(bench_summary, str(out / "fig_router_curve.png"))))

    # E. extra measurements
    outcomes = query_outcomes(eval_source, outcome_configs(params, has_d4), attacks, d4_model)
    summary = summarize_outcomes(outcomes)
    summary.insert(0, "tier", tier)
    summary.insert(1, "d4", d4_model.label if has_d4 else "none")
    tables["outcomes_per_query"] = outcomes
    tables["summary"] = summary
    auc = chunk_auc(eval_scores, tier)
    tables["defense_auc"] = auc
    if has_d4 and "d4_prob" in eval_scores:
        clean_d4 = eval_scores[(eval_scores["poisoning_ratio"] == 0)]["d4_prob"].dropna()
        pois_d4 = eval_scores[eval_scores["is_poisoned"] == 1]["d4_prob"].dropna()
        if len(clean_d4) and len(pois_d4):
            recall, fpr = float((pois_d4 >= 0.5).mean()), float((clean_d4 >= 0.5).mean())
            table = prevalence_table(recall, fpr, n_poison=len(pois_d4), n_clean=len(clean_d4))
            table.insert(0, "tier", tier)
            tables["d4_prevalence_precision"] = table
    for name, table in tables.items():
        if table is not None and len(table):
            table.to_csv(out / f"{name}.csv", index=False)
    return {"tier": tier, "params": params, "router_cal": router_cal, "summary": summary, "auc": auc,
            "bench": bench_summary, "n_eval": len(eval_source.trigger_queries)}


def write_report(path: Path, args, results: list, d4_model, problems: Problems, n_queries_total: int) -> None:
    lines = ["# ieee_5000 evaluation run", "",
             f"- corpus: 5,000 clean chunks + 100 poison chunks per attack (poison share {IEEE_RATIO:.2%}); queries: "
             f"{args.eval_queries} ({results[0]['n_eval'] if results else 0} evaluated, others used only for calibration)",
             f"- D4: `{d4_model.label if d4_model else 'none'}`  |  tiers: {[r['tier'] for r in results]}  |  seed 0 (collections are static)",
             f"- embedder: {'FAKE (hash) embedder - plumbing check, not findings' if getattr(args, '_fake_embedder', False) else 'as supplied by the API (MiniLM)'}",
             "", "## Read before quoting any number", ""]
    caveats = [
        "The poison is synthetic and written by the team in one style per attack; a real attacker is more varied.",
        "Only ONE poison ratio exists in this corpus (1.96%). The 1/5/10/20% sweep of the 316-chunk benchmark is not reproduced here.",
        "Provenance tiers: `as_stored` separates poison from clean through the experiment's own labels (poison is named `controlled_*` "
        "and has an empty timestamp) - treat D2/D3 under it as exploratory. `spoofed` is what D3's identity check is built for. "
        "`forged` is a compromised trusted source: D2, D3 and the router cannot see it by construction.",
        "D4 was trained on 80 of the 100 targets' poison. Quote only the held-out queries; do not read them as generalisation to "
        "unseen attackers or other corpora. Poison chunks are 2-4x longer than clean ones: see the `length_only` ablation in "
        f"`{D4_MANIFEST}` before quoting a D4 F1.",
        "Thresholds were calibrated on the non-held-out queries only. D4 keeps its predeclared 0.5.",
        "`ref` (one authoritative text per CVE) blocks poison that reuses a known CVE id. It does not help against an invented id "
        "or a compromised registry; it is the baseline a classifier has to beat, not a defense claim.",
        "Generation success exists only for the RPI attack (marker POISONBENCH_RPI_SUCCESS); it is blank for sm and obea.",
        "Labels (is_poisoned) are stripped before any defense runs and used only to score.",
    ]
    lines += [f"- {c}" for c in caveats]
    for r in results:
        s = r["summary"]
        lines += ["", f"## Tier `{r['tier']}`", "",
                  f"Calibrated parameters: `{ {k: round(v, 3) for k, v in r['params'].items()} }`  |  router threshold "
                  f"{r['router_cal'].threshold:.3f} (poison recall {r['router_cal'].recall:.2f}, attained={r['router_cal'].attained})", ""]
        if s.empty:
            lines.append("_no rows_")
            continue
        view = s[s["config"].isin(["none", "d1", "d2", "d3", "d1c", "ref", "d4", "d1>d2>d3", "d1>d2>d3>d4", "d1>d2>d3>d4[cal]",
                                   "d1>d4", "d1>d4[cal]", "ref>d4", "d3>d4"])]
        lines.append("| config | attack | raw ASR | defended ASR (95% CI) | poison blocked | legit kept | empty ctx | clean blocked/q | ms mean |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, row in view.iterrows():
            if row["attack"] == "none":
                continue
            lines.append(f"| {row['config']} | {row['attack']} | {row['raw_asr']:.2f} | "
                         f"{row['defended_asr']:.2f} ({row['defended_asr_lo']:.2f}-{row['defended_asr_hi']:.2f}) | "
                         f"{row['poison_blocked_rate']:.2f} | {row['legit_kept_rate']:.2f} | {row['empty_context_rate']:.2f} | "
                         f"{row['clean_blocked_per_query']:.2f} | {row['latency_ms_mean']:.1f} |")
        if not r["auc"].empty:
            overall = r["auc"][r["auc"]["attack"] == "all"][["defense", "auc"]]
            def label(defense, auc):
                note = " (inverted: poison scores LOWER than clean)" if auc < 0.4 else ""
                return f"{defense}={auc:.2f}{note}"
            lines += ["", "Score AUC over retrieved chunks (0.5 = no signal): " +
                      ", ".join(label(d, a) for d, a in zip(overall["defense"], overall["auc"]))]
            if r["tier"] == "as_stored":
                lines.append("D2/D3 AUC near 1.0 under `as_stored` comes from the experiment's labels, not from the defenses (see caveats).")
            if r["tier"] == "forged":
                lines.append("Under `forged`, D3 sits at 0.5 by construction. D2 can fall BELOW 0.5: the forged poison copies its target's CVE id, "
                             "so it earns corroboration from the legitimate chunk, while clean chunks retrieved alone earn none.")
    if problems:
        lines += ["", "## Sections that failed (results above may be incomplete)", ""]
        lines += [f"- `{p['tier']}/{p['section']}`: {p['error']}" for p in problems]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true", help="only check the four collections")
    ap.add_argument("--tier", default="all", choices=["all", *ANALYSIS_TIERS])
    ap.add_argument("--d4", default="random_forest",
                    choices=["none", "random_forest", "xgboost", "transfer_random_forest", "transfer_xgboost"])
    ap.add_argument("--d4-dir", default=str(MODEL_DIR), help="folder with the six-feature D4 models")
    ap.add_argument("--train-d4", action="store_true", help="train the six-feature models first if they are missing")
    ap.add_argument("--eval-queries", default="heldout", choices=["heldout", "all"],
                    help="'all' also evaluates the 80 queries D4 trained on (optimistic for D4)")
    ap.add_argument("--max-eval-queries", type=int, default=0, help="cap evaluation queries (quick runs)")
    ap.add_argument("--generate", action="store_true", help="also run the LLM for the RPI attack (needs Ollama)")
    ap.add_argument("--no-router-bench", action="store_true")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats in the router benchmark")
    ap.add_argument("--chroma-host", default=None)
    ap.add_argument("--chroma-port", type=int, default=None)
    ap.add_argument("--db", default=None, help="result store URL; default sqlite under --out")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None, client=None, embed_query=None, fake_embedder: bool = False) -> int:
    args = parse_args(argv)
    args._fake_embedder = fake_embedder
    logging.getLogger("defenses").setLevel(logging.ERROR)   # D1 warns when earlier stages shrink the batch; expected
    if client is None:
        import chromadb
        if args.chroma_host or args.chroma_port:
            client = chromadb.HttpClient(host=args.chroma_host or "localhost", port=args.chroma_port or 8001)
        else:
            from api.pipeline.vectorstore import get_client
            client = get_client()

    rows = preflight(client)
    for row in rows:
        status = "OK     " if row["ok"] else ("WRONG  " if row["exists"] else "MISSING")
        print(f"{status} {row['collection']:34s} chunks={row['chunks']:<6} (want {row['expected_chunks']})  "
              f"poisoned={row['poisoned']} (want {row['expected_poisoned']})")
    if args.preflight:
        return 0 if all(r["ok"] for r in rows) else 1
    if not all(r["exists"] for r in rows):
        print("Some collections are missing; build them with scripts/ingest_corpus_ieee.py and the three "
              "scripts/ingest_*_ieee.py attack scripts, then rerun with --preflight.")
        return 2
    if not all(r["ok"] for r in rows):
        print("WARNING: collection sizes differ from the build scripts' output; numbers will not match the paper's setup.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = args.db or f"sqlite:///{(out_dir / 'poisonbench_eval.db').as_posix()}"
    store = ResultStore(db)

    d4_model = None
    if args.d4 != "none":
        try:
            d4_model = load_d4(args.d4, Path(args.d4_dir))
        except FileNotFoundError:
            if args.train_d4 and not args.d4.startswith("transfer_"):
                print("training six-feature D4 models ...")
                train(Path(args.d4_dir), overwrite=True)
                d4_model = load_d4(args.d4, Path(args.d4_dir))
            else:
                print(f"D4 model '{args.d4}' not found. Run `python -m evaluation.ieee5000_d4 --train` "
                      "(or pass --train-d4), or use --d4 none.")
                return 2
    print(f"Defense 4: {d4_model.label if d4_model else 'none'}")

    if embed_query is not None:
        ref_embed = lambda texts: [embed_query(t) for t in texts]   # noqa: E731
    else:
        def ref_embed(texts):
            from api.pipeline.embedding import embed_texts
            return embed_texts(texts)
    reference = CorpusReference.from_collection(client.get_collection(CLEAN_COLLECTION), embed=ref_embed)
    set_reference(reference)
    print(f"Corpus reference: {reference.n_chunks} clean chunks, {len(reference.registry)} CVE ids "
          f"(vectors via {reference.method})")
    base_source = build_source(client, embed_query, tier="as_stored")
    tiers = list(ANALYSIS_TIERS) if args.tier == "all" else [args.tier]
    problems = Problems()
    results = []
    for tier in tiers:
        results.append(run_tier(tier, base_source, d4_model, args, store, out_dir, problems))

    all_summary = pd.concat([r["summary"] for r in results], ignore_index=True)
    all_summary.to_csv(out_dir / "summary.csv", index=False)
    write_report(out_dir / "report.md", args, results, d4_model, problems, len(base_source.trigger_queries))
    print(f"\nwrote {out_dir / 'report.md'} and {out_dir / 'summary.csv'}  (database: {db})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
