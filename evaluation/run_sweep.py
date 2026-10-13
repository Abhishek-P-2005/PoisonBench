"""
evaluation/run_sweep.py
-----------------------
One command that runs the whole evaluation and writes every table and figure.

    PYTHONPATH=. python -m evaluation.run_sweep --db sqlite:///poisonbench_eval.db --out eval_out
    PYTHONPATH=. python -m evaluation.run_sweep --db postgresql://user:pw@localhost:5432/poisonbench --out eval_out

Phases
  A  calibration   score every defense alone on calibration seeds, build
                   ROC / PR curves, pick thresholds (FPR <= 5%), pick the
                   router risk threshold                     (Module 1, 5)
  B  main sweeps   all defense subsets with default and calibrated
                   parameters, plus all orderings, on separate evaluation
                   seeds                                      (Modules 2, 3, 4)
  C  router        static vs dynamic benchmark                (Module 5)
  D  reports       CSV tables, figures, RESULTS_SUMMARY.md

Plugging in the real parts
  --d4-factory pkg.module:function   function() must return an object with
        predict_proba(rows) -> list[float]. Wrap Subash's XGBoost and
        Abhishek's Random Forest with evaluation.classifier.SklearnProbaModel.
  The synthetic source is for testing. For final numbers build a
  ChromaScenarioSource (see EVALUATION_README.md).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os

import pandas as pd

from evaluation.analysis import calibration, interaction, ordering, pareto
from evaluation.classifier import train_standin_model
from evaluation.config import ATTACK_TYPES, POISONING_RATIOS, make_config
from evaluation.generation import CachedGenerator, OllamaGenerator, SimulatedGenerator
from evaluation.plots import plot_embedding_map
from evaluation.router_benchmark import (
    benchmark_static_vs_dynamic,
    plot_router_curve,
    router_sensitivity,
    calibrate_from_scores,
    check_static_overhead,
    plot_router_benchmark,
    summarize_benchmark,
)
from evaluation.runner import collect_training_rows, run_sweep, score_pass
from evaluation.scenarios import SyntheticConfig, SyntheticScenarioSource
from evaluation.store import ResultStore, schema_sql


def load_factory(spec: str):
    module_name, function_name = spec.split(":")
    return getattr(importlib.import_module(module_name), function_name)()


def parse_args():
    parser = argparse.ArgumentParser(description="PoisonBench evaluation sweep")
    parser.add_argument("--db", default="sqlite:///poisonbench_eval.db")
    parser.add_argument("--out", default="eval_out")
    parser.add_argument("--quick", action="store_true", help="fewer ratios and seeds, for a fast check")
    parser.add_argument("--generation", choices=["none", "simulated", "ollama"], default="simulated")
    parser.add_argument("--ollama-model", default="phi3")
    parser.add_argument("--d4-factory", default=None, help="module:function returning the real Defense 4 model")
    parser.add_argument("--cal-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats in the router benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    # Defense 1 warns whenever an earlier stage shrinks the batch too far; that is
    # expected during ordering sweeps and is recorded in the logs instead.
    logging.getLogger("defenses").setLevel(logging.ERROR)

    ratios = (0.05, 0.20) if args.quick else POISONING_RATIOS
    cal_seeds = args.cal_seeds or ([100] if args.quick else [100, 101])
    eval_seeds = args.eval_seeds or ([1] if args.quick else [1, 2, 3])
    train_seeds = [200, 201]

    source = SyntheticScenarioSource()
    store = ResultStore(args.db)
    with open(os.path.join(args.out, "eval_schema.sql"), "w") as handle:
        handle.write(schema_sql())

    generator = None
    if args.generation == "simulated":
        generator = CachedGenerator(SimulatedGenerator())
    elif args.generation == "ollama":
        generator = CachedGenerator(OllamaGenerator(model=args.ollama_model))

    # Defense 4 model
    if args.d4_factory:
        d4_model = load_factory(args.d4_factory)
        d4_label = "real Defense 4 model"
    else:
        rows, labels = collect_training_rows(source, ATTACK_TYPES, ratios, train_seeds)
        d4_model = train_standin_model(rows, labels)
        d4_label = "STAND-IN random forest (not the real Defense 4)"
    print(f"Defense 4: {d4_label}")

    # ── Phase A: calibration ────────────────────────────────────────────
    print("Phase A: calibration")
    cal_scores = score_pass(source, ATTACK_TYPES, ratios, cal_seeds, group="calibration", store=store, d4_model=d4_model)
    reports = calibration.calibrate_all(cal_scores)
    params = calibration.thresholds_to_params(reports)
    # The router sees chunks after the cheap stages run with the calibrated parameters,
    # so its threshold is calibrated on risk scores produced the same way.
    router_scores = score_pass(source, ATTACK_TYPES, ratios, cal_seeds, group="calibration_router",
                               store=store, d4_model=None, params=params)
    router_cal = calibrate_from_scores(router_scores)
    calibration.plot_roc_pr(reports, os.path.join(args.out, "fig2_roc_pr.png"))

    with open(os.path.join(args.out, "calibration.json"), "w") as handle:
        json.dump({
            "params": params,
            "reports": {name: report.summary() for name, report in reports.items()},
            "router": router_cal.__dict__,
        }, handle, indent=2)
    print("  calibrated params:", {k: round(v, 3) for k, v in params.items()})
    print(f"  router threshold {router_cal.threshold:.3f} (poison recall {router_cal.recall:.2f}, "
          f"clean sent to heavy path {router_cal.clean_high_fraction:.2f}, attained={router_cal.attained})")

    # ── Phase B: main sweeps ────────────────────────────────────────────
    print("Phase B: sweeps")
    configs = [make_config([], label="")]
    for base_params, label in ((params, "cal"), ({}, "default")):
        configs.extend(interaction.subset_configs(params=base_params, label=label))
    for stages in (("d1", "d2", "d3", "d4"), ("d1", "d3", "d4")):
        configs.append(make_config(stages, params, use_router=True, router_threshold=router_cal.threshold, label="cal"))
    main = run_sweep(source, configs, ATTACK_TYPES, ratios, eval_seeds, group="main",
                     store=store, generator=generator, d4_model=d4_model, verbose=True)

    order_configs = ordering.permutation_configs(params=params, label="cal")
    order = run_sweep(source, order_configs, ATTACK_TYPES, ratios, eval_seeds, group="ordering",
                      store=store, d4_model=d4_model, verbose=True)

    cheap_order_configs = ordering.permutation_configs(params=params, tail=(), label="cal-cheap")
    cheap_order = run_sweep(source, cheap_order_configs, ATTACK_TYPES, ratios, eval_seeds, group="ordering_cheap",
                            store=store, d4_model=d4_model, verbose=True)

    eval_scores = score_pass(source, ATTACK_TYPES, ratios, eval_seeds, group="eval_scores", store=store, d4_model=d4_model)

    # ── Phase C: router benchmark ───────────────────────────────────────
    print("Phase C: router benchmark")
    sweep_thresholds = (0.05, 0.15, 0.25, 0.40, 0.60)
    bench = benchmark_static_vs_dynamic(
        source, d4_model, params, router_cal.threshold, ATTACK_TYPES, ratios, eval_seeds,
        repeats=args.repeats, extra_thresholds=sweep_thresholds,
    )
    bench_summary = summarize_benchmark(bench)
    overhead = check_static_overhead(bench)
    plot_router_benchmark(bench_summary, os.path.join(args.out, "fig5_router_benchmark.png"))
    plot_router_curve(bench_summary, os.path.join(args.out, "fig5b_router_operating_curve.png"))

    stealth_values = (0.0, 0.15, 0.30)
    variants = {}
    for value in stealth_values:
        # spoof_prob=1.0: every poisoned chunk that is not stealthy carries a
        # detectable provenance problem, so `value` is the share of poisoned
        # chunks whose provenance looks completely clean.
        variants[f"clean_looking={value:.2f}"] = SyntheticScenarioSource(SyntheticConfig(
            stealth_prob={attack: value for attack in ATTACK_TYPES},
            spoof_prob={attack: 1.0 for attack in ATTACK_TYPES},
        ))
    sensitivity = router_sensitivity(variants, d4_model, params, ATTACK_TYPES, ratios, cal_seeds, eval_seeds,
                                     repeats=max(2, args.repeats - 2))

    # ── Phase D: reports ────────────────────────────────────────────────
    print("Phase D: reports")
    runs = main.runs
    cal_runs = runs[runs["config_name"].str.contains(r"\[cal\]") | (runs["config_name"] == "none")]
    default_runs = runs[runs["config_name"].str.contains(r"\[default\]") | (runs["config_name"] == "none")]

    # Module 3
    uplift = interaction.marginal_uplift(cal_runs[~cal_runs["config_name"].str.contains("router")])
    redundancy = interaction.redundancy_table(cal_runs[~cal_runs["config_name"].str.contains("router")])
    flagged_scores = interaction.with_calibrated_flags(eval_scores, params)
    agreement = interaction.pairwise_agreement(flagged_scores, params.get("d4_threshold", 0.5))
    unique = interaction.unique_catches(flagged_scores, params.get("d4_threshold", 0.5))

    # Module 2
    cheap_names = [c.name for c in cheap_order_configs]
    cheap_success = ordering.ordering_matrix(cheap_order.runs, "attack_success_defended", cheap_names)
    cheap_best = ordering.best_ordering(cheap_order.runs, cheap_names)
    ordering.plot_permutation_heatmap(cheap_success, os.path.join(args.out, "fig3b_permutation_heatmap_no_d4.png"),
                                      "Attack success, defenses 1-3 only, by order", "attack success rate")
    order_names = [c.name for c in order_configs]
    order_security = ordering.ordering_matrix(order.runs, "security_blocked_pct", order_names)
    order_success = ordering.ordering_matrix(order.runs, "attack_success_defended", order_names)
    order_latency = ordering.ordering_matrix(order.runs, "latency_ms_mean", order_names)
    order_best = ordering.best_ordering(order.runs, order_names)
    order_drops = ordering.drop_rates_by_step(order.stages, order.runs, order_names)
    ordering.plot_permutation_heatmap(order_success, os.path.join(args.out, "fig3_permutation_heatmap.png"),
                                      "Attack success after defenses, by defense order", "attack success rate")

    # Module 4
    pareto_input = pd.concat([runs, order.runs], ignore_index=True)
    pareto_input = pareto_input[~pareto_input["config_name"].str.contains(r"\[default\]")]
    pareto_tbl = pareto.pareto_table(runs)
    pareto.plot_pareto_3d(pareto_tbl, os.path.join(args.out, "fig4_pareto_3d.html"), os.path.join(args.out, "fig4_pareto_3d.png"))

    # Figure 1
    plot_embedding_map(source, "optimized", 0.10, eval_seeds[0], params.get("d1_soft_z", 1.5),
                       os.path.join(args.out, "fig1_embedding_map.png"))

    tables = {
        "runs": runs, "ordering_runs": order.runs, "calibration_curves_summary": pd.DataFrame([r.summary() for r in reports.values()]),
        "marginal_uplift": uplift, "redundancy": redundancy, "pairwise_agreement": agreement, "unique_catches": unique,
        "ordering_attack_success": order_success.reset_index(), "ordering_security": order_security.reset_index(),
        "ordering_latency_ms": order_latency.reset_index(), "ordering_best": order_best.reset_index(),
        "ordering_drop_rates": order_drops, "pareto": pareto_tbl,
        "router_benchmark": bench, "router_benchmark_summary": bench_summary, "router_sensitivity": sensitivity,
        "ordering_no_d4_attack_success": cheap_success.reset_index(), "ordering_no_d4_best": cheap_best.reset_index(),
    }
    for name, table in tables.items():
        table.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)

    write_summary(args, d4_label, params, router_cal, reports, bench_summary, overhead,
                  runs, order_success, order_best, pareto_tbl, uplift, redundancy, agreement, unique, generator,
                  cheap_success, cheap_best, sensitivity)
    print(f"Done. Results in {args.out}/  (database: {args.db})")


def write_summary(args, d4_label, params, router_cal, reports, bench_summary, overhead,
                  runs, order_success, order_best, pareto_tbl, uplift, redundancy, agreement, unique, generator,
                  cheap_success, cheap_best, sensitivity):
    pd.set_option("display.width", 200)
    lines = ["# PoisonBench evaluation summary", ""]
    lines += [
        "> **Read first.** This run used the SYNTHETIC scenario source, "
        + ("a SIMULATED generator, " if args.generation == "simulated" else "")
        + f"and {d4_label}. The numbers show that the harness works end to end. "
        "They are not results about real attacks or the real Defense 4. Re-run with "
        "ChromaScenarioSource, Abhishek's attacks and the real classifiers for report numbers.",
        "",
        "## Module 1: calibrated thresholds (FPR ceiling 5%)",
        "```", pd.DataFrame([r.summary() for r in reports.values()]).round(3).to_string(index=False), "```",
        "Parameters: " + json.dumps({k: round(v, 4) for k, v in params.items()}), "",
        "## Module 2: best defense orderings (D4 last)", "```", order_best.round(3).to_string(), "```", "",
        "Attack success after defenses, by order and attack:", "```", order_success.round(3).to_string(), "```", "",
        "Orderings of Defenses 1-3 only (no classifier, where order can matter most):", "```",
        cheap_best.round(3).to_string(), "```", "```", cheap_success.round(3).to_string(), "```", "",
        "## Module 3: marginal uplift in security (% blocked)", "```", uplift.round(2).to_string(index=False), "```",
        "Redundancy (near 1 means the other defenses already cover it):", "```", redundancy.round(3).to_string(index=False), "```",
        "Pairwise agreement:", "```", agreement.round(3).to_string(index=False), "```",
        "Unique catches among poisoned chunks:", "```", unique.to_string(index=False), "```", "",
        "## Module 4: Pareto-optimal configurations", "```",
        pareto_tbl[pareto_tbl["pareto_optimal"]].round(2).to_string(index=False), "```", "",
        "## Module 5: adaptive router vs static pipelines",
        f"Router threshold {router_cal.threshold:.3f}: sends {router_cal.recall:.0%} of poisoned and "
        f"{router_cal.clean_high_fraction:.0%} of clean calibration chunks to the heavy path (target reached: {router_cal.attained}).",
        "```", bench_summary.round(3).to_string(index=False), "```",
        "Router sensitivity: share of poisoned chunks whose provenance looks completely clean (the router cannot see these):", "```",
        sensitivity.round(3).to_string(index=False), "```", "",
        "Static overhead check (limit 150 ms): " + json.dumps(overhead), "",
    ]
    with open(os.path.join(args.out, "RESULTS_SUMMARY.md"), "w") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
