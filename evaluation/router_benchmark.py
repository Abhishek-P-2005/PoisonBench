"""
evaluation/router_benchmark.py
------------------------------
Module 5 evaluation (PRD v2.0 Req 5.4): dynamic routing against static
pipelines, plus calibration of the router's risk threshold.

Three pipelines are compared on the same retrieval batches:
    static_cheap  d1>d2>d3          no classifier at all
    static_full   d1>d2>d3>d4       classifier on every chunk
    dynamic       d1>d2>d3>d4+router  classifier only on high-risk chunks

Latency is the median of several timed repeats per query. Security and
false-positive numbers come from one logged run of each pipeline.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.config import CONTEXT_SIZE, STATIC_OVERHEAD_LIMIT_MS, make_config
from evaluation.pipeline import DefensePipeline
from evaluation.runner import _scenarios, run_one
from evaluation.stages import build_stages


@dataclass
class RouterCalibration:
    threshold:               float
    min_recall:              float
    recall:                  float     # poisoned chunks sent to the heavy path
    clean_high_fraction:     float     # clean chunks sent to the heavy path (lost savings)
    attained:                bool
    n_clean:                 int
    n_poison:                int
    note:                    str = ""


def calibrate_risk_threshold(clean_risk, poison_risk, min_recall: float = 0.95) -> RouterCalibration:
    """
    Pick the HIGHEST threshold that still sends at least `min_recall` of the
    poisoned chunks to the heavy path. A higher threshold sends fewer clean
    chunks to the heavy path, which is where the latency saving comes from.

    If poisoned chunks with risk 0 make that recall impossible, the router
    cannot be made safe by its threshold: threshold 0 (everything heavy) is
    returned and attained=False.
    """
    clean = np.array([r for r in clean_risk if r is not None and not np.isnan(r)], dtype=float)
    poison = np.array([r for r in poison_risk if r is not None and not np.isnan(r)], dtype=float)
    if len(clean) == 0 or len(poison) == 0:
        # Nothing to calibrate against. Fail safe: send every chunk to the heavy path.
        return RouterCalibration(
            threshold=0.0, min_recall=min_recall, recall=1.0, clean_high_fraction=1.0,
            attained=False, n_clean=len(clean), n_poison=len(poison),
            note="no clean or no poisoned chunk reaches the router in the calibration data "
                 "(the cheap stages already removed them), so it cannot be calibrated; "
                 "fail-safe: every chunk is inspected",
        )

    candidates = sorted(set(np.round(np.concatenate([clean, poison]), 6).tolist()), reverse=True)
    for threshold in candidates:
        if threshold <= 0:
            continue
        recall = float((poison >= threshold).mean())
        if recall >= min_recall:
            return RouterCalibration(
                threshold=float(threshold), min_recall=min_recall, recall=recall,
                clean_high_fraction=float((clean >= threshold).mean()),
                attained=True, n_clean=len(clean), n_poison=len(poison),
            )

    return RouterCalibration(
        threshold=0.0, min_recall=min_recall, recall=1.0, clean_high_fraction=1.0,
        attained=False, n_clean=len(clean), n_poison=len(poison),
    )


def calibrate_from_scores(chunks_df: pd.DataFrame, min_recall: float = 0.95) -> RouterCalibration:
    clean = chunks_df[chunks_df["is_poisoned"] == 0]["risk_score"].tolist()
    poison = chunks_df[chunks_df["is_poisoned"] == 1]["risk_score"].tolist()
    return calibrate_risk_threshold(clean, poison, min_recall)


def benchmark_configs(params: dict, router_threshold: float, extra_thresholds=()) -> dict:
    configs = {
        "static_cheap": make_config(["d1", "d2", "d3"], params),
        "static_full": make_config(["d1", "d2", "d3", "d4"], params),
        "dynamic": make_config(["d1", "d2", "d3", "d4"], params, use_router=True, router_threshold=router_threshold),
    }
    for threshold in extra_thresholds:
        configs[f"dynamic@{threshold:.2f}"] = make_config(
            ["d1", "d2", "d3", "d4"], params, use_router=True, router_threshold=threshold,
        )
    return configs


def _median_latency_ms(pipeline, scenario_queries, repeats: int) -> float:
    per_query = []
    for scenario_query in scenario_queries:
        timings = []
        for _ in range(repeats):
            timings.append(pipeline.run(scenario_query.candidates).total_latency_ms)
        per_query.append(statistics.median(timings))
    return float(np.mean(per_query))


def benchmark_static_vs_dynamic(
    source,
    d4_model,
    params: dict,
    router_threshold: float,
    attacks,
    ratios,
    seeds,
    repeats: int = 5,
    context_size: int = CONTEXT_SIZE,
    include_clean_baseline: bool = True,
    extra_thresholds=(),
) -> pd.DataFrame:
    configs = benchmark_configs(params, router_threshold, extra_thresholds)
    rows = []

    for seed in seeds:
        for attack, ratio in _scenarios(attacks, ratios, include_clean_baseline):
            scenario_queries = source.iter_queries(attack, ratio, seed)
            for mode, config in configs.items():
                pipeline = DefensePipeline(build_stages(config, d4_model), context_size, config.name)
                pipeline.run(scenario_queries[0].candidates)   # warm-up
                latency_ms = _median_latency_ms(pipeline, scenario_queries, repeats)

                logged = run_one(
                    scenario_queries, config, attack, ratio, seed, "router_benchmark",
                    source.name, d4_model=d4_model, context_size=context_size,
                )
                run = logged["run"]
                routed_low = sum(r["routed_low"] or 0 for r in logged["stages"])
                routed_high = sum(r["routed_high"] or 0 for r in logged["stages"])
                d4_inspected = sum(r["clean_in"] + r["poison_in"] for r in logged["stages"] if r["stage"] == "d4")

                rows.append({
                    "mode": mode,
                    "attack_type": attack,
                    "poisoning_ratio": ratio,
                    "seed": seed,
                    "latency_ms": latency_ms,
                    "throughput_qps": 1000.0 / latency_ms if latency_ms > 0 else float("nan"),
                    "security_blocked_pct": run["security_blocked_pct"],
                    "attack_success_defended": run["attack_success_defended"],
                    "detection_rate": run["detection_rate"],
                    "fpr": run["fpr"],
                    "d4_chunks_inspected": d4_inspected,
                    "d4_chunks_skipped": routed_low,
                    "router_high": routed_high,
                })
    return pd.DataFrame(rows)


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per mode and attack type over attacked runs, plus an 'all attacks' row per mode."""
    attacked = df[df["poisoning_ratio"] > 0]
    columns = ["latency_ms", "throughput_qps", "security_blocked_pct", "attack_success_defended",
               "detection_rate", "d4_chunks_inspected", "d4_chunks_skipped"]
    by_attack = attacked.groupby(["mode", "attack_type"])[columns].mean().reset_index()
    overall = attacked.groupby("mode")[columns].mean().reset_index()
    overall["attack_type"] = "all"
    fpr = df.groupby("mode")["fpr"].mean()
    summary = pd.concat([by_attack, overall], ignore_index=True)
    summary["fpr_all_runs"] = summary["mode"].map(fpr)
    return summary


def check_static_overhead(df: pd.DataFrame) -> dict:
    """NFR check: static defense overhead must stay below the limit per query batch."""
    result = {}
    for mode in ("static_cheap", "static_full"):
        worst = float(df[df["mode"] == mode]["latency_ms"].max())
        result[mode] = {"worst_ms": worst, "limit_ms": STATIC_OVERHEAD_LIMIT_MS, "ok": worst < STATIC_OVERHEAD_LIMIT_MS}
    return result


def plot_router_benchmark(summary: pd.DataFrame, path: str) -> None:
    """Deliverable 5: grouped bars, static pipelines against the adaptive router."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ["static_cheap", "static_full", "dynamic"]
    colors = {"static_cheap": "#9aa0a6", "static_full": "#1a73e8", "dynamic": "#d93025"}
    summary = summary[summary["mode"].isin(modes)]
    attacks = [a for a in summary["attack_type"].unique() if a != "all"] + ["all"]
    panels = [
        ("latency_ms", "Latency per query batch (ms)"),
        ("security_blocked_pct", "Security (% poisoned context blocked)"),
        ("throughput_qps", "Throughput (queries / second)"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    width = 0.8 / len(modes)
    for ax, (column, label) in zip(axes, panels):
        for m, mode in enumerate(modes):
            values = []
            for attack in attacks:
                row = summary[(summary["mode"] == mode) & (summary["attack_type"] == attack)]
                values.append(float(row[column].iloc[0]) if len(row) else 0.0)
            positions = [i + m * width for i in range(len(attacks))]
            ax.bar(positions, values, width, label=mode, color=colors[mode])
        ax.set_xticks([i + width for i in range(len(attacks))])
        ax.set_xticklabels(attacks)
        ax.set_title(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(modes))
    fig.suptitle("Adaptive router vs static pipelines")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_router_curve(summary: pd.DataFrame, path: str) -> None:
    """Latency against security for every router threshold, with the static pipelines for reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = summary[summary["attack_type"] == "all"].sort_values("latency_ms")
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in overall.iterrows():
        is_static = row["mode"].startswith("static")
        ax.scatter(row["latency_ms"], row["security_blocked_pct"], s=70,
                   c="#1a73e8" if is_static else "#d93025", marker="s" if is_static else "o")
        ax.annotate(row["mode"], (row["latency_ms"], row["security_blocked_pct"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Latency per query batch (ms)")
    ax.set_ylabel("Security (% poisoned context blocked)")
    ax.set_title("Router operating points (red) vs static pipelines (blue)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def router_sensitivity(
    variants: dict,
    d4_model,
    params: dict,
    attacks,
    ratios,
    cal_seeds,
    eval_seeds,
    min_recall: float = 0.95,
    repeats: int = 3,
) -> pd.DataFrame:
    """
    How much the router can save depends on how well an attacker can forge
    provenance. `variants` maps a label to a scenario source (for the
    synthetic source: different stealth_prob). For each variant, calibrate
    the router at `min_recall`, then compare it with the static full pipeline.
    """
    from evaluation.runner import score_pass

    rows = []
    for label, source in variants.items():
        scores = score_pass(source, attacks, ratios, cal_seeds, group=f"sens_{label}", d4_model=d4_model, params=params)
        calibration = calibrate_from_scores(scores, min_recall)
        bench = benchmark_static_vs_dynamic(
            source, d4_model, params, calibration.threshold, attacks, ratios, eval_seeds, repeats=repeats,
        )
        summary = summarize_benchmark(bench)
        overall = summary[summary["attack_type"] == "all"].set_index("mode")
        full, dynamic = overall.loc["static_full"], overall.loc["dynamic"]
        total_d4 = dynamic["d4_chunks_inspected"] + dynamic["d4_chunks_skipped"]
        rows.append({
            "variant": label,
            "router_threshold": calibration.threshold,
            "target_recall_reached": calibration.attained,
            "poison_to_heavy_path": calibration.recall,
            "clean_to_heavy_path": calibration.clean_high_fraction,
            "latency_static_full_ms": full["latency_ms"],
            "latency_dynamic_ms": dynamic["latency_ms"],
            "latency_saving_pct": 100.0 * (1.0 - dynamic["latency_ms"] / full["latency_ms"]),
            "security_static_full": full["security_blocked_pct"],
            "security_dynamic": dynamic["security_blocked_pct"],
            "security_loss_pts": full["security_blocked_pct"] - dynamic["security_blocked_pct"],
            "d4_skipped_fraction": dynamic["d4_chunks_skipped"] / total_d4 if total_d4 else float("nan"),
            "calibration_note": calibration.note,
        })
    return pd.DataFrame(rows)
