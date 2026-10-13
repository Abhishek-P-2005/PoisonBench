"""
evaluation/analysis/ordering.py
-------------------------------
Module 2: Defense Ordering Optimization (PRD v2.0 Reqs 2.1 - 2.3).

Builds every ordering of the orderable defenses, so the sweep can show
whether execution order changes security, false positives or latency.
D4 (the classifier) is kept last as the final gate by default; set
include_tail=True to permute it as well (N! with N = 4 gives 24 orders).
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from evaluation.config import make_config


def permutation_configs(
    params: dict | None = None,
    orderable=("d1", "d2", "d3"),
    tail=("d4",),
    include_tail: bool = False,
    use_router: bool = False,
    router_threshold: float | None = None,
    label: str = "",
) -> list:
    configs = []
    if include_tail:
        all_stages = tuple(orderable) + tuple(tail)
        for order in permutations(all_stages):
            configs.append(make_config(order, params, use_router, router_threshold, label))
    else:
        for order in permutations(orderable):
            configs.append(make_config(tuple(order) + tuple(tail), params, use_router, router_threshold, label))
    return configs


def _attacked(runs_df: pd.DataFrame) -> pd.DataFrame:
    return runs_df[runs_df["poisoning_ratio"] > 0]


def ordering_matrix(runs_df: pd.DataFrame, metric: str, config_names: list | None = None) -> pd.DataFrame:
    """Rows = orderings, columns = attack types, cells = mean of `metric`."""
    frame = _attacked(runs_df)
    if config_names is not None:
        frame = frame[frame["config_name"].isin(config_names)]
    table = frame.pivot_table(index="config_name", columns="attack_type", values=metric, aggfunc="mean")
    table["mean"] = table.mean(axis=1)
    return table.sort_values("mean")


def order_sensitivity(matrix: pd.DataFrame) -> pd.Series:
    """Spread (max - min across orderings) per attack type. 0 means order does not matter."""
    values = matrix.drop(columns=["mean"], errors="ignore")
    return values.max() - values.min()


def drop_rates_by_step(stages_df: pd.DataFrame, runs_df: pd.DataFrame, config_names: list | None = None) -> pd.DataFrame:
    """Req 2.3: chunks in, share blocked and latency at each position of each ordering."""
    attacked_ids = set(_attacked(runs_df)["run_id"])
    frame = stages_df[stages_df["run_id"].isin(attacked_ids)].copy()
    names = runs_df.set_index("run_id")["config_name"]
    frame["config_name"] = frame["run_id"].map(names)
    if config_names is not None:
        frame = frame[frame["config_name"].isin(config_names)]

    grouped = frame.groupby(["config_name", "position", "stage"]).agg(
        mean_n_in=("n_in", "mean"),
        mean_blocked=("n_blocked", "mean"),
        mean_latency_ms=("latency_ms", "mean"),
    ).reset_index()
    grouped["drop_rate"] = grouped["mean_blocked"] / grouped["mean_n_in"].replace(0, np.nan)
    return grouped


def best_ordering(runs_df: pd.DataFrame, config_names: list) -> pd.DataFrame:
    """Orderings ranked by security (high), then false positives (low), then latency (low)."""
    frame = runs_df[runs_df["config_name"].isin(config_names)]
    attacked = _attacked(frame).groupby("config_name").agg(
        security=("security_blocked_pct", "mean"),
        attack_success_defended=("attack_success_defended", "mean"),
    )
    overall = frame.groupby("config_name").agg(
        fpr=("fpr", "mean"),
        latency_ms=("latency_ms_mean", "mean"),
    )
    table = attacked.join(overall)
    return table.sort_values(["security", "fpr", "latency_ms"], ascending=[False, True, True])


def plot_permutation_heatmap(matrix: pd.DataFrame, path: str, title: str, value_label: str, lower_is_better: bool = True) -> None:
    """Deliverable 3: N x M colour grid of orderings against attack types."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = matrix.drop(columns=["mean"], errors="ignore")
    fig, ax = plt.subplots(figsize=(1.6 * len(data.columns) + 4, 0.55 * len(data) + 2))
    sns.heatmap(
        data, annot=True, fmt=".2f", ax=ax,
        cmap="viridis_r" if lower_is_better else "viridis",
        cbar_kws={"label": value_label},
    )
    ax.set_xlabel("Attack type")
    ax.set_ylabel("Defense order")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
