"""
evaluation/analysis/pareto.py
-----------------------------
Module 4: Multi-Objective Pareto Trade-Off Profiling (PRD v2.0 Reqs 4.1, 4.2).

Every defense configuration gets three numbers:
    security   % of poisoned context chunks blocked        (higher is better)
    usability  100 - false positive rate in %               (higher is better)
    latency_ms mean defense latency per query batch        (lower is better)
and the non-dominated (Pareto-optimal) configurations are extracted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def config_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    """
    security : mean over attacked runs (poisoning ratio > 0)
    usability: 100 * (1 - mean FPR) over ALL runs that had clean chunks,
               clean baseline runs included
    """
    attacked = runs_df[runs_df["poisoning_ratio"] > 0]
    security = attacked.groupby("config_name")["security_blocked_pct"].mean()
    attack_success = attacked.groupby("config_name")["attack_success_defended"].mean()
    fpr = runs_df.groupby("config_name")["fpr"].mean()
    latency = runs_df.groupby("config_name")["latency_ms_mean"].mean()

    table = pd.DataFrame({
        "security": security,
        "attack_success": attack_success,
        "usability": 100.0 * (1.0 - fpr),
        "latency_ms": latency,
    })
    return table.dropna(subset=["security", "usability", "latency_ms"]).reset_index()


def pareto_front(table: pd.DataFrame) -> pd.Series:
    """True for configs that no other config beats on all three objectives."""
    values = table[["security", "usability", "latency_ms"]].to_numpy()
    n = len(values)
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            at_least_as_good = (
                values[j][0] >= values[i][0]
                and values[j][1] >= values[i][1]
                and values[j][2] <= values[i][2]
            )
            strictly_better = (
                values[j][0] > values[i][0]
                or values[j][1] > values[i][1]
                or values[j][2] < values[i][2]
            )
            if at_least_as_good and strictly_better:
                optimal[i] = False
                break
    return pd.Series(optimal, index=table.index)


def pareto_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    table = config_summary(runs_df)
    table["pareto_optimal"] = pareto_front(table)
    return table.sort_values(["pareto_optimal", "security"], ascending=[False, False]).reset_index(drop=True)


def plot_pareto_3d(table: pd.DataFrame, html_path: str, png_path: str) -> None:
    """Deliverable 4: interactive 3D scatter (HTML) plus a static PNG for the report."""
    import plotly.graph_objects as go

    figure = go.Figure()
    for optimal, color, name in ((False, "#9aa0a6", "dominated"), (True, "#d93025", "Pareto-optimal")):
        subset = table[table["pareto_optimal"] == optimal]
        figure.add_trace(go.Scatter3d(
            x=subset["security"], y=subset["usability"], z=subset["latency_ms"],
            mode="markers+text" if optimal else "markers",
            text=subset["config_name"] if optimal else None,
            textposition="top center",
            hovertext=subset["config_name"],
            marker=dict(size=7 if optimal else 4, color=color),
            name=name,
        ))
    figure.update_layout(
        scene=dict(
            xaxis_title="Security (% blocked)",
            yaxis_title="Usability (100 - FPR %)",
            zaxis_title="Latency (ms)",
        ),
        title="Security vs usability vs latency",
    )
    figure.write_html(html_path, include_plotlyjs=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for optimal, color, name in ((False, "#9aa0a6", "dominated"), (True, "#d93025", "Pareto-optimal")):
        subset = table[table["pareto_optimal"] == optimal]
        ax.scatter(subset["security"], subset["usability"], subset["latency_ms"], c=color, label=name, s=40 if optimal else 18)
    ax.set_xlabel("Security (% blocked)")
    ax.set_ylabel("Usability (100 - FPR %)")
    ax.set_zlabel("Latency (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
