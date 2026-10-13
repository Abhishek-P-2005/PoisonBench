"""
evaluation/analysis/interaction.py
----------------------------------
Module 3: Defense Interaction and Marginal Contribution (PRD v2.0 Reqs 3.1, 3.2).

  subset_configs      every single-defense, pairwise and larger combination
  marginal_uplift     gain in security from adding one defense to a base set
  redundancy_table    how much of each defense's solo value survives once the
                      other defenses are already present
  pairwise_agreement  how often two defenses agree on a chunk, and which one
                      is right more often when they disagree
  unique_catches      poisoned chunks only one defense flags
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from evaluation.config import make_config


def subset_configs(stages=("d1", "d2", "d3", "d4"), params: dict | None = None, label: str = "") -> list:
    """All non-empty subsets of the defenses, each run in canonical order."""
    configs = []
    for size in range(1, len(stages) + 1):
        for subset in combinations(stages, size):
            configs.append(make_config(subset, params, label=label))
    return configs


def _stage_set(config_json: str) -> frozenset:
    return frozenset(json.loads(config_json)["stages"])


def config_means(runs_df: pd.DataFrame, metric: str = "security_blocked_pct") -> dict:
    """{frozenset(stages): mean metric over attacked runs}, including the empty set."""
    attacked = runs_df[runs_df["poisoning_ratio"] > 0]
    result = {}
    for config_json, group in attacked.groupby("config_json"):
        result[_stage_set(config_json)] = float(group[metric].mean())
    if frozenset() not in result:
        result[frozenset()] = 0.0
    return result


def marginal_uplift(runs_df: pd.DataFrame, metric: str = "security_blocked_pct") -> pd.DataFrame:
    """One row per (base set, added defense) where both configs were run."""
    means = config_means(runs_df, metric)
    rows = []
    for base, base_value in means.items():
        for other, other_value in means.items():
            if len(other) == len(base) + 1 and base < other:
                added = sorted(other - base)[0]
                rows.append({
                    "base": ">".join(sorted(base)) or "none",
                    "added": added,
                    "base_value": base_value,
                    "new_value": other_value,
                    "uplift": other_value - base_value,
                })
    return pd.DataFrame(rows).sort_values(["added", "base"]).reset_index(drop=True)


def redundancy_table(runs_df: pd.DataFrame, metric: str = "security_blocked_pct") -> pd.DataFrame:
    """
    solo_value : metric with only this defense
    last_in    : uplift when this defense is added to all the others
    redundancy : 1 - last_in / solo_value  (near 1 = fully covered by the others)
    """
    means = config_means(runs_df, metric)
    all_stages = max(means.keys(), key=len)
    rows = []
    for defense in sorted(all_stages):
        solo = means.get(frozenset([defense]))
        others = all_stages - {defense}
        without = means.get(others)
        full = means.get(all_stages)
        if solo is None or without is None or full is None:
            continue
        last_in = full - without
        redundancy = None
        if solo > 0:
            redundancy = 1.0 - last_in / solo
        rows.append({"defense": defense, "solo_value": solo, "last_in_uplift": last_in, "redundancy": redundancy})
    return pd.DataFrame(rows)


def _flag_columns(chunks_df: pd.DataFrame, d4_threshold: float = 0.5) -> dict:
    flags = {
        "d1": chunks_df["d1_flagged"].astype(int),
        "d2": chunks_df["d2_flagged"].astype(int),
        "d3": chunks_df["d3_flagged"].astype(int),
    }
    if "d4_prob" in chunks_df.columns and chunks_df["d4_prob"].notna().any():
        flags["d4"] = (chunks_df["d4_prob"] >= d4_threshold).astype(int)
    return flags


def pairwise_agreement(chunks_df: pd.DataFrame, d4_threshold: float = 0.5) -> pd.DataFrame:
    """
    For each pair of defenses on the same chunks:
      agreement      share of chunks where both give the same flag
      a_only_precision / b_only_precision
                     when only A (or only B) flags a chunk, how often it is
                     really poisoned: the defense with the higher value is
                     right more often in a disagreement
    """
    flags = _flag_columns(chunks_df, d4_threshold)
    truth = chunks_df["is_poisoned"].astype(int)
    rows = []
    for a, b in combinations(flags.keys(), 2):
        fa, fb = flags[a], flags[b]
        a_only = (fa == 1) & (fb == 0)
        b_only = (fb == 1) & (fa == 0)
        rows.append({
            "pair": f"{a}-{b}",
            "agreement": float((fa == fb).mean()),
            "both_flag": int(((fa == 1) & (fb == 1)).sum()),
            "a_only": int(a_only.sum()),
            "b_only": int(b_only.sum()),
            "a_only_precision": float(truth[a_only].mean()) if a_only.sum() > 0 else np.nan,
            "b_only_precision": float(truth[b_only].mean()) if b_only.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def unique_catches(chunks_df: pd.DataFrame, d4_threshold: float = 0.5) -> pd.DataFrame:
    """Among poisoned chunks: how many each defense catches, and how many only it catches."""
    flags = _flag_columns(chunks_df, d4_threshold)
    poisoned = chunks_df["is_poisoned"] == 1
    n_poison = int(poisoned.sum())
    any_flag = sum(flags[name] for name in flags) > 0

    rows = []
    for name, column in flags.items():
        others = sum(flags[other] for other in flags if other != name)
        caught = poisoned & (column == 1)
        unique = caught & (others == 0)
        rows.append({
            "defense": name,
            "caught": int(caught.sum()),
            "caught_rate": float(caught.sum() / n_poison) if n_poison else np.nan,
            "unique_catches": int(unique.sum()),
        })
    rows.append({
        "defense": "none_of_them",
        "caught": int((poisoned & ~any_flag).sum()),
        "caught_rate": float((poisoned & ~any_flag).sum() / n_poison) if n_poison else np.nan,
        "unique_catches": int((poisoned & ~any_flag).sum()),
    })
    return pd.DataFrame(rows)


def with_calibrated_flags(chunks_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Recompute the flag columns from the logged scores using calibrated thresholds."""
    frame = chunks_df.copy()
    if "d1_soft_z" in params:
        frame["d1_flagged"] = (frame["d1_z"] >= params["d1_soft_z"]).astype(int)
    if "d2_flag_threshold" in params:
        frame["d2_flagged"] = ((1.0 - frame["d2_suspicion"]) < params["d2_flag_threshold"]).astype(int)
    if "d3_flag_below" in params:
        frame["d3_flagged"] = ((1.0 - frame["d3_suspicion"]) < params["d3_flag_below"]).astype(int)
    return frame
