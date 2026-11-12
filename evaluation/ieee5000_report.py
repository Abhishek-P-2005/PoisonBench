"""
evaluation/ieee5000_report.py
-----------------------------
Measurement helpers for runs on the ieee_5000 corpus. Read-only with respect to the
collections and the existing result tables: everything here either re-runs a pipeline on
retrieved candidates (cheap, in memory) or summarises a DataFrame it is given.

What it adds on top of the harness's own run rows:

  query_outcomes      per query and config: was the poison in the final context, did the
                      LEGITIMATE document survive, was the context emptied (the paper's
                      "worse than the attack" failure), per-query latency.
  summarize_outcomes  rates with Wilson intervals over queries and latency mean / p95.
  chunk_auc           threshold-free AUC of each defense's score from a score_pass frame.
  prevalence_table    precision of a detector at realistic poison prevalence. This corpus is
                      ~2% poison and the held-out classifier test set is ~6% poison; a deployed
                      store may be far cleaner than either.
  compare_summaries   diff two summary CSVs (for example a run with real MiniLM embeddings
                      against an earlier run, or before and after a code change).

    python -m evaluation.ieee5000_report --compare out_a/summary.csv out_b/summary.csv
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.config import CLEAN_ATTACK, CONTEXT_SIZE      # noqa: E402
from evaluation.pipeline import DefensePipeline               # noqa: E402
from evaluation.runner import truth_map                       # noqa: E402
from evaluation.stages import build_stages                    # noqa: E402


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion; (nan, nan) when n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap(per_query: pd.DataFrame, numerator: str, denominator: str,
                      n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """
    95% interval of sum(numerator)/sum(denominator) resampling QUERIES, not chunks.
    Chunks retrieved for one query are not independent, so a per-chunk interval is too narrow.
    """
    num = per_query[numerator].to_numpy(dtype=float)
    den = per_query[denominator].to_numpy(dtype=float)
    if len(num) == 0 or den.sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(num), len(num))
        total = den[pick].sum()
        if total > 0:
            stats.append(num[pick].sum() / total)
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


# ── per-query outcomes ────────────────────────────────────────────────────────

def query_outcomes(source, configs, attacks, d4_model=None, context_size: int = CONTEXT_SIZE,
                   include_clean: bool = True) -> pd.DataFrame:
    """
    Re-run each config on the source's retrieved candidates and record what the language
    model would have been given. `source` must expose iter_queries() and expected_doc().
    """
    scenarios = ([CLEAN_ATTACK] if include_clean else []) + list(attacks)
    rows = []
    for attack in scenarios:
        scenario_queries = source.iter_queries(attack, 0.0, 0)
        for config in configs:
            supports = getattr(source, "supports", None)
            if supports is not None and not supports(config):
                continue
            pipeline = DefensePipeline(build_stages(config, d4_model), context_size, config.name)
            if scenario_queries:
                pipeline.run(scenario_queries[0].candidates)          # warm-up, not recorded
            for sq in scenario_queries:
                result = pipeline.run(sq.candidates)
                truth = truth_map(sq.candidates)
                expected = source.expected_doc(sq.query_id)
                ranked = sorted(sq.candidates, key=lambda c: c["similarity"], reverse=True)
                raw_ids = [c["chunk_id"] for c in ranked[:context_size]]
                final_ids = [c["chunk_id"] for c in result.final_context]

                def is_legit(chunk_id):
                    return (not truth.get(chunk_id, False)) and chunk_id.split("::")[0] == expected

                n_poison = sum(1 for c in sq.candidates if truth[c["chunk_id"]])
                rows.append({
                    "config": config.name, "attack": attack, "query_id": sq.query_id,
                    "n_poison_candidates": n_poison,
                    "poison_in_raw_context": int(any(truth.get(i, False) for i in raw_ids)),
                    "poison_in_context": int(any(truth.get(i, False) for i in final_ids)),
                    "legit_in_raw_context": int(any(is_legit(i) for i in raw_ids)),
                    "legit_in_context": int(any(is_legit(i) for i in final_ids)),
                    "empty_context": int(len(final_ids) == 0),
                    "poison_blocked": sum(1 for i in result.blocked_ids if truth.get(i, False)),
                    "clean_blocked": sum(1 for i in result.blocked_ids if not truth.get(i, False)),
                    "poison_flagged": sum(1 for i in (result.flagged_ids | result.blocked_ids) if truth.get(i, False)),
                    "clean_flagged": sum(1 for i in (result.flagged_ids | result.blocked_ids) if not truth.get(i, False)),
                    "latency_ms": result.total_latency_ms,
                })
    return pd.DataFrame(rows)


def summarize_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, attack). Rates are over queries; intervals are 95% Wilson / bootstrap."""
    if outcomes.empty:
        return pd.DataFrame()
    out = []
    for (config, attack), g in outcomes.groupby(["config", "attack"], sort=False):
        n = len(g)
        raw = int(g["poison_in_raw_context"].sum())
        defended = int(g["poison_in_context"].sum())
        legit_raw = int(g["legit_in_raw_context"].sum())
        legit_kept = int(g.loc[g["legit_in_raw_context"] == 1, "legit_in_context"].sum())
        empty = int(g["empty_context"].sum())
        lo, hi = wilson(defended, n)
        det_lo, det_hi = cluster_bootstrap(g, "poison_flagged", "n_poison_candidates")
        n_poison = int(g["n_poison_candidates"].sum())
        out.append({
            "config": config, "attack": attack, "n_queries": n,
            "raw_asr": raw / n, "defended_asr": defended / n,
            "defended_asr_lo": lo, "defended_asr_hi": hi,
            "poison_blocked_rate": (g["poison_blocked"].sum() / n_poison) if n_poison else float("nan"),
            "poison_flagged_rate": (g["poison_flagged"].sum() / n_poison) if n_poison else float("nan"),
            "poison_flagged_lo": det_lo, "poison_flagged_hi": det_hi,
            "legit_kept_rate": (legit_kept / legit_raw) if legit_raw else float("nan"),
            "empty_context_rate": empty / n,
            "clean_blocked_per_query": float(g["clean_blocked"].mean()),
            "clean_flagged_per_query": float(g["clean_flagged"].mean()),
            "latency_ms_mean": float(g["latency_ms"].mean()),
            "latency_ms_p95": float(np.percentile(g["latency_ms"], 95)),
        })
    return pd.DataFrame(out)


# ── chunk-level scores ────────────────────────────────────────────────────────

SCORE_COLUMNS = {"d1": "d1_z", "d2": "d2_suspicion", "d3": "d3_suspicion", "d4": "d4_prob", "router_risk": "risk_score"}


def chunk_auc(chunks_df: pd.DataFrame, tier: str = "") -> pd.DataFrame:
    """ROC-AUC of every defense score against the label, overall and per attack. 0.5 = no signal."""
    from sklearn.metrics import roc_auc_score
    rows = []
    if chunks_df.empty:
        return pd.DataFrame(rows)
    clean = chunks_df[chunks_df["poisoning_ratio"] == 0]
    for attack in sorted(a for a in chunks_df["attack_type"].unique() if a != CLEAN_ATTACK) + ["all"]:
        poison = chunks_df[(chunks_df["is_poisoned"] == 1) & ((chunks_df["attack_type"] == attack) | (attack == "all"))]
        for name, column in SCORE_COLUMNS.items():
            if column not in chunks_df.columns:
                continue
            neg, pos = clean[column].dropna(), poison[column].dropna()
            if len(neg) == 0 or len(pos) == 0:
                continue
            y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
            rows.append({"tier": tier, "attack": attack, "defense": name, "auc": float(roc_auc_score(y, np.r_[neg, pos])),
                         "n_clean": len(neg), "n_poison": len(pos)})
    return pd.DataFrame(rows)


def prevalence_table(recall: float, fpr: float, prevalences=(0.005, 0.01, 0.02, 0.05, 0.10),
                     n_poison: int | None = None, n_clean: int | None = None) -> pd.DataFrame:
    """
    Precision of a detector with this recall / FPR if poison were `prevalence` of the chunks.
    With n_poison / n_clean given, also a conservative column using the worse end of each 95%
    Wilson interval (lower recall, higher FPR). A measured FPR of 0 on 200 clean chunks does not
    mean 0: the interval still reaches about 2%, which at 1% prevalence cuts precision a lot.
    """
    recall_lo = wilson(round(recall * n_poison), n_poison)[0] if n_poison else float("nan")
    fpr_hi = wilson(round(fpr * n_clean), n_clean)[1] if n_clean else float("nan")
    rows = []
    for prevalence in prevalences:
        tp, fp = prevalence * recall, (1 - prevalence) * fpr
        row = {"prevalence": prevalence, "recall": recall, "fpr": fpr,
               "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan")}
        if n_poison and n_clean:
            tp_c, fp_c = prevalence * recall_lo, (1 - prevalence) * fpr_hi
            row.update(recall_lo=recall_lo, fpr_hi=fpr_hi,
                       precision_conservative=tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# ── comparing two runs ────────────────────────────────────────────────────────

def compare_summaries(a: pd.DataFrame, b: pd.DataFrame, keys=("config", "attack"),
                      metrics=("defended_asr", "poison_blocked_rate", "legit_kept_rate",
                               "empty_context_rate", "latency_ms_mean")) -> pd.DataFrame:
    merged = a.merge(b, on=list(keys), suffixes=("_a", "_b"), how="outer")
    for metric in metrics:
        if f"{metric}_a" in merged and f"{metric}_b" in merged:
            merged[f"{metric}_delta"] = merged[f"{metric}_b"] - merged[f"{metric}_a"]
    keep = list(keys) + [c for c in merged.columns if c.endswith(("_a", "_b", "_delta")) and c.split("_a")[0].split("_b")[0].split("_delta")[0] in metrics]
    return merged[keep]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("SUMMARY_A", "SUMMARY_B"))
    args = ap.parse_args(argv)
    if args.compare:
        diff = compare_summaries(pd.read_csv(args.compare[0]), pd.read_csv(args.compare[1]))
        print(diff.round(3).to_string(index=False))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
