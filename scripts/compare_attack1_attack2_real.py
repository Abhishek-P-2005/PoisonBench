import json
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = REPO_ROOT / "data" / "results"

ATTACK1_CSV = RESULTS_DIR / "real_semantic_mimicry_top5_v1_results.csv"
ATTACK2_CSV = RESULTS_DIR / "real_optimized_embedding_top5_v1_results.csv"

OUTPUT_CSV = RESULTS_DIR / "attack1_vs_attack2_comparison_v1.csv"
OUTPUT_JSON = RESULTS_DIR / "attack1_vs_attack2_comparison_v1_summary.json"


# ============================================================
# Helpers
# ============================================================

def reciprocal_rank(rank):
    if pd.isna(rank):
        return 0.0

    rank = float(rank)

    if rank <= 0:
        return 0.0

    return 1.0 / rank


def validate_results(df, attack_name):
    required_columns = {
        "query_id",
        "query",
        "legitimate_doc_id",
        "poison_doc_id",
        "retrieved_doc_ids",
        "legitimate_rank",
        "poison_rank",
        "legitimate_top5",
        "poison_top5",
        "poison_rank1",
        "poison_beats_legitimate",
        "error",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{attack_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if df["query_id"].duplicated().any():
        duplicates = (
            df.loc[df["query_id"].duplicated(), "query_id"]
            .tolist()
        )
        raise ValueError(
            f"{attack_name} contains duplicate query IDs: {duplicates}"
        )

    if len(df) != 20:
        raise ValueError(
            f"{attack_name} should contain 20 frozen queries, "
            f"but found {len(df)}."
        )

    error_rows = df[df["error"].notna()]

    if len(error_rows) > 0:
        raise ValueError(
            f"{attack_name} contains {len(error_rows)} API/error rows."
        )


def attack_metrics(df):
    total = len(df)

    return {
        "queries": int(total),

        "poison_top5_hits": int(df["poison_top5"].sum()),
        "poison_top5_asr": float(df["poison_top5"].mean()),

        "poison_rank1_hits": int(df["poison_rank1"].sum()),
        "poison_rank1_asr": float(df["poison_rank1"].mean()),

        "poison_beats_legitimate_hits":
            int(df["poison_beats_legitimate"].sum()),

        "poison_beats_legitimate_rate":
            float(df["poison_beats_legitimate"].mean()),

        "legitimate_top5_hits":
            int(df["legitimate_top5"].sum()),

        "legitimate_top5_recall":
            float(df["legitimate_top5"].mean()),

        "legitimate_mrr":
            float(
                df["legitimate_rank"]
                .apply(reciprocal_rank)
                .mean()
            ),

        "mean_legitimate_rank":
            float(df["legitimate_rank"].mean()),

        "mean_poison_rank":
            float(df["poison_rank"].mean()),
    }


# ============================================================
# Main comparison
# ============================================================

def main():

    print("=" * 72)
    print("PHASE 9A - ATTACK 1 VS ATTACK 2 COMPARATIVE ANALYSIS")
    print("=" * 72)

    print(f"Attack 1 results : {ATTACK1_CSV}")
    print(f"Attack 2 results : {ATTACK2_CSV}")
    print(f"Output CSV       : {OUTPUT_CSV}")
    print(f"Output JSON      : {OUTPUT_JSON}")
    print()

    # --------------------------------------------------------
    # Load frozen results
    # --------------------------------------------------------

    attack1 = pd.read_csv(ATTACK1_CSV)
    attack2 = pd.read_csv(ATTACK2_CSV)

    validate_results(
        attack1,
        "Attack 1 - Semantic Mimicry",
    )

    validate_results(
        attack2,
        "Attack 2 - Optimized Embedding",
    )

    print(
        f"Attack 1 rows loaded: {len(attack1)}"
    )

    print(
        f"Attack 2 rows loaded: {len(attack2)}"
    )

    # --------------------------------------------------------
    # Validate benchmark alignment
    # --------------------------------------------------------

    attack1_ids = set(attack1["query_id"])
    attack2_ids = set(attack2["query_id"])

    if attack1_ids != attack2_ids:

        only_attack1 = sorted(
            attack1_ids - attack2_ids
        )

        only_attack2 = sorted(
            attack2_ids - attack1_ids
        )

        raise ValueError(
            "Attack result files do not contain the same "
            "query IDs.\n"
            f"Only Attack 1: {only_attack1}\n"
            f"Only Attack 2: {only_attack2}"
        )

    merged = attack1.merge(
        attack2,
        on="query_id",
        suffixes=("_attack1", "_attack2"),
        validate="one_to_one",
    )

    # Validate query text and legitimate document identity
    query_mismatch = (
        merged["query_attack1"]
        != merged["query_attack2"]
    )

    if query_mismatch.any():

        bad_ids = merged.loc[
            query_mismatch,
            "query_id"
        ].tolist()

        raise ValueError(
            "Query text mismatch between frozen results "
            f"for: {bad_ids}"
        )

    legitimate_mismatch = (
        merged["legitimate_doc_id_attack1"]
        != merged["legitimate_doc_id_attack2"]
    )

    if legitimate_mismatch.any():

        bad_ids = merged.loc[
            legitimate_mismatch,
            "query_id"
        ].tolist()

        raise ValueError(
            "Legitimate document mismatch between attacks "
            f"for: {bad_ids}"
        )

    # --------------------------------------------------------
    # Build query-level comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame()

    comparison["query_id"] = merged["query_id"]

    comparison["query"] = merged["query_attack1"]

    comparison["legitimate_doc_id"] = (
        merged["legitimate_doc_id_attack1"]
    )

    comparison["attack1_poison_doc_id"] = (
        merged["poison_doc_id_attack1"]
    )

    comparison["attack2_poison_doc_id"] = (
        merged["poison_doc_id_attack2"]
    )

    comparison["attack1_legitimate_rank"] = (
        merged["legitimate_rank_attack1"]
    )

    comparison["attack2_legitimate_rank"] = (
        merged["legitimate_rank_attack2"]
    )

    comparison["legitimate_rank_change"] = (
        comparison["attack2_legitimate_rank"]
        - comparison["attack1_legitimate_rank"]
    )

    comparison["attack1_poison_rank"] = (
        merged["poison_rank_attack1"]
    )

    comparison["attack2_poison_rank"] = (
        merged["poison_rank_attack2"]
    )

    comparison["poison_rank_improvement"] = (
        comparison["attack1_poison_rank"]
        - comparison["attack2_poison_rank"]
    )

    comparison["attack1_poison_top5"] = (
        merged["poison_top5_attack1"]
    )

    comparison["attack2_poison_top5"] = (
        merged["poison_top5_attack2"]
    )

    comparison["top5_failure_converted"] = (
        (~comparison["attack1_poison_top5"])
        & comparison["attack2_poison_top5"]
    )

    comparison["attack1_poison_rank1"] = (
        merged["poison_rank1_attack1"]
    )

    comparison["attack2_poison_rank1"] = (
        merged["poison_rank1_attack2"]
    )

    comparison["rank1_failure_converted"] = (
        (~comparison["attack1_poison_rank1"])
        & comparison["attack2_poison_rank1"]
    )

    comparison["attack1_beats_legitimate"] = (
        merged["poison_beats_legitimate_attack1"]
    )

    comparison["attack2_beats_legitimate"] = (
        merged["poison_beats_legitimate_attack2"]
    )

    comparison["beat_failure_converted"] = (
        (~comparison["attack1_beats_legitimate"])
        & comparison["attack2_beats_legitimate"]
    )

    comparison["attack1_legitimate_rr"] = (
        comparison["attack1_legitimate_rank"]
        .apply(reciprocal_rank)
    )

    comparison["attack2_legitimate_rr"] = (
        comparison["attack2_legitimate_rank"]
        .apply(reciprocal_rank)
    )

    comparison["legitimate_rr_change"] = (
        comparison["attack2_legitimate_rr"]
        - comparison["attack1_legitimate_rr"]
    )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    attack1_metrics = attack_metrics(attack1)
    attack2_metrics = attack_metrics(attack2)

    rank1_converted = comparison.loc[
        comparison["rank1_failure_converted"],
        "query_id"
    ].tolist()

    top5_converted = comparison.loc[
        comparison["top5_failure_converted"],
        "query_id"
    ].tolist()

    beat_converted = comparison.loc[
        comparison["beat_failure_converted"],
        "query_id"
    ].tolist()

    legitimate_rank_worsened = comparison.loc[
        comparison["legitimate_rank_change"] > 0,
        "query_id"
    ].tolist()

    legitimate_rank_same = comparison.loc[
        comparison["legitimate_rank_change"] == 0,
        "query_id"
    ].tolist()

    legitimate_rank_improved = comparison.loc[
        comparison["legitimate_rank_change"] < 0,
        "query_id"
    ].tolist()

    summary = {

        "phase": "Phase 9A",

        "comparison":
            "Semantic Mimicry vs Optimization-Based Embedding",

        "queries": int(len(comparison)),

        "attack1": {
            "name": "Semantic Mimicry",
            **attack1_metrics,
        },

        "attack2": {
            "name": "Optimization-Based Embedding",
            **attack2_metrics,
        },

        "attack2_minus_attack1": {

            "poison_top5_asr_change":
                attack2_metrics["poison_top5_asr"]
                - attack1_metrics["poison_top5_asr"],

            "poison_rank1_asr_change":
                attack2_metrics["poison_rank1_asr"]
                - attack1_metrics["poison_rank1_asr"],

            "poison_beats_legitimate_rate_change":
                attack2_metrics[
                    "poison_beats_legitimate_rate"
                ]
                - attack1_metrics[
                    "poison_beats_legitimate_rate"
                ],

            "legitimate_top5_recall_change":
                attack2_metrics["legitimate_top5_recall"]
                - attack1_metrics[
                    "legitimate_top5_recall"
                ],

            "legitimate_mrr_change":
                attack2_metrics["legitimate_mrr"]
                - attack1_metrics["legitimate_mrr"],
        },

        "query_level_transitions": {

            "rank1_failures_converted_count":
                len(rank1_converted),

            "rank1_failures_converted_queries":
                rank1_converted,

            "top5_failures_converted_count":
                len(top5_converted),

            "top5_failures_converted_queries":
                top5_converted,

            "beat_failures_converted_count":
                len(beat_converted),

            "beat_failures_converted_queries":
                beat_converted,
        },

        "legitimate_document_rank_effect": {

            "rank_worsened_count":
                len(legitimate_rank_worsened),

            "rank_worsened_queries":
                legitimate_rank_worsened,

            "rank_unchanged_count":
                len(legitimate_rank_same),

            "rank_unchanged_queries":
                legitimate_rank_same,

            "rank_improved_count":
                len(legitimate_rank_improved),

            "rank_improved_queries":
                legitimate_rank_improved,
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            summary,
            fh,
            indent=2,
        )

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("ATTACK COMPARISON")
    print("=" * 72)

    print()
    print("Attack 1 - Semantic Mimicry")
    print(
        f"  Poison Top-5 ASR       : "
        f"{attack1_metrics['poison_top5_asr']:.4f}"
    )
    print(
        f"  Poison Rank-1 ASR      : "
        f"{attack1_metrics['poison_rank1_asr']:.4f}"
    )
    print(
        f"  Beats legitimate       : "
        f"{attack1_metrics['poison_beats_legitimate_rate']:.4f}"
    )
    print(
        f"  Legitimate Top-5       : "
        f"{attack1_metrics['legitimate_top5_recall']:.4f}"
    )
    print(
        f"  Legitimate MRR         : "
        f"{attack1_metrics['legitimate_mrr']:.6f}"
    )

    print()
    print("Attack 2 - Optimization-Based Embedding")
    print(
        f"  Poison Top-5 ASR       : "
        f"{attack2_metrics['poison_top5_asr']:.4f}"
    )
    print(
        f"  Poison Rank-1 ASR      : "
        f"{attack2_metrics['poison_rank1_asr']:.4f}"
    )
    print(
        f"  Beats legitimate       : "
        f"{attack2_metrics['poison_beats_legitimate_rate']:.4f}"
    )
    print(
        f"  Legitimate Top-5       : "
        f"{attack2_metrics['legitimate_top5_recall']:.4f}"
    )
    print(
        f"  Legitimate MRR         : "
        f"{attack2_metrics['legitimate_mrr']:.6f}"
    )

    print()
    print("Attack 2 improvement over Attack 1")

    print(
        "  Top-5 ASR change       : "
        f"{summary['attack2_minus_attack1']['poison_top5_asr_change']:+.4f}"
    )

    print(
        "  Rank-1 ASR change      : "
        f"{summary['attack2_minus_attack1']['poison_rank1_asr_change']:+.4f}"
    )

    print(
        "  Beats-legitimate change: "
        f"{summary['attack2_minus_attack1']['poison_beats_legitimate_rate_change']:+.4f}"
    )

    print(
        "  Legitimate MRR change  : "
        f"{summary['attack2_minus_attack1']['legitimate_mrr_change']:+.6f}"
    )

    print()
    print("Query-level transitions")

    print(
        f"  Rank-1 failures converted : "
        f"{len(rank1_converted)} "
        f"{rank1_converted}"
    )

    print(
        f"  Top-5 failures converted  : "
        f"{len(top5_converted)} "
        f"{top5_converted}"
    )

    print(
        f"  Beat failures converted   : "
        f"{len(beat_converted)} "
        f"{beat_converted}"
    )

    print()
    print("Legitimate-document rank effect")

    print(
        f"  Rank worsened : "
        f"{len(legitimate_rank_worsened)} "
        f"{legitimate_rank_worsened}"
    )

    print(
        f"  Rank unchanged: "
        f"{len(legitimate_rank_same)}"
    )

    print(
        f"  Rank improved : "
        f"{len(legitimate_rank_improved)} "
        f"{legitimate_rank_improved}"
    )

    print()
    print("=" * 72)
    print("PHASE 9A COMPLETE")
    print("=" * 72)

    print(f"Comparison CSV saved : {OUTPUT_CSV}")
    print(f"Summary JSON saved   : {OUTPUT_JSON}")


if __name__ == "__main__":
    main()