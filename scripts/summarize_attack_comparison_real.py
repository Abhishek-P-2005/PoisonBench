import json
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results"

COMPARISON_CSV = (
    RESULTS_DIR / "attack1_vs_attack2_comparison_v1.csv"
)

COMPARISON_JSON = (
    RESULTS_DIR / "attack1_vs_attack2_comparison_v1_summary.json"
)

OUTPUT_CSV = (
    RESULTS_DIR / "attack1_vs_attack2_report_table_v1.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR / "attack1_vs_attack2_report_findings_v1.json"
)


# ============================================================
# Helpers
# ============================================================

def percentage(value):
    return round(float(value) * 100.0, 2)


def validate_inputs(df, summary):

    if len(df) != 20:
        raise ValueError(
            f"Expected 20 comparison rows, found {len(df)}."
        )

    if df["query_id"].duplicated().any():
        raise ValueError(
            "Duplicate query IDs found in comparison CSV."
        )

    if summary.get("queries") != 20:
        raise ValueError(
            "Comparison summary does not contain 20 queries."
        )

    required_columns = {
        "query_id",
        "attack1_legitimate_rank",
        "attack2_legitimate_rank",
        "legitimate_rank_change",
        "attack1_poison_rank",
        "attack2_poison_rank",
        "poison_rank_improvement",
        "attack1_poison_top5",
        "attack2_poison_top5",
        "top5_failure_converted",
        "attack1_poison_rank1",
        "attack2_poison_rank1",
        "rank1_failure_converted",
        "attack1_beats_legitimate",
        "attack2_beats_legitimate",
        "beat_failure_converted",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing comparison columns: {sorted(missing)}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("PHASE 9B - FORMAL ATTACK COMPARISON SUMMARY")
    print("=" * 72)

    print(f"Comparison CSV  : {COMPARISON_CSV}")
    print(f"Comparison JSON : {COMPARISON_JSON}")
    print()

    comparison = pd.read_csv(COMPARISON_CSV)

    with open(
        COMPARISON_JSON,
        "r",
        encoding="utf-8",
    ) as fh:
        summary = json.load(fh)

    validate_inputs(comparison, summary)

    print(f"Comparison rows loaded: {len(comparison)}")

    attack1 = summary["attack1"]
    attack2 = summary["attack2"]

    # ========================================================
    # Report-ready aggregate table
    # ========================================================

    report_rows = [
        {
            "metric": "Poison Top-5 ASR (%)",
            "semantic_mimicry":
                percentage(attack1["poison_top5_asr"]),
            "optimized_embedding":
                percentage(attack2["poison_top5_asr"]),
            "absolute_change":
                percentage(
                    attack2["poison_top5_asr"]
                    - attack1["poison_top5_asr"]
                ),
        },
        {
            "metric": "Poison Rank-1 ASR (%)",
            "semantic_mimicry":
                percentage(attack1["poison_rank1_asr"]),
            "optimized_embedding":
                percentage(attack2["poison_rank1_asr"]),
            "absolute_change":
                percentage(
                    attack2["poison_rank1_asr"]
                    - attack1["poison_rank1_asr"]
                ),
        },
        {
            "metric": "Poison Beats Legitimate (%)",
            "semantic_mimicry":
                percentage(
                    attack1["poison_beats_legitimate_rate"]
                ),
            "optimized_embedding":
                percentage(
                    attack2["poison_beats_legitimate_rate"]
                ),
            "absolute_change":
                percentage(
                    attack2["poison_beats_legitimate_rate"]
                    - attack1["poison_beats_legitimate_rate"]
                ),
        },
        {
            "metric": "Legitimate Top-5 Recall (%)",
            "semantic_mimicry":
                percentage(
                    attack1["legitimate_top5_recall"]
                ),
            "optimized_embedding":
                percentage(
                    attack2["legitimate_top5_recall"]
                ),
            "absolute_change":
                percentage(
                    attack2["legitimate_top5_recall"]
                    - attack1["legitimate_top5_recall"]
                ),
        },
        {
            "metric": "Legitimate MRR",
            "semantic_mimicry":
                round(attack1["legitimate_mrr"], 6),
            "optimized_embedding":
                round(attack2["legitimate_mrr"], 6),
            "absolute_change":
                round(
                    attack2["legitimate_mrr"]
                    - attack1["legitimate_mrr"],
                    6,
                ),
        },
        {
            "metric": "Mean Legitimate Rank",
            "semantic_mimicry":
                round(attack1["mean_legitimate_rank"], 4),
            "optimized_embedding":
                round(attack2["mean_legitimate_rank"], 4),
            "absolute_change":
                round(
                    attack2["mean_legitimate_rank"]
                    - attack1["mean_legitimate_rank"],
                    4,
                ),
        },
        {
            "metric": "Mean Poison Rank",
            "semantic_mimicry":
                round(attack1["mean_poison_rank"], 4),
            "optimized_embedding":
                round(attack2["mean_poison_rank"], 4),
            "absolute_change":
                round(
                    attack2["mean_poison_rank"]
                    - attack1["mean_poison_rank"],
                    4,
                ),
        },
    ]

    report_table = pd.DataFrame(report_rows)

    # ========================================================
    # Query-level findings
    # ========================================================

    rank1_converted = comparison.loc[
        comparison["rank1_failure_converted"],
        "query_id",
    ].tolist()

    top5_converted = comparison.loc[
        comparison["top5_failure_converted"],
        "query_id",
    ].tolist()

    beat_converted = comparison.loc[
        comparison["beat_failure_converted"],
        "query_id",
    ].tolist()

    legitimate_worsened = comparison.loc[
        comparison["legitimate_rank_change"] > 0,
        "query_id",
    ].tolist()

    legitimate_unchanged = comparison.loc[
        comparison["legitimate_rank_change"] == 0,
        "query_id",
    ].tolist()

    legitimate_improved = comparison.loc[
        comparison["legitimate_rank_change"] < 0,
        "query_id",
    ].tolist()

    poison_rank_better = comparison.loc[
        comparison["poison_rank_improvement"] > 0,
        "query_id",
    ].tolist()

    poison_rank_same = comparison.loc[
        comparison["poison_rank_improvement"] == 0,
        "query_id",
    ].tolist()

    poison_rank_worse = comparison.loc[
        comparison["poison_rank_improvement"] < 0,
        "query_id",
    ].tolist()

    findings = {
        "phase": "Phase 9B",
        "experiment":
            "Attack 1 vs Attack 2 comparative analysis",

        "benchmark_queries": 20,

        "aggregate_findings": {
            "semantic_mimicry_top5_asr":
                attack1["poison_top5_asr"],

            "optimized_embedding_top5_asr":
                attack2["poison_top5_asr"],

            "top5_asr_gain":
                attack2["poison_top5_asr"]
                - attack1["poison_top5_asr"],

            "semantic_mimicry_rank1_asr":
                attack1["poison_rank1_asr"],

            "optimized_embedding_rank1_asr":
                attack2["poison_rank1_asr"],

            "rank1_asr_gain":
                attack2["poison_rank1_asr"]
                - attack1["poison_rank1_asr"],

            "semantic_mimicry_beats_legitimate":
                attack1["poison_beats_legitimate_rate"],

            "optimized_embedding_beats_legitimate":
                attack2["poison_beats_legitimate_rate"],

            "beats_legitimate_gain":
                attack2["poison_beats_legitimate_rate"]
                - attack1["poison_beats_legitimate_rate"],

            "semantic_mimicry_legitimate_mrr":
                attack1["legitimate_mrr"],

            "optimized_embedding_legitimate_mrr":
                attack2["legitimate_mrr"],

            "legitimate_mrr_change":
                attack2["legitimate_mrr"]
                - attack1["legitimate_mrr"],
        },

        "query_level_findings": {
            "rank1_failures_converted": rank1_converted,
            "rank1_failures_converted_count":
                len(rank1_converted),

            "top5_failures_converted": top5_converted,
            "top5_failures_converted_count":
                len(top5_converted),

            "beat_failures_converted": beat_converted,
            "beat_failures_converted_count":
                len(beat_converted),

            "legitimate_rank_worsened":
                legitimate_worsened,

            "legitimate_rank_worsened_count":
                len(legitimate_worsened),

            "legitimate_rank_unchanged":
                legitimate_unchanged,

            "legitimate_rank_unchanged_count":
                len(legitimate_unchanged),

            "legitimate_rank_improved":
                legitimate_improved,

            "legitimate_rank_improved_count":
                len(legitimate_improved),

            "poison_rank_improved":
                poison_rank_better,

            "poison_rank_improved_count":
                len(poison_rank_better),

            "poison_rank_unchanged":
                poison_rank_same,

            "poison_rank_unchanged_count":
                len(poison_rank_same),

            "poison_rank_worsened":
                poison_rank_worse,

            "poison_rank_worsened_count":
                len(poison_rank_worse),
        },

        "interpretation": {
            "primary_result":
                "Optimization-based embedding poisoning "
                "outperformed semantic mimicry on retrieval "
                "attack success.",

            "rank1_result":
                "All Semantic Mimicry Rank-1 failures were "
                "converted into Rank-1 poison takeovers by "
                "the optimized attack.",

            "top5_result":
                "The optimized attack achieved poison "
                "retrieval in the Top-5 for every frozen "
                "benchmark query.",

            "legitimate_result":
                "Legitimate Top-5 recall remained unchanged "
                "while legitimate reciprocal-rank performance "
                "decreased, indicating stronger ranking "
                "displacement rather than complete removal "
                "of legitimate evidence.",
        },
    }

    # ========================================================
    # Save artifacts
    # ========================================================

    report_table.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            findings,
            fh,
            indent=2,
        )

    # ========================================================
    # Terminal output
    # ========================================================

    print()
    print("=" * 72)
    print("REPORT-READY COMPARISON")
    print("=" * 72)

    print()
    print(
        report_table.to_string(index=False)
    )

    print()
    print("=" * 72)
    print("QUERY-LEVEL FINDINGS")
    print("=" * 72)

    print(
        f"Rank-1 failures converted : "
        f"{len(rank1_converted)} "
        f"{rank1_converted}"
    )

    print(
        f"Top-5 failures converted  : "
        f"{len(top5_converted)} "
        f"{top5_converted}"
    )

    print(
        f"Beat failures converted   : "
        f"{len(beat_converted)} "
        f"{beat_converted}"
    )

    print(
        f"Legitimate rank worsened  : "
        f"{len(legitimate_worsened)} "
        f"{legitimate_worsened}"
    )

    print(
        f"Legitimate rank unchanged : "
        f"{len(legitimate_unchanged)}"
    )

    print(
        f"Legitimate rank improved  : "
        f"{len(legitimate_improved)} "
        f"{legitimate_improved}"
    )

    print()
    print("=" * 72)
    print("PHASE 9B COMPLETE")
    print("=" * 72)

    print(f"Report table saved : {OUTPUT_CSV}")
    print(f"Findings saved     : {OUTPUT_JSON}")


if __name__ == "__main__":
    main()