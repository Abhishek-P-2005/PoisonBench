import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

POISON_FILE = (
    REPO_ROOT
    / "data"
    / "poisoned"
    / "optimized_embedding_attack_real.jsonl"
)

OPTIMIZATION_STATS = (
    REPO_ROOT
    / "data"
    / "poisoned"
    / "optimized_embedding_attack_stats.jsonl"
)

ATTACK2_SUMMARY = (
    REPO_ROOT
    / "data"
    / "results"
    / "real_optimized_embedding_top5_v1_summary.json"
)

PHASE9_COMPARISON = (
    REPO_ROOT
    / "data"
    / "results"
    / "attack1_vs_attack2_comparison_v1.csv"
)

PHASE9_SUMMARY = (
    REPO_ROOT
    / "data"
    / "results"
    / "attack1_vs_attack2_comparison_v1_summary.json"
)

PHASE9_REPORT_TABLE = (
    REPO_ROOT
    / "data"
    / "results"
    / "attack1_vs_attack2_report_table_v1.csv"
)

PHASE9_FINDINGS = (
    REPO_ROOT
    / "data"
    / "results"
    / "attack1_vs_attack2_report_findings_v1.json"
)

MANIFEST_FILE = (
    REPO_ROOT
    / "data"
    / "results"
    / "optimized_embedding_attack_final_manifest_v1.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {exc}"
                )

    return rows


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def main():

    print("=" * 76)
    print("PHASE 10 - OPTIMIZED EMBEDDING ATTACK FINAL VALIDATION")
    print("=" * 76)

    required_files = [
        POISON_FILE,
        OPTIMIZATION_STATS,
        ATTACK2_SUMMARY,
        PHASE9_COMPARISON,
        PHASE9_SUMMARY,
        PHASE9_REPORT_TABLE,
        PHASE9_FINDINGS,
    ]

    print()
    print("Checking required artifacts...")

    for path in required_files:
        check(
            path.exists(),
            f"Artifact exists: {path.name}",
        )

    # --------------------------------------------------------
    # Validate optimized poison corpus
    # --------------------------------------------------------

    print()
    print("Validating optimized poison corpus...")

    poisons = load_jsonl(POISON_FILE)

    check(
        len(poisons) == 20,
        "Optimized poison corpus contains exactly 20 records",
    )

    query_ids = [row["query_id"] for row in poisons]
    poison_ids = [row["poison_doc_id"] for row in poisons]

    check(
        len(set(query_ids)) == 20,
        "All 20 query IDs are unique",
    )

    check(
        len(set(poison_ids)) == 20,
        "All 20 poison document IDs are unique",
    )

    expected_queries = {
        f"q{i:02d}"
        for i in range(1, 21)
    }

    check(
        set(query_ids) == expected_queries,
        "Poison corpus covers exactly q01-q20",
    )

    check(
        all(
            row.get("attack_type")
            == "optimization_embedding"
            for row in poisons
        ),
        "All records use attack_type=optimization_embedding",
    )

    check(
        all(
            row.get("is_poisoned") is True
            for row in poisons
        ),
        "All records are marked is_poisoned=true",
    )

    check(
        all(
            isinstance(row.get("poison_text"), str)
            and row["poison_text"].strip()
            for row in poisons
        ),
        "All poison records contain non-empty poison text",
    )

    # --------------------------------------------------------
    # Validate optimization effect
    # --------------------------------------------------------

    print()
    print("Validating optimization effect...")

    seed_scores = [
        float(row["seed_similarity"])
        for row in poisons
    ]

    optimized_scores = [
        float(row["optimized_similarity"])
        for row in poisons
    ]

    improvements = [
        optimized - seed
        for seed, optimized
        in zip(seed_scores, optimized_scores)
    ]

    check(
        all(
            optimized >= seed
            for seed, optimized
            in zip(seed_scores, optimized_scores)
        ),
        "No optimized poison is worse than its seed",
    )

    improved_count = sum(
        improvement > 0
        for improvement in improvements
    )

    check(
        improved_count == 20,
        "All 20 optimized poisons improved over their seeds",
    )

    mean_seed = sum(seed_scores) / len(seed_scores)

    mean_optimized = (
        sum(optimized_scores)
        / len(optimized_scores)
    )

    mean_improvement = (
        sum(improvements)
        / len(improvements)
    )

    check(
        mean_optimized > mean_seed,
        "Mean optimized similarity exceeds mean seed similarity",
    )

    # --------------------------------------------------------
    # Validate Phase 8 frozen evaluation
    # --------------------------------------------------------

    print()
    print("Validating frozen Attack 2 evaluation...")

    attack2 = load_json(ATTACK2_SUMMARY)

    check(
        attack2["queries"] == 20,
        "Phase 8 evaluated exactly 20 queries",
    )

    check(
        attack2["top_k"] == 5,
        "Phase 8 used frozen Top-k=5",
    )

    check(
        attack2["api_errors"] == 0,
        "Phase 8 contains zero API errors",
    )

    check(
        attack2["poison_top5_hits"] == 20,
        "Poison retrieved in Top-5 for 20/20 queries",
    )

    check(
        attack2["poison_rank1_hits"] == 20,
        "Poison achieved Rank-1 for 20/20 queries",
    )

    check(
        attack2["poison_beats_legitimate_hits"] == 20,
        "Poison beat legitimate document for 20/20 queries",
    )

    check(
        attack2["legitimate_top5_hits"] == 20,
        "Legitimate document retained in Top-5 for 20/20 queries",
    )

    # --------------------------------------------------------
    # Build final manifest
    # --------------------------------------------------------

    manifest = {
        "attack": "Optimization-Based Embedding Attack",
        "attack_type": "optimization_embedding",
        "status": "COMPLETE_AND_FROZEN",

        "benchmark": {
            "queries": 20,
            "top_k": 5,
        },

        "optimization": {
            "poison_documents": len(poisons),
            "improved_documents": improved_count,
            "mean_seed_similarity": mean_seed,
            "mean_optimized_similarity": mean_optimized,
            "mean_absolute_improvement": mean_improvement,
        },

        "retrieval_evaluation": {
            "api_errors": attack2["api_errors"],
            "poison_top5_hits":
                attack2["poison_top5_hits"],
            "poison_top5_asr":
                attack2["poison_top5_asr"],
            "poison_rank1_hits":
                attack2["poison_rank1_hits"],
            "poison_rank1_asr":
                attack2["poison_rank1_asr"],
            "poison_beats_legitimate_hits":
                attack2["poison_beats_legitimate_hits"],
            "poison_beats_legitimate_rate":
                attack2["poison_beats_legitimate_rate"],
            "legitimate_top5_recall":
                attack2["legitimate_top5_recall"],
            "legitimate_mrr_after_poisoning":
                attack2["legitimate_mrr_after_poisoning"],
        },

        "clean_reference":
            attack2["clean_reference"],

        "artifacts": {
            "optimized_poison_corpus":
                str(POISON_FILE.relative_to(REPO_ROOT)),

            "optimization_stats":
                str(
                    OPTIMIZATION_STATS.relative_to(REPO_ROOT)
                ),

            "attack2_summary":
                str(
                    ATTACK2_SUMMARY.relative_to(REPO_ROOT)
                ),

            "attack_comparison":
                str(
                    PHASE9_COMPARISON.relative_to(REPO_ROOT)
                ),

            "attack_comparison_summary":
                str(
                    PHASE9_SUMMARY.relative_to(REPO_ROOT)
                ),

            "report_table":
                str(
                    PHASE9_REPORT_TABLE.relative_to(REPO_ROOT)
                ),

            "report_findings":
                str(
                    PHASE9_FINDINGS.relative_to(REPO_ROOT)
                ),
        },
    }

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            manifest,
            fh,
            indent=2,
        )

    print()
    print("=" * 76)
    print("FINAL OPTIMIZATION STATISTICS")
    print("=" * 76)

    print(
        f"Mean seed similarity      : {mean_seed:.6f}"
    )

    print(
        f"Mean optimized similarity : {mean_optimized:.6f}"
    )

    print(
        f"Mean improvement          : {mean_improvement:+.6f}"
    )

    print()
    print("Retrieval evaluation:")
    print(
        f"  Poison Top-5 ASR        : "
        f"{attack2['poison_top5_asr']:.4f}"
    )
    print(
        f"  Poison Rank-1 ASR       : "
        f"{attack2['poison_rank1_asr']:.4f}"
    )
    print(
        f"  Beats legitimate        : "
        f"{attack2['poison_beats_legitimate_rate']:.4f}"
    )
    print(
        f"  Legitimate Top-5 recall : "
        f"{attack2['legitimate_top5_recall']:.4f}"
    )
    print(
        f"  Legitimate MRR          : "
        f"{attack2['legitimate_mrr_after_poisoning']:.6f}"
    )

    print()
    print("=" * 76)
    print("PHASE 10 COMPLETE - ATTACK 2 FROZEN")
    print("=" * 76)

    print(
        f"Final manifest saved: {MANIFEST_FILE}"
    )


if __name__ == "__main__":
    main()