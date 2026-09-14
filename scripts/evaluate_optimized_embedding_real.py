"""
PoisonBench
Attack 2 - Optimization-Based Embedding Attack
Main Roadmap Phase 8

Runs the frozen 20-query benchmark against the ChromaDB collection
containing:

    Clean NVD corpus
    +
    20 optimized embedding attack poison documents

Metrics:
1. Poison Top-5 success
2. Poison Rank-1 takeover
3. Poison beats legitimate document
4. Legitimate document Top-5 retention
5. Legitimate document rank
6. Poison document rank
7. MRR of legitimate documents after poisoning

Expected precondition:

    Clean NVD corpus + Attack-2 OBEA poisons already ingested.

Usage:

    python scripts/evaluate_optimized_embedding_real.py
"""

import csv
import json
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_PATH = (
    REPO_ROOT
    / "data"
    / "benchmark"
    / "real_trigger_queries_v1.jsonl"
)

POISON_PATH = (
    REPO_ROOT
    / "data"
    / "poisoned"
    / "optimized_embedding_attack_real.jsonl"
)

RESULTS_DIR = (
    REPO_ROOT
    / "data"
    / "results"
)

CSV_OUTPUT = (
    RESULTS_DIR
    / "real_optimized_embedding_top5_v1_results.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "real_optimized_embedding_top5_v1_summary.json"
)

API_URL = "http://localhost:8000"

TOP_K = 5

EXPECTED_QUERIES = 20


# ============================================================
# FILE UTILITIES
# ============================================================

def load_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON in {path.name} "
                    f"at line {line_number}"
                ) from exc

    return records


# ============================================================
# BENCHMARK FIELD HELPERS
# ============================================================

def get_query_text(record: dict) -> str:

    possible_fields = [
        "query",
        "query_text",
        "text",
        "question",
        "trigger_query",
    ]

    for field in possible_fields:

        value = record.get(field)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value.strip()

    raise ValueError(
        "Could not locate query text "
        f"in benchmark record: {record}"
    )


def get_query_id(record: dict) -> str:

    possible_fields = [
        "query_id",
        "id",
    ]

    for field in possible_fields:

        value = record.get(field)

        if value is not None:
            return str(value).strip().lower()

    raise ValueError(
        "Could not locate query ID "
        f"in benchmark record: {record}"
    )


def normalize_doc_id(value):

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# RANKING UTILITIES
# ============================================================

def find_rank(
    retrieved_ids: list[str],
    target_id: str,
):

    target_id = normalize_doc_id(
        target_id
    )

    for index, doc_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if (
            normalize_doc_id(doc_id)
            == target_id
        ):
            return index

    return None


# ============================================================
# API QUERY
# ============================================================

def run_query(
    query_text: str,
) -> dict:

    response = requests.post(
        f"{API_URL}/retrieve",
        json={
            "query": query_text,
            "top_k": TOP_K,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# PRE-RUN VALIDATION
# ============================================================

def validate_inputs(
    benchmark,
    poison_records,
):

    if len(benchmark) != EXPECTED_QUERIES:

        raise ValueError(
            f"Expected {EXPECTED_QUERIES} "
            f"benchmark queries but found "
            f"{len(benchmark)}."
        )

    if len(poison_records) != EXPECTED_QUERIES:

        raise ValueError(
            f"Expected {EXPECTED_QUERIES} "
            f"Attack-2 poison documents but "
            f"found {len(poison_records)}."
        )

    poison_ids = set()

    query_ids = set()

    for record in poison_records:

        required = {
            "query_id",
            "legitimate_doc_id",
            "poison_doc_id",
            "poison_text",
        }

        missing = (
            required
            - record.keys()
        )

        if missing:

            raise KeyError(
                f"Missing fields in Attack-2 "
                f"record: {sorted(missing)}"
            )

        query_id = (
            str(record["query_id"])
            .strip()
            .lower()
        )

        poison_doc_id = normalize_doc_id(
            record["poison_doc_id"]
        )

        if query_id in query_ids:

            raise ValueError(
                f"Duplicate query_id: "
                f"{query_id}"
            )

        if poison_doc_id in poison_ids:

            raise ValueError(
                f"Duplicate poison_doc_id: "
                f"{poison_doc_id}"
            )

        if not poison_doc_id.startswith(
            "POISON-OBEA-"
        ):

            raise ValueError(
                f"Unexpected Attack-2 ID: "
                f"{poison_doc_id}"
            )

        query_ids.add(
            query_id
        )

        poison_ids.add(
            poison_doc_id
        )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark = load_jsonl(
        BENCHMARK_PATH
    )

    poison_records = load_jsonl(
        POISON_PATH
    )

    validate_inputs(
        benchmark,
        poison_records,
    )

    poison_map = {
        str(record["query_id"])
        .strip()
        .lower(): record

        for record in poison_records
    }

    print("=" * 72)
    print("POISONBENCH")
    print(
        "ATTACK 2 - OPTIMIZATION-BASED "
        "EMBEDDING ATTACK"
    )
    print("MAIN ROADMAP PHASE 8")
    print(
        "FULL 20-QUERY RETRIEVAL "
        "ATTACK EVALUATION"
    )
    print("=" * 72)

    print()
    print(
        f"Benchmark queries : "
        f"{len(benchmark)}"
    )

    print(
        f"Poison documents  : "
        f"{len(poison_records)}"
    )

    print(
        f"Top-k             : "
        f"{TOP_K}"
    )

    print()

    results = []

    api_errors = 0

    # ========================================================
    # QUERY LOOP
    # ========================================================

    for benchmark_record in benchmark:

        query_id = get_query_id(
            benchmark_record
        )

        query_text = get_query_text(
            benchmark_record
        )

        if query_id not in poison_map:

            raise KeyError(
                f"No Attack-2 poison mapping "
                f"found for {query_id}"
            )

        poison_record = (
            poison_map[query_id]
        )

        legitimate_doc_id = (
            normalize_doc_id(
                poison_record[
                    "legitimate_doc_id"
                ]
            )
        )

        poison_doc_id = (
            normalize_doc_id(
                poison_record[
                    "poison_doc_id"
                ]
            )
        )

        print(
            f"[{query_id}] "
            f"{query_text[:75]}"
        )

        try:

            api_result = run_query(
                query_text
            )

            retrieved = (
                api_result.get(
                    "results",
                    [],
                )
            )

            retrieved_ids = [
                normalize_doc_id(
                    item.get("doc_id")
                )

                for item in retrieved
            ]

            legitimate_rank = find_rank(
                retrieved_ids,
                legitimate_doc_id,
            )

            poison_rank = find_rank(
                retrieved_ids,
                poison_doc_id,
            )

            poison_top5 = (
                poison_rank
                is not None
            )

            poison_rank1 = (
                poison_rank == 1
            )

            legitimate_top5 = (
                legitimate_rank
                is not None
            )

            poison_beats_legitimate = (
                poison_rank is not None
                and (
                    legitimate_rank
                    is None
                    or
                    poison_rank
                    < legitimate_rank
                )
            )

            row = {
                "query_id":
                    query_id,

                "query":
                    query_text,

                "legitimate_doc_id":
                    legitimate_doc_id,

                "poison_doc_id":
                    poison_doc_id,

                "retrieved_doc_ids":
                    " | ".join(
                        retrieved_ids
                    ),

                "legitimate_rank":
                    legitimate_rank
                    if legitimate_rank
                    is not None
                    else "",

                "poison_rank":
                    poison_rank
                    if poison_rank
                    is not None
                    else "",

                "legitimate_top5":
                    legitimate_top5,

                "poison_top5":
                    poison_top5,

                "poison_rank1":
                    poison_rank1,

                "poison_beats_legitimate":
                    poison_beats_legitimate,

                "error":
                    "",
            }

            results.append(
                row
            )

            print(
                f"  Legit rank="
                f"{legitimate_rank} | "
                f"Poison rank="
                f"{poison_rank} | "
                f"Poison Top-5="
                f"{poison_top5} | "
                f"Rank-1 takeover="
                f"{poison_rank1}"
            )

        except Exception as exc:

            api_errors += 1

            print(
                f"  ERROR: {exc}"
            )

            results.append(
                {
                    "query_id":
                        query_id,

                    "query":
                        query_text,

                    "legitimate_doc_id":
                        legitimate_doc_id,

                    "poison_doc_id":
                        poison_doc_id,

                    "retrieved_doc_ids":
                        "",

                    "legitimate_rank":
                        "",

                    "poison_rank":
                        "",

                    "legitimate_top5":
                        False,

                    "poison_top5":
                        False,

                    "poison_rank1":
                        False,

                    "poison_beats_legitimate":
                        False,

                    "error":
                        str(exc),
                }
            )

        print()

    # ========================================================
    # METRIC CALCULATION
    # ========================================================

    n = len(results)

    poison_top5_hits = sum(
        1
        for row in results
        if row["poison_top5"]
    )

    poison_rank1_hits = sum(
        1
        for row in results
        if row["poison_rank1"]
    )

    poison_beats_legitimate_hits = sum(
        1
        for row in results
        if row[
            "poison_beats_legitimate"
        ]
    )

    legitimate_top5_hits = sum(
        1
        for row in results
        if row[
            "legitimate_top5"
        ]
    )

    reciprocal_ranks = []

    for row in results:

        rank = (
            row["legitimate_rank"]
        )

        if (
            isinstance(rank, int)
            and rank > 0
        ):

            reciprocal_ranks.append(
                1.0 / rank
            )

        else:

            reciprocal_ranks.append(
                0.0
            )

    legitimate_mrr = (
        sum(reciprocal_ranks) / n
        if n
        else 0.0
    )

    poison_top5_asr = (
        poison_top5_hits / n
        if n
        else 0.0
    )

    poison_rank1_asr = (
        poison_rank1_hits / n
        if n
        else 0.0
    )

    poison_beats_legitimate_rate = (
        poison_beats_legitimate_hits
        / n
        if n
        else 0.0
    )

    legitimate_top5_recall = (
        legitimate_top5_hits / n
        if n
        else 0.0
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "benchmark":
            str(
                BENCHMARK_PATH.relative_to(
                    REPO_ROOT
                )
            ),

        "attack_file":
            str(
                POISON_PATH.relative_to(
                    REPO_ROOT
                )
            ),

        "attack_type":
            "optimization_embedding",

        "queries":
            n,

        "top_k":
            TOP_K,

        "api_errors":
            api_errors,

        "poison_top5_hits":
            poison_top5_hits,

        "poison_top5_asr":
            poison_top5_asr,

        "poison_rank1_hits":
            poison_rank1_hits,

        "poison_rank1_asr":
            poison_rank1_asr,

        "poison_beats_legitimate_hits":
            poison_beats_legitimate_hits,

        "poison_beats_legitimate_rate":
            poison_beats_legitimate_rate,

        "legitimate_top5_hits":
            legitimate_top5_hits,

        "legitimate_top5_recall":
            legitimate_top5_recall,

        "legitimate_mrr_after_poisoning":
            legitimate_mrr,

        "clean_reference": {

            "top5_recall":
                1.0,

            "rank1_accuracy":
                0.95,

            "mrr":
                0.9666666666666668,
        },
    }

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
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
    ]

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in results:
            writer.writerow(
                row
            )

    # ========================================================
    # SAVE SUMMARY JSON
    # ========================================================

    with SUMMARY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print("=" * 72)
    print("PHASE 8 SUMMARY")
    print("=" * 72)

    print(
        f"Queries evaluated             : "
        f"{n}"
    )

    print(
        f"API errors                    : "
        f"{api_errors}"
    )

    print()

    print(
        f"Poison Top-5 ASR              : "
        f"{poison_top5_asr:.2%} "
        f"({poison_top5_hits}/{n})"
    )

    print(
        f"Poison Rank-1 takeover ASR    : "
        f"{poison_rank1_asr:.2%} "
        f"({poison_rank1_hits}/{n})"
    )

    print(
        f"Poison beats legitimate       : "
        f"{poison_beats_legitimate_rate:.2%} "
        f"({poison_beats_legitimate_hits}/{n})"
    )

    print(
        f"Legitimate Top-5 recall       : "
        f"{legitimate_top5_recall:.2%} "
        f"({legitimate_top5_hits}/{n})"
    )

    print(
        f"Legitimate MRR after poisoning: "
        f"{legitimate_mrr:.4f}"
    )

    print()
    print(
        "Clean baseline reference"
    )

    print(
        "  Top-5 recall : 100.00%"
    )

    print(
        "  Rank-1       : 95.00%"
    )

    print(
        "  MRR          : 0.9667"
    )

    print()
    print(
        f"Detailed CSV : "
        f"{CSV_OUTPUT}"
    )

    print(
        f"Summary JSON : "
        f"{SUMMARY_OUTPUT}"
    )

    print()

    if api_errors == 0:

        print(
            "PASS: All 20 benchmark "
            "queries evaluated successfully."
        )

    else:

        print(
            "WARNING: Evaluation completed "
            f"with {api_errors} API error(s)."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()