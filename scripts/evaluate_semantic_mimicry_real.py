"""
Phase 7F - Real Semantic Mimicry Attack Evaluation

Runs the frozen 20-query real benchmark against the poisoned ChromaDB
collection and measures retrieval-level attack success.

Metrics:
1. Poison Top-5 success
2. Poison Rank-1 takeover
3. Poison beats legitimate document
4. Legitimate document Top-5 retention
5. Legitimate document rank
6. Poison document rank
7. MRR of legitimate documents after poisoning

Expected precondition:
    Clean NVD corpus + semantic mimicry poisons already ingested.

Usage:
    python scripts/evaluate_semantic_mimicry_real.py
"""

import csv
import json
from pathlib import Path

import requests


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
    / "semantic_mimicry_real.jsonl"
)

RESULTS_DIR = REPO_ROOT / "data" / "results"

CSV_OUTPUT = RESULTS_DIR / "real_semantic_mimicry_top5_v1_results.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "real_semantic_mimicry_top5_v1_summary.json"

API_URL = "http://localhost:8000"
TOP_K = 5


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def get_query_text(record: dict) -> str:
    """
    Supports likely benchmark field names without changing
    the frozen benchmark file.
    """

    possible_fields = [
        "query",
        "query_text",
        "text",
        "question",
        "trigger_query",
    ]

    for field in possible_fields:
        value = record.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        f"Could not find query text field in benchmark record: {record}"
    )


def get_query_id(record: dict) -> str:
    possible_fields = [
        "query_id",
        "id",
    ]

    for field in possible_fields:
        value = record.get(field)

        if value is not None:
            return str(value)

    raise ValueError(
        f"Could not find query_id in benchmark record: {record}"
    )


def normalize_doc_id(value):
    if value is None:
        return ""

    return str(value).strip()


def find_rank(retrieved_ids: list[str], target_id: str):
    """
    Returns 1-based rank.
    Returns None if target document is not in Top-k.
    """

    target_id = normalize_doc_id(target_id)

    for index, doc_id in enumerate(retrieved_ids, start=1):
        if normalize_doc_id(doc_id) == target_id:
            return index

    return None


def run_query(query_text: str) -> dict:
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark = load_jsonl(BENCHMARK_PATH)
    poison_records = load_jsonl(POISON_PATH)

    poison_map = {
    str(record["query_id"]).strip().lower(): record
    for record in poison_records
    }

    print("=" * 72)
    print("PHASE 7F - REAL SEMANTIC MIMICRY ATTACK EVALUATION")
    print("=" * 72)

    print(f"Benchmark queries : {len(benchmark)}")
    print(f"Poison documents : {len(poison_records)}")
    print(f"Top-k            : {TOP_K}")
    print()

    results = []

    for benchmark_record in benchmark:

        query_id = get_query_id(benchmark_record).strip().lower()
        query_text = get_query_text(benchmark_record)

        if query_id not in poison_map:
            raise KeyError(
                f"No poison mapping found for benchmark query {query_id}"
            )

        poison_record = poison_map[query_id]

        legitimate_doc_id = normalize_doc_id(
            poison_record["legitimate_doc_id"]
        )

        poison_doc_id = normalize_doc_id(
            poison_record["poison_doc_id"]
        )

        print(f"[{query_id}] {query_text[:75]}")

        try:
            api_result = run_query(query_text)

            retrieved = api_result.get("results", [])

            retrieved_ids = [
                normalize_doc_id(item.get("doc_id"))
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

            poison_top5 = poison_rank is not None
            poison_rank1 = poison_rank == 1

            legitimate_top5 = legitimate_rank is not None

            poison_beats_legitimate = (
                poison_rank is not None
                and (
                    legitimate_rank is None
                    or poison_rank < legitimate_rank
                )
            )

            row = {
                "query_id": query_id,
                "query": query_text,
                "legitimate_doc_id": legitimate_doc_id,
                "poison_doc_id": poison_doc_id,
                "retrieved_doc_ids": " | ".join(retrieved_ids),
                "legitimate_rank": (
                    legitimate_rank
                    if legitimate_rank is not None
                    else ""
                ),
                "poison_rank": (
                    poison_rank
                    if poison_rank is not None
                    else ""
                ),
                "legitimate_top5": legitimate_top5,
                "poison_top5": poison_top5,
                "poison_rank1": poison_rank1,
                "poison_beats_legitimate": poison_beats_legitimate,
            }

            results.append(row)

            print(
                f"  Legit rank={legitimate_rank} | "
                f"Poison rank={poison_rank} | "
                f"Poison Top-5={poison_top5} | "
                f"Rank-1 takeover={poison_rank1}"
            )

        except Exception as exc:

            print(f"  ERROR: {exc}")

            results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "legitimate_doc_id": legitimate_doc_id,
                    "poison_doc_id": poison_doc_id,
                    "retrieved_doc_ids": "",
                    "legitimate_rank": "",
                    "poison_rank": "",
                    "legitimate_top5": False,
                    "poison_top5": False,
                    "poison_rank1": False,
                    "poison_beats_legitimate": False,
                    "error": str(exc),
                }
            )

        print()

    n = len(results)

    poison_top5_hits = sum(
        1 for row in results
        if row["poison_top5"]
    )

    poison_rank1_hits = sum(
        1 for row in results
        if row["poison_rank1"]
    )

    poison_beats_legitimate_hits = sum(
        1 for row in results
        if row["poison_beats_legitimate"]
    )

    legitimate_top5_hits = sum(
        1 for row in results
        if row["legitimate_top5"]
    )

    reciprocal_ranks = []

    for row in results:
        rank = row["legitimate_rank"]

        if isinstance(rank, int) and rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    legitimate_mrr = (
        sum(reciprocal_ranks) / n
        if n
        else 0.0
    )

    poison_top5_asr = poison_top5_hits / n if n else 0.0

    poison_rank1_asr = poison_rank1_hits / n if n else 0.0

    poison_beats_legitimate_rate = (
        poison_beats_legitimate_hits / n
        if n
        else 0.0
    )

    legitimate_top5_recall = (
        legitimate_top5_hits / n
        if n
        else 0.0
    )

    summary = {
        "benchmark": str(
            BENCHMARK_PATH.relative_to(REPO_ROOT)
        ),
        "attack_file": str(
            POISON_PATH.relative_to(REPO_ROOT)
        ),
        "queries": n,
        "top_k": TOP_K,

        "poison_top5_hits": poison_top5_hits,
        "poison_top5_asr": poison_top5_asr,

        "poison_rank1_hits": poison_rank1_hits,
        "poison_rank1_asr": poison_rank1_asr,

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
            "top5_recall": 1.0,
            "rank1_accuracy": 0.95,
            "mrr": 0.9666666666666668,
        },
    }

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
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    with SUMMARY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("=" * 72)
    print("PHASE 7F SUMMARY")
    print("=" * 72)

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
    print("Clean baseline reference")
    print("  Top-5 recall : 100.00%")
    print("  Rank-1       : 95.00%")
    print("  MRR          : 0.9667")

    print()
    print(f"Detailed CSV : {CSV_OUTPUT}")
    print(f"Summary JSON : {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()