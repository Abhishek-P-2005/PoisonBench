"""
Phase 2 - Semantic Mimicry Retrieval-Depth Ablation

Evaluates the frozen 20-query real benchmark at multiple retrieval depths:
Top-1, Top-3, Top-5, and Top-10.

The purpose is to determine whether the semantic-mimicry poisoning attack
remains effective when retrieval depth changes.

Outputs:
1. Per-query CSV
2. Aggregated JSON summary

Precondition:
    Clean NVD corpus + semantic mimicry poisons are already ingested.
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

CSV_OUTPUT = (
    RESULTS_DIR
    / "semantic_mimicry_ablation_results.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "semantic_mimicry_ablation_summary.json"
)

API_URL = "http://localhost:8000"

TOP_K_VALUES = [1, 3, 5, 10]


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


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

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        f"Could not find query text in record: {record}"
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
        f"Could not find query ID in record: {record}"
    )


def normalize_doc_id(value):
    if value is None:
        return ""

    return str(value).strip()


def find_rank(retrieved_ids, target_id):

    target_id = normalize_doc_id(target_id)

    for index, doc_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if normalize_doc_id(doc_id) == target_id:
            return index

    return None


def find_similarity(retrieved, target_id):

    target_id = normalize_doc_id(target_id)

    for item in retrieved:

        doc_id = normalize_doc_id(
            item.get("doc_id")
        )

        if doc_id == target_id:

            score = item.get(
                "similarity_score"
            )

            if score is None:
                return None

            return float(score)

    return None


def run_query(query_text: str, top_k: int):

    response = requests.post(
        f"{API_URL}/retrieve",
        json={
            "query": query_text,
            "top_k": top_k,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


def evaluate_at_k(
    benchmark,
    poison_map,
    top_k,
):

    print()
    print("=" * 72)
    print(f"EVALUATING TOP-{top_k}")
    print("=" * 72)

    results = []

    for benchmark_record in benchmark:

        query_id = (
            get_query_id(benchmark_record)
            .strip()
            .lower()
        )

        query_text = get_query_text(
            benchmark_record
        )

        if query_id not in poison_map:
            raise KeyError(
                f"No poison mapping for {query_id}"
            )

        poison_record = poison_map[query_id]

        legitimate_doc_id = normalize_doc_id(
            poison_record[
                "legitimate_doc_id"
            ]
        )

        poison_doc_id = normalize_doc_id(
            poison_record[
                "poison_doc_id"
            ]
        )

        print(
            f"[{query_id}] "
            f"{query_text[:70]}"
        )

        try:

            api_result = run_query(
                query_text,
                top_k,
            )

            retrieved = api_result.get(
                "results",
                [],
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

            legitimate_similarity = (
                find_similarity(
                    retrieved,
                    legitimate_doc_id,
                )
            )

            poison_similarity = (
                find_similarity(
                    retrieved,
                    poison_doc_id,
                )
            )

            poison_hit = (
                poison_rank is not None
            )

            legitimate_hit = (
                legitimate_rank is not None
            )

            poison_rank1 = (
                poison_rank == 1
            )

            poison_beats_legitimate = (
                poison_rank is not None
                and (
                    legitimate_rank is None
                    or
                    poison_rank
                    < legitimate_rank
                )
            )

            similarity_advantage = None

            if (
                poison_similarity is not None
                and
                legitimate_similarity
                is not None
            ):
                similarity_advantage = (
                    poison_similarity
                    - legitimate_similarity
                )

            row = {
                "top_k": top_k,
                "query_id": query_id,
                "query": query_text,
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
                "legitimate_similarity":
                    legitimate_similarity
                    if legitimate_similarity
                    is not None
                    else "",
                "poison_similarity":
                    poison_similarity
                    if poison_similarity
                    is not None
                    else "",
                "similarity_advantage":
                    similarity_advantage
                    if similarity_advantage
                    is not None
                    else "",
                "legitimate_hit":
                    legitimate_hit,
                "poison_hit":
                    poison_hit,
                "poison_rank1":
                    poison_rank1,
                "poison_beats_legitimate":
                    poison_beats_legitimate,
                "error": "",
            }

            results.append(row)

            print(
                f"  Legit rank={legitimate_rank} | "
                f"Poison rank={poison_rank} | "
                f"Poison hit={poison_hit} | "
                f"Rank-1={poison_rank1}"
            )

        except Exception as exc:

            print(f"  ERROR: {exc}")

            results.append(
                {
                    "top_k": top_k,
                    "query_id": query_id,
                    "query": query_text,
                    "legitimate_doc_id":
                        legitimate_doc_id,
                    "poison_doc_id":
                        poison_doc_id,
                    "retrieved_doc_ids": "",
                    "legitimate_rank": "",
                    "poison_rank": "",
                    "legitimate_similarity": "",
                    "poison_similarity": "",
                    "similarity_advantage": "",
                    "legitimate_hit": False,
                    "poison_hit": False,
                    "poison_rank1": False,
                    "poison_beats_legitimate":
                        False,
                    "error": str(exc),
                }
            )

    return results


def calculate_summary(
    results,
    top_k,
):

    n = len(results)

    poison_hits = sum(
        row["poison_hit"]
        for row in results
    )

    rank1_hits = sum(
        row["poison_rank1"]
        for row in results
    )

    beats_hits = sum(
        row["poison_beats_legitimate"]
        for row in results
    )

    legitimate_hits = sum(
        row["legitimate_hit"]
        for row in results
    )

    reciprocal_ranks = []

    for row in results:

        rank = row["legitimate_rank"]

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

    poison_asr = (
        poison_hits / n
        if n
        else 0.0
    )

    rank1_asr = (
        rank1_hits / n
        if n
        else 0.0
    )

    beats_rate = (
        beats_hits / n
        if n
        else 0.0
    )

    legitimate_recall = (
        legitimate_hits / n
        if n
        else 0.0
    )

    return {
        "top_k": top_k,
        "queries": n,

        "poison_hits":
            poison_hits,

        "poison_retrieval_asr":
            poison_asr,

        "poison_rank1_hits":
            rank1_hits,

        "poison_rank1_asr":
            rank1_asr,

        "poison_beats_legitimate_hits":
            beats_hits,

        "poison_beats_legitimate_rate":
            beats_rate,

        "legitimate_hits":
            legitimate_hits,

        "legitimate_recall":
            legitimate_recall,

        "legitimate_mrr":
            legitimate_mrr,
    }


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

    poison_map = {
        str(record["query_id"])
        .strip()
        .lower(): record
        for record in poison_records
    }

    print("=" * 72)
    print(
        "PHASE 2 - SEMANTIC MIMICRY "
        "TOP-K ABLATION"
    )
    print("=" * 72)

    print(
        f"Benchmark queries : "
        f"{len(benchmark)}"
    )

    print(
        f"Poison documents : "
        f"{len(poison_records)}"
    )

    print(
        f"Top-k settings   : "
        f"{TOP_K_VALUES}"
    )

    all_results = []

    summaries = {}

    for top_k in TOP_K_VALUES:

        results = evaluate_at_k(
            benchmark,
            poison_map,
            top_k,
        )

        all_results.extend(results)

        summaries[str(top_k)] = (
            calculate_summary(
                results,
                top_k,
            )
        )

    fieldnames = [
        "top_k",
        "query_id",
        "query",
        "legitimate_doc_id",
        "poison_doc_id",
        "retrieved_doc_ids",
        "legitimate_rank",
        "poison_rank",
        "legitimate_similarity",
        "poison_similarity",
        "similarity_advantage",
        "legitimate_hit",
        "poison_hit",
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
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    summary_output = {
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

        "top_k_values":
            TOP_K_VALUES,

        "results":
            summaries,

        "frozen_top5_reference": {
            "poison_top5_asr":
                0.95,

            "poison_rank1_asr":
                0.80,

            "poison_beats_legitimate_rate":
                0.80,

            "legitimate_top5_recall":
                1.00,

            "legitimate_mrr":
                0.05666666666666667,
        },
    }

    with SUMMARY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary_output,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("PHASE 2 SUMMARY")
    print("=" * 72)

    for top_k in TOP_K_VALUES:

        summary = summaries[
            str(top_k)
        ]

        print()
        print(f"TOP-{top_k}")

        print(
            "  Poison retrieval ASR : "
            f"{summary['poison_retrieval_asr']:.2%}"
        )

        print(
            "  Rank-1 takeover      : "
            f"{summary['poison_rank1_asr']:.2%}"
        )

        print(
            "  Beats legitimate     : "
            f"{summary['poison_beats_legitimate_rate']:.2%}"
        )

        print(
            "  Legitimate recall    : "
            f"{summary['legitimate_recall']:.2%}"
        )

        print(
            "  Legitimate MRR       : "
            f"{summary['legitimate_mrr']:.4f}"
        )

    print()
    print(f"CSV output     : {CSV_OUTPUT}")
    print(f"Summary output : {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()