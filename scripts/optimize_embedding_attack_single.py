"""
Phase 3 - Optimization-Based Embedding Attack
Single-query prototype.

This prototype searches over candidate poison texts for one benchmark query
and keeps the text whose embedding is most similar to the target query.

No corpus ingestion occurs in this phase.
This is optimization/search only.
"""

import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.pipeline.embedding import embed_texts



BENCHMARK_PATH = (
    REPO_ROOT
    / "data"
    / "benchmark"
    / "real_trigger_queries_v1.jsonl"
)

TARGET_QUERY_ID = "q05"


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def get_query_id(record: dict) -> str:
    for field in ["query_id", "id"]:
        value = record.get(field)

        if value is not None:
            return str(value).strip().lower()

    raise ValueError(
        f"Could not find query id in record: {record}"
    )


def get_query_text(record: dict) -> str:
    for field in [
        "query",
        "query_text",
        "text",
        "question",
        "trigger_query",
    ]:
        value = record.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        f"Could not find query text in record: {record}"
    )


def cosine_similarity(vec_a, vec_b) -> float:
    """
    Embeddings are already normalized by embedding.py,
    so cosine similarity is simply the dot product.
    """
    return float(
        np.dot(
            np.asarray(vec_a),
            np.asarray(vec_b),
        )
    )


def main():
    benchmark = load_jsonl(BENCHMARK_PATH)

    target_record = None

    for record in benchmark:
        if get_query_id(record) == TARGET_QUERY_ID:
            target_record = record
            break

    if target_record is None:
        raise ValueError(
            f"Query {TARGET_QUERY_ID} not found"
        )

    query_text = get_query_text(target_record)

    print("=" * 72)
    print("PHASE 3 - OPTIMIZATION-BASED EMBEDDING ATTACK")
    print("SINGLE-QUERY PROTOTYPE")
    print("=" * 72)

    print(f"Target query ID : {TARGET_QUERY_ID}")
    print(f"Query           : {query_text}")
    print()

    candidates = [
        (
            "UltraVNC security vulnerability."
        ),

        (
            "UltraVNC authentication vulnerability."
        ),

        (
            "UltraVNC authentication vulnerability "
            "involving challenge generation."
        ),

        (
            "UltraVNC authentication vulnerability "
            "involving weak pseudo-random challenge generation."
        ),

        (
            "UltraVNC authentication vulnerability involving "
            "weak pseudo-random challenge generation and "
            "predictable authentication challenge bytes."
        ),

        (
            "UltraVNC vulnerability using a weak pseudo-random "
            "generator for authentication challenge bytes, "
            "allowing predictable VNC authentication challenges."
        ),
    ]

    query_embedding = embed_texts(
        [query_text]
    )[0]

    candidate_embeddings = embed_texts(
        candidates
    )

    scored_candidates = []

    for index, (
        candidate,
        candidate_embedding,
    ) in enumerate(
        zip(
            candidates,
            candidate_embeddings,
        ),
        start=1,
    ):

        similarity = cosine_similarity(
            query_embedding,
            candidate_embedding,
        )

        scored_candidates.append(
            {
                "iteration": index,
                "text": candidate,
                "similarity": similarity,
            }
        )

        print(
            f"Candidate {index}: "
            f"similarity={similarity:.6f}"
        )

        print(
            f"  {candidate}"
        )

        print()

    scored_candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    best = scored_candidates[0]

    print("=" * 72)
    print("BEST CANDIDATE")
    print("=" * 72)

    print(
        f"Similarity : "
        f"{best['similarity']:.6f}"
    )

    print(
        f"Text       : "
        f"{best['text']}"
    )

    print()

    print("=" * 72)
    print("RANKING")
    print("=" * 72)

    for rank, item in enumerate(
        scored_candidates,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{item['similarity']:.6f} | "
            f"{item['text']}"
        )


if __name__ == "__main__":
    main()