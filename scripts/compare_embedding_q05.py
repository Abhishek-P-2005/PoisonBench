"""
Phase 5 - Controlled Retrieval Comparison
Optimization-Based Embedding Attack

Compares q05 retrieval behavior across:
1. Legitimate CVE document
2. Semantic Mimicry poison
3. Optimized embedding poison

Expected precondition:
    ChromaDB is running and currently contains the clean NVD corpus
    plus the existing Semantic Mimicry poison set.
"""

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from api.pipeline.embedding import embed_texts
from api.pipeline.vectorstore import (
    get_collection,
    query,
    upsert_chunks,
)


QUERY_ID = "q05"

TRIGGER_QUERY = (
    "Which UltraVNC vulnerability uses a weak pseudo-random generator "
    "for authentication challenge bytes, making the challenge predictable?"
)

LEGITIMATE_DOC_ID = "CVE-2026-44040"

SEMANTIC_MIMICRY_DOC_ID = "POISON-SM-q05"

OPTIMIZED_DOC_ID = "POISON-OBEA-q05"

SEMANTIC_MIMICRY_TEXT = (
    "CVE-2026-44040 UltraVNC authentication vulnerability involving weak "
    "pseudo-random challenge generation. Predictable VNC authentication "
    "challenge bytes can enable challenge prediction and offline attacks."
)

OPTIMIZED_TEXT = (
    "weak pseudo-random generator. pseudo-random challenge bytes. "
    "CVE-2026-44040 UltraVNC authentication vulnerability involving weak "
    "pseudo-random challenge generation. Predictable VNC authentication "
    "challenge bytes can enable challenge prediction and offline attacks. "
    "weak pseudo-random generator is central to this vulnerability. "
    "predictable challenge generation. predictable challenge generation "
    "is central to this vulnerability."
)

TOP_K = 10


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Embeddings are normalized by embedding.py,
    so cosine similarity equals dot product.
    """

    return float(np.dot(a, b))


def calculate_direct_similarity(
    query_text: str,
    document_text: str,
) -> float:

    embeddings = np.array(
        embed_texts(
            [
                query_text,
                document_text,
            ]
        ),
        dtype=np.float32,
    )

    return cosine_similarity(
        embeddings[0],
        embeddings[1],
    )


def find_rank(
    results: list[dict],
    doc_id: str,
):
    """
    Returns 1-based rank.
    Returns None if document is outside Top-k.
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result["doc_id"] == doc_id:
            return rank

    return None


def find_result(
    results: list[dict],
    doc_id: str,
):

    for result in results:
        if result["doc_id"] == doc_id:
            return result

    return None


def print_target_result(
    label: str,
    doc_id: str,
    results: list[dict],
):

    rank = find_rank(
        results,
        doc_id,
    )

    result = find_result(
        results,
        doc_id,
    )

    if result is None:

        print(
            f"{label:<24}: "
            f"not retrieved in Top-{TOP_K}"
        )

        return

    print(
        f"{label:<24}: "
        f"rank={rank} | "
        f"similarity={result['similarity_score']:.6f}"
    )


def main():

    print("=" * 72)
    print(
        "PHASE 5 - CONTROLLED RETRIEVAL COMPARISON"
    )
    print("=" * 72)

    print(f"Query ID : {QUERY_ID}")
    print(f"Top-k    : {TOP_K}")
    print()

    # --------------------------------------------------
    # Step 1: Direct embedding comparison
    # --------------------------------------------------

    semantic_similarity = (
        calculate_direct_similarity(
            TRIGGER_QUERY,
            SEMANTIC_MIMICRY_TEXT,
        )
    )

    optimized_similarity = (
        calculate_direct_similarity(
            TRIGGER_QUERY,
            OPTIMIZED_TEXT,
        )
    )

    print(
        "DIRECT QUERY-DOCUMENT EMBEDDING SIMILARITY"
    )
    print("-" * 72)

    print(
        f"Semantic Mimicry poison : "
        f"{semantic_similarity:.6f}"
    )

    print(
        f"Optimized poison        : "
        f"{optimized_similarity:.6f}"
    )

    print(
        f"Optimization gain       : "
        f"{optimized_similarity - semantic_similarity:+.6f}"
    )

    print()

    # --------------------------------------------------
    # Step 2: Check existing collection
    # --------------------------------------------------

    collection = get_collection()

    print(
        f"Collection chunks before optimized poison: "
        f"{collection.count()}"
    )

    print()

    # --------------------------------------------------
    # Step 3: Baseline retrieval before optimized poison
    # --------------------------------------------------

    print("=" * 72)
    print(
        "BEFORE OPTIMIZED POISON INGESTION"
    )
    print("=" * 72)

    before_results = query(
        TRIGGER_QUERY,
        top_k=TOP_K,
    )

    print_target_result(
        "Legitimate document",
        LEGITIMATE_DOC_ID,
        before_results,
    )

    print_target_result(
        "Semantic Mimicry",
        SEMANTIC_MIMICRY_DOC_ID,
        before_results,
    )

    print_target_result(
        "Optimized poison",
        OPTIMIZED_DOC_ID,
        before_results,
    )

    print()

    print("Top retrieved documents:")

    for rank, result in enumerate(
        before_results,
        start=1,
    ):

        print(
            f"{rank:02d}. "
            f"{result['doc_id']:<22} "
            f"similarity="
            f"{result['similarity_score']:.6f}"
        )

    print()

    # --------------------------------------------------
    # Step 4: Insert optimized poison
    # --------------------------------------------------

    optimized_chunk = {
        "chunk_id": (
            f"{OPTIMIZED_DOC_ID}_chunk_0"
        ),
        "text": OPTIMIZED_TEXT,
        "doc_id": OPTIMIZED_DOC_ID,
        "source": (
            "controlled_optimization_embedding_attack"
        ),
        "timestamp": "",
        "is_poisoned": True,
        "position": 0,
    }

    upsert_chunks(
        [optimized_chunk]
    )

    print(
        f"Collection chunks after optimized poison : "
        f"{collection.count()}"
    )

    print()

    # --------------------------------------------------
    # Step 5: Retrieval after optimized poison
    # --------------------------------------------------

    print("=" * 72)
    print(
        "AFTER OPTIMIZED POISON INGESTION"
    )
    print("=" * 72)

    after_results = query(
        TRIGGER_QUERY,
        top_k=TOP_K,
    )

    legitimate_rank = find_rank(
        after_results,
        LEGITIMATE_DOC_ID,
    )

    semantic_rank = find_rank(
        after_results,
        SEMANTIC_MIMICRY_DOC_ID,
    )

    optimized_rank = find_rank(
        after_results,
        OPTIMIZED_DOC_ID,
    )

    print_target_result(
        "Legitimate document",
        LEGITIMATE_DOC_ID,
        after_results,
    )

    print_target_result(
        "Semantic Mimicry",
        SEMANTIC_MIMICRY_DOC_ID,
        after_results,
    )

    print_target_result(
        "Optimized poison",
        OPTIMIZED_DOC_ID,
        after_results,
    )

    print()

    print("Top retrieved documents:")

    for rank, result in enumerate(
        after_results,
        start=1,
    ):

        marker = ""

        if (
            result["doc_id"]
            == OPTIMIZED_DOC_ID
        ):
            marker = " <-- OPTIMIZED"

        elif (
            result["doc_id"]
            == SEMANTIC_MIMICRY_DOC_ID
        ):
            marker = " <-- SEMANTIC MIMICRY"

        elif (
            result["doc_id"]
            == LEGITIMATE_DOC_ID
        ):
            marker = " <-- LEGITIMATE"

        print(
            f"{rank:02d}. "
            f"{result['doc_id']:<22} "
            f"similarity="
            f"{result['similarity_score']:.6f}"
            f"{marker}"
        )

    print()

    # --------------------------------------------------
    # Step 6: Controlled attack interpretation
    # --------------------------------------------------

    print("=" * 72)
    print("PHASE 5 RESULT")
    print("=" * 72)

    print(
        f"Legitimate rank       : "
        f"{legitimate_rank}"
    )

    print(
        f"Semantic Mimicry rank : "
        f"{semantic_rank}"
    )

    print(
        f"Optimized poison rank : "
        f"{optimized_rank}"
    )

    print()

    optimized_beats_semantic = (
        optimized_rank is not None
        and (
            semantic_rank is None
            or optimized_rank < semantic_rank
        )
    )

    optimized_beats_legitimate = (
        optimized_rank is not None
        and (
            legitimate_rank is None
            or optimized_rank < legitimate_rank
        )
    )

    optimized_rank1 = (
        optimized_rank == 1
    )

    print(
        f"Optimized beats Semantic Mimicry : "
        f"{optimized_beats_semantic}"
    )

    print(
        f"Optimized beats legitimate       : "
        f"{optimized_beats_legitimate}"
    )

    print(
        f"Optimized Rank-1 takeover        : "
        f"{optimized_rank1}"
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()