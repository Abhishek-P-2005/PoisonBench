"""
defenses/defense1_outlier_detection/test_smoke.py
-----------------------------------------------------
Standalone smoke test using synthetic embeddings — no ChromaDB or embedding
model required. Demonstrates TWO cases on purpose:

  1. A naive semantic mimicry attack (PRD Attack 1) with an embedding that
     is clearly off-topic relative to the retrieval batch — Defense 1
     SHOULD catch this. Cheapest attack, weakest against defenses.

  2. An optimization-based attack (PRD Attack 2) with an embedding
     deliberately close to the batch centroid, simulating an attacker who
     optimized their document to maximize similarity — Defense 1 is
     EXPECTED to miss this, per the documented limitation in config.py.
     This is not a bug — it is the reason the pipeline doesn't stop at
     Defense 1.

Usage
-----
    python -m defenses.defense1_outlier_detection.test_smoke
"""

from __future__ import annotations

from defenses.defense1_outlier_detection.filter import apply_outlier_filter


def build_synthetic_batch() -> list[dict]:
    """
    4-dimensional toy embeddings. Legitimate documents cluster tightly
    around direction (1, 1, 0, 0). The optimization attack embedding is
    deliberately placed almost exactly on that same direction (blending
    in). The naive attack embedding points in an unrelated direction
    (0, 0, 1, 1) — orthogonal to the legitimate cluster.
    """
    legit_embeddings = [
        [1.00, 1.00, 0.00, 0.00],
        [0.93, 1.06, 0.04, -0.03],
        [1.07, 0.94, -0.04, 0.03],
        [0.96, 1.04, 0.03, -0.04],
    ]

    optimization_attack_embedding = [1.03, 0.97, 0.02, -0.02]   # blends in
    naive_attack_embedding        = [0.00, 0.00, 1.00, 1.00]    # off-topic

    results = []
    for i, emb in enumerate(legit_embeddings, start=1):
        results.append({
            "chunk_id": f"legit-{i}",
            "text": f"Legitimate CVE advisory chunk {i}",
            "similarity": 0.80,
            "embedding": emb,
        })

    results.append({
        "chunk_id": "chunk-OPTIMIZATION-ATTACK",
        "text": "Attacker text optimized to match embedding of legitimate content",
        "similarity": 0.95,
        "embedding": optimization_attack_embedding,
    })

    results.append({
        "chunk_id": "chunk-NAIVE-ATTACK",
        "text": "Attacker text that is topically unrelated to the query",
        "similarity": 0.60,
        "embedding": naive_attack_embedding,
    })

    return results


def run_smoke_test() -> None:
    print("=" * 74)
    print("Defense 1 (Embedding Outlier Detection) — Smoke Test")
    print("=" * 74)

    results = build_synthetic_batch()
    passed, blocked = apply_outlier_filter(results)

    print(f"\nInput: {len(results)} chunks")
    print(f"Passed: {len(passed)}   Blocked: {len(blocked)}")

    print("\n--- All chunks, sorted by z-score (most anomalous first) ---")
    all_scored = sorted(passed + blocked, key=lambda r: r["outlier_zscore"], reverse=True)
    for r in all_scored:
        status = "BLOCKED" if r["outlier_blocked"] else ("flagged" if r["outlier_flagged"] else "ok")
        print(
            f"  {r['chunk_id']:<28} distance={r['outlier_distance']:.3f}  "
            f"zscore={r['outlier_zscore']:+.2f}  [{status}]"
        )

    blocked_ids = {r["chunk_id"] for r in blocked}
    passed_ids  = {r["chunk_id"] for r in passed}

    print("\n--- Interpretation ---")
    print(
        "Naive attack (off-topic embedding): "
        f"{'BLOCKED, as expected' if 'chunk-NAIVE-ATTACK' in blocked_ids else 'NOT blocked — investigate'}"
    )
    print(
        "Optimization attack (embedding blended in): "
        f"{'passed through undetected, as documented in config.py' if 'chunk-OPTIMIZATION-ATTACK' in passed_ids else 'was blocked — unexpectedly caught, verify thresholds'}"
    )

    assert "chunk-NAIVE-ATTACK" in blocked_ids, (
        "FAIL: the off-topic naive attack should be blocked as an outlier."
    )
    assert "chunk-OPTIMIZATION-ATTACK" in passed_ids, (
        "FAIL (of the test's own expectation, not the defense): the "
        "optimization attack was blocked, meaning it wasn't blended in "
        "tightly enough in this synthetic example — the defense working "
        "'too well' here would misrepresent its documented real-world "
        "limitation."
    )

    print(
        "\nPASS: Defense 1 caught the naive attack and, as documented, missed "
        "the optimization-based attack — which is exactly why Defense 2 and "
        "Defense 3 exist downstream (see defenses/demo_review3.py for the "
        "full pipeline catching it)."
    )
    print(
        "\nNote for panel: these are 4-dimensional TOY embeddings for "
        "illustration only, so z-scores here are exaggerated compared to "
        "real 384-dim MiniLM embeddings, which naturally have more spread. "
        "SOFT_FLAG_ZSCORE / HARD_BLOCK_ZSCORE in config.py will need "
        "recalibration once run against real corpus embeddings — same "
        "'not yet calibrated' status as Defense 2's RERANK_ALPHA."
    )
    print("=" * 74)


if __name__ == "__main__":
    run_smoke_test()
