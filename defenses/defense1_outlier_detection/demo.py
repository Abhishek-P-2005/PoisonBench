"""
defenses/defense1_outlier_detection/demo.py
===============================================
DETAILED WALKTHROUGH DEMO for Defense 1, shown on its own (not chained with
Defense 2/3 — see defenses/demo_review3.py for the chained version).

Difference from test_smoke.py: test_smoke.py asserts pass/fail for CI-style
verification. This script is for a LIVE audience — it narrates every step
of how the z-score is actually computed, not just the final verdict, so you
can point at specific numbers while explaining the method.

Run:
    export PYTHONPATH=.
    python -m defenses.defense1_outlier_detection.demo
"""

from __future__ import annotations

from defenses.defense1_outlier_detection.config import (
    HARD_BLOCK_ZSCORE,
    MIN_STD_FLOOR,
    SOFT_FLAG_ZSCORE,
)
from defenses.defense1_outlier_detection.filter import apply_outlier_filter
from defenses.defense1_outlier_detection.outlier_detector import (
    _centroid,       # noqa: used here only for the worked-example printout
    _cosine_distance,
    _mean,
    _stdev,
)

SEP = "=" * 74


def build_batch() -> list[dict]:
    """
    6 documents: 4 legitimate CVE chunks clustered tightly around direction
    (1, 1, 0, 0), one naive attack (off-topic embedding), one
    optimization-based attack (embedding deliberately close to the
    legitimate cluster). Toy 4-dimensional vectors for illustration —
    real embeddings from MiniLM are 384-dimensional.
    """
    return [
        {"chunk_id": "legit-1", "text": "CVE-2023-1111 advisory chunk 1", "similarity": 0.80, "embedding": [1.00, 1.00, 0.00, 0.00]},
        {"chunk_id": "legit-2", "text": "CVE-2023-1111 advisory chunk 2", "similarity": 0.78, "embedding": [0.93, 1.06, 0.04, -0.03]},
        {"chunk_id": "legit-3", "text": "CVE-2023-1111 advisory chunk 3", "similarity": 0.76, "embedding": [1.07, 0.94, -0.04, 0.03]},
        {"chunk_id": "legit-4", "text": "CVE-2023-1111 advisory chunk 4", "similarity": 0.74, "embedding": [0.96, 1.04, 0.03, -0.04]},
        {"chunk_id": "chunk-OPTIMIZATION-ATTACK", "text": "Attacker text optimized to match legitimate embeddings", "similarity": 0.95, "embedding": [1.03, 0.97, 0.02, -0.02]},
        {"chunk_id": "chunk-NAIVE-ATTACK", "text": "Attacker text that is topically unrelated", "similarity": 0.60, "embedding": [0.00, 0.00, 1.00, 1.00]},
    ]


def run_demo() -> None:
    print(SEP)
    print("Defense 1 — Embedding Outlier Detection — Detailed Walkthrough")
    print(SEP)

    print(
        "\nMETHOD:\n"
        "  1. For each document, take every OTHER document in the batch.\n"
        "  2. Compute their centroid (average embedding vector).\n"
        "  3. Measure this document's cosine distance from that centroid.\n"
        "  4. Compare that distance to the mean and spread (std dev) of\n"
        "     everyone else's distance from the same centroid, as a z-score:\n"
        "         z = (this_distance - mean_distance) / std_distance\n"
        "  5. z-score >= {:.2f} -> flagged as suspicious (soft)\n"
        "     z-score >= {:.2f} -> blocked outright (hard)\n"
        "\n"
        "  Note: the document being scored is EXCLUDED from its own\n"
        "  reference statistics ('leave-one-out') — otherwise an attack\n"
        "  document could drag the centroid toward itself and partially\n"
        "  hide its own signal.\n"
        "\n"
        "  A minimum standard-deviation floor of {:.3f} is also applied, so\n"
        "  that when legitimate documents happen to cluster extremely\n"
        "  tightly, tiny natural variation doesn't get amplified into a\n"
        "  meaningless huge z-score purely from dividing by a near-zero\n"
        "  number."
        .format(SOFT_FLAG_ZSCORE, HARD_BLOCK_ZSCORE, MIN_STD_FLOOR)
    )

    batch = build_batch()

    print(f"\n{SEP}")
    print("WORKED EXAMPLE — showing the arithmetic for two specific documents")
    print(SEP)

    ids = [b["chunk_id"] for b in batch]
    embeddings = [b["embedding"] for b in batch]

    for target_id in ("chunk-NAIVE-ATTACK", "chunk-OPTIMIZATION-ATTACK"):
        idx = ids.index(target_id)
        others = embeddings[:idx] + embeddings[idx + 1:]
        centroid = _centroid(others)
        other_distances = [_cosine_distance(e, centroid) for e in others]
        mean_d = _mean(other_distances)
        std_d = max(_stdev(other_distances, mean_d), MIN_STD_FLOOR)
        dist = _cosine_distance(embeddings[idx], centroid)
        zscore = (dist - mean_d) / std_d

        print(f"\n  {target_id}")
        print(f"    embedding                  = {embeddings[idx]}")
        print(f"    centroid of other 5 docs   = [{', '.join(f'{c:.3f}' for c in centroid)}]")
        print(f"    this doc's distance        = {dist:.4f}")
        print(f"    other docs' mean distance  = {mean_d:.4f}")
        print(f"    other docs' std (floored)  = {std_d:.4f}")
        print(f"    z-score = ({dist:.4f} - {mean_d:.4f}) / {std_d:.4f} = {zscore:+.2f}")

    print(f"\n{SEP}")
    print("FULL BATCH RESULT (via the actual production function)")
    print(SEP)

    passed, blocked = apply_outlier_filter(batch)

    all_scored = sorted(passed + blocked, key=lambda r: r["outlier_zscore"], reverse=True)
    print(f"\n{'chunk_id':<28} {'distance':>10} {'z-score':>10}  verdict")
    print("-" * 62)
    for r in all_scored:
        verdict = "BLOCKED" if r["outlier_blocked"] else ("flagged" if r["outlier_flagged"] else "passed")
        print(f"{r['chunk_id']:<28} {r['outlier_distance']:>10.4f} {r['outlier_zscore']:>+10.2f}  {verdict}")

    print(
        f"\nRESULT: {len(blocked)} document(s) blocked, {len(passed)} passed through.\n"
        "The naive attack (off-topic embedding) is caught. The optimization\n"
        "attack (embedding deliberately close to legitimate content) is NOT\n"
        "caught here — that is a documented, expected limitation (see\n"
        "config.py), and it's exactly why Defenses 2 and 3 exist downstream."
    )
    print(SEP)


if __name__ == "__main__":
    run_demo()
