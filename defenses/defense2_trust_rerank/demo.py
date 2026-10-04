"""
defenses/defense2_trust_rerank/demo.py
===========================================
DETAILED WALKTHROUGH DEMO for Defense 2, shown on its own.

Difference from test_smoke.py: this script prints the actual arithmetic
behind the trust score and the final blended ranking score, so you can
point at specific numbers while explaining the method to a live audience.

Run:
    export PYTHONPATH=.
    python -m defenses.defense2_trust_rerank.demo
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from defenses.defense2_trust_rerank.config import (
    DEFAULT_TRUST_WEIGHTS,
    LOW_TRUST_FLAG_THRESHOLD,
    RERANK_ALPHA,
)
from defenses.defense2_trust_rerank.reranker import rerank_results

SEP = "=" * 74


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


def build_batch() -> list[dict]:
    """
    4 candidates for one query:
      chunk-1, chunk-2, chunk-4  — legitimate, verified, established
      chunk-3-POISONED            — unverified, brand new, uncorroborated,
                                     but with the HIGHEST raw similarity
                                     (simulating an attack that achieved
                                     strong embedding similarity)
    """
    return [
        {
            "chunk_id": "chunk-1", "text": "CVE-2023-1111: SQL injection advisory (primary)",
            "similarity": 0.91,
            "metadata": {"doc_id": "CVE-2023-1111", "verified": True,
                          "published_date": _days_ago(400), "modified_date": _days_ago(395)},
        },
        {
            "chunk_id": "chunk-2", "text": "CVE-2023-1111: additional advisory detail",
            "similarity": 0.85,
            "metadata": {"doc_id": "CVE-2023-1111", "verified": True,
                          "published_date": _days_ago(395), "modified_date": _days_ago(390)},
        },
        {
            "chunk_id": "chunk-3-POISONED", "text": "[attacker-crafted text optimized for similarity]",
            "similarity": 0.95,
            "metadata": {"doc_id": "CVE-2023-1111-fake", "verified": False,
                          "published_date": _days_ago(0), "modified_date": _days_ago(0)},
        },
        {
            "chunk_id": "chunk-4", "text": "CVE-2023-1111: vendor patch notes",
            "similarity": 0.60,
            "metadata": {"doc_id": "CVE-2023-1111", "verified": True,
                          "published_date": _days_ago(380), "modified_date": _days_ago(375)},
        },
    ]


def run_demo() -> None:
    print(SEP)
    print("Defense 2 — Trust-Scored Re-ranking — Detailed Walkthrough")
    print(SEP)

    w = DEFAULT_TRUST_WEIGHTS
    print(
        "\nMETHOD:\n"
        "  Every document gets a trust score built from three components:\n"
        f"    trust = {w.source:.1f} * source_reliability   "
        "(1.0 if source is on the trusted allow-list, 0.4 otherwise)\n"
        f"          + {w.age:.1f} * age_score               "
        "(ramps from 0.5 at 0 days old to 1.0 at 30+ days old)\n"
        f"          + {w.corroboration:.1f} * corroboration_score "
        "(fraction of other verified chunks sharing the same doc_id,\n"
        "                                        saturating at 3)\n"
        "\n"
        "  That trust score is then blended with retrieval similarity:\n"
        f"    final_score = {RERANK_ALPHA} * normalized_similarity "
        f"+ {1 - RERANK_ALPHA:.1f} * trust_score\n"
        "\n"
        "  Documents are re-sorted by final_score. Anything with trust_score\n"
        f"  below {LOW_TRUST_FLAG_THRESHOLD} is flagged as low-trust for logging, "
        "independent of\n"
        "  whether it gets pushed out of the top rank."
    )

    batch = build_batch()

    print(f"\n{SEP}")
    print("BEFORE — raw retrieval, sorted by similarity only")
    print(SEP)
    for r in sorted(batch, key=lambda x: x["similarity"], reverse=True):
        print(f"  {r['chunk_id']:<20} similarity={r['similarity']:.2f}")

    reranked = rerank_results(batch)

    print(f"\n{SEP}")
    print("AFTER — Defense 2 re-ranking, with the full arithmetic shown")
    print(SEP)
    for rank, r in enumerate(reranked, start=1):
        bd = r["trust_breakdown"]
        flag = "  <-- LOW TRUST" if r["is_low_trust"] else ""
        print(f"\n  #{rank}  {r['chunk_id']}{flag}")
        print(f"      raw similarity        = {r['similarity']:.3f}")
        print(f"      normalized similarity = {r['similarity_norm']:.3f}")
        print(
            f"      trust components      -> source={bd['source_score']:.2f}  "
            f"age={bd['age_score']:.2f}  corroboration={bd['corroboration_score']:.2f}"
        )
        print(f"      trust_score            = {r['trust_score']:.3f}")
        print(
            f"      final_score = {RERANK_ALPHA} * {r['similarity_norm']:.3f} + "
            f"{1 - RERANK_ALPHA:.1f} * {r['trust_score']:.3f} = {r['final_score']:.3f}"
        )

    print(f"\n{SEP}")
    top_before = max(batch, key=lambda x: x["similarity"])["chunk_id"]
    top_after = reranked[0]["chunk_id"]
    print(f"RESULT: highest raw similarity was '{top_before}'.")
    print(f"        highest final_score after Defense 2 is '{top_after}'.")
    if top_before != top_after:
        print(
            "        Defense 2 successfully demoted the highest-similarity "
            "document because\n        it had no trust signal backing it up "
            "(unverified source, brand new, no corroboration)."
        )
    print(SEP)


if __name__ == "__main__":
    run_demo()
