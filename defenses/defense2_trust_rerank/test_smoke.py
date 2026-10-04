"""
defenses/defense2_trust_rerank/test_smoke.py
-----------------------------------------------
Standalone smoke test using synthetic retrieval results — does not require
ChromaDB, PostgreSQL, or the attack module to be ready. Run this to verify
Defense 2 works correctly in isolation before wiring it into the full pipeline.

Usage
-----
    python -m defenses.defense2_trust_rerank.test_smoke
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from defenses.defense2_trust_rerank.reranker import rerank_results
from defenses.defense2_trust_rerank.trust_scorer import TrustScorer


def _make_metadata(
    doc_id: str,
    verified: bool,
    days_old: int,
) -> dict:
    pub_date = (datetime.now(timezone.utc) - timedelta(days=days_old)).strftime("%Y-%m-%d")
    return {
        "doc_id": doc_id,
        "verified": verified,
        "published_date": pub_date,
        "modified_date": pub_date,
    }


def build_synthetic_results() -> list[dict]:
    """
    Four synthetic retrieval results simulating a realistic top-4:
      1. Legit, verified, old, well-corroborated, high similarity
      2. Legit, verified, old, well-corroborated, medium similarity
      3. Suspicious: unverified, brand-new, no corroboration, HIGH similarity
         (this is the interesting case — an optimization-based attack that
         achieved high cosine similarity but has no trust signal backing it)
      4. Legit, verified, old, but low similarity
    """
    return [
        {
            "chunk_id": "chunk-1",
            "text": "CVE-2023-1111: SQL injection in ExampleCorp WebApp v2.",
            "similarity": 0.91,
            "metadata": _make_metadata("CVE-2023-1111", verified=True, days_old=400),
        },
        {
            "chunk_id": "chunk-2",
            "text": "CVE-2023-1111: Additional advisory detail on the same SQLi.",
            "similarity": 0.85,
            "metadata": _make_metadata("CVE-2023-1111", verified=True, days_old=395),
        },
        {
            "chunk_id": "chunk-3-POISONED",
            "text": "CVE-2023-1111: [attacker-crafted text optimized for similarity]",
            "similarity": 0.95,   # deliberately the HIGHEST similarity
            "metadata": _make_metadata("CVE-2023-1111-fake", verified=False, days_old=0),
        },
        {
            "chunk_id": "chunk-4",
            "text": "CVE-2023-1111: Vendor patch notes.",
            "similarity": 0.60,
            "metadata": _make_metadata("CVE-2023-1111", verified=True, days_old=380),
        },
    ]


def run_smoke_test() -> None:
    print("=" * 70)
    print("Defense 2 (Trust-Scored Re-ranking) — Smoke Test")
    print("=" * 70)

    results = build_synthetic_results()

    print("\n--- BEFORE re-ranking (sorted by raw similarity) ---")
    for r in sorted(results, key=lambda x: x["similarity"], reverse=True):
        print(f"  {r['chunk_id']:<20} similarity={r['similarity']:.2f}")

    reranked = rerank_results(results)

    print("\n--- AFTER Defense 2 re-ranking (sorted by final_score) ---")
    for r in reranked:
        flag = " [LOW TRUST]" if r["is_low_trust"] else ""
        print(
            f"  {r['chunk_id']:<20} "
            f"sim_norm={r['similarity_norm']:.2f}  "
            f"trust={r['trust_score']:.2f}  "
            f"final={r['final_score']:.2f}{flag}"
        )
        bd = r["trust_breakdown"]
        print(
            f"      breakdown -> source={bd['source_score']:.2f}  "
            f"age={bd['age_score']:.2f}  "
            f"corroboration={bd['corroboration_score']:.2f}"
        )

    # ── assertions ────────────────────────────────────────────────────────
    top_id = reranked[0]["chunk_id"]
    poisoned_rank = next(
        i for i, r in enumerate(reranked) if r["chunk_id"] == "chunk-3-POISONED"
    )

    print(f"\nTop-ranked after Defense 2: {top_id}")
    print(f"Poisoned chunk rank after Defense 2: {poisoned_rank + 1} of {len(reranked)}")

    assert top_id != "chunk-3-POISONED", (
        "FAIL: the highest-similarity poisoned chunk should NOT be top-ranked "
        "after trust re-ranking, given it is unverified, brand-new, and "
        "uncorroborated."
    )
    assert reranked[poisoned_rank]["is_low_trust"], (
        "FAIL: the poisoned chunk should be flagged as low-trust."
    )

    print("\nPASS: Defense 2 successfully demoted the high-similarity, "
          "low-trust poisoned chunk out of the top rank.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
