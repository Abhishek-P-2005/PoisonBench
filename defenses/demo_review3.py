# """
# defenses/demo_review3.py
# ===========================
# LIVE DEMO SCRIPT for the panel review.

# Demonstrates the three completed defense modules working together as an
# ordered pipeline, exactly as specified in PRD Section 5:

#     Retriever (top-k cosine similarity)
#         -> Defense 1: Embedding Outlier Detection
#         -> Defense 2: Trust-Scored Re-ranking
#         -> Defense 3: Provenance Verification
#         -> Filtered/re-ranked context (would go to LLM generation next)

# Scenario simulated
# -------------------
# A user asks a security question. The retriever returns 7 candidate chunks.
# Among them is a poisoned document injected by an optimization-based attack
# (PRD Section 6.2, Attack 2) — it was specifically crafted to:
#   (a) maximize embedding similarity to the query (highest raw similarity),
#   (b) blend into the embedding space so Defense 1 does not flag it, and
#   (c) reuse a real CVE ID so it can inherit corroboration for Defense 2.
# This is deliberately the HARD case, not a strawman — it is designed to
# survive the first two defense layers.

# This script shows, step by step, how each defense stage changes the outcome:

#   STAGE 0 — Raw retrieval ranking (no defenses)          -> attack SUCCEEDS
#   STAGE 1 — After Defense 1 (outlier detection)          -> attack SURVIVES (by design, see config.py)
#   STAGE 2 — After Defense 2 (trust re-ranking)           -> attack SURVIVES, score reduced
#   STAGE 3 — After Defense 3 (provenance verification)    -> attack BLOCKED

# Run this live for the panel with:
#     PYTHONPATH=. python -m defenses.demo_review3

# No external services required — ChromaDB, PostgreSQL, and the attack
# module are NOT needed to run this. It uses hand-built synthetic data
# (including toy 4-dimensional embeddings) so it can be demonstrated
# standalone, independent of whether Subash's pipeline or Abhishek's attack
# module are wired up yet.
# """

# from __future__ import annotations

# from corpus.embeddings import Embedder
# from corpus.sample_cve_corpus import (
#     build_clean_corpus, 
#     build_poisoned_chunk,
# )

# from defenses.defense1_outlier_detection.filter import apply_outlier_filter
# from defenses.defense2_trust_rerank.reranker import rerank_results
# from defenses.defense3_provenance.filter import apply_provenance_filter


# SEP = "=" * 78
# ATTACK_ID = None


# def build_scenario() -> list[dict]:
#     clean = build_clean_corpus()
#     poisoned = build_poisoned_chunk()

#     # Use a small retrieval subset for the Review-2 demonstration.
#     candidates = clean[:7]

#     candidates.append(poisoned)

#     embedder = Embedder()

#     texts = [item["text"] for item in candidates]
#     embeddings = embedder.embed(texts)

#     query = (
#         "How can the SQL injection vulnerability in ExampleCorp WebApp "
#         "be mitigated?"
#     )

#     query_embedding = embedder.embed_one(query)

#     def cosine_similarity(a, b):
#         return sum(x * y for x, y in zip(a, b))

#     for item, embedding in zip(candidates, embeddings):
#         item["embedding"] = embedding
#         item["similarity"] = cosine_similarity(
#             query_embedding,
#             embedding,
#         )

#     global ATTACK_ID
#     ATTACK_ID = poisoned["chunk_id"]

#     return candidates

# def print_ranking(title: str, ranked: list[dict], score_key: str = "similarity") -> None:
#     print(f"\n{title}")
#     print("-" * len(title))
#     for rank, r in enumerate(ranked, start=1):
#         marker = "  <-- ATTACK DOCUMENT" if r["chunk_id"] == ATTACK_ID else ""
#         print(f"  #{rank}  {r['chunk_id']:<20} {score_key}={r[score_key]:.3f}{marker}")


# def run_demo() -> None:
#     print(SEP)
#     print("PoisonBench — Live Demo")
#     print("Module 3 (Akilan) — Defense 1 + Defense 2 + Defense 3, chained as a pipeline")
#     print(SEP)
#     print(
#         "\nQuery: \"How is the ExampleCorp WebApp SQL injection vulnerability "
#         "mitigated?\"\n"
#         "Simulated retriever has returned 7 candidate chunks, top-k similarity "
#         "search.\nOne of them (chunk-3-ATTACK) is a poisoned document from an "
#         "optimization-based\nattack (PRD Attack 2) — deliberately crafted to "
#         "win the top similarity rank,\nblend into the embedding space, AND "
#         "reuse a real CVE ID for corroboration.\nThis is the hard case: it is "
#         "designed to survive the first two defense layers."
#     )

#     candidates = build_scenario()

#     # ── STAGE 0: raw retrieval, no defenses ─────────────────────────────────
#     raw_ranked = sorted(candidates, key=lambda r: r["similarity"], reverse=True)
#     print_ranking("STAGE 0 — Raw retrieval ranking (NO DEFENSES)", raw_ranked)
#     if raw_ranked[0]["chunk_id"] == ATTACK_ID:
#         print(
#             "\n  >>> ATTACK SUCCEEDS: poisoned document is ranked #1 and would "
#             "be passed to the LLM. <<<"
#         )

#     # ── STAGE 1: Defense 1 — Embedding Outlier Detection ────────────────────
#     stage1_passed, stage1_blocked = apply_outlier_filter(candidates)
#     attack_row_0 = next(r for r in stage1_passed + stage1_blocked if r["chunk_id"] == ATTACK_ID)
#     print("\nSTAGE 1 — After Defense 1 (Embedding Outlier Detection)")
#     print("-" * 58)
#     print(f"  Chunks passed: {len(stage1_passed)}   Blocked as outliers: {len(stage1_blocked)}")
#     print(
#         f"  Attack document: z-score={attack_row_0['outlier_zscore']:.2f}  "
#         f"blocked={attack_row_0['outlier_blocked']}"
#     )
#     print(
#         "\n  NOTE FOR PANEL: this attack's embedding was specifically placed close to "
#         "the\n  legitimate cluster (blending in), so Defense 1 does NOT catch it — "
#         "this is\n  the documented limitation in config.py, not a defect. Defense 1's "
#         "job is to\n  catch naive attacks cheaply; see defenses/defense1_outlier_detection/"
#         "test_smoke.py\n  for a case where it does catch an off-topic naive attack."
#     )
#     # Continue the pipeline with everything Defense 1 passed through.
#     stage1 = stage1_passed

#     # ── STAGE 2: Defense 2 — Trust-Scored Re-ranking ────────────────────────
#     stage2 = rerank_results(stage1)
#     print_ranking("STAGE 2 — After Defense 2 (Trust-Scored Re-ranking)", stage2, score_key="final_score")
#     attack_rank_2 = next(i for i, r in enumerate(stage2, start=1) if r["chunk_id"] == ATTACK_ID)
#     attack_row_2 = next(r for r in stage2 if r["chunk_id"] == ATTACK_ID)
#     bd = attack_row_2["trust_breakdown"]
#     print(
#         f"\n  Attack document: final_score dropped from raw similarity 0.970 to "
#         f"{attack_row_2['final_score']:.3f}, still rank #{attack_rank_2}.\n"
#         f"  Trust breakdown -> source={bd['source_score']:.2f}  age={bd['age_score']:.2f}  "
#         f"corroboration={bd['corroboration_score']:.2f}  "
#         f"(low_trust_flag={attack_row_2['is_low_trust']})\n\n"
#         f"  NOTE FOR PANEL: Defense 2 alone does not fully catch this attack either. The "
#         f"attacker reused\n  the real CVE ID 'CVE-2023-1111' instead of inventing a new "
#         f"one, so it inherits\n  corroboration score from the 3 legitimate chunks sharing "
#         f"that ID (corroboration=1.00).\n  This is realistic adversarial behavior, not a bug "
#         f"in the defense — and it is exactly why\n  the pipeline does not stop at Defense 2. "
#         f"Defense 3 checks source identity and content\n  integrity directly, independent of "
#         f"corroboration, which is what catches this case next."
#     )

#     # ── STAGE 3: Defense 3 — Provenance Verification ────────────────────────
#     passed, blocked = apply_provenance_filter(stage2)
#     print("\nSTAGE 3 — After Defense 3 (Provenance Verification)")
#     print("-" * 58)
#     print(f"  Chunks passed into final context: {len(passed)}")
#     for r in passed:
#         print(f"    - {r['chunk_id']}")
#     print(f"  Chunks BLOCKED (removed from context): {len(blocked)}")
#     for r in blocked:
#         print(f"    - {r['chunk_id']}  reasons: {r['provenance_reasons']}")

#     attack_blocked = any(r["chunk_id"] == ATTACK_ID for r in blocked)

#     print("\n" + SEP)
#     if attack_blocked:
#         print(
#             "RESULT: The attack survived Defense 1 (blended embedding) and Defense 2\n"
#             "(borrowed corroboration), exactly as it was designed to. Defense 3 then\n"
#             "BLOCKED it outright on identity and integrity grounds before it reached\n"
#             "the LLM. Final context sent to generation contains only legitimate,\n"
#             "verified documents. No single defense stopped this attack — the layered\n"
#             "pipeline did. This is the actual argument for PRD Section 5's design,\n"
#             "demonstrated rather than asserted."
#         )
#     else:
#         print("RESULT: Attack document was NOT blocked — investigate defense logic.")
#     print(SEP)

#     print("\nFinal context that would be passed to LLM generation:")
#     for r in passed:
#         print(f"  [{r['chunk_id']}] {r['text'][:80]}...")

#     assert attack_blocked, "Demo invariant violated: attack should be blocked by Defense 3"


# if __name__ == "__main__":
#     run_demo()



"""
defenses/demo_review3.py
========================

PoisonBench — Review 2 Live Defense Demo

Demonstrates the three implemented defense modules working together
as an ordered RAG defense pipeline:

    Retriever
        ↓
    Defense 1 — Embedding Outlier Detection
        ↓
    Defense 2 — Trust-Scored Re-ranking
        ↓
    Defense 3 — Provenance Verification
        ↓
    Final Context

The demonstration uses:
    - A small CVE-like synthetic corpus
    - MiniLM embeddings (all-MiniLM-L6-v2)
    - Legitimate CVE-like chunks
    - One simulated poisoned chunk
    - The actual Defense 1, Defense 2 and Defense 3 implementations

The poisoned document represents a difficult semantic-blending case:
    - it is semantically related to the query
    - its embedding is close to the legitimate cluster
    - it reuses a real CVE ID for corroboration
    - it claims to be from an official source
    - its source identity is not trusted
    - its stored hash does not match its actual content

The purpose of this demo is to demonstrate why multiple complementary
defense layers are useful.

Important:
    This is a retrieval-level defense demonstration.
    Generation-level attack success is evaluated separately in the
    complete RAG pipeline.

Run:

    python -m defenses.demo_review3

No external ChromaDB or PostgreSQL service is required.
"""

from __future__ import annotations

from corpus.embeddings import Embedder
from corpus.sample_cve_corpus import (
    build_clean_corpus,
    build_poisoned_chunk,
)

from defenses.defense1_outlier_detection.filter import (
    apply_outlier_filter,
)

from defenses.defense2_trust_rerank.reranker import (
    rerank_results,
)

from defenses.defense3_provenance.filter import (
    apply_provenance_filter,
)


SEP = "=" * 78

ATTACK_ID = None


def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    """
    Calculate cosine similarity.

    The Embedder returns normalized MiniLM embeddings, so the cosine
    similarity is equal to their dot product.
    """

    return sum(
        x * y
        for x, y in zip(a, b)
    )


def build_scenario() -> tuple[list[dict], str]:
    """
    Build the Review-2 demonstration scenario.

    The scenario intentionally uses the complete small corpus returned
    by build_clean_corpus(), together with one poisoned chunk.

    Returns:
        candidates, query
    """

    clean = build_clean_corpus()
    poisoned = build_poisoned_chunk()

    # Use the existing clean corpus and add exactly one poisoned chunk.
    candidates = list(clean)
    candidates.append(poisoned)

    if not candidates:
        raise RuntimeError(
            "Demo corpus is empty."
        )

    if len(candidates) < 2:
        raise RuntimeError(
            "Demo requires at least two candidate chunks."
        )

    global ATTACK_ID
    ATTACK_ID = poisoned["chunk_id"]

    # Query used throughout the pipeline.
    query = (
        "How is the ExampleCorp WebApp SQL injection vulnerability "
        "mitigated?"
    )

    # Load the actual MiniLM embedding model.
    embedder = Embedder()

    texts = [
        item["text"]
        for item in candidates
    ]

    # Generate document embeddings.
    embeddings = embedder.embed(texts)

    # Generate query embedding.
    query_embedding = embedder.embed_one(query)

    # Attach embeddings and retrieval similarities.
    for item, embedding in zip(
        candidates,
        embeddings,
    ):

        item["embedding"] = embedding

        item["similarity"] = cosine_similarity(
            query_embedding,
            embedding,
        )

    return candidates, query


def print_ranking(
    title: str,
    ranked: list[dict],
    score_key: str = "similarity",
) -> None:
    """
    Print a ranked list of documents.
    """

    print(f"\n{title}")
    print("-" * len(title))

    for rank, result in enumerate(
        ranked,
        start=1,
    ):

        marker = ""

        if result["chunk_id"] == ATTACK_ID:
            marker = "  <-- ATTACK DOCUMENT"

        print(
            f"  #{rank:<2} "
            f"{result['chunk_id']:<24} "
            f"{score_key}={result[score_key]:.3f}"
            f"{marker}"
        )


def print_stage_summary(
    stage_name: str,
    passed: list[dict],
    blocked: list[dict],
) -> None:
    """
    Print a basic summary of a defense stage.
    """

    print(f"\n{stage_name}")
    print("-" * 58)

    print(
        f"  Chunks passed  : {len(passed)}"
    )

    print(
        f"  Chunks blocked : {len(blocked)}"
    )

    if blocked:

        print(
            "\n  Blocked chunks:"
        )

        for result in blocked:

            print(
                f"    - {result['chunk_id']}"
            )


def find_attack(
    results: list[dict],
) -> dict | None:
    """
    Find the attack document in a list of results.
    """

    for result in results:

        if result["chunk_id"] == ATTACK_ID:
            return result

    return None


def find_rank(
    results: list[dict],
    chunk_id: str,
) -> int | None:
    """
    Return the 1-based rank of a chunk.
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if result["chunk_id"] == chunk_id:
            return rank

    return None


def run_demo() -> None:
    """
    Execute the complete Review-2 defense pipeline.
    """

    print(SEP)

    print(
        "PoisonBench — Review 2 Defense Demo"
    )

    print(
        "Module 3 (Akilan) — "
        "Defense 1 + Defense 2 + Defense 3"
    )

    print(
        "Chained as a layered defense pipeline"
    )

    print(SEP)

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    query = (
        "How is the ExampleCorp WebApp SQL injection vulnerability "
        "mitigated?"
    )

    print(
        "\nQUERY"
    )

    print(
        "-----"
    )

    print(
        f'"{query}"'
    )

    # ------------------------------------------------------------------
    # BUILD SCENARIO
    # ------------------------------------------------------------------

    candidates, query = build_scenario()

    clean_count = len(candidates) - 1

    print(
        "\nSimulated retriever has returned "
        f"{len(candidates)} candidate chunks."
    )

    print(
        "These candidates are ranked using cosine similarity "
        "between the query embedding and document embeddings."
    )

    print(
        "\nDemonstration composition:"
    )

    print(
        f"  - {clean_count} legitimate CVE-like chunks"
    )

    print(
        "  - 1 simulated poisoned chunk"
    )

    print(
        "\nThe poisoned document is designed to represent a "
        "semantic-blending attack:"
    )

    print(
        "  - high semantic relevance to the query"
    )

    print(
        "  - embedding close to the legitimate cluster"
    )

    print(
        "  - reused legitimate CVE identity"
    )

    print(
        "  - untrusted source identity"
    )

    print(
        "  - integrity/hash mismatch"
    )

    print(
        "\nPipeline:"
    )

    print(
        "  Retriever"
        "\n      ↓"
        "\n  Defense 1 — Embedding Outlier Detection"
        "\n      ↓"
        "\n  Defense 2 — Trust-Scored Re-ranking"
        "\n      ↓"
        "\n  Defense 3 — Provenance Verification"
        "\n      ↓"
        "\n  Final Context"
    )

    # ------------------------------------------------------------------
    # STAGE 0 — RAW RETRIEVAL
    # ------------------------------------------------------------------

    raw_ranked = sorted(
        candidates,
        key=lambda result: result["similarity"],
        reverse=True,
    )

    print_ranking(
        "STAGE 0 — Raw Retrieval Ranking (NO DEFENSES)",
        raw_ranked,
        score_key="similarity",
    )

    attack_raw = find_attack(
        raw_ranked
    )

    if attack_raw is None:

        print(
            "\nERROR: Attack document was not found "
            "in the raw retrieval results."
        )

        return

    attack_raw_rank = find_rank(
        raw_ranked,
        ATTACK_ID,
    )

    print(
        "\n  Attack document raw similarity: "
        f"{attack_raw['similarity']:.3f}"
    )

    print(
        "  Attack document raw rank: "
        f"#{attack_raw_rank}"
    )

    if attack_raw_rank == 1:

        print(
            "\n  >>> The attack achieved rank #1 "
            "in this run."
        )

    else:

        print(
            "\n  >>> The attack entered the retrieved "
            f"candidate set at rank #{attack_raw_rank}."
        )

        print(
            "      A poisoning document does not need to "
            "be ranked #1 to require defense."
        )

        print(
            "      It only needs to enter the candidate "
            "context considered by the RAG pipeline."
        )

    # ------------------------------------------------------------------
    # STAGE 1 — DEFENSE 1
    # ------------------------------------------------------------------

    stage1_passed, stage1_blocked = apply_outlier_filter(
        candidates
    )

    print_stage_summary(
        "STAGE 1 — Defense 1: Embedding Outlier Detection",
        stage1_passed,
        stage1_blocked,
    )

    attack_stage1 = find_attack(
        stage1_passed + stage1_blocked
    )

    if attack_stage1 is None:

        print(
            "\nERROR: Attack document disappeared during "
            "Defense 1 processing."
        )

        return

    print(
        "\n  Attack document:"
    )

    print(
        f"    outlier z-score : "
        f"{attack_stage1['outlier_zscore']:.3f}"
    )

    print(
        f"    outlier score   : "
        f"{attack_stage1['outlier_score']:.3f}"
    )

    print(
        f"    flagged         : "
        f"{attack_stage1['outlier_flagged']}"
    )

    print(
        f"    blocked         : "
        f"{attack_stage1['outlier_blocked']}"
    )

    if attack_stage1["outlier_blocked"]:

        print(
            "\n  >>> Defense 1 BLOCKED the poisoned document."
        )

        print(
            "      The document will not continue to "
            "Defense 2."
        )

    else:

        print(
            "\n  >>> Defense 1 did NOT block the attack."
        )

        print(
            "      This is consistent with a semantically "
            "blended poisoning case."
        )

        print(
            "      Its embedding is close enough to the "
            "legitimate cluster to avoid the outlier threshold."
        )

    # Only documents passing Defense 1 continue.
    stage1 = stage1_passed

    # ------------------------------------------------------------------
    # STAGE 2 — DEFENSE 2
    # ------------------------------------------------------------------

    if not stage1:

        print(
            "\nSTAGE 2 — Defense 2: Trust-Scored Re-ranking"
        )

        print(
            "-" * 58
        )

        print(
            "  No documents remain after Defense 1."
        )

        stage2 = []

    else:

        stage2 = rerank_results(
            stage1
        )

        print_ranking(
            "STAGE 2 — Defense 2: Trust-Scored Re-ranking",
            stage2,
            score_key="final_score",
        )

    attack_stage2 = find_attack(
        stage2
    )

    if attack_stage2 is not None:

        attack_rank_2 = find_rank(
            stage2,
            ATTACK_ID,
        )

        breakdown = attack_stage2.get(
            "trust_breakdown",
            {},
        )

        source_score = breakdown.get(
            "source_score",
            0.0,
        )

        age_score = breakdown.get(
            "age_score",
            0.0,
        )

        corroboration_score = breakdown.get(
            "corroboration_score",
            0.0,
        )

        print(
            "\n  Attack document after Defense 2:"
        )

        print(
            f"    raw similarity       : "
            f"{attack_stage2['similarity']:.3f}"
        )

        print(
            f"    normalized similarity: "
            f"{attack_stage2.get('normalized_similarity', 0.0):.3f}"
        )

        print(
            f"    final score          : "
            f"{attack_stage2['final_score']:.3f}"
        )

        print(
            f"    rank                 : "
            f"#{attack_rank_2}"
        )

        print(
            "\n  Trust breakdown:"
        )

        print(
            f"    source score         : "
            f"{source_score:.2f}"
        )

        print(
            f"    age score            : "
            f"{age_score:.2f}"
        )

        print(
            f"    corroboration score  : "
            f"{corroboration_score:.2f}"
        )

        print(
            "\n  Low-trust flag: "
            f"{attack_stage2.get('is_low_trust', False)}"
        )

        print(
            "\n  Attack score comparison:"
        )

        print(
            f"    Retrieval similarity : "
            f"{attack_stage2['similarity']:.3f}"
        )

        print(
            f"    Trust-aware score    : "
            f"{attack_stage2['final_score']:.3f}"
        )

        if attack_rank_2 == 1:

            print(
                "\n  >>> The attack remains ranked #1 "
                "after Defense 2."
            )

        else:

            print(
                "\n  >>> The attack remains in the "
                f"candidate set at rank #{attack_rank_2}."
            )

            print(
                "      Defense 2 changes the ranking using "
                "trust information in addition to similarity."
            )

    else:

        print(
            "\n  Attack document is no longer present "
            "after Defense 2."
        )

    # ------------------------------------------------------------------
    # STAGE 3 — DEFENSE 3
    # ------------------------------------------------------------------

    if stage2:

        passed, blocked = apply_provenance_filter(
            stage2
        )

    else:

        passed = []
        blocked = []

    print_stage_summary(
        "STAGE 3 — Defense 3: Provenance Verification",
        passed,
        blocked,
    )

    if blocked:

        print(
            "\n  Provenance verification results:"
        )

        for result in blocked:

            reasons = result.get(
                "provenance_reasons",
                [],
            )

            print(
                f"    - {result['chunk_id']}"
            )

            if reasons:

                for reason in reasons:

                    print(
                        f"        reason: {reason}"
                    )

            else:

                print(
                    "        reason: provenance verification failed"
                )

    # ------------------------------------------------------------------
    # ATTACK STATUS AFTER D3
    # ------------------------------------------------------------------

    attack_blocked = any(
        result["chunk_id"] == ATTACK_ID
        for result in blocked
    )

    attack_in_final_context = any(
        result["chunk_id"] == ATTACK_ID
        for result in passed
    )

    # ------------------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------------------

    print(
        "\n" + SEP
    )

    print(
        "FINAL RESULT"
    )

    print(
        SEP
    )

    if attack_blocked:

        print(
            "\n  ATTACK BLOCKED"
        )

        print(
            "\n  The poisoned document was removed "
            "before reaching the final context."
        )

        print(
            "\n  Defense 3 rejected it based on provenance "
            "and/or integrity verification."
        )

        print(
            "\n  Layered behavior demonstrated:"
        )

        print(
            "    - D1 checks embedding-space anomalies."
        )

        print(
            "    - D2 incorporates source trust, age "
            "and corroboration."
        )

        print(
            "    - D3 independently verifies provenance "
            "and content integrity."
        )

    elif attack_in_final_context:

        print(
            "\n  WARNING: ATTACK REMAINS IN FINAL CONTEXT"
        )

        print(
            "\n  The poisoned document passed all currently "
            "configured defense checks."
        )

        print(
            "\n  This run should be investigated before "
            "using the demonstration as a successful blocking case."
        )

    else:

        print(
            "\n  ATTACK REMOVED BEFORE FINAL CONTEXT"
        )

        print(
            "\n  The poisoned document was blocked by "
            "an earlier defense layer."
        )

    # ------------------------------------------------------------------
    # FINAL CONTEXT
    # ------------------------------------------------------------------

    print(
        "\nFinal context that would be passed "
        "to LLM generation:"
    )

    print(
        "-" * 58
    )

    if passed:

        for result in passed:

            preview = result["text"].replace(
                "\n",
                " ",
            )

            if len(preview) > 100:

                preview = (
                    preview[:100]
                    + "..."
                )

            print(
                f"  [{result['chunk_id']}] "
                f"{preview}"
            )

    else:

        print(
            "  No chunks remain."
        )

    # ------------------------------------------------------------------
    # FINAL COUNTS
    # ------------------------------------------------------------------

    print(
        "\nPipeline summary:"
    )

    print(
        f"  Initial candidates : {len(candidates)}"
    )

    print(
        f"  After D1           : {len(stage1)}"
    )

    print(
        f"  After D2           : {len(stage2)}"
    )

    print(
        f"  Final context      : {len(passed)}"
    )

    print(
        f"  Attack blocked     : {attack_blocked}"
    )

    # ------------------------------------------------------------------
    # END
    # ------------------------------------------------------------------

    print(
        "\n" + SEP
    )

    print(
        "Demo completed."
    )

    print(
        "\nThis Review-2 demonstration evaluates "
        "retrieval-level defense behavior."
    )

    print(
        "Generation-level attack success will be evaluated "
        "later with the complete RAG pipeline."
    )

    print(
        SEP
    )


if __name__ == "__main__":
    run_demo()

