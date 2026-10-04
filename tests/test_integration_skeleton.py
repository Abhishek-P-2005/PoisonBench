# """
# tests/test_integration_skeleton.py
# =====================================
# NOT runnable until Subash's retriever and Abhishek's attack module exist.
# This is the executable version of INTEGRATION_PLAN.md — every place a real
# component needs to plug in is marked "# TODO: INTEGRATION" with a note on
# exactly what's expected there.

# Purpose: prove, mechanically, that the three module boundaries described in
# INTEGRATION_PLAN.md actually hold — not just that each module works alone.

# Run once the pieces exist:
#     pytest tests/test_integration_skeleton.py -v
# """

# from __future__ import annotations

# import pytest

# from corpus.ingestor import get_chroma_client, get_or_create_collection
# from defenses.defense2_trust_rerank.reranker import rerank_chroma_result
# from defenses.defense3_provenance.filter import filter_chroma_result


# # ── fixtures ──────────────────────────────────────────────────────────────────

# @pytest.fixture(scope="module")
# def chroma_collection():
#     """
#     Assumes `python -m corpus.pipeline` has already been run against a live
#     ChromaDB instance, AND that Abhishek's attack injector has already
#     written poisoned documents into the same collection.

#     # TODO: INTEGRATION — confirm with Abhishek which collection name his
#     # injector writes to. It MUST be the same collection Module 1 ingests
#     # clean data into ("nvd_corpus" by default, see corpus/ingestor.py).
#     """
#     client = get_chroma_client()
#     return get_or_create_collection(client, "nvd_corpus")


# @pytest.fixture(scope="module")
# def trigger_queries():
#     """
#     # TODO: INTEGRATION — replace with Abhishek's actual fixed trigger-query
#     # benchmark (PRD §6.2: "Each attack needs a fixed, reusable trigger
#     # query set so results are comparable across attacks and defenses").
#     # Import it directly from his module rather than hardcoding here, e.g.:
#     #     from attacks.trigger_queries import TRIGGER_QUERIES
#     """
#     return [
#         "How is the ExampleCorp WebApp SQL injection vulnerability mitigated?",
#     ]


# # ── contract test 1: collection contains both clean and poisoned docs ────────

# def test_collection_has_poisoned_documents(chroma_collection):
#     """
#     Verifies Abhishek's injector actually wrote into the shared collection
#     with the agreed metadata schema, not a separate collection or a
#     different field set.
#     """
#     count = chroma_collection.count()
#     assert count > 0, "Collection is empty — has corpus.pipeline been run?"

#     # Sample a batch and check at least one poisoned doc exists with the
#     # expected metadata fields intact.
#     sample = chroma_collection.get(limit=min(count, 500), include=["metadatas"])
#     metadatas = sample.get("metadatas", [])

#     poisoned = [m for m in metadatas if m.get("is_poisoned") is True]
#     assert poisoned, (
#         "No documents with is_poisoned=True found in a sample of the "
#         "collection. Either no attack has been injected yet, or the "
#         "injector is not setting is_poisoned in its metadata — check "
#         "with Abhishek that his injector uses corpus.metadata.build_metadata() "
#         "or an equivalent that sets this field."
#     )

#     # Every poisoned doc must still carry the full required field set —
#     # this is the actual schema-compatibility check.
#     required_fields = {
#         "chunk_id", "doc_id", "source_type", "source_id", "verified",
#         "published_date", "modified_date", "chunk_hash",
#     }
#     for meta in poisoned:
#         missing = required_fields - set(meta.keys())
#         assert not missing, f"Poisoned doc metadata missing fields: {missing}"


# # ── contract test 2: real retrieval output flows through Defense 2/3 untouched ─

# def test_real_retrieval_passes_through_defense_pipeline(chroma_collection, trigger_queries):
#     """
#     Takes REAL ChromaDB query output (not synthetic dicts) and runs it
#     through the exact same Defense 2 -> Defense 3 functions used in
#     demo_review3.py, with zero adapter code required beyond what already
#     exists in reranker.py / filter.py.

#     # TODO: INTEGRATION — once Subash's FastAPI retrieval endpoint exists,
#     # prefer calling it directly instead of querying ChromaDB here, so this
#     # test also exercises his API layer:
#     #     response = requests.get(f"{BACKEND_URL}/retrieve", params={"q": query, "k": 5})
#     #     chroma_result = response.json()["raw_chroma_result"]  # or equivalent
#     """
#     query = trigger_queries[0]

#     # This IS Subash's retrieval mechanism (top-k cosine similarity) —
#     # calling it directly here until his API wrapper exists.
#     raw_result = chroma_collection.query(query_texts=[query], n_results=10)

#     stage1 = rerank_chroma_result(raw_result)
#     assert len(stage1) > 0, "Defense 2 produced no output from real retrieval results"
#     assert all("final_score" in r for r in stage1), "Defense 2 output missing final_score"

#     passed, blocked = filter_chroma_result(raw_result)
#     assert len(passed) + len(blocked) == len(stage1) - len(stage1) + len(raw_result["ids"][0]) or True
#     # ^ the exact count relationship depends on whether Defense 3 runs on
#     # Defense 2's output or raw retrieval directly — pin this down once
#     # the real pipeline wiring is decided (see note below).

#     # TODO: INTEGRATION — decide and assert the actual pipeline wiring:
#     # should Defense 3 consume Defense 2's re-ranked list (recommended,
#     # matches PRD §5 ordering) or run independently on raw retrieval?
#     # Once decided, replace the loose assertion above with a precise one,
#     # e.g.:
#     #     passed, blocked = apply_provenance_filter(stage1)
#     #     assert len(passed) + len(blocked) == len(stage1)


# # ── contract test 3: end-to-end attack scenario matches the demo pattern ─────

# def test_end_to_end_attack_is_caught(chroma_collection, trigger_queries):
#     """
#     The real-data equivalent of demo_review3.py's synthetic scenario:
#     for at least one trigger query, if a poisoned document is retrieved in
#     the raw top-k, it should be demoted and/or blocked by the time it exits
#     the Defense 2 -> Defense 3 pipeline.

#     # TODO: INTEGRATION — this test can only pass once Abhishek's attacks
#     # are actually injected AND at least one trigger query is known to
#     # retrieve a poisoned document in raw top-k. Coordinate with him on
#     # which (query, attack_type, poisoning_ratio) combination to use as
#     # the canonical integration test case.
#     """
#     query = trigger_queries[0]
#     raw_result = chroma_collection.query(query_texts=[query], n_results=10)

#     raw_ids = raw_result["ids"][0]
#     raw_metadatas = raw_result["metadatas"][0]
#     poisoned_in_raw = any(m.get("is_poisoned") for m in raw_metadatas)

#     if not poisoned_in_raw:
#         pytest.skip(
#             f"No poisoned document in raw top-10 for query '{query}' — "
#             f"choose a trigger query known to surface an injected attack, "
#             f"or increase poisoning ratio for this integration run."
#         )

#     stage1 = rerank_chroma_result(raw_result)
#     passed, blocked = filter_chroma_result(raw_result)  # see wiring TODO above

#     blocked_ids = {r["chunk_id"] for r in blocked}
#     poisoned_ids = {
#         chunk_id for chunk_id, meta in zip(raw_ids, raw_metadatas)
#         if meta.get("is_poisoned")
#     }

#     caught = poisoned_ids & blocked_ids
#     assert caught, (
#         f"Poisoned document(s) {poisoned_ids} were retrieved but NONE were "
#         f"blocked by Defense 3. Either the attack evaded both defenses "
#         f"(a genuine, reportable result — check attack_type before treating "
#         f"this as a bug) or the metadata contract has drifted."
#     )






"""
tests/test_integration_skeleton.py
==================================

PoisonBench — Defense Integration Tests

Purpose:
    Verify that real ChromaDB retrieval output can flow through the
    implemented Defense 2 -> Defense 3 pipeline.

Expected pipeline:

    ChromaDB Retrieval
            |
            v
    Defense 2 — Trust-Scored Re-ranking
            |
            v
    Defense 3 — Provenance Verification
            |
            v
    Final Context

Prerequisites:
    1. ChromaDB must be available.
    2. The NVD corpus must already be ingested.
    3. At least one poisoned document must be present in the collection
       for the attack-specific test.

Run:

    pytest tests/test_integration_skeleton.py -v

Important:
    These tests exercise the real ChromaDB adapter path.

    They are different from the synthetic demo in:

        defenses/demo_review3.py

    The demo demonstrates the defense logic using the controlled
    Review-2 scenario, while these tests verify that real ChromaDB
    retrieval output can cross the defense module boundaries.
"""

from __future__ import annotations

import pytest

from corpus.ingestor import (
    get_chroma_client,
    get_or_create_collection,
)

from defenses.defense2_trust_rerank.reranker import (
    rerank_chroma_result,
)

from defenses.defense3_provenance.filter import (
    apply_provenance_filter,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def chroma_collection():
    """
    Connect to the shared NVD ChromaDB collection.

    The expected collection name is:

        nvd_corpus

    This is the same collection used by the corpus ingestion pipeline.
    """

    try:
        client = get_chroma_client()

        collection = get_or_create_collection(
            client,
            "nvd_corpus",
        )

    except Exception as exc:
        pytest.fail(
            "Could not connect to the ChromaDB collection 'nvd_corpus'. "
            "Make sure ChromaDB is running and the corpus has been "
            f"ingested before running integration tests.\nError: {exc}"
        )

    return collection


@pytest.fixture(scope="module")
def trigger_queries():
    """
    Fixed trigger-query set used for the integration tests.

    In the final project this can be replaced by the attack module's
    reusable benchmark query set.
    """

    return [
        (
            "How is the ExampleCorp WebApp SQL injection "
            "vulnerability mitigated?"
        ),
    ]


# ============================================================================
# Helper functions
# ============================================================================

def get_raw_retrieval(
    chroma_collection,
    query: str,
    n_results: int = 10,
) -> dict:
    """
    Execute a real ChromaDB similarity search.
    """

    return chroma_collection.query(
        query_texts=[query],
        n_results=n_results,
    )


def get_retrieved_metadata(
    raw_result: dict,
) -> list[dict]:
    """
    Safely extract metadata records from a ChromaDB result.
    """

    metadatas = raw_result.get("metadatas", [])

    if not metadatas:
        return []

    return metadatas[0] or []


def get_retrieved_ids(
    raw_result: dict,
) -> list[str]:
    """
    Safely extract document IDs from a ChromaDB result.
    """

    ids = raw_result.get("ids", [])

    if not ids:
        return []

    return ids[0] or []


# ============================================================================
# Contract Test 1
# ============================================================================

def test_collection_has_documents(
    chroma_collection,
):
    """
    Verify that the shared ChromaDB collection contains documents.

    This confirms that the corpus ingestion stage has produced data
    that can be consumed by the defense pipeline.
    """

    count = chroma_collection.count()

    assert count > 0, (
        "Collection 'nvd_corpus' is empty. "
        "Run the corpus ingestion pipeline before running "
        "the integration tests."
    )


# ============================================================================
# Contract Test 2
# ============================================================================

def test_collection_metadata_schema(
    chroma_collection,
):
    """
    Verify that retrieved documents contain the metadata fields
    required by the trust and provenance defenses.
    """

    count = chroma_collection.count()

    assert count > 0, (
        "Collection is empty. Cannot verify metadata schema."
    )

    sample_size = min(
        count,
        500,
    )

    sample = chroma_collection.get(
        limit=sample_size,
        include=["metadatas"],
    )

    metadatas = sample.get(
        "metadatas",
        [],
    )

    assert metadatas, (
        "No metadata was returned from the ChromaDB collection."
    )

    required_fields = {
        "chunk_id",
        "doc_id",
        "source_type",
        "source_id",
        "verified",
        "published_date",
        "modified_date",
        "chunk_hash",
    }

    documents_checked = 0

    for metadata in metadatas:

        if not metadata:
            continue

        documents_checked += 1

        missing = (
            required_fields
            - set(metadata.keys())
        )

        assert not missing, (
            "Document metadata is missing required fields: "
            f"{missing}\n"
            f"Metadata received: {metadata}"
        )

    assert documents_checked > 0, (
        "No valid metadata records were found in the collection."
    )


# ============================================================================
# Contract Test 3
# ============================================================================

def test_real_retrieval_passes_through_defense_pipeline(
    chroma_collection,
    trigger_queries,
):
    """
    Verify that real ChromaDB retrieval output can pass through:

        ChromaDB
           ->
        Defense 2
           ->
        Defense 3

    without requiring a separate synthetic adapter.
    """

    query = trigger_queries[0]

    raw_result = get_raw_retrieval(
        chroma_collection,
        query,
        n_results=10,
    )

    raw_ids = get_retrieved_ids(
        raw_result
    )

    assert raw_ids, (
        f"ChromaDB returned no documents for query: '{query}'"
    )

    # ------------------------------------------------------------------
    # Defense 2
    # ------------------------------------------------------------------

    stage2 = rerank_chroma_result(
        raw_result
    )

    assert stage2, (
        "Defense 2 produced no output from the real "
        "ChromaDB retrieval result."
    )

    assert all(
        "final_score" in result
        for result in stage2
    ), (
        "Defense 2 output is missing 'final_score' "
        "from one or more results."
    )

    # ------------------------------------------------------------------
    # Defense 3
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Defense 3 consumes the output of Defense 2.
    # This matches the intended layered pipeline:
    #
    #       Retrieval -> D2 -> D3
    #
    # rather than running D3 independently on the raw retrieval.
    # ------------------------------------------------------------------

    passed, blocked = apply_provenance_filter(
        stage2
    )

    total_after_d3 = (
        len(passed)
        + len(blocked)
    )

    assert total_after_d3 == len(stage2), (
        "Defense 3 did not preserve the expected result count.\n"
        f"Defense 2 results : {len(stage2)}\n"
        f"D3 passed         : {len(passed)}\n"
        f"D3 blocked        : {len(blocked)}"
    )

    # Every result must have enough information to identify it.
    assert all(
        "chunk_id" in result
        for result in passed + blocked
    ), (
        "Defense 3 output contains a result without 'chunk_id'."
    )


# ============================================================================
# Contract Test 4
# ============================================================================

def test_end_to_end_attack_is_caught(
    chroma_collection,
    trigger_queries,
):
    """
    Verify the real-data attack path.

    If the fixed trigger query retrieves a poisoned document in the
    raw ChromaDB top-k result, that document must be blocked by the
    final Defense 3 stage.

    If no poisoned document is retrieved, the test is skipped because
    there is no attack instance available to evaluate for this query.
    """

    query = trigger_queries[0]

    raw_result = get_raw_retrieval(
        chroma_collection,
        query,
        n_results=10,
    )

    raw_ids = get_retrieved_ids(
        raw_result
    )

    raw_metadatas = get_retrieved_metadata(
        raw_result
    )

    assert len(raw_ids) == len(raw_metadatas), (
        "ChromaDB returned a different number of IDs and metadata "
        "records."
    )

    # ------------------------------------------------------------------
    # Identify poisoned documents in raw retrieval.
    # ------------------------------------------------------------------

    poisoned_ids = {
        chunk_id
        for chunk_id, metadata in zip(
            raw_ids,
            raw_metadatas,
        )
        if metadata
        and metadata.get("is_poisoned") is True
    }

    if not poisoned_ids:
        pytest.skip(
            "No poisoned document was retrieved in the raw top-10 "
            f"for query '{query}'. "
            "Use a trigger query known to surface an injected attack "
            "before treating this as a failed attack-defense test."
        )

    # ------------------------------------------------------------------
    # Defense 2
    # ------------------------------------------------------------------

    stage2 = rerank_chroma_result(
        raw_result
    )

    assert stage2, (
        "Defense 2 returned no documents even though the raw "
        "retrieval contained a poisoned document."
    )

    # ------------------------------------------------------------------
    # Defense 3
    # ------------------------------------------------------------------

    passed, blocked = apply_provenance_filter(
        stage2
    )

    blocked_ids = {
        result["chunk_id"]
        for result in blocked
        if "chunk_id" in result
    }

    passed_ids = {
        result["chunk_id"]
        for result in passed
        if "chunk_id" in result
    }

    caught = (
        poisoned_ids
        & blocked_ids
    )

    # ------------------------------------------------------------------
    # Final assertion
    # ------------------------------------------------------------------

    assert caught, (
        "A poisoned document was retrieved but Defense 3 did not "
        "block it.\n\n"
        f"Poisoned IDs retrieved : {poisoned_ids}\n"
        f"Passed IDs             : {passed_ids}\n"
        f"Blocked IDs            : {blocked_ids}\n\n"
        "Check the poisoned document's source_type, source_id, "
        "verified flag and chunk_hash. If the attack legitimately "
        "evades provenance verification, record it as a defense "
        "limitation rather than silently weakening the test."
    )


# ============================================================================
# Optional reporting test
# ============================================================================

def test_pipeline_preserves_expected_flow(
    chroma_collection,
    trigger_queries,
):
    """
    Simple pipeline accounting check.

    Verifies:

        Raw retrieval
             |
             v
        Defense 2
             |
             v
        Defense 3
             |
             v
        Passed + Blocked

    This test does not require a poisoned document.
    """

    query = trigger_queries[0]

    raw_result = get_raw_retrieval(
        chroma_collection,
        query,
        n_results=10,
    )

    raw_count = len(
        get_retrieved_ids(raw_result)
    )

    assert raw_count > 0, (
        "Raw retrieval returned no documents."
    )

    stage2 = rerank_chroma_result(
        raw_result
    )

    assert len(stage2) == raw_count, (
        "Defense 2 unexpectedly changed the number of "
        "retrieved documents.\n"
        f"Raw retrieval : {raw_count}\n"
        f"After D2      : {len(stage2)}"
    )

    passed, blocked = apply_provenance_filter(
        stage2
    )

    assert (
        len(passed)
        + len(blocked)
        == len(stage2)
    ), (
        "Defense 3 result accounting is inconsistent.\n"
        f"After D2 : {len(stage2)}\n"
        f"Passed   : {len(passed)}\n"
        f"Blocked  : {len(blocked)}"
    )