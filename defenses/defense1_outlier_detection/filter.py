"""
defense1_outlier_detection/filter.py
---------------------------------------
Applies OutlierDetector across a retrieval result batch. This is the first
stage of the defense pipeline (PRD §5 ordering: outlier detection first,
as the cheapest filter, before trust re-ranking and provenance checks).

Result format expected
------------------------
Same shape as Defense 2 / Defense 3, with one additional required field:
    {
        "chunk_id":  str,
        "text":      str,
        "similarity": float,
        "metadata":  dict,
        "embedding": list[float],   # <-- required for this defense specifically
    }

Embeddings are not part of ChunkMetadata (corpus/metadata.py) because
ChromaDB stores them separately from document metadata. Callers must
request embeddings explicitly from ChromaDB (include=["embeddings", ...])
— see `from_chroma_query_result()` below.
"""

from __future__ import annotations

import logging

from defenses.defense1_outlier_detection.outlier_detector import OutlierDetector

log = logging.getLogger(__name__)


def apply_outlier_filter(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Run outlier detection across every result in the batch.

    Parameters
    ----------
    results : retrieval results, each with an "embedding" key

    Returns
    -------
    (passed, blocked)
        passed  : results NOT hard-blocked, each augmented with
                  outlier_zscore, outlier_distance, outlier_flagged,
                  outlier_score
        blocked : results that WERE hard-blocked, same augmentation, kept
                  separately so the evaluation harness can log them as
                  catches rather than silently dropping them
    """
    if not results:
        return [], []

    missing_embeddings = [r["chunk_id"] for r in results if "embedding" not in r]
    if missing_embeddings:
        raise ValueError(
            f"apply_outlier_filter requires an 'embedding' field on every "
            f"result. Missing for: {missing_embeddings}. If calling this "
            f"from a ChromaDB query, make sure you passed "
            f"include=['embeddings', 'documents', 'metadatas', 'distances']."
        )

    detector = OutlierDetector()
    chunk_ids = [r["chunk_id"] for r in results]
    embeddings = [r["embedding"] for r in results]
    verdicts = detector.score_batch(chunk_ids, embeddings)

    passed, blocked = [], []
    for result, verdict in zip(results, verdicts):
        augmented = {
            **result,
            "outlier_zscore":    verdict.zscore,
            "outlier_distance":  verdict.distance_from_centroid,
            "outlier_flagged":   verdict.is_outlier_flagged,
            "outlier_blocked":   verdict.is_outlier_blocked,
            "outlier_reason":    verdict.reason,
            "outlier_score":     verdict.outlier_score,
        }
        if verdict.is_outlier_blocked:
            blocked.append(augmented)
        else:
            passed.append(augmented)

    if blocked:
        log.info(
            "Defense 1: blocked %d/%d retrieved chunks as embedding outliers",
            len(blocked), len(results),
        )

    return passed, blocked


# ── ChromaDB adapter ───────────────────────────────────────────────────────────

def from_chroma_query_result(chroma_result: dict, query_index: int = 0) -> list[dict]:
    """
    Convert a raw ChromaDB `.query()` response into the format
    `apply_outlier_filter()` expects. Requires the query to have been made
    with `include=["embeddings", "documents", "metadatas", "distances"]` —
    ChromaDB does NOT return embeddings by default, unlike documents and
    metadatas.
    """
    ids        = chroma_result.get("ids", [[]])[query_index]
    documents  = chroma_result.get("documents", [[]])[query_index]
    metadatas  = chroma_result.get("metadatas", [[]])[query_index]
    distances  = chroma_result.get("distances", [[]])[query_index]
    embeddings = chroma_result.get("embeddings", [[]])[query_index]

    if not embeddings:
        raise ValueError(
            "ChromaDB result has no embeddings. Re-run the query with "
            "include=['embeddings', 'documents', 'metadatas', 'distances']."
        )

    results = []
    for chunk_id, text, metadata, distance, embedding in zip(
        ids, documents, metadatas, distances, embeddings
    ):
        results.append({
            "chunk_id":   chunk_id,
            "text":       text,
            "similarity": 1.0 - distance,
            "metadata":   metadata,
            "embedding":  list(embedding),
        })
    return results


def filter_chroma_result(
    chroma_result: dict,
    query_index: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Convenience one-shot: adapt a raw ChromaDB result and filter it."""
    results = from_chroma_query_result(chroma_result, query_index=query_index)
    return apply_outlier_filter(results)
