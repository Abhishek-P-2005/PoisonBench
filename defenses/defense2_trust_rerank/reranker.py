"""
defense2_trust_rerank/reranker.py
------------------------------------
Takes a top-k retrieval result set (from the baseline cosine-similarity
retriever) and re-ranks it using trust scores computed by TrustScorer.

Retrieval result format expected
---------------------------------
A list of dicts, one per retrieved chunk:
    {
        "chunk_id":  str,
        "text":      str,
        "similarity": float,   # cosine similarity, higher = more relevant
        "metadata":  dict,     # ChunkMetadata.to_chroma_dict() output
    }

This is deliberately decoupled from ChromaDB's raw response shape (which
returns "distances", not similarities, and nested lists-of-lists for
multi-query calls) — see `from_chroma_query_result()` below for the
adapter that converts a raw ChromaDB response into this format.
"""

from __future__ import annotations

import logging
from typing import Any

from defenses.defense2_trust_rerank.config import (
    DEFAULT_TRUST_WEIGHTS,
    LOW_TRUST_FLAG_THRESHOLD,
    RERANK_ALPHA,
    TrustWeights,
)
from defenses.defense2_trust_rerank.trust_scorer import TrustScorer

log = logging.getLogger(__name__)


def _normalize_similarities(results: list[dict]) -> list[float]:
    """
    Min-max normalize similarity scores within this result set to [0, 1].
    Necessary because raw cosine similarity from different embedding spaces
    or distance metrics may not already be bounded to [0, 1], and blending
    with a [0, 1] trust score requires comparable scales.
    """
    sims = [r["similarity"] for r in results]
    if not sims:
        return []
    lo, hi = min(sims), max(sims)
    if hi - lo < 1e-9:
        # All identical — avoid divide-by-zero, treat as uniformly relevant.
        return [1.0 for _ in sims]
    return [(s - lo) / (hi - lo) for s in sims]


def rerank_results(
    results: list[dict],
    *,
    alpha: float = RERANK_ALPHA,
    weights: TrustWeights = DEFAULT_TRUST_WEIGHTS,
    flag_threshold: float = LOW_TRUST_FLAG_THRESHOLD,
) -> list[dict]:
    """
    Re-rank a retrieval result set by blending normalized similarity with
    trust score.

    final_score = alpha * normalized_similarity + (1 - alpha) * trust_score

    Parameters
    ----------
    results        : retrieval results in the format described above
    alpha           : blend weight favoring similarity vs trust (see config.py)
    weights         : TrustWeights for source/age/corroboration components
    flag_threshold  : trust scores below this get is_low_trust=True

    Returns
    -------
    A new list of dicts, sorted descending by final_score, each augmented with:
        trust_score       : float
        trust_breakdown    : dict (source_score, age_score, corroboration_score)
        similarity_norm    : float (normalized similarity used in the blend)
        final_score        : float
        is_low_trust       : bool (for defense_flags logging)
    """
    if not results:
        return []

    scorer = TrustScorer(weights=weights)
    metadata_list = [r["metadata"] for r in results]
    breakdowns    = scorer.score_batch(metadata_list)
    sims_norm     = _normalize_similarities(results)

    augmented = []
    for result, breakdown, sim_norm in zip(results, breakdowns, sims_norm):
        final_score = alpha * sim_norm + (1 - alpha) * breakdown.final_score
        augmented.append({
            **result,
            "trust_score": breakdown.final_score,
            "trust_breakdown": {
                "source_score":       breakdown.source_score,
                "age_score":          breakdown.age_score,
                "corroboration_score": breakdown.corroboration_score,
            },
            "similarity_norm": sim_norm,
            "final_score": final_score,
            "is_low_trust": breakdown.final_score < flag_threshold,
        })

    augmented.sort(key=lambda r: r["final_score"], reverse=True)

    n_flagged = sum(1 for r in augmented if r["is_low_trust"])
    if n_flagged:
        log.info(
            "Defense 2: flagged %d/%d retrieved chunks as low-trust (threshold=%.2f)",
            n_flagged, len(augmented), flag_threshold,
        )

    return augmented


# ── ChromaDB adapter ───────────────────────────────────────────────────────────

def from_chroma_query_result(chroma_result: dict, query_index: int = 0) -> list[dict]:
    """
    Convert a raw ChromaDB `.query()` response into the result format
    `rerank_results()` expects.

    ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite), not
    similarity. We convert via similarity = 1 - distance, which is the
    standard mapping when the collection was created with
    metadata={"hnsw:space": "cosine"} (see corpus/ingestor.py).

    Parameters
    ----------
    chroma_result : the dict returned by collection.query(...)
    query_index   : which query's results to extract, if query_texts had
                    more than one entry (default: the first/only query)
    """
    ids        = chroma_result.get("ids", [[]])[query_index]
    documents  = chroma_result.get("documents", [[]])[query_index]
    metadatas  = chroma_result.get("metadatas", [[]])[query_index]
    distances  = chroma_result.get("distances", [[]])[query_index]

    results = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        results.append({
            "chunk_id":   chunk_id,
            "text":       text,
            "similarity": 1.0 - distance,
            "metadata":   metadata,
        })
    return results


def rerank_chroma_result(
    chroma_result: dict,
    *,
    query_index: int = 0,
    alpha: float = RERANK_ALPHA,
    weights: TrustWeights = DEFAULT_TRUST_WEIGHTS,
    flag_threshold: float = LOW_TRUST_FLAG_THRESHOLD,
) -> list[dict]:
    """
    Convenience one-shot: adapt a raw ChromaDB query result and re-rank it
    in a single call. This is the function most callers (e.g. the FastAPI
    retrieval endpoint, or the evaluation harness) should use directly.
    """
    results = from_chroma_query_result(chroma_result, query_index=query_index)
    return rerank_results(
        results,
        alpha=alpha,
        weights=weights,
        flag_threshold=flag_threshold,
    )
