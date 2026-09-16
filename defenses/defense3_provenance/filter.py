"""
defense3_provenance/filter.py
--------------------------------
Applies ProvenanceVerifier across a full retrieval result set (the output
of Defense 2's re-ranking, per the pipeline order in PRD Section 5), removing
hard-blocked documents entirely and flagging soft-failures for downstream
use (logging, Defense 4 features).

Result format expected — same shape as defense2_trust_rerank.reranker:
    {
        "chunk_id":  str,
        "text":      str,
        "similarity": float,
        "metadata":  dict,
        ... (any fields added by earlier defenses, e.g. trust_score)
    }
"""

from __future__ import annotations

import logging

from defenses.defense3_provenance.provenance_verifier import ProvenanceVerifier

log = logging.getLogger(__name__)


def apply_provenance_filter(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Run provenance verification across every result in the set.

    Parameters
    ----------
    results : retrieval results (post Defense 1 / Defense 2 if applicable)

    Returns
    -------
    (passed, blocked)
        passed  : results that are NOT hard-blocked, each augmented with
                  provenance_score, provenance_flagged, provenance_reasons
        blocked : results that WERE hard-blocked, same augmentation, kept
                  separately so the evaluation harness can log them as
                  defense catches rather than silently dropping them
    """
    if not results:
        return [], []

    verifier = ProvenanceVerifier()
    passed, blocked = [], []

    for result in results:
        verdict = verifier.verify(result["text"], result["metadata"])

        augmented = {
            **result,
            "provenance_score":    verdict.provenance_score,
            "provenance_flagged":  verdict.flagged,
            "provenance_blocked":  verdict.blocked,
            "provenance_reasons":  verdict.reasons,
            "provenance_failed_checks": verdict.failed_checks,
        }

        if verdict.blocked:
            blocked.append(augmented)
        else:
            passed.append(augmented)

    if blocked:
        log.info(
            "Defense 3: blocked %d/%d retrieved chunks on provenance grounds",
            len(blocked), len(results),
        )

    return passed, blocked


# ── ChromaDB adapter (mirrors defense2_trust_rerank.reranker) ─────────────────

def from_chroma_query_result(chroma_result: dict, query_index: int = 0) -> list[dict]:
    """
    Convert a raw ChromaDB `.query()` response into the format
    `apply_provenance_filter()` expects. Duplicated (not imported) from
    defense2_trust_rerank.reranker so Defense 3 can run standalone without
    requiring Defense 2 to be present — useful for isolated evaluation runs
    that test provenance verification alone (see PRD ablation sweep design).
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


def filter_chroma_result(
    chroma_result: dict,
    query_index: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Convenience one-shot: adapt a raw ChromaDB result and filter it."""
    results = from_chroma_query_result(chroma_result, query_index=query_index)
    return apply_provenance_filter(results)
