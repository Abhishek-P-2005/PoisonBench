"""
defense2_trust_rerank/trust_scorer.py
---------------------------------------
Computes a trust score in [0, 1] for a single document chunk, based on three
independent signals:

  1. Source reliability  — is the claimed source on the trusted list?
  2. Document age         — has it had time to be scrutinized / corroborated?
  3. Corroboration        — do other verified chunks reference the same doc_id?

Design note
-----------
None of these signals use `is_poisoned` or `attack_type` from ChunkMetadata.
Those fields are ground-truth labels set by the attack module for evaluation
purposes only. A real defense cannot see them at inference time, and using
them here would silently make Defense 2's numbers meaningless — it would be
scoring using the answer key. Trust is computed only from information a
defense would legitimately have: claimed source, timestamps, and corpus
co-occurrence.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from defenses.defense2_trust_rerank.config import (
    AGE_FULL_TRUST_DAYS,
    AGE_MIN_TRUST_SCORE,
    CORROBORATION_SATURATION_COUNT,
    DEFAULT_TRUST_WEIGHTS,
    TrustWeights,
    UNVERIFIED_SOURCE_SCORE,
    VERIFIED_SOURCE_SCORE,
)

log = logging.getLogger(__name__)


@dataclass
class TrustBreakdown:
    """Component scores behind a final trust score — kept for logging/debugging."""
    source_score:        float
    age_score:           float
    corroboration_score: float
    final_score:         float


class TrustScorer:
    """
    Stateful scorer: holds a corroboration index (doc_id -> count of verified
    chunks) built once per batch/collection, so scoring many chunks doesn't
    require a fresh corpus scan per chunk.
    """

    def __init__(self, weights: TrustWeights = DEFAULT_TRUST_WEIGHTS):
        weights.validate()
        self.weights = weights
        self._corroboration_index: Counter[str] = Counter()
        self._index_built = False

    # ── corroboration index ────────────────────────────────────────────────

    def build_corroboration_index(self, all_metadata: list[dict]) -> None:
        """
        Build a doc_id -> count-of-verified-chunks index from a full batch
        of chunk metadata (e.g. everything currently in a ChromaDB collection,
        or everything in the current retrieval result set — see reranker.py
        for which scope is used in practice).

        Call this once per batch before scoring individual chunks with
        `score()`, so corroboration counts reflect the whole batch rather
        than being computed independently (and redundantly) per chunk.
        """
        self._corroboration_index.clear()
        for meta in all_metadata:
            if meta.get("verified", False):
                self._corroboration_index[meta["doc_id"]] += 1
        self._index_built = True
        log.debug(
            "Corroboration index built: %d unique doc_ids",
            len(self._corroboration_index),
        )

    def _corroboration_count_for(self, doc_id: str, self_verified: bool) -> int:
        """
        Count of OTHER verified chunks (excluding this chunk itself) that
        reference the same doc_id.
        """
        count = self._corroboration_index.get(doc_id, 0)
        if self_verified:
            count -= 1   # don't let a chunk corroborate itself
        return max(count, 0)

    # ── component scores ──────────────────────────────────────────────────

    @staticmethod
    def _source_score(verified: bool) -> float:
        return VERIFIED_SOURCE_SCORE if verified else UNVERIFIED_SOURCE_SCORE

    @staticmethod
    def _age_score(published_date: str, modified_date: str) -> float:
        """
        Linear ramp from AGE_MIN_TRUST_SCORE (age = 0 days) to 1.0
        (age >= AGE_FULL_TRUST_DAYS). Uses the more recent of published /
        modified date, since a recently-modified old CVE is effectively
        "fresh" content again.
        """
        try:
            pub_dt = datetime.strptime(published_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pub_dt = None
        try:
            mod_dt = datetime.strptime(modified_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            mod_dt = None

        candidates = [d for d in (pub_dt, mod_dt) if d is not None]
        if not candidates:
            # No usable date — neutral score, don't punish or reward.
            return (1.0 + AGE_MIN_TRUST_SCORE) / 2

        most_recent = max(candidates)
        age_days = (datetime.now(timezone.utc) - most_recent).days

        age_days = max(age_days, 0)

        if age_days >= AGE_FULL_TRUST_DAYS:
            return 1.0

        # linear interpolation between (0 days -> AGE_MIN_TRUST_SCORE)
        # and (AGE_FULL_TRUST_DAYS -> 1.0)
        fraction = age_days / AGE_FULL_TRUST_DAYS
        return AGE_MIN_TRUST_SCORE + fraction * (1.0 - AGE_MIN_TRUST_SCORE)

    def _corroboration_score(self, doc_id: str, self_verified: bool) -> float:
        if not self._index_built:
            log.warning(
                "Corroboration index not built — call build_corroboration_index() "
                "first. Defaulting to 0 corroboration for doc_id=%s",
                doc_id,
            )
            return 0.0
        count = self._corroboration_count_for(doc_id, self_verified)
        return min(count / CORROBORATION_SATURATION_COUNT, 1.0)

    # ── public scoring entry point ────────────────────────────────────────

    def score(self, metadata: dict) -> TrustBreakdown:
        """
        Compute the trust score for a single chunk given its ChromaDB
        metadata dict (as produced by ChunkMetadata.to_chroma_dict()).

        Parameters
        ----------
        metadata : dict with at least the keys:
            doc_id, verified, published_date, modified_date

        Returns
        -------
        TrustBreakdown with component scores and the final weighted trust score.
        """
        doc_id     = metadata.get("doc_id", "UNKNOWN")
        verified   = bool(metadata.get("verified", False))
        published  = metadata.get("published_date", "")
        modified   = metadata.get("modified_date", "")

        source_score        = self._source_score(verified)
        age_score            = self._age_score(published, modified)
        corroboration_score  = self._corroboration_score(doc_id, verified)

        final_score = (
            self.weights.source        * source_score
            + self.weights.age         * age_score
            + self.weights.corroboration * corroboration_score
        )
        final_score = max(0.0, min(1.0, final_score))   # clamp defensively

        return TrustBreakdown(
            source_score        = source_score,
            age_score            = age_score,
            corroboration_score  = corroboration_score,
            final_score          = final_score,
        )

    def score_batch(self, metadata_list: list[dict]) -> list[TrustBreakdown]:
        """
        Convenience: build the corroboration index from this batch, then
        score every chunk in it. Use this when re-ranking a single
        retrieval result set (top-k documents) so corroboration reflects
        "how many of the documents retrieved for this query support each
        other" rather than the whole corpus — which is the correct scope
        for a per-query re-ranking defense.
        """
        self.build_corroboration_index(metadata_list)
        return [self.score(meta) for meta in metadata_list]
