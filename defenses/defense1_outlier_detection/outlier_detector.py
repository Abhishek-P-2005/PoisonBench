"""
defense1_outlier_detection/outlier_detector.py
--------------------------------------------------
Computes an outlier verdict for each chunk in a retrieval batch, based on
how far its embedding sits from the batch centroid, expressed as a z-score
of cosine distance.

Why z-score of distance, not raw distance
-------------------------------------------
Raw cosine distance from a centroid has no fixed "this is too far" cutoff —
it depends entirely on how tightly the legitimate documents in this
particular batch cluster together (a batch of near-duplicate CVE chunks
clusters far tighter than a batch of five unrelated CVEs about different
vulnerability classes). Z-score normalizes for that: it asks "how unusual is
this distance RELATIVE TO the spread of distances in this specific batch?"
rather than applying one global threshold everywhere.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from defenses.defense1_outlier_detection.config import (
    HARD_BLOCK_ZSCORE,
    MIN_NEIGHBORHOOD_SIZE,
    MIN_STD_FLOOR,
    SOFT_FLAG_ZSCORE,
)

log = logging.getLogger(__name__)


@dataclass
class OutlierVerdict:
    chunk_id:                str
    distance_from_centroid:  float
    zscore:                  float
    is_outlier_flagged:      bool
    is_outlier_blocked:      bool
    reason:                  str
    outlier_score:           float   # normalized 0-1, for Defense 4 features


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine_similarity. 0 = identical direction, up to 2 = opposite."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0   # degenerate vector — treat as maximally uninformative
    cosine_sim = dot / (norm_a * norm_b)
    cosine_sim = max(-1.0, min(1.0, cosine_sim))   # clamp for float error
    return 1.0 - cosine_sim


def _centroid(embeddings: list[list[float]]) -> list[float]:
    dim = len(embeddings[0])
    sums = [0.0] * dim
    for emb in embeddings:
        for i, val in enumerate(emb):
            sums[i] += val
    n = len(embeddings)
    return [s / n for s in sums]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: list[float], mean_val: float) -> float:
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


class OutlierDetector:
    """Stateless — safe to reuse across batches and threads."""

    def __init__(
        self,
        soft_zscore: float = SOFT_FLAG_ZSCORE,
        hard_zscore: float = HARD_BLOCK_ZSCORE,
        min_neighborhood: int = MIN_NEIGHBORHOOD_SIZE,
    ):
        self.soft_zscore = soft_zscore
        self.hard_zscore = hard_zscore
        self.min_neighborhood = min_neighborhood

    def score_batch(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
    ) -> list[OutlierVerdict]:
        """
        Score every chunk using a LEAVE-ONE-OUT centroid: for chunk i, the
        reference centroid and distance distribution are computed from
        every OTHER chunk in the batch, excluding chunk i itself.

        This matters: if the centroid were computed over the whole batch
        including the candidate being scored, a single anomalous embedding
        drags the centroid slightly toward itself and dilutes its own
        z-score — the outlier partially hides itself just by being counted
        in its own reference statistics. Leave-one-out avoids this.

        Parameters
        ----------
        chunk_ids  : chunk identifiers, same order as embeddings
        embeddings : one embedding vector per chunk, same order as chunk_ids

        Returns
        -------
        One OutlierVerdict per chunk, same order as input.
        """
        n = len(embeddings)

        # Need at least min_neighborhood OTHER points to compute reliable
        # leave-one-out statistics, i.e. n - 1 >= min_neighborhood.
        if n - 1 < self.min_neighborhood:
            log.warning(
                "Defense 1: batch size %d gives fewer than "
                "MIN_NEIGHBORHOOD_SIZE=%d reference points per chunk after "
                "leave-one-out — cannot compute reliable outlier statistics, "
                "returning neutral verdicts for all chunks",
                n, self.min_neighborhood,
            )
            return [
                OutlierVerdict(
                    chunk_id=cid, distance_from_centroid=0.0, zscore=0.0,
                    is_outlier_flagged=False, is_outlier_blocked=False,
                    reason="batch too small to compute outlier statistics",
                    outlier_score=0.0,
                )
                for cid in chunk_ids
            ]

        verdicts = []
        for i in range(n):
            others = embeddings[:i] + embeddings[i + 1:]
            centroid = _centroid(others)
            other_distances = [_cosine_distance(e, centroid) for e in others]
            mean_d = _mean(other_distances)
            std_d = max(_stdev(other_distances, mean_d), MIN_STD_FLOOR)

            dist = _cosine_distance(embeddings[i], centroid)
            zscore = 0.0 if std_d < 1e-9 else (dist - mean_d) / std_d
            blocked = zscore >= self.hard_zscore
            flagged = zscore >= self.soft_zscore

            reason = ""
            if blocked:
                reason = (
                    f"embedding distance z-score {zscore:.2f} exceeds "
                    f"hard-block threshold {self.hard_zscore}"
                )
            elif flagged:
                reason = (
                    f"embedding distance z-score {zscore:.2f} exceeds "
                    f"soft-flag threshold {self.soft_zscore}"
                )

            outlier_score = max(0.0, min(1.0, zscore / self.hard_zscore)) if self.hard_zscore > 0 else 0.0

            verdicts.append(OutlierVerdict(
                chunk_id=chunk_ids[i],
                distance_from_centroid=dist,
                zscore=zscore,
                is_outlier_flagged=flagged,
                is_outlier_blocked=blocked,
                reason=reason,
                outlier_score=outlier_score,
            ))

        n_blocked = sum(1 for v in verdicts if v.is_outlier_blocked)
        if n_blocked:
            log.info("Defense 1: blocked %d/%d chunks as embedding outliers", n_blocked, n)

        return verdicts
