"""
optimization_layer/risk_scorer.py
----------------------------------------
Req 5.1: compute a fast initial risk score for every chunk using
  - chunk provenance metadata   (Defense 3 checks)
  - vector centroid distance    (Defense 1 leave-one-out z-score)

If an earlier pipeline stage already annotated the chunk with
`provenance_score` or `outlier_score`, those values are reused, so the
router adds almost no extra work inside the full pipeline. If a field is
missing, it is computed here with the same Defense 1 / Defense 3 code.

This module never reads `is_poisoned` or `attack_type`.
"""

from __future__ import annotations

from dataclasses import dataclass

from defenses.defense1_outlier_detection.config import MIN_NEIGHBORHOOD_SIZE
from defenses.defense1_outlier_detection.outlier_detector import OutlierDetector
from defenses.defense2_trust_rerank.trust_scorer import TrustScorer
from defenses.defense3_provenance.provenance_verifier import ProvenanceVerifier
from optimization_layer.config import (
    INCLUDE_TRUST_SIGNAL,
    RISK_METHOD,
    UNAVAILABLE_SIGNAL_RISK,
    WEIGHT_CENTROID,
    WEIGHT_PROVENANCE,
)


@dataclass
class RiskVerdict:
    chunk_id:        str
    provenance_risk: float
    centroid_risk:   float
    trust_risk:      float
    risk_score:      float   # combined, in [0, 1]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


class RiskScorer:
    def __init__(
        self,
        method: str = RISK_METHOD,
        weight_provenance: float = WEIGHT_PROVENANCE,
        weight_centroid: float = WEIGHT_CENTROID,
        include_trust: bool = INCLUDE_TRUST_SIGNAL,
    ):
        if method not in ("noisy_or", "weighted"):
            raise ValueError(f"unknown risk method: {method}")
        self.method = method
        self.weight_provenance = weight_provenance
        self.weight_centroid = weight_centroid
        self.include_trust = include_trust

    # ── individual signals ────────────────────────────────────────────────

    def _provenance_risks(self, batch: list[dict]) -> list[float]:
        verifier = ProvenanceVerifier()
        risks = []
        for chunk in batch:
            if "provenance_score" in chunk:
                score = chunk["provenance_score"]
            else:
                score = verifier.verify(chunk["text"], chunk["metadata"]).provenance_score
            risks.append(_clip(1.0 - score))
        return risks

    def _centroid_risks(self, batch: list[dict]) -> list[float]:
        # Reuse Defense 1 output when every chunk already has it.
        if all("outlier_score" in chunk for chunk in batch):
            return [_clip(chunk["outlier_score"]) for chunk in batch]

        has_embeddings = all("embedding" in chunk for chunk in batch)
        big_enough = len(batch) - 1 >= MIN_NEIGHBORHOOD_SIZE
        if not has_embeddings or not big_enough:
            return [UNAVAILABLE_SIGNAL_RISK for _ in batch]

        chunk_ids = [chunk["chunk_id"] for chunk in batch]
        embeddings = [chunk["embedding"] for chunk in batch]
        verdicts = OutlierDetector().score_batch(chunk_ids, embeddings)
        return [_clip(v.outlier_score) for v in verdicts]

    def _trust_risks(self, batch: list[dict]) -> list[float]:
        if not self.include_trust:
            return [0.0 for _ in batch]
        if all("trust_score" in chunk for chunk in batch):
            return [_clip(1.0 - chunk["trust_score"]) for chunk in batch]
        breakdowns = TrustScorer().score_batch([chunk["metadata"] for chunk in batch])
        return [_clip(1.0 - b.final_score) for b in breakdowns]

    # ── combination ───────────────────────────────────────────────────────

    def _combine(self, prov: float, cent: float, trust: float) -> float:
        if self.method == "noisy_or":
            safe = (1.0 - prov) * (1.0 - cent)
            if self.include_trust:
                safe = safe * (1.0 - trust)
            return _clip(1.0 - safe)

        total_weight = self.weight_provenance + self.weight_centroid
        risk = (self.weight_provenance * prov + self.weight_centroid * cent) / total_weight
        if self.include_trust:
            risk = max(risk, trust)
        return _clip(risk)

    def score_batch(self, batch: list[dict]) -> list[RiskVerdict]:
        if not batch:
            return []
        prov_risks = self._provenance_risks(batch)
        cent_risks = self._centroid_risks(batch)
        trust_risks = self._trust_risks(batch)

        verdicts = []
        for i, chunk in enumerate(batch):
            verdicts.append(RiskVerdict(
                chunk_id=chunk["chunk_id"],
                provenance_risk=prov_risks[i],
                centroid_risk=cent_risks[i],
                trust_risk=trust_risks[i],
                risk_score=self._combine(prov_risks[i], cent_risks[i], trust_risks[i]),
            ))
        return verdicts
