"""
optimization_layer/router.py
-----------------------------------
Reqs 5.2 and 5.3: route each chunk along a fast path or a heavy path.

  LOW  risk (risk <  threshold) -> bypass the heavy stage, go straight to
                                   prompt-context construction   (Req 5.2)
  HIGH risk (risk >= threshold) -> run the heavy stage (the full defense
                                   pipeline for this chunk)       (Req 5.3)

The heavy stage is passed in as a function, so this module does not depend
on how Defense 4 is built:

    inspector(batch, only_ids) -> (passed, blocked)

`inspector` must inspect only the chunks whose chunk_id is in `only_ids`
and return every other chunk unchanged in `passed`. The full batch is still
given to it so batch-level features stay the same as in the static pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from optimization_layer.config import RISK_THRESHOLD
from optimization_layer.risk_scorer import RiskScorer, RiskVerdict

Inspector = Callable[[list[dict], set], tuple]


@dataclass
class RoutingDecision:
    verdicts:      list[RiskVerdict]
    high_risk_ids: set = field(default_factory=set)
    low_risk_ids:  set = field(default_factory=set)

    @property
    def n_high(self) -> int:
        return len(self.high_risk_ids)

    @property
    def n_low(self) -> int:
        return len(self.low_risk_ids)


class AdaptiveRouter:
    def __init__(
        self,
        risk_threshold: float = RISK_THRESHOLD,
        scorer: RiskScorer | None = None,
    ):
        self.risk_threshold = risk_threshold
        self.scorer = scorer or RiskScorer()

    def route(self, batch: list[dict]) -> RoutingDecision:
        verdicts = self.scorer.score_batch(batch)
        decision = RoutingDecision(verdicts=verdicts)
        for verdict in verdicts:
            if verdict.risk_score >= self.risk_threshold:
                decision.high_risk_ids.add(verdict.chunk_id)
            else:
                decision.low_risk_ids.add(verdict.chunk_id)
        return decision


def apply_adaptive_routing(
    batch: list[dict],
    inspector: Inspector,
    router: AdaptiveRouter | None = None,
) -> tuple[list[dict], list[dict], RoutingDecision]:
    """
    Route the batch, run the heavy inspector on high-risk chunks only, and
    return (passed, blocked, decision). Every returned chunk is annotated
    with `risk_score` and `routed_path` ("fast" or "heavy").
    """
    if not batch:
        return [], [], RoutingDecision(verdicts=[])

    router = router or AdaptiveRouter()
    decision = router.route(batch)
    risk_by_id = {v.chunk_id: v.risk_score for v in decision.verdicts}

    annotated = []
    for chunk in batch:
        path = "heavy" if chunk["chunk_id"] in decision.high_risk_ids else "fast"
        annotated.append({
            **chunk,
            "risk_score": risk_by_id[chunk["chunk_id"]],
            "routed_path": path,
        })

    if decision.high_risk_ids:
        passed, blocked = inspector(annotated, decision.high_risk_ids)
    else:
        passed, blocked = annotated, []

    return passed, blocked, decision
