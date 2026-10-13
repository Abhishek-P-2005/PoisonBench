"""
optimization_layer
-------------------------
Optimization layer (Module 5): Adaptive Risk-Based Defense Routing (PRD v2.0).

Computes a fast risk score per chunk from provenance metadata and embedding
centroid distance, then sends only high-risk chunks to the heavy inspection
stage (the supervised ML classifier). Low-risk chunks skip it.

Public API:
    from optimization_layer import (
        AdaptiveRouter, RiskScorer, apply_adaptive_routing,
    )
"""

from optimization_layer.risk_scorer import RiskScorer, RiskVerdict
from optimization_layer.router import (
    AdaptiveRouter,
    RoutingDecision,
    apply_adaptive_routing,
)

__all__ = [
    "AdaptiveRouter",
    "RiskScorer",
    "RiskVerdict",
    "RoutingDecision",
    "apply_adaptive_routing",
]
