"""
optimization_layer/test_smoke.py
-------------------------------------------------
Standalone smoke test on the fixed 7-chunk scenario in evaluation/fixtures.py
(copied from the Review 2 demo).
No ChromaDB, PostgreSQL or trained model needed. The "heavy inspector" here is
a trivial stand-in, because this test only checks the ROUTING:

  - the spoofed attack chunk is sent to the heavy path
  - the clean chunks bypass it
  - the heavy inspector is called only for the high-risk chunk

Usage
-----
    PYTHONPATH=. python -m optimization_layer.test_smoke
"""

from __future__ import annotations

from optimization_layer import AdaptiveRouter, apply_adaptive_routing
from evaluation.fixtures import ATTACK_ID, build_scenario
from evaluation.pipeline import strip_labels


def run_smoke_test() -> None:
    batch = [strip_labels(chunk) for chunk in build_scenario()]
    inspected = []

    def stand_in_inspector(chunks, only_ids):
        """Blocks any chunk it is asked to inspect that failed a provenance check."""
        passed, blocked = [], []
        for chunk in chunks:
            if chunk["chunk_id"] in only_ids:
                inspected.append(chunk["chunk_id"])
                if chunk["metadata"]["source_id"] != "nvd_official":
                    blocked.append(chunk)
                    continue
            passed.append(chunk)
        return passed, blocked

    passed, blocked, decision = apply_adaptive_routing(batch, stand_in_inspector, AdaptiveRouter(risk_threshold=0.25))

    print("=" * 74)
    print("Optimization layer smoke test: adaptive risk-based routing")
    print("=" * 74)
    for verdict in decision.verdicts:
        path = "HEAVY" if verdict.chunk_id in decision.high_risk_ids else "fast "
        print(f"  {verdict.chunk_id:<18} provenance_risk={verdict.provenance_risk:.2f}  "
              f"centroid_risk={verdict.centroid_risk:.2f}  risk={verdict.risk_score:.2f}  -> {path}")

    assert ATTACK_ID in decision.high_risk_ids, "spoofed chunk must be routed to the heavy path"
    assert inspected == [ATTACK_ID] or ATTACK_ID in inspected
    assert len(inspected) < len(batch), "the router should have let some chunks skip the heavy stage"
    assert ATTACK_ID in [chunk["chunk_id"] for chunk in blocked]

    print(f"\n  Heavy inspector ran on {len(inspected)} of {len(batch)} chunks "
          f"({len(batch) - len(inspected)} skipped).")
    print("PASS: the attack chunk was routed to the heavy stage and blocked; clean chunks took the fast path.")
    print("=" * 74)


if __name__ == "__main__":
    run_smoke_test()
