"""
optimization_layer/config.py
-----------------------------------
Configuration for Optimization layer (Module 5): Adaptive Risk-Based Defense Routing
(PRD v2.0, Module 5, Reqs 5.1 - 5.4).

The router does NOT replace the defense pipeline. It decides, per chunk,
whether a chunk needs the HEAVY inspection stage (the supervised ML
classifier, Defense 4) or can go straight to prompt-context construction.

    cheap signals (provenance + embedding centroid distance)
        -> risk score in [0, 1]
        -> risk < RISK_THRESHOLD   : LOW risk  -> bypass heavy stage
        -> risk >= RISK_THRESHOLD  : HIGH risk -> full defense pipeline

Nothing here is final. RISK_THRESHOLD is a placeholder until it is
calibrated on labeled data (see evaluation/router_benchmark.py,
calibrate_risk_threshold).
"""

from __future__ import annotations

# ── threshold ─────────────────────────────────────────────────────────────────
# A chunk with risk >= this value is sent to the heavy stage.
# Placeholder. Lower value = safer (more chunks inspected, less latency saved).
RISK_THRESHOLD = 0.25

# ── how the cheap signals are combined ────────────────────────────────────────
# "noisy_or"  : risk = 1 - (1 - provenance_risk) * (1 - centroid_risk)
#               A chunk is risky if EITHER signal fires. This also covers the
#               "signals disagree -> inspect closely" rule from the final
#               review plan.
# "weighted"  : risk = w_prov * provenance_risk + w_cent * centroid_risk
RISK_METHOD = "noisy_or"
WEIGHT_PROVENANCE = 0.5
WEIGHT_CENTROID = 0.5

# Optional third signal from Defense 2 (trust). PRD v2.0 Req 5.1 only names
# provenance metadata and centroid distance, so this is OFF by default.
INCLUDE_TRUST_SIGNAL = False

# ── missing signals ───────────────────────────────────────────────────────────
# If a signal cannot be computed (no embedding on the chunk, or the batch is
# too small for Defense 1 to produce statistics), we do not pretend the chunk
# is safe. The signal is replaced by this value, so unknown means "somewhat
# risky" rather than "clean".
UNAVAILABLE_SIGNAL_RISK = 0.5
