"""
defense1_outlier_detection/config.py
---------------------------------------
Configuration for Defense 1: Embedding Outlier Detection (PRD §6.3).

Method: flag documents whose embedding sits unusually far from the centroid
of their semantic neighborhood, measured as a z-score of cosine distance
from the centroid, computed over the current retrieval batch.

Explicit, documented limitation (this is in the PRD, not a bug to hide):
this defense is expected to be weakest against the optimization-based
embedding attack (PRD Attack 2), because that attack is specifically
designed to maximize embedding similarity to legitimate content — i.e. to
NOT be an outlier. Defense 1 is a cheap first-pass filter that catches
naive attacks; it is not expected to catch sophisticated ones alone. That
is the argument for the layered pipeline (Defense 1 -> 2 -> 3 -> 4).
"""

from __future__ import annotations

# ── neighborhood scope ────────────────────────────────────────────────────────
# Outlier statistics (mean/std of centroid distance) are computed over the
# current retrieval batch — the top-k candidates for one query — not the
# whole corpus. This matches how Defense 2's corroboration score is scoped
# (per-query, not corpus-wide) and is the practical choice for a retrieval-
# time defense: you don't want to recompute a corpus-wide centroid on every
# query.
MIN_NEIGHBORHOOD_SIZE = 4   # need at least this many candidates to compute
                             # a statistically meaningful mean/std; below
                             # this, verdicts are neutral (cannot determine)

# ── numerical stability ────────────────────────────────────────────────────────
# When the legitimate documents in a batch are extremely tightly clustered
# (near-identical embeddings), the standard deviation of their centroid
# distances can be tiny — and dividing by a near-zero std turns even
# negligible natural variation into an enormous, meaningless z-score. This
# floor prevents that: std is never treated as smaller than this value when
# computing z-scores, so genuinely tiny differences don't get amplified into
# false outlier signals. Real 384-dim MiniLM embeddings are extremely
# unlikely to cluster this tightly, so this mostly matters for small
# synthetic test batches — but it costs nothing to have in production too.
MIN_STD_FLOOR = 0.01
# Two thresholds, same hard/soft split used in Defense 3: an extreme outlier
# (very unusual embedding — most likely a naive, poorly-blended attack) is
# blocked outright. A moderate outlier is flagged but passed downstream,
# since Defense 1 alone catching a moderate anomaly is not strong enough
# evidence to remove a document from context by itself.
SOFT_FLAG_ZSCORE  = 1.5   # flag as suspicious above this
HARD_BLOCK_ZSCORE = 2.75  # block outright above this
