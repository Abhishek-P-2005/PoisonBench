"""
defense2_trust_rerank/config.py
--------------------------------
All tunable parameters for Defense 2 live here so they can be swept during
the ablation study and calibrated via ROC / Precision-Recall analysis
(PRD Section 3.3.5 — Defense Analysis Framework) without touching logic code.

Nothing here is final. TRUST_WEIGHTS and RERANK_ALPHA are placeholder
starting points to be calibrated once labeled poisoned/clean data exists
from the attack module.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── trust score component weights ─────────────────────────────────────────────
# trust_score = w_source * source_reliability
#             + w_age    * age_score
#             + w_corrob * corroboration_score
# Must sum to 1.0 — enforced by TrustWeights.validate().
@dataclass(frozen=True)
class TrustWeights:
    source: float = 0.5     # weight for source-reliability component
    age:    float = 0.2     # weight for document-age component
    corroboration: float = 0.3   # weight for cross-document corroboration

    def validate(self) -> None:
        total = self.source + self.age + self.corroboration
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"TrustWeights must sum to 1.0, got {total}")


DEFAULT_TRUST_WEIGHTS = TrustWeights()


# ── source reliability base scores ────────────────────────────────────────────
# Keyed on the `verified` boolean from ChunkMetadata (True = source_id is in
# corpus.metadata.TRUSTED_SOURCES). We deliberately do NOT key on `is_poisoned`
# or `source_type == POISONED` — that would be using the attack module's
# ground-truth label directly, which is cheating for a defense that is
# supposed to work without knowing which documents are poisoned.
VERIFIED_SOURCE_SCORE   = 1.0   # source_id in TRUSTED_SOURCES
UNVERIFIED_SOURCE_SCORE = 0.4   # source_id not in TRUSTED_SOURCES
                                 # (not 0 — legitimate but unverified
                                 #  community contributions should not be
                                 #  automatically zeroed out)


# ── age scoring ────────────────────────────────────────────────────────────────
# Very recently published/modified documents have not had time to be
# corroborated by other sources or scrutinized by the community, so they
# get a mild trust discount. This is a soft signal, not a hard filter.
AGE_FULL_TRUST_DAYS   = 30   # documents older than this get full age trust
AGE_MIN_TRUST_DAYS    = 0    # documents this fresh get the minimum age trust
AGE_MIN_TRUST_SCORE   = 0.5  # trust floor for brand-new documents


# ── corroboration scoring ─────────────────────────────────────────────────────
# Corroboration = how many OTHER chunks referencing the same doc_id (CVE ID)
# exist in the corpus from verified sources. A document that is the sole
# chunk making a claim is inherently less corroborated than one backed by
# multiple independent verified chunks.
CORROBORATION_SATURATION_COUNT = 3   # corroboration score saturates at this count
                                      # (3 independent verified chunks = full trust)


# ── re-ranking blend ──────────────────────────────────────────────────────────
# final_score = ALPHA * normalized_similarity + (1 - ALPHA) * trust_score
# ALPHA close to 1.0 means Defense 2 barely changes the baseline ranking;
# ALPHA close to 0.0 means trust dominates over semantic relevance.
# Start conservative — trust nudges the ranking, it does not override
# relevance outright. Recalibrate via the ablation sweep.
RERANK_ALPHA = 0.7

# ── flagging threshold ─────────────────────────────────────────────────────────
# Documents with trust_score below this are flagged as "low trust" for
# logging into the false_positives / defense_flags tables, independent of
# whether they get re-ranked out of the top-k. This threshold is a starting
# point — Section 3.3.5 calls for ROC/PR-based calibration once labeled
# data is available; see calibration.py.
LOW_TRUST_FLAG_THRESHOLD = 0.45
