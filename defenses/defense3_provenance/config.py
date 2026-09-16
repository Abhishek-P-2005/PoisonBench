"""
defense3_provenance/config.py
--------------------------------
Configuration for Defense 3: Provenance Verification.

This defense is an access-control style validation layer (PRD Section 6.5):
each document carries a claimed source identity, and only sources on a
trusted list can contribute to retrieval without being flagged. Conceptually
this mirrors role-based access control, applied to document sources instead
of user accounts.

Where Defense 2 (Trust Re-ranking) computes a soft, continuous trust score,
Defense 3 runs discrete pass/fail checks on document identity and integrity.
The two are complementary: Defense 2 asks "how much should I trust this,
relatively speaking?"; Defense 3 asks "is this document's claimed identity
even internally consistent and on the allow-list?"
"""

from __future__ import annotations

import re

from corpus.metadata import TRUSTED_SOURCES, SourceType

# ── source allow-list ──────────────────────────────────────────────────────────
# Re-exported from corpus.metadata so there is exactly one source of truth
# for "which sources are trusted" across Defense 2 and Defense 3.
ALLOWED_SOURCE_IDS: frozenset[str] = TRUSTED_SOURCES

# source_type values that are permitted to claim membership in the trusted
# allow-list at all. A document claiming source_type == POISONED should never
# reach this check in a real deployment (that label doesn't exist outside
# evaluation), but we defend against it anyway for safety.
ALLOWED_CLAIMING_SOURCE_TYPES: frozenset[str] = frozenset({
    SourceType.NVD_OFFICIAL,
})

# ── document ID format ────────────────────────────────────────────────────────
# Official NVD CVE identifiers always match this pattern. A document claiming
# to be NVD-sourced but with a malformed or non-conforming doc_id is a strong
# spoofing signal — e.g. "CVE-2023-1111-fake" as used in an injection attempt.
CVE_ID_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,7}$")

# ── temporal sanity ────────────────────────────────────────────────────────────
# How much clock skew to tolerate before flagging a published/modified date
# as "in the future" (handles minor timezone/clock differences, not meant to
# be lenient toward genuinely bogus dates).
MAX_FUTURE_SKEW_DAYS = 1

# ── verdict policy ─────────────────────────────────────────────────────────────
# Which individual check failures are severe enough to BLOCK a document
# outright (removed from context entirely) versus FLAG it (kept but marked
# suspicious for downstream defenses / logging, e.g. Defense 4's classifier
# can use the flag as a feature).
#
# Rationale: identity spoofing (claiming trust you don't have) and integrity
# failure (tampered content) are hard failures — there is no legitimate
# reason for either to occur, so we block. Format and temporal irregularities
# can occur in real dirty data, so they are soft signals — flag, don't block.
HARD_BLOCK_CHECKS: frozenset[str] = frozenset({
    "identity_spoofing",
    "integrity_mismatch",
})
