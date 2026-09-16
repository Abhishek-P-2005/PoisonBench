"""
defense3_provenance/provenance_verifier.py
---------------------------------------------
Runs four independent, discrete checks against a document chunk's claimed
identity and integrity:

  1. Identity check    — is source_id on the trusted allow-list, and is it
                          internally consistent with the claimed source_type?
                          (catches spoofing: claiming to be NVD-official while
                          using a source_id that isn't on the allow-list)
  2. Format check       — does doc_id match the expected CVE-YYYY-NNNNN pattern
                          for anything claiming an official source?
  3. Temporal check     — are published/modified dates sane (not in the future,
                          modified not before published)?
  4. Integrity check    — does the chunk's content hash match what's recorded
                          in its own metadata? (tamper detection)

Design note — consistent with Defense 2
-----------------------------------------
Like Defense 2, none of these checks use `is_poisoned` or `attack_type`.
They only look at fields a document legitimately carries and that a real
defense would have access to at inference time: source_id, source_type,
doc_id, timestamps, and content hash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from corpus.metadata import ChunkMetadata
from defenses.defense3_provenance.config import (
    ALLOWED_CLAIMING_SOURCE_TYPES,
    ALLOWED_SOURCE_IDS,
    CVE_ID_PATTERN,
    HARD_BLOCK_CHECKS,
    MAX_FUTURE_SKEW_DAYS,
)

log = logging.getLogger(__name__)


@dataclass
class ProvenanceVerdict:
    """Result of running all provenance checks against a single chunk."""
    chunk_id:          str
    passed_checks:     list[str] = field(default_factory=list)
    failed_checks:     list[str] = field(default_factory=list)   # check names
    reasons:           list[str] = field(default_factory=list)   # human-readable
    provenance_score:  float = 1.0   # fraction of checks passed, in [0, 1]
    blocked:           bool  = False  # True if any HARD_BLOCK_CHECKS failed
    flagged:           bool  = False  # True if any check failed (block or soft)

    @property
    def allowed(self) -> bool:
        """A document is allowed into context iff it is not hard-blocked."""
        return not self.blocked


def _stringify(value) -> str:
    """
    Normalize enum-or-str metadata values to their plain string form for
    human-readable messages. SourceType is a (str, Enum) mixin — depending
    on Python version, str(enum_member) can render as "SourceType.X" rather
    than "x". `.value` always gives the clean underlying string.
    """
    return getattr(value, "value", value)


class ProvenanceVerifier:
    """Stateless verifier — safe to reuse across chunks and threads."""

    CHECK_NAMES = (
        "identity_spoofing",
        "format_mismatch",
        "temporal_anomaly",
        "integrity_mismatch",
    )

    def verify(self, text: str, metadata: dict) -> ProvenanceVerdict:
        """
        Run all four checks against one chunk.

        Parameters
        ----------
        text     : the raw chunk text (needed for integrity check)
        metadata : ChunkMetadata.to_chroma_dict() output — must contain
                   chunk_id, source_id, source_type, doc_id, published_date,
                   modified_date, chunk_hash

        Returns
        -------
        ProvenanceVerdict
        """
        chunk_id = metadata.get("chunk_id", "UNKNOWN")
        verdict = ProvenanceVerdict(chunk_id=chunk_id)

        checks = {
            "identity_spoofing":  self._check_identity(metadata),
            "format_mismatch":    self._check_format(metadata),
            "temporal_anomaly":   self._check_temporal(metadata),
            "integrity_mismatch": self._check_integrity(text, metadata),
        }

        for check_name, (passed, reason) in checks.items():
            if passed:
                verdict.passed_checks.append(check_name)
            else:
                verdict.failed_checks.append(check_name)
                verdict.reasons.append(reason)
                if check_name in HARD_BLOCK_CHECKS:
                    verdict.blocked = True

        verdict.flagged = len(verdict.failed_checks) > 0
        verdict.provenance_score = len(verdict.passed_checks) / len(self.CHECK_NAMES)

        if verdict.blocked:
            log.info(
                "Defense 3: BLOCKED chunk_id=%s reasons=%s",
                chunk_id, verdict.reasons,
            )
        elif verdict.flagged:
            log.info(
                "Defense 3: flagged chunk_id=%s (soft) reasons=%s",
                chunk_id, verdict.reasons,
            )

        return verdict

    # ── individual checks ────────────────────────────────────────────────
    # Each returns (passed: bool, reason: str). reason is only meaningful
    # when passed is False.

    @staticmethod
    def _check_identity(metadata: dict) -> tuple[bool, str]:
        """
        Two-part identity check:
          a) if source_type claims to be an allowed claiming type (currently
             only NVD_OFFICIAL), source_id MUST be on the allow-list —
             claiming official status with a non-allow-listed source_id is
             spoofing.
          b) source_id being on the allow-list should imply `verified=True`
             in the metadata already set by corpus/metadata.py; a mismatch
             here indicates metadata was tampered with after ingestion.
        """
        source_type = metadata.get("source_type", "")
        source_id   = metadata.get("source_id", "")
        verified    = metadata.get("verified", False)

        source_id_on_allowlist = source_id in ALLOWED_SOURCE_IDS

        if source_type in ALLOWED_CLAIMING_SOURCE_TYPES and not source_id_on_allowlist:
            return False, (
                f"source_type='{_stringify(source_type)}' claims official status but "
                f"source_id='{source_id}' is not on the trusted allow-list"
            )

        if source_id_on_allowlist and not verified:
            return False, (
                f"source_id='{source_id}' is allow-listed but metadata "
                f"verified flag is False — possible metadata tampering"
            )

        return True, ""

    @staticmethod
    def _check_format(metadata: dict) -> tuple[bool, str]:
        """
        Anything claiming an allowed official source_type must have a
        well-formed CVE ID. A malformed doc_id on an official-claiming
        document is a spoofing / injection signal — e.g. "CVE-2023-1111-fake".
        """
        source_type = metadata.get("source_type", "")
        doc_id      = metadata.get("doc_id", "")

        if source_type not in ALLOWED_CLAIMING_SOURCE_TYPES:
            return True, ""   # format rule only applies to official-claiming docs

        if not CVE_ID_PATTERN.match(doc_id):
            return False, f"doc_id='{doc_id}' does not match expected CVE-YYYY-NNNNN format"

        return True, ""

    @staticmethod
    def _check_temporal(metadata: dict) -> tuple[bool, str]:
        """
        published_date and modified_date must both be parseable, modified
        must not precede published, and neither may be unreasonably far
        in the future (allowing for minor clock skew).
        """
        published_str = metadata.get("published_date", "")
        modified_str  = metadata.get("modified_date", "")

        try:
            pub_dt = datetime.strptime(published_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return False, f"published_date='{published_str}' is unparseable"

        try:
            mod_dt = datetime.strptime(modified_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return False, f"modified_date='{modified_str}' is unparseable"

        now = datetime.now(timezone.utc)
        max_future = now + timedelta(days=MAX_FUTURE_SKEW_DAYS)

        if pub_dt > max_future:
            return False, f"published_date='{published_str}' is in the future"
        if mod_dt > max_future:
            return False, f"modified_date='{modified_str}' is in the future"
        if mod_dt < pub_dt:
            return False, (
                f"modified_date='{modified_str}' precedes "
                f"published_date='{published_str}'"
            )

        return True, ""

    @staticmethod
    def _check_integrity(text: str, metadata: dict) -> tuple[bool, str]:
        """
        Recompute the SHA-256 hash of the chunk text and compare against
        the hash recorded in metadata at ingestion time. A mismatch means
        either the text or the metadata was modified after ingestion —
        both are tamper signals.
        """
        recorded_hash = metadata.get("chunk_hash", "")
        if not recorded_hash:
            return False, "no chunk_hash present in metadata"

        actual_hash = ChunkMetadata.make_chunk_hash(text)
        if actual_hash != recorded_hash:
            return False, (
                f"chunk_hash mismatch — recorded='{recorded_hash[:12]}...' "
                f"actual='{actual_hash[:12]}...'"
            )

        return True, ""
