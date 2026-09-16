"""
defense3_provenance
---------------------
Defense 3: Provenance Verification.

Access-control style validation layer: checks whether a document's claimed
source identity is on the trusted allow-list and internally consistent
(identity), well-formed (format), temporally sane (temporal), and
untampered (integrity). Hard failures (identity spoofing, integrity
mismatch) block the document from context; soft failures (format,
temporal irregularities) flag it for downstream logging and Defense 4
feature engineering, but do not remove it.

Public API:
    from defenses.defense3_provenance import apply_provenance_filter, ProvenanceVerifier
"""

from defenses.defense3_provenance.filter import apply_provenance_filter
from defenses.defense3_provenance.provenance_verifier import (
    ProvenanceVerdict,
    ProvenanceVerifier,
)

__all__ = ["apply_provenance_filter", "ProvenanceVerifier", "ProvenanceVerdict"]
