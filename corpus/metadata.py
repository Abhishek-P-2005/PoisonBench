"""
corpus/metadata.py
------------------
Canonical metadata schema for every document chunk in the corpus.
Reproduced here (Module 1) so Defense 2 and other modules import from
a single source of truth. See Module 1 delivery for full docstring detail.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    NVD_OFFICIAL = "nvd_official"
    VENDOR_ADVISORY = "vendor_advisory"
    SECURITY_RESEARCHER = "security_researcher"
    UNVERIFIED = "unverified"
    POISONED = "poisoned"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    NONE     = "NONE"
    UNKNOWN  = "UNKNOWN"


TRUSTED_SOURCES: frozenset[str] = frozenset({
    "nvd_official",
    "nist_nvd_api_v2",
})


@dataclass
class ChunkMetadata:
    chunk_id:       str
    doc_id:         str
    chunk_index:    int
    chunk_hash:     str
    source_type:    str
    source_id:      str
    verified:       bool
    published_date: str
    modified_date:  str
    ingested_at:    str
    severity:       str
    cvss_score:     float
    cwe_ids:        str
    cpe_affected:   str
    token_count:    int  = 0
    char_count:     int  = 0
    is_poisoned:    bool = False
    attack_type:    str  = ""

    def to_chroma_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def make_chunk_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def make_chunk_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_metadata(
    *,
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    published_date: str,
    modified_date: str,
    severity: str = Severity.UNKNOWN,
    cvss_score: float = 0.0,
    cwe_ids: list[str] = None,
    cpe_affected: list[str] = None,
    source_type: str = SourceType.NVD_OFFICIAL,
    source_id: str = "nvd_official",
    is_poisoned: bool = False,
    attack_type: str = "",
) -> ChunkMetadata:
    cwe_ids      = cwe_ids or []
    cpe_affected = cpe_affected or []
    return ChunkMetadata(
        chunk_id       = ChunkMetadata.make_chunk_id(),
        doc_id         = doc_id,
        chunk_index    = chunk_index,
        chunk_hash     = ChunkMetadata.make_chunk_hash(chunk_text),
        source_type    = source_type,
        source_id      = source_id,
        verified       = source_id in TRUSTED_SOURCES,
        published_date = published_date,
        modified_date  = modified_date,
        ingested_at    = datetime.utcnow().isoformat(timespec="seconds") + "Z",
        severity       = severity,
        cvss_score     = float(cvss_score),
        cwe_ids        = ",".join(cwe_ids),
        cpe_affected   = ",".join(cpe_affected[:5]),
        token_count    = 0,
        char_count     = len(chunk_text),
        is_poisoned    = is_poisoned,
        attack_type    = attack_type,
    )
