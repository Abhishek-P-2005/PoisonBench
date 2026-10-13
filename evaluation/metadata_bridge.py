"""
evaluation/metadata_bridge.py
-----------------------------
How the harness treats chunk metadata that was stored by the original pipeline.

The Chroma collections built by this repo's ingestion scripts carry only
    doc_id, source, timestamp, is_poisoned, position   (+ attack_type on some)
but Defenses 2 and 3 need provenance fields (source_type, source_id, verified,
published_date, modified_date, chunk_hash). The Phase E scripts of this repo
deliberately SKIP D2/D3 on such collections rather than invent provenance. The
harness keeps that stance as the default and offers an explicit, labelled
alternative:

  provenance="thin"     metadata untouched. D2, D3 and the router are NOT
                        supported; the sweep skips configs that need them.
  provenance="stored"   metadata untouched, but assumed complete (collections
                        ingested with the full corpus.metadata schema).
  provenance="derived"  EXPLORATORY. Missing fields are derived from the chunk's
                        own claims (`source`, `timestamp`). Because the hash is
                        computed from the stored text at read time, D3's
                        integrity check cannot fail, and a chunk whose timestamp
                        is empty fails D3's soft temporal check regardless of
                        whether it is poisoned. Results are tagged
                        "chroma[derived]" in eval_runs.source_name.
  provenance="auto"     "stored" if the first retrieved chunk already carries the
                        required fields, otherwise "thin".

is_poisoned / attack_type are never read to derive any provenance value.
"""
from __future__ import annotations

import hashlib
import re

REQUIRED_FIELDS = ("chunk_hash", "source_type", "source_id", "published_date", "modified_date")
OFFICIAL_SOURCE_LABELS = frozenset({"nvd", "nvd_official", "nist_nvd_api_v2"})
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_date(value) -> str:
    """'2026-03-05T12:00:00.000' -> '2026-03-05'; empty/unparseable -> ''."""
    match = _DATE_RE.match(str(value)) if value else None
    return match.group(1) if match else ""


def has_canonical_provenance(metadata: dict) -> bool:
    return all(metadata.get(field) not in (None, "") for field in REQUIRED_FIELDS[:3])


def enrich_metadata(chunk_id: str, text: str, metadata: dict | None) -> dict:
    """Return metadata with every field the defenses read. Existing values win."""
    meta = dict(metadata or {})
    meta.setdefault("chunk_id", chunk_id)
    meta.setdefault("doc_id", chunk_id.split("::")[0])
    meta.setdefault("chunk_index", int(meta.get("position", 0) or 0))
    if not meta.get("source_type") or not meta.get("source_id"):
        label = (meta.get("source") or "").strip()
        if label.lower() in OFFICIAL_SOURCE_LABELS:
            meta.update(source_type="nvd_official", source_id="nvd_official", verified=True)
        else:
            meta.update(source_type="unverified", source_id=label or "unknown", verified=False)
    else:
        meta["verified"] = bool(meta.get("verified", False))
    published = normalize_date(meta.get("published_date") or meta.get("timestamp"))
    meta["published_date"] = published
    meta["modified_date"] = normalize_date(meta.get("modified_date")) or published
    if not meta.get("chunk_hash"):
        meta["chunk_hash"] = chunk_hash(text)
    return meta
