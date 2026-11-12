"""
evaluation/reference.py
-----------------------
Two baselines that use a reference the attacker does not control: a snapshot of the clean
corpus taken at ingestion time.

  d1c  CorpusD1Stage   D1's idea (distance from a centre, as a z-score) but the centre and
                       spread come from the WHOLE clean corpus, not from the retrieved batch.
                       The batch version fails when poison is most of the batch (the paper's
                       section VIII-I); this version cannot be dragged by the batch.

  ref  RegistryStage   "One authoritative text per CVE". A chunk that claims a CVE id (the first
                       one in its text) must match the registry's text for that id, byte for byte (SHA-256).
                       A mismatch is blocked. A chunk with no claim, or a CVE id the registry
                       has never seen, is only flagged (new CVEs are legitimate), so this rule
                       is blocked by an attacker who invents a new id or who can write to the
                       registry. It is the trivial baseline any content classifier has to beat.

Both stages read a module-level reference set with set_reference(). The run script builds it
from the clean collection; the stage raises if none was set. Neither stage reads is_poisoned.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from evaluation.stages import StageOutput

_CVE_RE = re.compile(r"(CVE-\d{4}-\d{4,7})")   # first CVE id anywhere: OBEA poison puts keywords before it
STD_FLOOR = 0.01


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CorpusReference:
    centroid: np.ndarray
    mean_distance: float
    std_distance: float
    registry: dict = field(default_factory=dict)      # CVE id -> set of authentic text hashes
    n_chunks: int = 0
    method: str = "chunks"                            # how the vectors were obtained

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "CorpusReference":
        """chunks: [{chunk_id, text, embedding}] of the CLEAN corpus."""
        if not chunks:
            raise ValueError("cannot build a reference from zero chunks")
        matrix = np.asarray([c["embedding"] for c in chunks], dtype=float)
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        centroid = matrix.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        distance = 1.0 - matrix @ centroid
        registry: dict[str, set] = {}
        for chunk in chunks:
            match = _CVE_RE.search(chunk["text"])
            if match:
                registry.setdefault(match.group(1), set()).add(text_hash(chunk["text"]))
        return cls(centroid, float(distance.mean()), float(distance.std()), registry, len(chunks))

    @classmethod
    def from_collection(cls, collection, batch: int = 100, embed=None) -> "CorpusReference":
        """
        Read the clean collection by id (not by offset). If this Chroma build refuses to return
        stored embeddings ('Error finding id' has been seen on Windows with the in-process client),
        fall back to fetching the documents and re-embedding them with `embed(list_of_texts) ->
        list_of_vectors`. Without `embed` the original error is raised with a hint.
        """
        ids = collection.get(include=[])["ids"]
        chunks, reembedded = [], False
        for start in range(0, len(ids), batch):
            sub = ids[start:start + batch]
            try:
                got = collection.get(ids=sub, include=["documents", "embeddings"])
                vectors = [list(v) for v in got["embeddings"]]
            except Exception as exc:   # noqa: BLE001
                if embed is None:
                    raise RuntimeError(
                        f"Chroma could not return stored embeddings ({exc}). Pass embed=<function> so the reference "
                        "can be rebuilt by re-embedding the stored text.") from exc
                got = collection.get(ids=sub, include=["documents"])
                vectors = [list(v) for v in embed(got["documents"])]
                reembedded = True
            for chunk_id, text, vector in zip(got["ids"], got["documents"], vectors):
                chunks.append({"chunk_id": chunk_id, "text": text, "embedding": vector})
        reference = cls.from_chunks(chunks)
        reference.method = "re-embedded" if reembedded else "chroma"
        return reference

    def z_score(self, embedding) -> float:
        vector = np.asarray(embedding, dtype=float)
        vector = vector / max(np.linalg.norm(vector), 1e-12)
        distance = 1.0 - float(vector @ self.centroid)
        return (distance - self.mean_distance) / max(self.std_distance, STD_FLOOR)


_REFERENCE: CorpusReference | None = None


def set_reference(reference: CorpusReference | None) -> None:
    global _REFERENCE
    _REFERENCE = reference


def get_reference() -> CorpusReference | None:
    return _REFERENCE


def _require_reference() -> CorpusReference:
    if _REFERENCE is None:
        raise RuntimeError("no corpus reference set; call evaluation.reference.set_reference("
                           "CorpusReference.from_collection(<clean collection>)) first")
    return _REFERENCE


class CorpusD1Stage:
    name = "d1c"

    def __init__(self, soft_z: float = 1.5, hard_z: float | None = None):
        self.soft_z, self.hard_z = soft_z, hard_z

    def run(self, batch: list[dict]) -> StageOutput:
        out = StageOutput()
        reference = _require_reference()
        for chunk in batch:
            if "embedding" not in chunk:
                raise ValueError(f"CorpusD1Stage needs an 'embedding' on every chunk ({chunk['chunk_id']})")
            z = reference.z_score(chunk["embedding"])
            out.suspicion[chunk["chunk_id"]] = z
            if z > self.soft_z:
                out.flagged_ids.add(chunk["chunk_id"])
            if self.hard_z is not None and z > self.hard_z:
                out.blocked.append({**chunk, "corpus_zscore": z})
            else:
                out.passed.append({**chunk, "corpus_zscore": z})
        return out


class RegistryStage:
    name = "ref"

    def run(self, batch: list[dict]) -> StageOutput:
        out = StageOutput()
        reference = _require_reference()
        for chunk in batch:
            match = _CVE_RE.search(chunk["text"])
            known = reference.registry.get(match.group(1)) if match else None
            if known is not None and text_hash(chunk["text"]) not in known:
                verdict, score = "mismatch", 1.0          # claims a known CVE, text is not the NVD text
            elif known is None:
                verdict, score = "unverifiable", 0.5      # no claim, or an id the registry never saw
            else:
                verdict, score = "authentic", 0.0
            out.suspicion[chunk["chunk_id"]] = score
            augmented = {**chunk, "registry_verdict": verdict}
            if verdict == "mismatch":
                out.flagged_ids.add(chunk["chunk_id"])
                out.blocked.append(augmented)
            else:
                if verdict == "unverifiable":
                    out.flagged_ids.add(chunk["chunk_id"])
                out.passed.append(augmented)
        return out
