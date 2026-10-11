"""PoisonBench Phase B4 — isolated ChromaDB preflight, READ ONLY.

Run from PoisonBench_Shared repository root:
    python scripts/phase_b4_collection_preflight.py

Checks existing frozen Chroma collections and twelve NEW, separate, reserved
Phase B target collection names. Does not create, modify, or delete collections,
write files, compute embeddings, or invoke Ollama.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN = {
    "poisonbench_corpus": (336, "controlled_optimized_embedding"),
    "poisonbench_rpi_clean_v1": (316, None),
    "poisonbench_rpi_poisoned_v1": (336, "controlled_retrieved_prompt_injection"),
}
CLEAN_REF = "poisonbench_rpi_clean_v1"
TARGETS = {
    (attack, ratio): f"poisonbench_b_{attack.lower()}_{ratio}pct_v1"
    for ratio in (1, 5, 10, 20)
    for attack in ("SM", "OBEA", "RPI")
}
EXPECTED_NVD = 316
EXPECTED_NVD_DOCS = 300
EXPECTED_POISON = 20
EXPECTED_DIM = 384


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def snapshot(collection):
    """Read IDs, documents, metadata, NOT embeddings; no DB writes."""
    count = collection.count()
    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids")
    docs = result.get("documents")
    metas = result.get("metadatas")
    require(ids is not None and docs is not None and metas is not None,
            f"{collection.name}: incomplete get() response")
    require(len(ids) == len(docs) == len(metas) == count,
            f"{collection.name}: inconsistent count/response lengths")
    require(len(set(ids)) == len(ids), f"{collection.name}: duplicate chunk IDs")
    clean, poisoned = {}, {}
    for cid, text, meta in zip(ids, docs, metas):
        require(isinstance(cid, str) and cid and isinstance(text, str) and text.strip(),
                f"{collection.name}: invalid ID or empty chunk text at {cid!r}")
        require(isinstance(meta, dict), f"{collection.name}: missing metadata for {cid!r}")
        doc_id = meta.get("doc_id")
        require(isinstance(doc_id, str) and cid.startswith(doc_id + "::chunk"),
                f"{collection.name}: ID/doc_id mismatch for {cid!r}")
        if meta.get("is_poisoned") is False:
            require(meta.get("source") == "NVD" and doc_id.startswith("CVE-"),
                    f"{collection.name}: unexpected clean metadata at {cid!r}")
            clean[cid] = (text, meta)
        elif meta.get("is_poisoned") is True:
            poisoned[cid] = (text, meta)
        else:
            raise RuntimeError(f"{collection.name}: invalid poisoning flag at {cid!r}")
    return clean, poisoned


def main() -> None:
    print("=== Phase B4 — Isolated ChromaDB preflight (READ ONLY) ===")
    require(len(TARGETS) == 12 and len(set(TARGETS.values())) == 12,
            "Unexpected Phase B target name collisions")
    from api.config import settings
    # get_client() only constructs/reads an HTTP client; no model inference.
    from api.pipeline.vectorstore import get_client

    print(f"Configured Chroma endpoint: {settings.chroma_host}:{settings.chroma_port}")
    print(f"Configured default collection: {settings.chroma_collection}")
    require(settings.chroma_collection == "poisonbench_corpus",
            "Unexpected default collection setting; stop and inspect .env")
    client = get_client()
    names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    print(f"Existing collection names ({len(names)}): {', '.join(sorted(names)) or '[none]'}")
    missing = sorted(set(FROZEN) - names)
    require(not missing, f"Missing frozen collections: {missing}")
    collisions = sorted(set(TARGETS.values()) & names)
    require(not collisions, f"Reserved Phase B target names ALREADY EXIST: {collisions}. "
            "Stop; do not overwrite or reset them.")
    print("PASS: All 12 reserved Phase B target names are unused")

    snapshots = {}
    for name, (expected_count, expected_poison_source) in FROZEN.items():
        collection = client.get_collection(name=name)  # get, never get_or_create
        clean, poisoned = snapshot(collection)
        require(len(clean) == EXPECTED_NVD,
                f"{name}: expected {EXPECTED_NVD} NVD chunks; got {len(clean)}")
        require(len({m['doc_id'] for _txt, m in clean.values()}) == EXPECTED_NVD_DOCS,
                f"{name}: expected {EXPECTED_NVD_DOCS} different NVD documents")
        if expected_poison_source is None:
            require(not poisoned and collection.count() == expected_count,
                    f"{name}: clean reference is not clean")
        else:
            require(len(poisoned) == EXPECTED_POISON and collection.count() == expected_count,
                    f"{name}: frozen count/poison partition mismatch")
            require(all(m.get("source") == expected_poison_source and
                        m.get("doc_id", "").startswith("POISON-")
                        for _text, m in poisoned.values()),
                    f"{name}: unexpected poisoned source/ID")
        snapshots[name] = clean
        print(f"PASS: {name}: {len(clean)} NVD + {len(poisoned)} frozen poison = "
              f"{collection.count()} chunks")

    reference = snapshots[CLEAN_REF]
    for name in FROZEN:
        require(snapshots[name] == reference,
                f"{name}: clean NVD IDs, text or metadata differ from {CLEAN_REF}")
    print("PASS: all three frozen collections share identical NVD IDs/text/metadata")

    # Check embedding shape on a single clean chunk; does NOT load MiniLM.
    ref_collection = client.get_collection(name=CLEAN_REF)
    sample_id = sorted(reference)[0]
    sample = ref_collection.get(ids=[sample_id], include=["embeddings"])
    vectors = sample.get("embeddings")
    require(vectors is not None and len(vectors) == 1 and
            len(vectors[0]) == EXPECTED_DIM,
            f"Unexpected embedding dimensionality for {sample_id}")
    print(f"PASS: reference clean embedding dimension = {EXPECTED_DIM}")

    print("Reserved NEW Phase B collection names (creation NOT performed):")
    for (attack, ratio), name in TARGETS.items():
        expected_poison = {1: 3, 5: 17, 10: 35, 20: 79}[ratio]
        print(f"  {attack:4} {ratio:2}%  {name:31} "
              f"planned {EXPECTED_NVD}+{expected_poison}={EXPECTED_NVD+expected_poison}")
    print("PASS: PHASE B4 COLLECTION PREFLIGHT COMPLETE")
    print("SAFETY: READ ONLY — no collection creation, ingestion, reset, "
          "file writes, embedding generation or LLM calls")
    print("NEXT: Review the output before running an isolated collection builder.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
