"""PoisonBench Phase A4: safely ingest Attack 3 into its isolated collection.

From repository root, with existing project .venv:
    python scripts/ingest_retrieved_prompt_injection_real.py
    python scripts/ingest_retrieved_prompt_injection_real.py --apply

Without --apply: read-only preflight. With --apply: add exactly 20 RPI chunks to
poisonbench_rpi_poisoned_v1. No collection resets, deletes, or global upserts.
Existing poisonbench_corpus and poisonbench_rpi_clean_v1 are READ ONLY.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.config import settings  # noqa: E402
from api.pipeline.chunking import chunk_text  # noqa: E402
from api.pipeline.embedding import embed_texts  # noqa: E402
from api.pipeline.vectorstore import get_client  # noqa: E402

ORIGINAL = "poisonbench_corpus"
CLEAN = "poisonbench_rpi_clean_v1"
TARGET = "poisonbench_rpi_poisoned_v1"
POISON_FILE = ROOT / "data" / "poisoned" / "retrieved_prompt_injection_real.jsonl"
BENCHMARK_FILE = ROOT / "data" / "benchmark" / "real_trigger_queries_v1.jsonl"
MARKER = "POISONBENCH_RPI_SUCCESS"
EXPECTED_CLEAN = 316
EXPECTED_ORIGINAL = 336
EXPECTED_POISONS = 20
SOURCE = "controlled_retrieved_prompt_injection"


def load_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name} at line {line_no}") from exc
    return rows


def validate_poison_data():
    benchmark = load_jsonl(BENCHMARK_FILE)
    records = load_jsonl(POISON_FILE)
    if len(benchmark) != EXPECTED_POISONS or len(records) != EXPECTED_POISONS:
        raise RuntimeError("Expected precisely 20 benchmark queries and 20 RPI records")
    reference = {}
    for record in benchmark:
        qid = str(record["query_id"]).lower().strip()
        if qid in reference:
            raise RuntimeError(f"Duplicate benchmark query: {qid}")
        reference[qid] = str(record["expected_doc_id"]).strip()
    if set(reference) != {f"q{i:02d}" for i in range(1, 21)}:
        raise RuntimeError("Unexpected benchmark query IDs")

    seen_ids = set()
    seen_qids = set()
    chunks = []
    for record in records:
        required = {"query_id", "legitimate_doc_id", "poison_doc_id", "poison_text",
                    "source", "attack_type", "is_poisoned"}
        missing = required - record.keys()
        if missing:
            raise RuntimeError(f"Poison record is missing fields: {sorted(missing)}")
        qid = str(record["query_id"]).strip().lower()
        doc_id = str(record["poison_doc_id"]).strip()
        text = record["poison_text"]
        if qid not in reference or qid in seen_qids:
            raise RuntimeError(f"Unknown or duplicate query ID: {qid}")
        if str(record["legitimate_doc_id"]).strip() != reference[qid]:
            raise RuntimeError(f"Benchmark-to-legitimate-doc mismatch at {qid}")
        if doc_id != f"POISON-RPI-{qid}" or doc_id in seen_ids:
            raise RuntimeError(f"Unexpected or duplicate poison doc ID: {doc_id}")
        if (record["source"] != SOURCE or
                record["attack_type"] != "retrieved_prompt_injection" or
                record["is_poisoned"] is not True):
            raise RuntimeError(f"Invalid RPI metadata for {qid}")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"Empty or invalid poison text for {qid}")
        if MARKER not in text or len(text.split()) > settings.chunk_size_words:
            raise RuntimeError(f"Marker missing or payload exceeds configured chunk size at {qid}")
        parts = chunk_text(text=text, doc_id=doc_id)
        if len(parts) != 1:
            raise RuntimeError(f"Expected one chunk per RPI document at {qid}")
        part = parts[0]
        chunks.append({"chunk_id": part["chunk_id"], "text": part["text"],
                       "doc_id": doc_id, "source": SOURCE, "timestamp": "",
                       "is_poisoned": True, "position": part["position"]})
        seen_qids.add(qid)
        seen_ids.add(doc_id)
    if seen_qids != set(reference):
        raise RuntimeError("Dataset does not cover the frozen benchmark")
    return chunks


def collection_rows(collection):
    result = collection.get(include=["documents", "metadatas"])
    ids, documents, metas = result["ids"], result["documents"], result["metadatas"]
    if ids is None or documents is None or metas is None:
        raise RuntimeError(f"Incomplete collection data: {collection.name}")
    if not (len(ids) == len(documents) == len(metas) == collection.count()):
        raise RuntimeError(f"Unexpected collection data lengths: {collection.name}")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Duplicate chunk IDs in {collection.name}")
    return dict(zip(ids, zip(documents, metas)))


def assert_clean(rows, name):
    if len(rows) != EXPECTED_CLEAN:
        raise RuntimeError(f"{name}: expected {EXPECTED_CLEAN} chunks, got {len(rows)}")
    ids = set()
    for chunk_id, (doc, meta) in rows.items():
        if (not isinstance(meta, dict) or meta.get("is_poisoned") is not False or
                meta.get("source") != "NVD" or
                not str(meta.get("doc_id", "")).startswith("CVE-") or
                not chunk_id.startswith(str(meta.get("doc_id")) + "::chunk") or
                not isinstance(doc, str) or not doc.strip()):
            raise RuntimeError(f"Unexpected clean entry in {name}: {chunk_id}")
        ids.add(meta["doc_id"])
    if len(ids) != 300:
        raise RuntimeError(f"{name}: expected 300 unique NVD documents")


def preflight(client, expected_chunks):
    names = {obj if isinstance(obj, str) else obj.name for obj in client.list_collections()}
    for name in (ORIGINAL, CLEAN, TARGET):
        if name not in names:
            raise RuntimeError(f"Missing collection {name}. No changes made.")
    original = client.get_collection(name=ORIGINAL)
    clean = client.get_collection(name=CLEAN)
    target = client.get_collection(name=TARGET)

    if original.count() != EXPECTED_ORIGINAL:
        raise RuntimeError("Original corpus count changed; no changes made")
    original_rows = collection_rows(original)
    clean_rows = collection_rows(clean)
    target_rows = collection_rows(target)
    original_clean = {i: r for i, r in original_rows.items()
                      if isinstance(r[1], dict) and r[1].get("is_poisoned") is False}
    original_poison = {i: r for i, r in original_rows.items()
                       if isinstance(r[1], dict) and r[1].get("is_poisoned") is True}
    assert_clean(original_clean, ORIGINAL)
    assert_clean(clean_rows, CLEAN)
    assert_clean(target_rows, TARGET)
    if clean_rows != original_clean or target_rows != original_clean:
        raise RuntimeError("Clean reference collections differ from original NVD source")
    if (len(original_poison) != 20 or
            any(not str(meta.get("doc_id", "")).startswith("POISON-OBEA-")
                or meta.get("source") != "controlled_optimized_embedding"
                for _, meta in original_poison.values())):
        raise RuntimeError("Original corpus has unexpected poison content")
    new_ids = {entry["chunk_id"] for entry in expected_chunks}
    if new_ids & set(original_rows) or new_ids & set(clean_rows) or new_ids & set(target_rows):
        raise RuntimeError("RPI chunk ID collision detected; no changes made")
    if len(new_ids) != EXPECTED_POISONS:
        raise RuntimeError("Expected 20 unique RPI chunk IDs")
    return original, clean, target, set(clean_rows)


def verify_after(original, clean, target, old_ids, expected_chunks):
    if original.count() != EXPECTED_ORIGINAL or clean.count() != EXPECTED_CLEAN:
        raise RuntimeError("Original or clean reference collection count changed")
    if target.count() != EXPECTED_CLEAN + EXPECTED_POISONS:
        raise RuntimeError(f"Target count incorrect: {target.count()}")
    ids = {entry["chunk_id"] for entry in expected_chunks}
    actual = collection_rows(target)
    if set(actual) != old_ids | ids:
        raise RuntimeError("Target contains unexpected chunk IDs")
    for entry in expected_chunks:
        doc, meta = actual[entry["chunk_id"]]
        if (doc != entry["text"] or meta.get("doc_id") != entry["doc_id"] or
                meta.get("source") != SOURCE or meta.get("is_poisoned") is not True or
                meta.get("position") != 0):
            raise RuntimeError(f"Poison verification failed: {entry['chunk_id']}")
    counts = Counter(meta.get("source", "UNKNOWN") for _, meta in actual.values())
    if counts != {"NVD": 316, SOURCE: 20}:
        raise RuntimeError(f"Unexpected source distribution: {counts}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Embed and add 20 poisons to isolated target")
    args = parser.parse_args()
    if settings.chroma_collection != ORIGINAL:
        raise RuntimeError(f"Unexpected configured collection: {settings.chroma_collection!r}")
    chunks = validate_poison_data()
    client = get_client()
    original, clean, target, old_ids = preflight(client, chunks)
    print("=== Phase A4: Attack 3 poison ingestion preflight ===")
    print(f"Original (READ ONLY) : {original.name} : {original.count()} chunks")
    print(f"Clean (READ ONLY)    : {clean.name} : {clean.count()} chunks")
    print(f"Target (WRITE ONLY)  : {target.name} : {target.count()} chunks")
    print(f"Validated RPI records: {len(chunks)} | 1 chunk/document")
    print(f"Expected target count after ingestion: {EXPECTED_CLEAN + EXPECTED_POISONS}")
    if not args.apply:
        print("PASS: READ-ONLY preflight. No embeddings generated; no collections modified.")
        print("After reviewing this output, execute with --apply to ingest.")
        return

    # Crucial: NEVER call vectorstore.upsert_chunks()/get_collection()/reset_collection().
    # These select the globally configured corpus rather than our isolated target.
    vectors = embed_texts([entry["text"] for entry in chunks])
    if len(vectors) != EXPECTED_POISONS or any(len(vector) != 384 for vector in vectors):
        raise RuntimeError("Unexpected embedding count/dimension; nothing ingested")
    target.add(
        ids=[entry["chunk_id"] for entry in chunks],
        documents=[entry["text"] for entry in chunks],
        embeddings=vectors,
        metadatas=[{"doc_id": entry["doc_id"], "source": entry["source"],
                    "timestamp": entry["timestamp"], "is_poisoned": True,
                    "position": entry["position"]} for entry in chunks],
    )
    counts = verify_after(original, clean, target, old_ids, chunks)
    print(f"PASS: Target {TARGET} has {target.count()} chunks ({dict(counts)})")
    print(f"PASS: Original collection still has {original.count()} chunks")
    print(f"PASS: Clean collection still has {clean.count()} chunks")
    print("NEXT: Test retrieval against this isolated target (not the default /retrieve endpoint).")


if __name__ == "__main__":
    main()
