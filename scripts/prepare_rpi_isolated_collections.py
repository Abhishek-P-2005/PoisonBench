"""PoisonBench Phase A4: safely clone ONLY clean NVD chunks for Attack 3.

Run from the repository root, using the existing project .venv:
    python scripts/prepare_rpi_isolated_collections.py
    python scripts/prepare_rpi_isolated_collections.py --apply

Default = read-only preflight. --apply creates two NEW collections (no resets).
The legacy poisonbench_corpus collection is read-only in both modes.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.pipeline.vectorstore import get_client  # noqa: E402
from api.config import settings  # noqa: E402

SOURCE_NAME = "poisonbench_corpus"
CLEAN_NAME = "poisonbench_rpi_clean_v1"
POISONED_NAME = "poisonbench_rpi_poisoned_v1"
EXPECTED_CLEAN_CHUNKS = 316
EXPECTED_CLEAN_DOCS = 300
EXPECTED_OBEA_CHUNKS = 20
EXPECTED_TOTAL_CHUNKS = EXPECTED_CLEAN_CHUNKS + EXPECTED_OBEA_CHUNKS
BATCH_SIZE = 64


def list_names(client):
    return {item if isinstance(item, str) else item.name
            for item in client.list_collections()}


def read_source_and_validate(source):
    count = source.count()
    if count != EXPECTED_TOTAL_CHUNKS:
        raise RuntimeError(
            f"Source has {count} chunks, expected {EXPECTED_TOTAL_CHUNKS}. "
            "No changes made."
        )

    clean = []
    poison_ids = []
    clean_doc_ids = set()
    seen_chunk_ids = set()
    source_counts = Counter()

    for offset in range(0, count, BATCH_SIZE):
        block = source.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = block.get("ids")
        texts = block.get("documents")
        metadatas = block.get("metadatas")
        embeddings = block.get("embeddings")
        if any(item is None for item in (ids, texts, metadatas, embeddings)):
            raise RuntimeError("Chroma returned incomplete source data. No changes made.")
        if not (len(ids) == len(texts) == len(metadatas) == len(embeddings)):
            raise RuntimeError("Inconsistent Chroma batch lengths. No changes made.")

        for chunk_id, text, metadata, embedding in zip(ids, texts, metadatas, embeddings):
            if chunk_id in seen_chunk_ids:
                raise RuntimeError(f"Duplicate chunk ID: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
            if not isinstance(metadata, dict):
                raise RuntimeError(f"Invalid metadata for {chunk_id}")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"Missing text for {chunk_id}")
            doc_id = str(metadata.get("doc_id", ""))
            source_counts[str(metadata.get("source", "MISSING"))] += 1
            poison_flag = metadata.get("is_poisoned")
            if poison_flag is False:
                if metadata.get("source") != "NVD" or not doc_id.startswith("CVE-"):
                    raise RuntimeError(f"Unexpected clean metadata for {chunk_id}")
                if not chunk_id.startswith(doc_id + "::chunk"):
                    raise RuntimeError(f"Unexpected clean chunk ID: {chunk_id}")
                vector = [float(v) for v in embedding]
                if len(vector) != 384:
                    raise RuntimeError(f"Embedding dimensions not 384 for {chunk_id}")
                clean.append((chunk_id, text, dict(metadata), vector))
                clean_doc_ids.add(doc_id)
            elif poison_flag is True:
                if not doc_id.startswith("POISON-OBEA-") or metadata.get("source") != "controlled_optimized_embedding":
                    raise RuntimeError(f"Unexpected poison document: {doc_id}")
                poison_ids.append(doc_id)
            else:
                raise RuntimeError(f"Invalid poison flag for {chunk_id}: {poison_flag!r}")

    if len(clean) != EXPECTED_CLEAN_CHUNKS:
        raise RuntimeError(f"Expected 316 clean chunks; found {len(clean)}")
    if len(clean_doc_ids) != EXPECTED_CLEAN_DOCS:
        raise RuntimeError(f"Expected 300 clean documents; found {len(clean_doc_ids)}")
    if len(poison_ids) != EXPECTED_OBEA_CHUNKS or len(set(poison_ids)) != EXPECTED_OBEA_CHUNKS:
        raise RuntimeError("Expected exactly 20 distinct OBEA poison chunks")
    if len(seen_chunk_ids) != count:
        raise RuntimeError("Not all source chunks were read")

    return clean, poison_ids, source_counts


def verify_target(collection, expected_ids):
    if collection.count() != len(expected_ids):
        raise RuntimeError(f"Target {collection.name} count mismatch")
    response = collection.get(include=["metadatas"])
    if set(response["ids"]) != expected_ids:
        raise RuntimeError(f"Target {collection.name} chunk ID mismatch")
    if any(m is None or m.get("is_poisoned") is not False or m.get("source") != "NVD"
           for m in response["metadatas"]):
        raise RuntimeError(f"Target {collection.name} contains non-clean metadata")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Create both isolated clean-only collections after verification")
    args = parser.parse_args()

    if settings.chroma_collection != SOURCE_NAME:
        raise RuntimeError(
            f"Configured collection is {settings.chroma_collection!r}, not {SOURCE_NAME!r}. "
            "Stop and review settings before proceeding."
        )

    client = get_client()
    names = list_names(client)
    if SOURCE_NAME not in names:
        raise RuntimeError(f"Source collection {SOURCE_NAME!r} is missing")
    existing_targets = sorted({CLEAN_NAME, POISONED_NAME} & names)
    if existing_targets:
        raise RuntimeError(
            f"Target collection(s) already exist: {existing_targets}. "
            "Stopping without changing them; inspect with your supervisor first."
        )

    source = client.get_collection(name=SOURCE_NAME)
    clean, poison_ids, sources = read_source_and_validate(source)
    print("=== Phase A4: Attack 3 collection preflight ===")
    print(f"Original collection     : {SOURCE_NAME} (READ ONLY)")
    print(f"Original chunk count    : {source.count()}")
    print(f"Sources                 : {dict(sources)}")
    print(f"Clean NVD chunks to copy: {len(clean)}")
    print(f"Excluded OBEA chunks    : {len(poison_ids)}")
    print(f"New clean collection    : {CLEAN_NAME}")
    print(f"New poisoned collection : {POISONED_NAME} (starts clean)")

    if not args.apply:
        print("PASS: Preflight validated. READ-ONLY: no collections created.")
        print("Once reviewed, run this script again with --apply.")
        return

    expected_ids = {row[0] for row in clean}
    for name in (CLEAN_NAME, POISONED_NAME):
        collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        for start in range(0, len(clean), BATCH_SIZE):
            batch = clean[start:start + BATCH_SIZE]
            collection.add(
                ids=[item[0] for item in batch],
                documents=[item[1] for item in batch],
                metadatas=[item[2] for item in batch],
                embeddings=[item[3] for item in batch],
            )
        verify_target(collection, expected_ids)
        print(f"PASS: {name}: {collection.count()} clean NVD chunks")

    if source.count() != EXPECTED_TOTAL_CHUNKS:
        raise RuntimeError("Original collection count changed unexpectedly")
    print("PASS: Both isolated collections initialized and verified.")
    print("PASS: Original collection was never written or reset.")
    print("NEXT: Add Attack 3 poisons ONLY to the isolated poisoned collection.")


if __name__ == "__main__":
    main()
