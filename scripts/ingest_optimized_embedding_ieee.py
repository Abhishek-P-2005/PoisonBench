import os
os.environ["CHROMA_COLLECTION"] = "poisonbench_ieee_5000_obea"

import json
import chromadb
from pathlib import Path
from api.pipeline.chunking import chunk_text
from api.pipeline.vectorstore import upsert_chunks, get_collection

POISON_PATH = Path("ieee_5000/poisoned/optimized_embedding_100.jsonl")
SOURCE_COLLECTION = "poisonbench_ieee_5000_clean"
TARGET_COLLECTION = "poisonbench_ieee_5000_obea"

def main():
    client = chromadb.HttpClient(host="localhost", port=8001)
    source = client.get_collection(SOURCE_COLLECTION)

    records = [
        json.loads(line)
        for line in POISON_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(records) != 100:
        raise ValueError(f"Expected 100 poison documents, found {len(records)}")

    query_ids = set()
    poison_ids = set()
    chunks = []

    for record in records:
        if record["is_poisoned"] is not True:
            raise ValueError("Invalid poison record")
        if record["attack_type"] != "optimization_embedding":
            raise ValueError("Unexpected attack type")
        if not record["poison_doc_id"].startswith("POISON-OBEA-"):
            raise ValueError("Unexpected poison document ID")

        query_ids.add(record["query_id"])
        poison_ids.add(record["poison_doc_id"])

        for chunk in chunk_text(record["poison_text"], record["poison_doc_id"]):
            chunks.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "doc_id": record["poison_doc_id"],
                "source": record["source"],
                "timestamp": "",
                "is_poisoned": True,
                "position": chunk["position"]
            })

    if len(query_ids) != 100 or len(poison_ids) != 100:
        raise ValueError("Duplicate query IDs or poison document IDs")

    if source.count() != 5000:
        raise RuntimeError("Unexpected clean corpus size")

    if TARGET_COLLECTION in [c.name for c in client.list_collections()]:
        raise RuntimeError("OBEA collection already exists. Refusing to overwrite.")

    target = get_collection()

    print("Copying 5,000 clean chunks...")
    offset = 0

    while offset < source.count():
        batch = source.get(
            limit=500,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )

        target.upsert(
            ids=batch["ids"],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
            embeddings=batch["embeddings"]
        )

        offset += len(batch["ids"])

    print("Clean chunks copied:", target.count())
    print("Optimized poison documents:", len(records))
    print("Optimized poison chunks:", len(chunks))

    upsert_chunks(chunks)

    print("Final OBEA collection count:", target.count())
    print("Original clean collection count:", source.count())

if __name__ == "__main__":
    main()
