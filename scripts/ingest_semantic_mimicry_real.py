import json
from pathlib import Path

from api.pipeline.chunking import chunk_text
from api.pipeline.vectorstore import upsert_chunks, get_collection


POISON_PATH = Path("data/poisoned/semantic_mimicry_real.jsonl")


def load_poison_records():
    records = []

    with POISON_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if record.get("is_poisoned") is not True:
                raise ValueError(
                    f"{record.get('poison_doc_id')} is not marked as poisoned."
                )

            records.append(record)

    return records


def build_chunks(records):
    chunks = []

    for record in records:
        poison_doc_id = record["poison_doc_id"]
        poison_text = record["poison_text"]

        text_chunks = chunk_text(
            text=poison_text,
            doc_id=poison_doc_id,
        )

        for chunk in text_chunks:
            chunks.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "doc_id": poison_doc_id,
                    "source": record["source"],
                    "timestamp": "",
                    "is_poisoned": True,
                    "position": chunk["position"],
                }
            )

    return chunks


def main():
    collection = get_collection()

    before_count = collection.count()

    records = load_poison_records()
    chunks = build_chunks(records)

    print(f"Poison documents loaded: {len(records)}")
    print(f"Poison chunks generated: {len(chunks)}")
    print(f"Collection count before ingestion: {before_count}")

    upsert_chunks(chunks)

    after_count = collection.count()

    print(f"Collection count after ingestion: {after_count}")
    print(f"Net new chunks: {after_count - before_count}")


if __name__ == "__main__":
    main()