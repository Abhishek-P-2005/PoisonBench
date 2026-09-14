"""
PoisonBench
Attack 2 - Optimization-Based Embedding Attack
Main Roadmap Phase 7

Ingests the frozen 20-document optimized embedding attack corpus
produced during Phase 6 into ChromaDB.

Expected precondition:

    Clean NVD corpus is already present in ChromaDB.
    Attack-1 semantic-mimicry poison documents must NOT be present.

This script DOES NOT reset the collection.
It only validates, chunks, and ingests the Attack-2 documents.
"""

import json
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from api.pipeline.chunking import chunk_text
from api.pipeline.vectorstore import (
    upsert_chunks,
    get_collection,
)


# ============================================================
# CONFIGURATION
# ============================================================

POISON_PATH = (
    PROJECT_ROOT
    / "data"
    / "poisoned"
    / "optimized_embedding_attack_real.jsonl"
)

EXPECTED_DOCUMENTS = 20

EXPECTED_ATTACK_TYPE = "optimization_embedding"

EXPECTED_SOURCE = "controlled_optimized_embedding"

POISON_ID_PREFIX = "POISON-OBEA-"


# ============================================================
# LOAD ATTACK-2 DOCUMENTS
# ============================================================

def load_poison_records():
    """
    Load and validate the frozen Phase-6 optimized poison corpus.
    """

    if not POISON_PATH.exists():
        raise FileNotFoundError(
            f"Attack-2 poison corpus not found:\n"
            f"{POISON_PATH}"
        )

    records = []

    seen_query_ids = set()
    seen_poison_ids = set()

    with POISON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}"
                ) from exc

            required_fields = {
                "query_id",
                "legitimate_doc_id",
                "poison_doc_id",
                "source",
                "attack_type",
                "is_poisoned",
                "poison_text",
            }

            missing = (
                required_fields
                - record.keys()
            )

            if missing:
                raise KeyError(
                    f"Record on line {line_number} "
                    f"is missing fields: "
                    f"{sorted(missing)}"
                )

            query_id = str(
                record["query_id"]
            ).strip().lower()

            poison_doc_id = str(
                record["poison_doc_id"]
            ).strip()

            poison_text = str(
                record["poison_text"]
            ).strip()

            # --------------------------------------------
            # Ground-truth validation
            # --------------------------------------------

            if record["is_poisoned"] is not True:
                raise ValueError(
                    f"{poison_doc_id} is not "
                    f"marked is_poisoned=True."
                )

            if (
                record["attack_type"]
                != EXPECTED_ATTACK_TYPE
            ):
                raise ValueError(
                    f"Unexpected attack_type for "
                    f"{poison_doc_id}: "
                    f"{record['attack_type']}"
                )

            if (
                record["source"]
                != EXPECTED_SOURCE
            ):
                raise ValueError(
                    f"Unexpected source for "
                    f"{poison_doc_id}: "
                    f"{record['source']}"
                )

            if not poison_doc_id.startswith(
                POISON_ID_PREFIX
            ):
                raise ValueError(
                    f"Unexpected poison ID: "
                    f"{poison_doc_id}"
                )

            if not poison_text:
                raise ValueError(
                    f"Empty poison text for "
                    f"{poison_doc_id}"
                )

            # --------------------------------------------
            # Duplicate validation
            # --------------------------------------------

            if query_id in seen_query_ids:
                raise ValueError(
                    f"Duplicate query_id: "
                    f"{query_id}"
                )

            if poison_doc_id in seen_poison_ids:
                raise ValueError(
                    f"Duplicate poison_doc_id: "
                    f"{poison_doc_id}"
                )

            seen_query_ids.add(
                query_id
            )

            seen_poison_ids.add(
                poison_doc_id
            )

            records.append(
                record
            )

    if len(records) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_DOCUMENTS} "
            f"Attack-2 documents but found "
            f"{len(records)}."
        )

    return records


# ============================================================
# BUILD CHUNKS
# ============================================================

def build_chunks(records):
    """
    Convert the optimized poison documents into the same chunk
    representation used by the baseline RAG pipeline.
    """

    chunks = []

    for record in records:

        poison_doc_id = str(
            record["poison_doc_id"]
        )

        poison_text = str(
            record["poison_text"]
        )

        text_chunks = chunk_text(
            text=poison_text,
            doc_id=poison_doc_id,
        )

        if not text_chunks:
            raise ValueError(
                f"No chunks generated for "
                f"{poison_doc_id}"
            )

        for chunk in text_chunks:

            chunks.append(
                {
                    "chunk_id": (
                        chunk["chunk_id"]
                    ),

                    "text": (
                        chunk["text"]
                    ),

                    "doc_id": (
                        poison_doc_id
                    ),

                    "source": (
                        record["source"]
                    ),

                    "timestamp": "",

                    "is_poisoned": True,

                    "position": (
                        chunk["position"]
                    ),
                }
            )

    return chunks


# ============================================================
# COLLECTION INSPECTION
# ============================================================

def inspect_collection(collection):
    """
    Inspect existing document IDs before ingestion.

    This prevents accidental evaluation with Attack-1 and
    Attack-2 poisons simultaneously present.
    """

    existing = collection.get(
        include=["metadatas"]
    )

    metadata_rows = (
        existing.get("metadatas")
        or []
    )

    attack1_ids = set()

    attack2_ids = set()

    for metadata in metadata_rows:

        if not metadata:
            continue

        doc_id = str(
            metadata.get(
                "doc_id",
                "",
            )
        )

        if doc_id.startswith(
            "POISON-SM-"
        ):
            attack1_ids.add(
                doc_id
            )

        if doc_id.startswith(
            POISON_ID_PREFIX
        ):
            attack2_ids.add(
                doc_id
            )

    return (
        attack1_ids,
        attack2_ids,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("POISONBENCH")
    print(
        "ATTACK 2 - OPTIMIZATION-BASED "
        "EMBEDDING ATTACK"
    )
    print("MAIN ROADMAP PHASE 7")
    print(
        "INGEST OPTIMIZED POISON CORPUS"
    )
    print("=" * 72)

    print()
    print(
        f"Attack corpus : "
        f"{POISON_PATH}"
    )

    # --------------------------------------------------------
    # Load + validate Phase-6 corpus
    # --------------------------------------------------------

    records = load_poison_records()

    print()
    print(
        f"Validated Attack-2 documents : "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Generate chunks
    # --------------------------------------------------------

    chunks = build_chunks(
        records
    )

    print(
        f"Attack-2 chunks generated    : "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # Inspect current Chroma collection
    # --------------------------------------------------------

    collection = get_collection()

    before_count = (
        collection.count()
    )

    attack1_ids, attack2_ids = (
        inspect_collection(
            collection
        )
    )

    print(
        f"Collection count before      : "
        f"{before_count}"
    )

    print(
        f"Existing Attack-1 docs       : "
        f"{len(attack1_ids)}"
    )

    print(
        f"Existing Attack-2 docs       : "
        f"{len(attack2_ids)}"
    )

    # --------------------------------------------------------
    # Critical experiment-isolation check
    # --------------------------------------------------------

    if attack1_ids:

        print()
        print("=" * 72)
        print("INGESTION BLOCKED")
        print("=" * 72)

        print(
            "Attack-1 semantic-mimicry "
            "documents are still present "
            "in ChromaDB."
        )

        print()
        print(
            "Attack 2 must be evaluated "
            "against the clean NVD corpus, "
            "not against a collection that "
            "also contains Attack 1 poisons."
        )

        print()
        print(
            "No Attack-2 documents were "
            "ingested."
        )

        print()
        print(
            "Restore/reset the clean NVD "
            "baseline before rerunning "
            "Phase 7."
        )

        print("=" * 72)

        return

    # --------------------------------------------------------
    # Idempotency information
    # --------------------------------------------------------

    if attack2_ids:

        print()
        print(
            "NOTE: Existing Attack-2 "
            "documents were detected."
        )

        print(
            "upsert_chunks() is idempotent "
            "for matching chunk IDs."
        )

    # --------------------------------------------------------
    # Ingest Attack 2
    # --------------------------------------------------------

    stored_count = upsert_chunks(
        chunks
    )

    after_count = (
        collection.count()
    )

    # --------------------------------------------------------
    # Post-ingestion validation
    # --------------------------------------------------------

    attack1_after, attack2_after = (
        inspect_collection(
            collection
        )
    )

    print()
    print("=" * 72)
    print("PHASE 7 INGESTION RESULT")
    print("=" * 72)

    print(
        f"Chunks submitted             : "
        f"{stored_count}"
    )

    print(
        f"Collection count before      : "
        f"{before_count}"
    )

    print(
        f"Collection count after       : "
        f"{after_count}"
    )

    print(
        f"Net collection change        : "
        f"{after_count - before_count:+d}"
    )

    print(
        f"Attack-1 documents present   : "
        f"{len(attack1_after)}"
    )

    print(
        f"Attack-2 documents present   : "
        f"{len(attack2_after)}"
    )

    # --------------------------------------------------------
    # Final assertions
    # --------------------------------------------------------

    if attack1_after:
        raise RuntimeError(
            "Experiment contamination: "
            "Attack-1 documents detected "
            "after Attack-2 ingestion."
        )

    if len(attack2_after) != EXPECTED_DOCUMENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_DOCUMENTS} "
            f"Attack-2 documents after "
            f"ingestion but detected "
            f"{len(attack2_after)}."
        )

    print()
    print(
        "PASS: Attack-2 optimized poison "
        "corpus is present in ChromaDB."
    )

    print(
        "PASS: No Attack-1 semantic-mimicry "
        "documents are present."
    )

    print()
    print(
        "Phase 7 ingestion completed."
    )

    print(
        "Next: Main Roadmap Phase 8 - "
        "full 20-query retrieval attack "
        "evaluation."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()