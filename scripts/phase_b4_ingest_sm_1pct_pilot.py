"""PoisonBench B4 — one-collection, fail-closed SM 1% ingestion pilot.

Run from repository root:
    python scripts/phase_b4_ingest_sm_1pct_pilot.py           # READ ONLY
    python scripts/phase_b4_ingest_sm_1pct_pilot.py --apply   # ONE NEW COLLECTION ONLY

New collection: poisonbench_b_sm_1pct_v1 (316 clean + 3 SM poison).
Existing frozen collections are never written to. No resets, deletes, upserts,
get_or_create_collection, or modification of source files. On a write failure,
the *new* collection may be left partially populated: DO NOT manually rerun,
delete it or use --apply again; report the error and inspect first.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_CLEAN = "poisonbench_rpi_clean_v1"
FROZEN = {
    "poisonbench_corpus": (336, "controlled_optimized_embedding"),
    "poisonbench_rpi_clean_v1": (316, None),
    "poisonbench_rpi_poisoned_v1": (336, "controlled_retrieved_prompt_injection"),
}
TARGET = "poisonbench_b_sm_1pct_v1"
DATASET = ROOT / "data/poisoned/phase_b/phase_b_sm_1pct_v1.jsonl"
SOURCE = "controlled_semantic_mimicry"
EXPECTED_POISON = 3
EXPECTED_CLEAN = 316
EXPECTED_DIM = 384
EXPECTED_NVD_DOCUMENTS = 300


def require(condition: bool, msg: str) -> None:
    if not condition:
        raise RuntimeError(msg)


def as_vector_list(vectors):
    """Convert NumPy arrays/Chroma response to ordinary float vectors."""
    require(vectors is not None, "Chroma returned no stored embeddings")
    converted = []
    for v in vectors:
        v = v.tolist() if hasattr(v, "tolist") else list(v)
        require(len(v) == EXPECTED_DIM, f"Expected {EXPECTED_DIM}-dim embedding")
        converted.append(v)
    return converted


def read_clean_and_frozen(client):
    """Read original collections; return clean records with original embeddings."""
    from scripts.phase_b4_collection_preflight import snapshot

    existing = {x if isinstance(x, str) else x.name for x in client.list_collections()}
    require(set(FROZEN) <= existing, f"Missing frozen collection(s): {sorted(set(FROZEN)-existing)}")
    require(TARGET not in existing,
            f"STOP: {TARGET} ALREADY EXISTS. Do not overwrite, delete, or rerun --apply.")
    clean_snapshots = {}
    for name, (expected_count, expected_poison_source) in FROZEN.items():
        c = client.get_collection(name=name)
        clean, poisons = snapshot(c)
        require(c.count() == expected_count and len(clean) == EXPECTED_CLEAN,
                f"Frozen {name}: count or clean partition mismatch")
        require(len({meta["doc_id"] for _txt, meta in clean.values()}) == EXPECTED_NVD_DOCUMENTS,
                f"Frozen {name}: unexpected clean NVD doc count")
        if expected_poison_source is None:
            require(len(poisons) == 0, f"Frozen clean collection {name} has poisons")
        else:
            require(len(poisons) == 20 and all(
                meta.get("source") == expected_poison_source for _text, meta in poisons.values()),
                f"Frozen {name}: poison source or count mismatch")
        clean_snapshots[name] = clean
        print(f"PASS: frozen {name}: {len(clean)} clean + {len(poisons)} poison")

    reference = clean_snapshots[SOURCE_CLEAN]
    require(all(s == reference for s in clean_snapshots.values()),
            "Clean NVD IDs/text/metadata mismatch across frozen collections")
    clean = client.get_collection(name=SOURCE_CLEAN)
    # Import original 316 clean vectors rather than regenerating them.
    get_response = clean.get(include=["documents", "metadatas", "embeddings"])
    ids = get_response.get("ids")
    docs = get_response.get("documents")
    metas = get_response.get("metadatas")
    require(ids is not None and docs is not None and metas is not None,
            "Reference Chroma .get response incomplete")
    vectors = as_vector_list(get_response.get("embeddings"))
    require(len(ids) == len(docs) == len(metas) == len(vectors) == EXPECTED_CLEAN,
            "Reference clean row or vector count inconsistent")
    positions = {id_: i for i, id_ in enumerate(ids)}
    require(set(positions) == set(reference), "Reference clean IDs differ")
    ordered_ids = sorted(positions)
    clean_records = []
    for id_ in ordered_ids:
        i = positions[id_]
        require((docs[i], metas[i]) == reference[id_],
                f"Reference clean text/metadata mismatch: {id_}")
        clean_records.append((id_, docs[i], metas[i], vectors[i]))
    print("PASS: 316 unchanged clean chunks; stored MiniLM embeddings are 384-dimensional")
    return clean_records


def read_poisons():
    from scripts.phase_b_validate_all_datasets import (
        dataset_path, load_jsonl, load_manifest, sha,
    )
    from api.pipeline.chunking import chunk_text
    from api.config import settings
    require(settings.chunk_size_words == 180 and settings.chunk_overlap_words == 30,
            "Unexpected chunk settings; check configuration before proceeding")
    require(dataset_path("SM", 1) == DATASET, "Unexpected B3 dataset path")
    manifest = load_manifest(ROOT / "data/poisoned/phase_b/phase_b_1pct_5pct_allocation_v1.json")
    item = next((x for x in manifest["files"]
                 if x.get("attack") == "SM" and x.get("target_ratio_pct") == 1), None)
    require(item is not None and sha(DATASET) == item["output_sha256"],
            "SM 1% dataset SHA-256 mismatch against B2A manifest")
    rows = load_jsonl(DATASET)
    require(len(rows) == EXPECTED_POISON, "SM 1% dataset expected 3 records")
    benchmark = load_jsonl(ROOT / "data/benchmark/real_trigger_queries_v1.jsonl")
    bench = {str(r["query_id"]).strip().lower(): r["expected_doc_id"] for r in benchmark}
    require(len(bench) == 20, "Benchmark IDs unexpected")
    chunks = []
    seen_docs = set()
    for r in rows:
        qid = r["query_id"]
        doc_id = r["poison_doc_id"]
        require(qid in bench and r["legitimate_doc_id"] == bench[qid],
                f"Poison query/CVE mapping wrong: {qid}")
        require(doc_id == f"POISON-SM-{qid}" and doc_id not in seen_docs,
                f"Unexpected/duplicate SM poison ID: {doc_id}")
        require(r.get("source") == SOURCE and r.get("is_poisoned") is True and
                r.get("attack_type") == "semantic_mimicry", f"Invalid SM poison metadata: {doc_id}")
        require(isinstance(r.get("poison_text"), str) and bool(r["poison_text"].strip()),
                f"Invalid poison text at {doc_id}")
        split = chunk_text(r["poison_text"], doc_id)
        require(len(split) == 1 and split[0]["position"] == 0,
                f"SM poison does not map to single chunk: {doc_id}")
        chunk = split[0]
        chunks.append({
            "chunk_id": chunk["chunk_id"], "text": chunk["text"],
            "metadata": {"doc_id": doc_id, "source": SOURCE, "timestamp": "",
                         "is_poisoned": True, "position": 0},
        })
        seen_docs.add(doc_id)
    require(len({p["chunk_id"] for p in chunks}) == EXPECTED_POISON,
            "Duplicate poison chunk IDs")
    print(f"PASS: SM 1% dataset integrity and manifest hash: {len(chunks)} single-chunk poisons")
    return chunks


def read_target_rows(target):
    response = target.get(include=["documents", "metadatas"])
    require(response.get("ids") is not None and response.get("documents") is not None
            and response.get("metadatas") is not None, "Incomplete target collection response")
    ids, docs, metas = response["ids"], response["documents"], response["metadatas"]
    require(len(ids) == len(docs) == len(metas) == target.count(), "Target response count mismatch")
    return dict(zip(ids, zip(docs, metas)))


def assert_target(target, clean_records, poison_chunks):
    actual = read_target_rows(target)
    expected_ids = {r[0] for r in clean_records} | {r["chunk_id"] for r in poison_chunks}
    require(target.count() == EXPECTED_CLEAN + EXPECTED_POISON and
            set(actual) == expected_ids, "Target collection count or IDs incorrect")
    for id_, doc, meta, _vector in clean_records:
        require(actual[id_] == (doc, meta), f"Copied clean chunk mismatch at {id_}")
    for poison in poison_chunks:
        require(actual[poison["chunk_id"]] == (poison["text"], poison["metadata"]),
                f"Poison chunk mismatch at {poison['chunk_id']}")
    counts = Counter(meta["source"] for _doc, meta in actual.values())
    require(counts == {"NVD": 316, SOURCE: 3}, f"Unexpected source distribution: {counts}")
    # Spot-check all copied clean embedding vectors. Only 316 vectors, no model inference.
    copied = target.get(ids=[r[0] for r in clean_records], include=["embeddings"])
    copied_ids = copied["ids"]
    copied_vectors = as_vector_list(copied.get("embeddings"))
    vector_lookup = dict(zip(copied_ids, copied_vectors))
    require(len(vector_lookup) == EXPECTED_CLEAN, "Missing copied clean embeddings")
    maxdiff = max(abs(a-b) for id_, _doc, _meta, embedding in clean_records
                  for a, b in zip(vector_lookup[id_], embedding))
    require(maxdiff <= 1e-6, f"Clean embedding copy deviation: {maxdiff}")
    print(f"PASS: Target {TARGET} has 316 exact clean + 3 SM poison = 319 chunks")
    print(f"PASS: All 316 copied clean embedding vectors preserved (max abs diff {maxdiff:.3g})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write to exactly one new isolated collection")
    args = parser.parse_args()
    print("=== Phase B4 — SM 1% isolated collection pilot ===")

    from api.config import settings
    from api.pipeline.vectorstore import get_client
    from scripts.phase_b_validate_all_datasets import main as validate_all
    require(settings.chroma_collection == "poisonbench_corpus", "Unexpected default collection")
    print("Validating all 12 frozen Phase B datasets (read-only)...")
    validate_all()    # fail closed if any local source file has changed
    print("Connecting to Chroma (existing frozen collections: READ ONLY)...")
    client = get_client()
    clean_records = read_clean_and_frozen(client)
    poisons = read_poisons()
    require({p["chunk_id"] for p in poisons}.isdisjoint({r[0] for r in clean_records}),
            "Poison ID collides with clean ID")
    print(f"Target NEW collection: {TARGET} — must be absent")
    print(f"Planned: {len(clean_records)} clean + {len(poisons)} SM poison = 319 chunks (0.94%)")
    if not args.apply:
        print("PASS: SM 1% INGESTION PILOT PREFLIGHT COMPLETE")
        print("SAFETY: READ ONLY — no file writes, embeddings computed or collections changed")
        print("NEXT: Review with assistant BEFORE running --apply")
        return

    from api.pipeline.embedding import embed_texts
    print("Embedding exactly 3 SM poison texts with project MiniLM (no DB writes yet)...")
    poison_vectors = embed_texts([p["text"] for p in poisons])
    require(len(poison_vectors) == EXPECTED_POISON and
            all(len(v) == EXPECTED_DIM for v in poison_vectors),
            "SM poison embedding shape unexpected")
    require(TARGET not in {x if isinstance(x, str) else x.name for x in client.list_collections()},
            "Target appeared during embedding; refusing to create or modify")
    print("Creating only NEW target collection; this is the first DB write...")
    # No reset or delete. If later step fails, leave target as-is and inspect.
    target = client.create_collection(name=TARGET, metadata={"hnsw:space": "cosine"})
    try:
        for offset in range(0, len(clean_records), 64):
            batch = clean_records[offset:offset + 64]
            target.add(
                ids=[r[0] for r in batch],
                documents=[r[1] for r in batch],
                metadatas=[r[2] for r in batch],
                embeddings=[r[3] for r in batch],
            )
            print(f"  Copied {min(offset + len(batch), EXPECTED_CLEAN)}/{EXPECTED_CLEAN} clean chunks")
        target.add(
            ids=[p["chunk_id"] for p in poisons],
            documents=[p["text"] for p in poisons],
            metadatas=[p["metadata"] for p in poisons],
            embeddings=poison_vectors,
        )
        assert_target(target, clean_records, poisons)
        for name, (expected_count, _src) in FROZEN.items():
            require(client.get_collection(name=name).count() == expected_count,
                    f"Frozen reference collection count changed unexpectedly: {name}")
        print("PASS: Three frozen collections unchanged in count")
        print("PASS: PHASE B4 SM 1% PILOT INGESTION COMPLETE")
        print("NEXT: Review output; verify before scaling other 11 experimental collections")
    except Exception:
        print(f"STOP: Write/verification interrupted. {TARGET} MAY BE PARTIALLY POPULATED.",
              file=sys.stderr)
        print("SAFETY: Do not delete/reset, rerun --apply, or continue to other collections.",
              file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
