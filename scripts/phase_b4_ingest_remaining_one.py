"""PoisonBench Phase B4 — stage ONE of 11 remaining isolated collections.

From repository root (start with OBEA 1%):
    python scripts/phase_b4_ingest_remaining_one.py --attack OBEA --ratio 1
    python scripts/phase_b4_ingest_remaining_one.py --attack OBEA --ratio 1 --apply

NO FLAGS / no --apply: READ ONLY (including Chroma queries, no embeddings computed).
--apply: explicitly creates ONE new collection, never deletes, resets, upserts, or overwrites.
Always independently inspects the completed SM 1% pilot and three frozen collections.
If --apply fails, DO NOT rerun, delete, reset or advance; inspect the possibly partial collection.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_b_validate_all_datasets import (  # noqa: E402
    RATIOS, SOURCES, dataset_path, load_jsonl, load_manifest, sha,
)

FROZEN = {
    "poisonbench_corpus": (336, 20, "controlled_optimized_embedding"),
    "poisonbench_rpi_clean_v1": (316, 0, None),
    "poisonbench_rpi_poisoned_v1": (336, 20, "controlled_retrieved_prompt_injection"),
}
PILOT = "poisonbench_b_sm_1pct_v1"
REF = "poisonbench_rpi_clean_v1"
DIM = 384
CLEAN_COUNT = 316
CHUNK_BATCH = 64


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def collection_names(client):
    return {x if isinstance(x, str) else x.name for x in client.list_collections()}


def vectors_to_list(vectors, count):
    need(vectors is not None and len(vectors) == count, "Missing/incomplete Chroma embeddings")
    result = []
    for v in vectors:
        v = list(v)
        need(len(v) == DIM and all(math.isfinite(float(x)) for x in v),
             "Invalid embedding dimensions or non-finite values")
        need(any(float(x) != 0 for x in v), "Zero vector detected")
        result.append(v)
    return result


def read_collection(collection):
    res = collection.get(include=["documents", "metadatas", "embeddings"])
    ids, docs, metas = res.get("ids"), res.get("documents"), res.get("metadatas")
    need(ids is not None and docs is not None and metas is not None,
         f"Incomplete Chroma response for {collection.name}")
    count = collection.count()
    need(len(ids) == len(docs) == len(metas) == count and len(set(ids)) == count,
         f"Chroma count/ID mismatch for {collection.name}")
    vectors = vectors_to_list(res.get("embeddings"), count)
    rows = {}
    for cid, doc, meta, vec in zip(ids, docs, metas, vectors):
        need(isinstance(meta, dict) and isinstance(doc, str) and doc.strip(),
             f"Invalid stored document/metadata at {cid}")
        need(cid == f"{meta.get('doc_id')}::chunk{meta.get('position')}",
             f"Stored chunk ID vs metadata mismatch: {cid}")
        rows[cid] = (doc, meta, vec)
    return rows


def split_rows(rows):
    clean = {cid: (doc, meta) for cid, (doc, meta, _v) in rows.items()
             if meta.get("is_poisoned") is False}
    poisons = {cid: (doc, meta) for cid, (doc, meta, _v) in rows.items()
               if meta.get("is_poisoned") is True}
    need(len(clean) + len(poisons) == len(rows), "Unknown poisoned flags detected")
    return clean, poisons


def expected_poisons(attack, ratio):
    from api.pipeline.chunking import chunk_text
    dataset = dataset_path(attack, ratio)
    manifest_file = (ROOT / "data/poisoned/phase_b/phase_b_1pct_5pct_allocation_v1.json"
                     if ratio in (1, 5) else
                     ROOT / f"data/poisoned/phase_b/phase_b_{ratio}pct_allocation_v1.json")
    manifest = load_manifest(manifest_file)
    if ratio in (1, 5):
        entries = [v for v in manifest["files"] if v.get("attack") == attack
                   and v.get("target_ratio_pct") == ratio]
        need(len(entries) == 1 and sha(dataset) == entries[0]["output_sha256"],
             f"Dataset hash does not match low-ratio manifest: {dataset}")
    else:
        need(sha(dataset) == manifest["datasets"][attack]["sha256"],
             f"Dataset hash does not match high-ratio manifest: {dataset}")
    records = load_jsonl(dataset)
    need(len(records) == RATIOS[ratio], "Unexpected poison record count")
    expected = {}
    ids = []
    texts = []
    for record in records:
        did = record["poison_doc_id"]
        pieces = chunk_text(record["poison_text"], did)
        need(len(pieces) == 1 and pieces[0]["position"] == 0,
             f"Not a single-chunk poison document: {did}")
        cid = pieces[0]["chunk_id"]
        need(cid not in expected and cid == f"{did}::chunk0", f"Duplicate/unexpected chunk ID {cid}")
        meta = {"doc_id": did, "source": SOURCES[attack], "timestamp": "",
                "is_poisoned": True, "position": 0}
        expected[cid] = (pieces[0]["text"], meta)
        ids.append(cid)
        texts.append(pieces[0]["text"])
    need(len(set(texts)) == len(texts), "Duplicate poison text in expected records")
    return expected


def read_and_check_existing(client, target):
    names = collection_names(client)
    reserved = {f"poisonbench_b_{attack.lower()}_{ratio}pct_v1"
                for ratio in (1, 5, 10, 20) for attack in ("SM", "OBEA", "RPI")}
    need(set(FROZEN) | {PILOT} <= names,
         f"Missing frozen/pilot collections: {sorted((set(FROZEN) | {PILOT}) - names)}")
    need(target not in names, f"STOP: {target} ALREADY EXISTS; do not overwrite or rerun --apply")
    need(not (names - set(FROZEN) - reserved),
         f"Unexpected non-Phase-B collection(s): {sorted(names - set(FROZEN) - reserved)}")
    # Enforce staged progress; previously created Phase B collections must not have been modified.
    ref = None
    ref_clean = None
    frozen = {}
    for name, (total, poison_count, source) in FROZEN.items():
        collection = client.get_collection(name=name)
        rows = read_collection(collection)
        clean, poisons = split_rows(rows)
        need(len(rows) == total and len(clean) == CLEAN_COUNT and len(poisons) == poison_count,
             f"Frozen collection count mismatch: {name}")
        need(all(m.get("source") == "NVD" for _t, m in clean.values()) and
             len({m["doc_id"] for _t, m in clean.values()}) == 300,
             f"Frozen clean corpus metadata mismatch: {name}")
        need(all(m.get("source") == source for _t, m in poisons.values()),
             f"Frozen poisoned source mismatch: {name}")
        frozen[name] = rows
        if name == REF:
            ref, ref_clean = rows, clean
        print(f"PASS: frozen {name}: {len(clean)} clean + {len(poisons)} poison")
    need(ref is not None, "Missing reference rows")
    need(all(split_rows(rows)[0] == ref_clean for rows in frozen.values()),
         "Frozen clean ID/text/metadata mismatch")
    print("PASS: all three frozen collections share 316 identical clean chunks")
    # Independent re-check of the first pilot, including its embeddings.
    pilot_rows = read_collection(client.get_collection(name=PILOT))
    expected = expected_poisons("SM", 1)
    verify_target_rows(PILOT, pilot_rows, ref, ref_clean, expected, quiet=False)
    # Preserve all existing Phase B data: compare each completed collection against its dataset.
    completed = sorted((names & reserved) - {PILOT})
    for name in completed:
        matched = [(a, r) for r in (1, 5, 10, 20) for a in ("SM", "OBEA", "RPI")
                   if name == f"poisonbench_b_{a.lower()}_{r}pct_v1"]
        need(len(matched) == 1, f"Could not map existing collection: {name}")
        attack, ratio = matched[0]
        rows = read_collection(client.get_collection(name=name))
        verify_target_rows(name, rows, ref, ref_clean,
                           expected_poisons(attack, ratio), quiet=True)
    if completed:
        print(f"PASS: {len(completed)} other existing Phase B collections verified exactly")
    return ref, ref_clean


def verify_target_rows(name, rows, clean_ref_rows, clean_ref, poisons_expected, quiet=False):
    clean, poisons = split_rows(rows)
    need(len(rows) == CLEAN_COUNT + len(poisons_expected), f"{name}: wrong total count")
    need(clean == clean_ref and poisons == poisons_expected,
         f"{name}: clean or poison IDs/text/metadata mismatch")
    max_diff = 0.0
    for cid in clean_ref:
        before, after = clean_ref_rows[cid][2], rows[cid][2]
        need(len(before) == len(after) == DIM, "Clean embedding dimension differs")
        max_diff = max(max_diff, max(abs(float(a) - float(b))
                                     for a, b in zip(before, after)))
    need(max_diff <= 1e-6, f"{name}: clean embeddings changed (max diff {max_diff})")
    if not quiet:
        print(f"PASS: {name}: 316 original clean + {len(poisons_expected)} exact poisons; "
              f"clean vector max diff {max_diff:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", choices=["SM", "OBEA", "RPI"], default="OBEA")
    parser.add_argument("--ratio", type=int, choices=[1, 5, 10, 20], default=1)
    parser.add_argument("--apply", action="store_true",
                        help="Create EXACTLY ONE previously unused experimental collection")
    args = parser.parse_args()
    need(not (args.attack == "SM" and args.ratio == 1),
         "SM 1% is already completed and verified; never recreate it")
    target = f"poisonbench_b_{args.attack.lower()}_{args.ratio}pct_v1"
    print("=== Phase B4 — Isolated single-collection builder ===")
    print(f"Selected: {args.attack} {args.ratio}%; collection {target}")

    from api.config import settings
    from api.pipeline.vectorstore import get_client
    from scripts.phase_b_validate_all_datasets import main as validate_all
    need(settings.chroma_collection == "poisonbench_corpus", "Default Chroma collection changed")
    need(settings.chunk_size_words == 180 and settings.chunk_overlap_words == 30,
         "Chunk configuration differs from frozen setup")
    print("Validating all 12 poison datasets (READ ONLY)...")
    validate_all()
    client = get_client()
    ref_rows, ref_clean = read_and_check_existing(client, target)
    poisons = expected_poisons(args.attack, args.ratio)
    need(not (set(ref_rows) & set(poisons)), "Poison chunk IDs collide with clean reference")
    print(f"PASS: {len(poisons)} {args.attack} poison chunks; planned 316+{len(poisons)}="
          f"{CLEAN_COUNT+len(poisons)} ({100*len(poisons)/(CLEAN_COUNT+len(poisons)):.2f}%)")
    if not args.apply:
        print("PASS: PHASE B4 SINGLE-COLLECTION PREFLIGHT COMPLETE")
        print("SAFETY: READ ONLY — no writes, model inference, Ollama or collection changes")
        print("NEXT: Review output with assistant BEFORE adding --apply")
        return

    from api.pipeline.embedding import embed_texts
    poison_ids = list(poisons)
    print(f"Embedding exactly {len(poison_ids)} poison chunks (no database write yet)...")
    poison_vectors = vectors_to_list(embed_texts([poisons[cid][0] for cid in poison_ids]),
                                     len(poison_ids))
    need(target not in collection_names(client), "Target appeared during embedding; refusing write")
    print(f"Creating only NEW collection {target} (first database write)...")
    collection = client.create_collection(name=target, metadata={"hnsw:space": "cosine"})
    try:
        clean_ids = sorted(ref_clean)
        for offset in range(0, len(clean_ids), CHUNK_BATCH):
            batch_ids = clean_ids[offset:offset + CHUNK_BATCH]
            collection.add(ids=batch_ids,
                           documents=[ref_rows[cid][0] for cid in batch_ids],
                           metadatas=[ref_rows[cid][1] for cid in batch_ids],
                           embeddings=[ref_rows[cid][2] for cid in batch_ids])
            print(f"  Copied clean: {offset+len(batch_ids)}/{CLEAN_COUNT}")
        for offset in range(0, len(poison_ids), CHUNK_BATCH):
            ids = poison_ids[offset:offset + CHUNK_BATCH]
            collection.add(ids=ids,
                           documents=[poisons[cid][0] for cid in ids],
                           metadatas=[poisons[cid][1] for cid in ids],
                           embeddings=poison_vectors[offset:offset + len(ids)])
            print(f"  Inserted poisons: {offset+len(ids)}/{len(poison_ids)}")
        need(collection.metadata is not None and
             collection.metadata.get("hnsw:space") == "cosine", "Target metric is not cosine")
        verify_target_rows(target, read_collection(collection), ref_rows, ref_clean, poisons)
        for name, (count, _pois, _source) in FROZEN.items():
            need(client.get_collection(name=name).count() == count,
                 f"Frozen collection count changed unexpectedly: {name}")
        print(f"PASS: PHASE B4 SINGLE COLLECTION COMPLETE: {target}")
        print("SAFETY: No frozen collections modified or reset")
        print("NEXT: Stop here; get independent verification before creating another collection")
    except Exception:
        print(f"STOP: {target} may be partially populated. DO NOT rerun --apply, "
              "delete/reset anything or start another ingestion.", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
