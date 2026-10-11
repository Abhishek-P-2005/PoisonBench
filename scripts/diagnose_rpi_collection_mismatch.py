"""Read-only diagnosis of NVD chunk differences across three PoisonBench Chroma collections.
Run from PoisonBench_Shared: python scripts/diagnose_rpi_collection_mismatch.py
Does not modify collections, embeddings or results.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from api.pipeline.vectorstore import get_client

ORIGINAL = "poisonbench_corpus"
CLEAN = "poisonbench_rpi_clean_v1"
POISONED = "poisonbench_rpi_poisoned_v1"


def clean_records(collection):
    response = collection.get(include=["documents", "metadatas", "embeddings"])
    records = {}
    for cid, txt, meta, emb in zip(response["ids"], response["documents"], response["metadatas"], response["embeddings"]):
        if meta and meta.get("source") == "NVD":
            records[cid] = (txt, meta, [float(x) for x in emb])
    return records


def compare(source, target, name):
    ids_a = set(source)
    ids_b = set(target)
    missing = sorted(ids_a - ids_b)
    extra = sorted(ids_b - ids_a)
    text_diffs = []
    meta_diffs = []
    emb_diffs = []
    max_abs_delta = 0.0
    max_id = None
    sample_dim = None
    for cid in sorted(ids_a & ids_b):
        ta, ma, ea = source[cid]
        tb, mb, eb = target[cid]
        if ta != tb:
            text_diffs.append(cid)
        if ma != mb:
            meta_diffs.append(cid)
        if len(ea) != len(eb):
            emb_diffs.append(cid)
            continue
        delta = max((abs(x-y) for x, y in zip(ea,eb)), default=0.0)
        if delta:
            emb_diffs.append(cid)
        if delta > max_abs_delta:
            max_abs_delta = delta
            max_id = cid
            sample_dim = next((i for i,(x,y) in enumerate(zip(ea,eb)) if abs(x-y)==delta), None)
    print(f"\n=== Original vs {name} ===")
    print(f"Clean chunk IDs: {len(ids_a)} vs {len(ids_b)}")
    print(f"Missing IDs: {len(missing)}; extra IDs: {len(extra)}")
    print(f"Text mismatches: {len(text_diffs)}; metadata mismatches: {len(meta_diffs)}")
    print(f"Embedding mismatches (exact): {len(emb_diffs)}")
    print(f"Maximum absolute embedding difference: {max_abs_delta:.12g}")
    if max_id is not None:
        print(f"Largest embedding difference: chunk {max_id}, dimension {sample_dim}")
    for label, samples in (("Missing",missing),("Extra",extra),("Text",text_diffs),("Metadata",meta_diffs),("Embeddings",emb_diffs)):
        if samples:
            print(f"{label} example: {samples[0]}")
            if label == "Metadata":
                key = samples[0]
                print("  Original metadata:", source[key][1])
                print("  Target metadata:  ", target[key][1])
    if not missing and not extra and not text_diffs and not meta_diffs and not emb_diffs:
        print("PASS: Exact match")
    else:
        print("DIAGNOSIS ONLY: no data was modified")


def main():
    client = get_client()
    a = clean_records(client.get_collection(name=ORIGINAL))
    b = clean_records(client.get_collection(name=CLEAN))
    c = clean_records(client.get_collection(name=POISONED))
    compare(a,b,CLEAN)
    compare(a,c,POISONED)
    print("\nREAD ONLY: No collection or file changes made.")

if __name__ == '__main__':
    main()
