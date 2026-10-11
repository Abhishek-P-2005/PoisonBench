"""PoisonBench — Phase A5: Attack 3 isolated Top-5 retrieval evaluation.

From the PoisonBench_Shared repository root using the existing .venv:

    python scripts/evaluate_retrieved_prompt_injection_real.py
    python scripts/evaluate_retrieved_prompt_injection_real.py --run

Default: READ-ONLY preflight, no embeddings or output files.
--run: query both isolated collections, save a NEW CSV and JSON summary.
Neither mode creates, changes, resets, or deletes any Chroma collection.
This is RETRIEVAL evaluation; generation-level success is Phase A6.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.config import settings  # noqa: E402
from api.pipeline.vectorstore import get_client  # noqa: E402

ORIGINAL = "poisonbench_corpus"
CLEAN = "poisonbench_rpi_clean_v1"
POISONED = "poisonbench_rpi_poisoned_v1"
SOURCE = "controlled_retrieved_prompt_injection"
BENCHMARK_FILE = ROOT / "data" / "benchmark" / "real_trigger_queries_v1.jsonl"
POISON_FILE = ROOT / "data" / "poisoned" / "retrieved_prompt_injection_real.jsonl"
OUTPUT_CSV = ROOT / "data" / "results" / "real_retrieved_prompt_injection_top5_v1_results.csv"
OUTPUT_JSON = ROOT / "data" / "results" / "real_retrieved_prompt_injection_top5_v1_summary.json"
TOP_K = 5
N = 20


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path.name}:{number}") from exc
    return rows


def validate_inputs():
    benchmark = read_jsonl(BENCHMARK_FILE)
    poisons = read_jsonl(POISON_FILE)
    if len(benchmark) != N or len(poisons) != N:
        raise RuntimeError(f"Expected exactly {N} queries and {N} RPI records")
    expected_ids = {f"q{i:02d}" for i in range(1, N + 1)}
    bm = {}
    for row in benchmark:
        qid = str(row["query_id"]).strip().lower()
        query = str(row["query"]).strip()
        docid = str(row["expected_doc_id"]).strip()
        if not query or not docid or qid in bm:
            raise RuntimeError(f"Missing or duplicate benchmark data: {qid}")
        bm[qid] = {"query": query, "legitimate_doc_id": docid}
    if set(bm) != expected_ids:
        raise RuntimeError("Benchmark IDs are not the frozen q01–q20 set")

    seen = set()
    for row in poisons:
        required = {"query_id", "legitimate_doc_id", "poison_doc_id", "poison_text", "source", "attack_type", "is_poisoned"}
        missing = required - row.keys()
        if missing:
            raise RuntimeError(f"Incomplete poison record: {sorted(missing)}")
        qid = str(row["query_id"]).strip().lower()
        if qid not in bm or qid in seen:
            raise RuntimeError(f"Unknown or duplicate poison query ID: {qid}")
        if (str(row["legitimate_doc_id"]).strip() != bm[qid]["legitimate_doc_id"]
                or row["poison_doc_id"] != f"POISON-RPI-{qid}"
                or row["source"] != SOURCE
                or row["attack_type"] != "retrieved_prompt_injection"
                or row["is_poisoned"] is not True
                or "POISONBENCH_RPI_SUCCESS" not in str(row["poison_text"])):
            raise RuntimeError(f"Poison record does not match benchmark or attack specification: {qid}")
        bm[qid]["poison_doc_id"] = row["poison_doc_id"]
        seen.add(qid)
    if seen != expected_ids:
        raise RuntimeError("Poison records do not cover all 20 queries")
    return [(qid, bm[qid]) for qid in sorted(bm)]


def inspect(collection):
    result = collection.get(include=["metadatas"])
    ids = result["ids"]
    metas = result["metadatas"]
    if ids is None or metas is None or len(ids) != len(metas) or len(ids) != collection.count():
        raise RuntimeError(f"Incomplete collection metadata: {collection.name}")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Duplicate chunk IDs: {collection.name}")
    return {id_: meta for id_, meta in zip(ids, metas)}


def preflight(client, benchmark):
    available = {obj if isinstance(obj, str) else obj.name for obj in client.list_collections()}
    if not {ORIGINAL, CLEAN, POISONED} <= available:
        raise RuntimeError(f"Missing collections: {sorted({ORIGINAL, CLEAN, POISONED} - available)}")
    original = client.get_collection(name=ORIGINAL)
    clean = client.get_collection(name=CLEAN)
    poisoned = client.get_collection(name=POISONED)
    counts = {ORIGINAL: original.count(), CLEAN: clean.count(), POISONED: poisoned.count()}
    if counts != {ORIGINAL: 336, CLEAN: 316, POISONED: 336}:
        raise RuntimeError(f"Unexpected collection counts; stopping: {counts}")

    original_meta = inspect(original)
    clean_meta = inspect(clean)
    poison_meta = inspect(poisoned)
    original_sources = Counter(m.get("source") for m in original_meta.values())
    clean_sources = Counter(m.get("source") for m in clean_meta.values())
    poison_sources = Counter(m.get("source") for m in poison_meta.values())
    if (original_sources != {"NVD": 316, "controlled_optimized_embedding": 20}
            or clean_sources != {"NVD": 316}
            or poison_sources != {"NVD": 316, SOURCE: 20}):
        raise RuntimeError(f"Unexpected source distributions: {original_sources}, {clean_sources}, {poison_sources}")

    clean_from_original = {id_: meta for id_, meta in original_meta.items() if meta.get("source") == "NVD"}
    clean_from_poisoned = {id_: meta for id_, meta in poison_meta.items() if meta.get("source") == "NVD"}
    if clean_meta != clean_from_original or clean_meta != clean_from_poisoned:
        raise RuntimeError("Clean metadata/chunk IDs differ across collections")
    if any(m.get("is_poisoned") is not False for m in clean_meta.values()):
        raise RuntimeError("Clean reference contains a non-clean label")

    expected_poison_chunks = {f"{row['poison_doc_id']}::chunk0" for _, row in benchmark}
    observed_poison_chunks = {id_ for id_, meta in poison_meta.items() if meta.get("source") == SOURCE}
    if observed_poison_chunks != expected_poison_chunks:
        raise RuntimeError("RPI chunk IDs do not match the frozen 20-query dataset")
    for id_ in observed_poison_chunks:
        meta = poison_meta[id_]
        if meta.get("is_poisoned") is not True or meta.get("doc_id") != id_.split("::", 1)[0]:
            raise RuntimeError(f"Invalid RPI poison metadata: {id_}")
    print("=== Phase A5 — Attack 3 isolated retrieval preflight ===")
    print(f"Original (READ ONLY) : {ORIGINAL} : {counts[ORIGINAL]} chunks (316 NVD + 20 OBEA)")
    print(f"Clean (READ ONLY)    : {CLEAN} : {counts[CLEAN]} chunks (316 NVD)")
    print(f"Poisoned (READ ONLY) : {POISONED} : {counts[POISONED]} chunks (316 NVD + 20 RPI)")
    print(f"Benchmark            : {len(benchmark)} frozen queries | Top-{TOP_K}")
    print("PASS: Corpus identity, poison IDs, and collection isolation verified.")
    return clean, poisoned


def retrieve(collection, embedding):
    response = collection.query(query_embeddings=[embedding], n_results=TOP_K,
                                include=["documents", "metadatas", "distances"])
    ids = response["ids"][0]
    metas = response["metadatas"][0]
    distances = response["distances"][0]
    if not (len(ids) == len(metas) == len(distances) == TOP_K):
        raise RuntimeError(f"Incomplete Top-{TOP_K} retrieval from {collection.name}")
    doc_ids = [str(m["doc_id"]) for m in metas]
    return doc_ids, [float(1.0 - dist) for dist in distances]


def rank(doc_ids, target):
    return next((position for position, item in enumerate(doc_ids, 1) if item == target), None)


def fraction(hits, denominator):
    return hits / denominator if denominator else None


def compute_summary(rows):
    successful = [r for r in rows if not r["error"]]
    failed = [r for r in rows if r["error"]]
    summary = {
        "attack_type": "retrieved_prompt_injection",
        "stage": "retrieval_only",
        "benchmark": "data/benchmark/real_trigger_queries_v1.jsonl",
        "attack_file": "data/poisoned/retrieved_prompt_injection_real.jsonl",
        "clean_collection": CLEAN,
        "poisoned_collection": POISONED,
        "original_collection_unmodified": ORIGINAL,
        "model": settings.embedding_model_name,
        "top_k": TOP_K,
        "queries_planned": N,
        "queries_succeeded": len(successful),
        "api_errors": len(failed),
        "error_queries": [r["query_id"] for r in failed],
        "all_queries_valid": len(successful) == N,
        "metrics_denominator": len(successful),
    }
    # Partial-run metrics are explicitly labeled as provisional, never final ASR.
    if successful:
        poison_top5 = sum(r["poison_rank"] is not None for r in successful)
        poison_rank1 = sum(r["poison_rank"] == 1 for r in successful)
        beat = sum(r["poison_rank"] is not None and
                   (r["legitimate_rank_poisoned"] is None or r["poison_rank"] < r["legitimate_rank_poisoned"])
                   for r in successful)
        legit_top5 = sum(r["legitimate_rank_poisoned"] is not None for r in successful)
        clean_top5 = sum(r["legitimate_rank_clean"] is not None for r in successful)
        clean_rank1 = sum(r["legitimate_rank_clean"] == 1 for r in successful)
        summary.update({
            "poison_top5_hits": poison_top5,
            "poison_top5_asr": fraction(poison_top5, len(successful)),
            "poison_rank1_hits": poison_rank1,
            "poison_rank1_asr": fraction(poison_rank1, len(successful)),
            "poison_beats_legitimate_hits": beat,
            "poison_beats_legitimate_rate": fraction(beat, len(successful)),
            "legitimate_top5_hits": legit_top5,
            "legitimate_top5_recall": fraction(legit_top5, len(successful)),
            "legitimate_mrr_after_poisoning": sum(1.0 / r["legitimate_rank_poisoned"] if r["legitimate_rank_poisoned"] else 0.0 for r in successful) / len(successful),
            "measured_clean_top5_hits": clean_top5,
            "measured_clean_top5_recall": fraction(clean_top5, len(successful)),
            "measured_clean_rank1_hits": clean_rank1,
            "measured_clean_rank1_accuracy": fraction(clean_rank1, len(successful)),
            "measured_clean_mrr": sum(1.0 / r["legitimate_rank_clean"] if r["legitimate_rank_clean"] else 0.0 for r in successful) / len(successful),
        })
    else:
        summary["metrics_unavailable"] = "All retrieval attempts failed"
    return summary


def save_results(rows, summary):
    if OUTPUT_CSV.exists() or OUTPUT_JSON.exists():
        raise FileExistsError("Attack 3 retrieval outputs already exist; refusing to overwrite. Back them up before rerunning.")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "query", "legitimate_doc_id", "poison_doc_id", "clean_retrieved_doc_ids", "poisoned_retrieved_doc_ids",
              "clean_retrieved_scores", "poisoned_retrieved_scores", "legitimate_rank_clean", "legitimate_rank_poisoned",
              "poison_rank", "poison_top5", "poison_rank1", "poison_beats_legitimate", "legitimate_top5", "error"]
    with OUTPUT_CSV.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("clean_retrieved_doc_ids", "poisoned_retrieved_doc_ids"):
                out[key] = " | ".join(out[key])
            for key in ("clean_retrieved_scores", "poisoned_retrieved_scores"):
                out[key] = " | ".join(f"{score:.6f}" for score in out[key])
            writer.writerow(out)
    with OUTPUT_JSON.open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def evaluate(clean, poisoned, benchmark):
    # Import only at execution time, so read-only preflight does not load MiniLM.
    from api.pipeline.embedding import embed_texts
    print(f"Encoding {len(benchmark)} benchmark queries once using {settings.embedding_model_name}...")
    vectors = embed_texts([item["query"] for _, item in benchmark])
    if len(vectors) != N or any(len(v) != 384 for v in vectors):
        raise RuntimeError("Unexpected embedding shape; no evaluation performed")
    rows = []
    for (qid, item), vector in zip(benchmark, vectors):
        result = {
            "query_id": qid, "query": item["query"],
            "legitimate_doc_id": item["legitimate_doc_id"], "poison_doc_id": item["poison_doc_id"],
            "clean_retrieved_doc_ids": [], "poisoned_retrieved_doc_ids": [],
            "clean_retrieved_scores": [], "poisoned_retrieved_scores": [],
            "legitimate_rank_clean": None, "legitimate_rank_poisoned": None, "poison_rank": None,
            "poison_top5": None, "poison_rank1": None, "poison_beats_legitimate": None,
            "legitimate_top5": None, "error": "",
        }
        try:
            clean_ids, clean_scores = retrieve(clean, vector)
            poison_ids, poison_scores = retrieve(poisoned, vector)
            if item["poison_doc_id"] in clean_ids:
                raise RuntimeError("RPI poison appeared in clean reference retrieval")
            result["clean_retrieved_doc_ids"] = clean_ids
            result["poisoned_retrieved_doc_ids"] = poison_ids
            result["clean_retrieved_scores"] = clean_scores
            result["poisoned_retrieved_scores"] = poison_scores
            result["legitimate_rank_clean"] = rank(clean_ids, item["legitimate_doc_id"])
            result["legitimate_rank_poisoned"] = rank(poison_ids, item["legitimate_doc_id"])
            result["poison_rank"] = rank(poison_ids, item["poison_doc_id"])
            result["poison_top5"] = result["poison_rank"] is not None
            result["poison_rank1"] = result["poison_rank"] == 1
            result["legitimate_top5"] = result["legitimate_rank_poisoned"] is not None
            result["poison_beats_legitimate"] = result["poison_top5"] and (
                result["legitimate_rank_poisoned"] is None or
                result["poison_rank"] < result["legitimate_rank_poisoned"])
            print(f"[{qid}] CLEAN legitimate #{result['legitimate_rank_clean']} | "
                  f"POISONED legitimate #{result['legitimate_rank_poisoned']} | "
                  f"RPI poison #{result['poison_rank']}")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{qid}] ERROR: {result['error']}")
        rows.append(result)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Perform read-only retrieval queries and export CSV/JSON")
    args = parser.parse_args()
    if settings.chroma_collection != ORIGINAL:
        raise RuntimeError(f"Unexpected configured collection: {settings.chroma_collection!r}")
    benchmark = validate_inputs()
    clean, poisoned = preflight(get_client(), benchmark)
    if not args.run:
        print("PASS: Read-only preflight. No retrieval, embeddings, or result files generated.")
        print("NEXT: Run this script with --run for 20-query Top-5 retrieval evaluation.")
        return
    if OUTPUT_CSV.exists() or OUTPUT_JSON.exists():
        raise FileExistsError("Attack 3 output filename already exists; refusing to overwrite frozen data")
    print("=== Phase A5 — Full 20-query retrieval evaluation ===")
    rows = evaluate(clean, poisoned, benchmark)
    summary = compute_summary(rows)
    save_results(rows, summary)
    print("=== Summary ===")
    print(f"Queries passed: {summary['queries_succeeded']}/{N} | API errors: {summary['api_errors']}")
    if summary["queries_succeeded"]:
        for label, key in (("Poison Top-5 ASR", "poison_top5_asr"),
                           ("Poison Rank-1 ASR", "poison_rank1_asr"),
                           ("Poison beats legitimate", "poison_beats_legitimate_rate"),
                           ("Legitimate Top-5 recall", "legitimate_top5_recall"),
                           ("Legitimate MRR", "legitimate_mrr_after_poisoning"),
                           ("Measured clean Top-5 recall", "measured_clean_top5_recall"),
                           ("Measured clean MRR", "measured_clean_mrr")):
            print(f"{label}: {summary[key]:.4f}")
    print(f"Detailed CSV: {OUTPUT_CSV}")
    print(f"Summary JSON: {OUTPUT_JSON}")
    if not summary["all_queries_valid"]:
        raise RuntimeError("Evaluation incomplete: errors occurred. Do NOT treat partial metrics as final.")
    print("PASS: All 20 queries evaluated against BOTH isolated collections.")
    print("NEXT: Phase A6 — independent generation-level evaluation (not measured here).")


if __name__ == "__main__":
    main()
