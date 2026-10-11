"""PoisonBench Phase B1: strictly read-only corpus-ratio design preflight.

Run from repository root:
    python scripts/phase_b_ratio_preflight.py

No model loading, Chroma access, network requests, or writes.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmark/real_trigger_queries_v1.jsonl"
ATTACKS = {
    "SM": ROOT / "data/poisoned/semantic_mimicry_real.jsonl",
    "OBEA": ROOT / "data/poisoned/optimized_embedding_attack_real.jsonl",
    "RPI": ROOT / "data/poisoned/retrieved_prompt_injection_real.jsonl",
}
EXPECTED_PREFIX = {"SM": "POISON-SM-", "OBEA": "POISON-OBEA-", "RPI": "POISON-RPI-"}
EXPECTED_NVD_CHUNKS = 316
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 30
RATIOS = (Decimal("0.01"), Decimal("0.05"), Decimal("0.10"), Decimal("0.20"))
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required existing file: {path}")
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path.name}:{i}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object: {path.name}:{i}")
            records.append(obj)
    return records


def chunk_count(text: str) -> int:
    length = len(text.split())
    if length == 0:
        return 0
    if length <= CHUNK_SIZE_WORDS:
        return 1
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    return 1 + (length - CHUNK_SIZE_WORDS + step - 1) // step


def approx_chunk_target(ratio: Decimal) -> int:
    # P/(C+P)=r  ->  P=C*r/(1-r). Round to nearest attainable chunk count.
    exact = Decimal(EXPECTED_NVD_CHUNKS) * ratio / (Decimal(1) - ratio)
    return int(exact.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def validate() -> list[str]:
    benchmark = load_jsonl(BENCHMARK)
    if len(benchmark) != 20:
        raise ValueError(f"Expected 20 benchmark queries; found {len(benchmark)}")
    expected = {}
    for item in benchmark:
        qid = str(item.get("query_id", "")).strip().lower()
        cve = str(item.get("expected_doc_id", "")).strip()
        query = str(item.get("query", "")).strip()
        if not qid or not cve or not query or qid in expected:
            raise ValueError(f"Invalid or duplicate benchmark query: {item}")
        expected[qid] = cve
    if set(expected) != {f"q{i:02d}" for i in range(1, 21)}:
        raise ValueError("Benchmark query IDs are not q01 through q20")

    print("=== Phase B1 — Poisoning-ratio design preflight (READ ONLY) ===")
    print(f"Frozen benchmark: {len(benchmark)} queries; clean baseline: {EXPECTED_NVD_CHUNKS} chunks")
    all_counts = {}
    for attack, path in ATTACKS.items():
        records = load_jsonl(path)
        seen_queries, seen_docs = set(), set()
        counts = Counter()
        for row in records:
            qid = str(row.get("query_id", "")).strip().lower()
            doc = str(row.get("poison_doc_id", "")).strip()
            legit = str(row.get("legitimate_doc_id", "")).strip()
            poison_text = row.get("poison_text")
            if (qid not in expected or qid in seen_queries or
                legit != expected[qid] or doc in seen_docs or
                not doc.startswith(EXPECTED_PREFIX[attack]) or
                row.get("is_poisoned") is not True or
                not isinstance(poison_text, str) or not poison_text.strip()):
                raise ValueError(f"{attack}: invalid mapping, label, or duplicate: {qid} / {doc}")
            pieces = chunk_count(poison_text)
            if pieces == 0:
                raise ValueError(f"{attack}: empty chunk output for {qid}")
            counts[pieces] += 1
            seen_queries.add(qid)
            seen_docs.add(doc)
        if len(records) != 20 or seen_queries != set(expected):
            raise ValueError(f"{attack}: expected precisely 20 unique targeted documents")
        total_chunks = sum(pieces * n for pieces, n in counts.items())
        all_counts[attack] = total_chunks
        print(f"PASS: {attack:4} | 20 unique targets | {total_chunks} predicted poison chunks | chunks/document {dict(sorted(counts.items()))}")

    selection = list(sorted(expected))
    random.Random(SEED).shuffle(selection)
    print(f"Fixed-seed ({SEED}) shared target order: {', '.join(selection)}")
    print("\nRatio design (final-corpus CHUNK percentage; one poison chunk per variant):")
    for ratio in RATIOS:
        p = approx_chunk_target(ratio)
        achieved = 100 * p / (EXPECTED_NVD_CHUNKS + p)
        distinct = min(p, len(selection))
        extras = max(0, p - len(selection))
        print(f"  {int(ratio * 100):2d}% -> {p:2d} poison + 316 clean = {EXPECTED_NVD_CHUNKS+p:3d} chunks | actual {achieved:5.2f}% | initially targeted queries {distinct:2d}/20 | extra variants {extras:2d}")
    if any(n != 20 for n in all_counts.values()):
        print("NOTE: At least one frozen attack document splits into multiple chunks; allocation requires chunk-aware handling.")
    print("\nSAFETY: No database connection, ingestion, model execution, result overwrite or file writes occurred.")
    print("NEXT: Review the allocation protocol and variant-generation strategy before Phase B2.")
    return selection


if __name__ == "__main__":
    try:
        validate()
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
