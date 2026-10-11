"""PoisonBench Phase B3 — read-only verification of all 12 ratio datasets.

Run from repository root:
    python scripts/phase_b_validate_all_datasets.py

This checks JSONL records, mapping/uniqueness, actual 180-word chunk limits,
manifest SHA-256 hashes, frozen base records, target coverage and nested ratios.
No ChromaDB connection, embeddings, Ollama, or file writes.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "poisoned" / "phase_b"
BENCHMARK = ROOT / "data" / "benchmark" / "real_trigger_queries_v1.jsonl"
ORIGINALS = {
    "SM": ROOT / "data" / "poisoned" / "semantic_mimicry_real.jsonl",
    "OBEA": ROOT / "data" / "poisoned" / "optimized_embedding_attack_real.jsonl",
    "RPI": ROOT / "data" / "poisoned" / "retrieved_prompt_injection_real.jsonl",
}
RATIOS = {1: 3, 5: 17, 10: 35, 20: 79}
PREFIXES = {"SM": "POISON-SM-", "OBEA": "POISON-OBEA-", "RPI": "POISON-RPI-"}
SOURCES = {
    "SM": "controlled_semantic_mimicry",
    "OBEA": "controlled_optimized_embedding",
    "RPI": "controlled_retrieved_prompt_injection",
}
RPI_MARKER = "POISONBENCH_RPI_SUCCESS"
CHUNK_WORD_LIMIT = 180
BASE_CLEAN_CHUNKS = 316


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    require(path.is_file(), f"Missing file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    require(path.is_file(), f"Missing JSONL: {path}")
    records = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
        require(isinstance(row, dict), f"Non-object at {path}:{line_no}")
        records.append(row)
    return records


def load_manifest(path: Path) -> dict:
    require(path.is_file(), f"Missing manifest: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(obj, dict), f"Expected JSON manifest at {path}")
    return obj


def dataset_path(attack: str, ratio: int) -> Path:
    return DATA / f"phase_b_{attack.lower()}_{ratio}pct_v1.jsonl"


def validate_rows(rows: list[dict], attack: str, ratio: int, lookup: dict[str, str],
                  base: dict[str, dict], selection: list[str]) -> None:
    count = RATIOS[ratio]
    require(len(rows) == count, f"{attack}/{ratio}%: expected {count} records, got {len(rows)}")
    ids: set[str] = set()
    texts: set[str] = set()
    per_query: Counter[str] = Counter()
    for index, row in enumerate(rows):
        qid = str(row.get("query_id", "")).strip().lower()
        doc_id = row.get("poison_doc_id")
        text = row.get("poison_text")
        where = f"{attack}/{ratio}% row {index + 1}"
        require(qid in lookup, f"{where}: unknown query {qid!r}")
        require(row.get("legitimate_doc_id") == lookup[qid], f"{where}: wrong CVE mapping")
        require(isinstance(doc_id, str) and doc_id.startswith(PREFIXES[attack]),
                f"{where}: invalid poison_doc_id")
        require(doc_id == f"{PREFIXES[attack]}{qid}" or
                (doc_id.startswith(f"{PREFIXES[attack]}{qid}-v") and doc_id[-2:].isdigit()),
                f"{where}: poison document ID/query mismatch")
        require(doc_id not in ids, f"{where}: repeated document ID {doc_id}")
        require(isinstance(text, str) and bool(text.strip()), f"{where}: empty poison text")
        require(text not in texts, f"{where}: duplicate poison text")
        require(len(text.split()) <= CHUNK_WORD_LIMIT, f"{where}: would split into >1 chunk")
        require(row.get("is_poisoned") is True, f"{where}: wrong poison label")
        require(row.get("source") == SOURCES[attack], f"{where}: wrong source label")
        require(isinstance(row.get("attack_type"), str) and bool(row["attack_type"]),
                f"{where}: missing attack_type")
        if attack == "RPI":
            require(text.count(RPI_MARKER) == 1, f"{where}: RPI marker missing/repeated")
        if "variant_index" in row:
            vi = row["variant_index"]
            require(isinstance(vi, int) and 2 <= vi <= 4, f"{where}: invalid variant index")
            require(doc_id == f"{PREFIXES[attack]}{qid}-v{vi:02d}", f"{where}: wrong variant ID")
            require(row.get("variant_of") == f"{PREFIXES[attack]}{qid}", f"{where}: wrong parent")
            require(bool(row.get("variant_method")), f"{where}: missing variant method")
            if attack == "OBEA":
                seed, opt, improvement = (row.get(k) for k in
                                         ("seed_similarity", "optimized_similarity", "improvement"))
                require(all(isinstance(x, (float, int)) for x in (seed, opt, improvement)),
                        f"{where}: incomplete optimizer scores")
                require(improvement > 0 and opt >= seed and abs((opt - seed) - improvement) < 1e-6,
                        f"{where}: invalid optimizer improvement")
                require(row.get("total_rounds") == 8 and row.get("candidates_per_round") == 12,
                        f"{where}: inconsistent optimizer configuration")
                require(isinstance(row.get("optimization_seed"), int),
                        f"{where}: missing optimizer seed")
        else:
            require(row == base[qid], f"{where}: frozen base record changed")
        per_query[qid] += 1
        ids.add(doc_id)
        texts.add(text)
    expected_targets = set(selection[:min(count, 20)])
    require(set(per_query) == expected_targets,
            f"{attack}/{ratio}%: unexpected target coverage")
    if ratio in (1, 5):
        require(all(n == 1 for n in per_query.values()),
                f"{attack}/{ratio}%: low-ratio dataset contains variants")
    if ratio == 10:
        require(Counter(per_query.values()) == Counter({1: 5, 2: 15}),
                f"{attack}/10%: incorrect query variant distribution")
    if ratio == 20:
        require(Counter(per_query.values()) == Counter({3: 1, 4: 19}),
                f"{attack}/20%: incorrect query variant distribution")


def main() -> None:
    print("=== Phase B3 — All 12 dataset integrity audit (READ ONLY) ===")
    bench = load_jsonl(BENCHMARK)
    require(len(bench) == 20, "Expected 20 frozen benchmark queries")
    lookup: dict[str, str] = {}
    for row in bench:
        qid = str(row.get("query_id", "")).strip().lower()
        require(qid not in lookup and bool(row.get("query")) and bool(row.get("expected_doc_id")),
                "Invalid/duplicate benchmark query")
        lookup[qid] = row["expected_doc_id"]
    require(set(lookup) == {f"q{i:02d}" for i in range(1, 21)}, "Unexpected benchmark query IDs")
    selection = sorted(lookup)
    random.Random(42).shuffle(selection)

    originals: dict[str, dict[str, dict]] = {}
    for attack, path in ORIGINALS.items():
        base_rows = load_jsonl(path)
        require(len(base_rows) == 20, f"{attack}: frozen source must have 20 rows")
        mapped = {str(r["query_id"]).strip().lower(): r for r in base_rows}
        require(set(mapped) == set(lookup), f"{attack}: source query coverage incorrect")
        originals[attack] = mapped

    low = load_manifest(DATA / "phase_b_1pct_5pct_allocation_v1.json")
    high10 = load_manifest(DATA / "phase_b_10pct_allocation_v1.json")
    high20 = load_manifest(DATA / "phase_b_20pct_allocation_v1.json")
    require(low.get("selection_seed") == 42 and low.get("selection_order") == selection,
            "B2A selection seed/order mismatch")
    for pct, manifest in ((10, high10), (20, high20)):
        require(manifest.get("ratio_target_pct") == pct and manifest.get("clean_chunks") == BASE_CLEAN_CHUNKS and
                manifest.get("poison_chunks") == RATIOS[pct] and
                manifest.get("final_chunks") == BASE_CLEAN_CHUNKS + RATIOS[pct],
                f"{pct}%: incorrect ratio allocation manifest")
        require(manifest.get("selection_seed") == 42 and manifest.get("selection_order") == selection,
                f"{pct}%: seed/order mismatch")
        require(manifest.get("targeted_query_count") == 20,
                f"{pct}%: incorrect target coverage in manifest")
    require(low.get("benchmark_sha256") == sha(BENCHMARK), "Frozen benchmark changed since B2A")
    for attack, path in ORIGINALS.items():
        require(low["source_dataset_sha256"][attack] == sha(path),
                f"Frozen {attack} input changed since B2A")
        for pct, manifest in ((10, high10), (20, high20)):
            rel = path.relative_to(ROOT).as_posix()
            require(manifest["input_sha256"][rel] == sha(path),
                    f"{pct}%: input hash mismatch for frozen {attack}")

    low_entries = {(r["attack"], r["target_ratio_pct"]): r for r in low["files"]}
    checked: dict[tuple[str, int], list[dict]] = {}
    print(f"PASS: frozen benchmark, 3 attack sources and allocation manifests; seed=42")
    for ratio in RATIOS:
        for attack in ("SM", "OBEA", "RPI"):
            path = dataset_path(attack, ratio)
            rows = load_jsonl(path)
            validate_rows(rows, attack, ratio, lookup, originals[attack], selection)
            digest = sha(path)
            if ratio in (1, 5):
                record = low_entries.get((attack, ratio))
                require(record is not None and digest == record["output_sha256"],
                        f"{attack}/{ratio}%: B2A hash mismatch")
                require(record["output_dataset"] == path.relative_to(ROOT).as_posix(),
                        f"{attack}/{ratio}%: B2A path mismatch")
            else:
                manifest = high10 if ratio == 10 else high20
                item = manifest["datasets"][attack]
                require(digest == item["sha256"], f"{attack}/{ratio}%: B2B hash mismatch")
                require(item["path"] == path.relative_to(ROOT).as_posix(),
                        f"{attack}/{ratio}%: B2B path mismatch")
                require(set(item["doc_ids"]) == {r["poison_doc_id"] for r in rows},
                        f"{attack}/{ratio}%: manifest/document ID mismatch")
                require(len(item["extra_variants"]) == RATIOS[ratio] - 20,
                        f"{attack}/{ratio}%: manifest variant count mismatch")
            checked[attack, ratio] = rows
            count = RATIOS[ratio]
            print(f"PASS: {attack:4} {ratio:2}% | {count:2} unique poison chunks | "
                  f"{BASE_CLEAN_CHUNKS+count} final | "
                  f"{100*count/(BASE_CLEAN_CHUNKS+count):5.2f}% achieved | "
                  f"targets {min(count,20)}/20 | SHA-256 OK")

    for attack in ("SM", "OBEA", "RPI"):
        for smaller, bigger in ((1, 5), (5, 10), (10, 20)):
            a = {row["poison_doc_id"]: row for row in checked[attack, smaller]}
            b = {row["poison_doc_id"]: row for row in checked[attack, bigger]}
            require(all(doc in b and row == b[doc] for doc, row in a.items()),
                    f"{attack}: {smaller}% is not an unchanged subset of {bigger}%")
    print("PASS: all 12 datasets nested correctly (1% ⊂ 5% ⊂ 10% ⊂ 20%)")
    print("PASS: benchmark mappings, record schemas, single chunks, variant provenance, frozen seeds and hashes")
    print("PASS: PHASE B3 DATASET VALIDATION COMPLETE")
    print("SAFETY: READ ONLY — no DB, embeddings, Ollama, output writes or corpus resets")
    print("NEXT: Phase B4 isolated ChromaDB collection preflight; do not ingest yet.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
