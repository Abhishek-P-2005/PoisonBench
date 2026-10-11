"""PoisonBench Phase B2A: deterministic 1%/5% poison subsets for all 3 attacks.

Usage (from repository root):
    python scripts/phase_b_build_low_ratio_datasets.py          # read-only
    python scripts/phase_b_build_low_ratio_datasets.py --build  # write NEW files only

Creates six independent attack/ratio JSONL datasets and a provenance manifest.
Never mutates the frozen original datasets, ChromaDB, or past result files.
The 10% and 20% variant-generation protocol is deliberately deferred until
we verify how the Attack 2 embedding optimizer generates *new* candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / 'data' / 'benchmark' / 'real_trigger_queries_v1.jsonl'
SOURCE_FILES = {
    'SM': ROOT / 'data' / 'poisoned' / 'semantic_mimicry_real.jsonl',
    'OBEA': ROOT / 'data' / 'poisoned' / 'optimized_embedding_attack_real.jsonl',
    'RPI': ROOT / 'data' / 'poisoned' / 'retrieved_prompt_injection_real.jsonl',
}
PREFIX = {'SM': 'POISON-SM-', 'OBEA': 'POISON-OBEA-', 'RPI': 'POISON-RPI-'}
OUT_DIR = ROOT / 'data' / 'poisoned' / 'phase_b'
MANIFEST = OUT_DIR / 'phase_b_1pct_5pct_allocation_v1.json'
N_CLEAN = 316
N_TARGETS = 20
SEED = 42
PLANS = [(1, 3), (5, 17)]


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Required frozen input missing: {path}')
    out = []
    for lineno, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if line.strip():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path}:{lineno}') from exc
            if not isinstance(obj, dict):
                raise ValueError(f'Expected object at {path}:{lineno}')
            out.append(obj)
    return out


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hash(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def validate_benchmark() -> dict[str, dict]:
    records = load_jsonl(BENCHMARK_PATH)
    expected = {}
    for row in records:
        qid = str(row.get('query_id', '')).strip().lower()
        if qid in expected or not qid or not str(row.get('query', '')).strip() or not str(row.get('expected_doc_id', '')).strip():
            raise ValueError(f'Invalid or duplicate benchmark query {qid!r}')
        expected[qid] = row
    if len(records) != N_TARGETS or set(expected) != {f'q{x:02d}' for x in range(1, N_TARGETS + 1)}:
        raise ValueError('Frozen benchmark must have q01–q20, one each')
    return expected


def validate_sources(benchmark: dict[str, dict]) -> dict[str, dict[str, dict]]:
    output = {}
    for attack, path in SOURCE_FILES.items():
        rows = load_jsonl(path)
        by_id = {}
        poison_ids = set()
        for row in rows:
            qid = str(row.get('query_id', '')).strip().lower()
            doc_id = str(row.get('poison_doc_id', '')).strip()
            text = row.get('poison_text')
            if qid not in benchmark or qid in by_id or doc_id in poison_ids:
                raise ValueError(f'{attack}: invalid/duplicate query or poison ID: {qid}/{doc_id}')
            if (not doc_id.startswith(PREFIX[attack]) or
                str(row.get('legitimate_doc_id', '')).strip() != str(benchmark[qid]['expected_doc_id']).strip() or
                row.get('is_poisoned') is not True or
                not isinstance(text, str) or not text.strip() or
                len(text.split()) > 180 or
                not str(row.get('source', '')).strip() or
                not str(row.get('attack_type', '')).strip()):
                raise ValueError(f'{attack}: invalid schema or non-single-chunk text at {qid}')
            by_id[qid] = row
            poison_ids.add(doc_id)
        if len(rows) != N_TARGETS or set(by_id) != set(benchmark):
            raise ValueError(f'{attack}: expected exactly 20 mapped seed poisons')
        output[attack] = by_id
    return output


def prepare() -> tuple[dict[Path, bytes], dict]:
    benchmark = validate_benchmark()
    sources = validate_sources(benchmark)
    order = sorted(benchmark)
    random.Random(SEED).shuffle(order)
    new_files: dict[Path, bytes] = {}
    rows_manifest = []
    for ratio, n_poison in PLANS:
        selected = order[:n_poison]
        actual_ratio = n_poison / (N_CLEAN + n_poison)
        for attack, indexed in sources.items():
            selected_rows = [indexed[qid] for qid in selected]
            ids = [row['poison_doc_id'] for row in selected_rows]
            if len(set(ids)) != n_poison:
                raise RuntimeError(f'{attack}/{ratio}: duplicate poison doc IDs')
            output_path = OUT_DIR / f'phase_b_{attack.lower()}_{ratio}pct_v1.jsonl'
            data = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in selected_rows).encode('utf-8')
            new_files[output_path] = data
            rows_manifest.append({
                'attack': attack,
                'target_ratio_pct': ratio,
                'clean_chunks': N_CLEAN,
                'poison_chunks': n_poison,
                'final_chunks': N_CLEAN + n_poison,
                'achieved_ratio_pct': round(actual_ratio * 100, 8),
                'targeted_query_ids': selected,
                'target_coverage': f'{len(selected)}/{N_TARGETS}',
                'original_poison_doc_ids': ids,
                'output_dataset': output_path.relative_to(ROOT).as_posix(),
                'output_sha256': sha_bytes(data),
            })
    metadata = {
        'phase': 'B2A',
        'protocol': 'final-corpus poisoning fraction = poison chunks/(316 clean + poison chunks)',
        'benchmark_path': BENCHMARK_PATH.relative_to(ROOT).as_posix(),
        'benchmark_sha256': source_hash(BENCHMARK_PATH),
        'source_dataset_sha256': {attack: source_hash(path) for attack, path in SOURCE_FILES.items()},
        'selection_seed': SEED,
        'selection_order': order,
        'selection_protocol': 'same first N query IDs in shuffled order for all 3 attacks; nested 1%-5% target coverage',
        'generation_method': 'unmodified, frozen one-chunk seed poison documents; no new attack variants at 1%/5%',
        'clean_chunk_count': N_CLEAN,
        'files': rows_manifest,
        'limitations': [
            'Overall benchmark success is coverage-limited (only 3 or 17 targeted queries); report success conditional on targeting too.',
            '10% and 20% require additional mechanism-preserving attack variants; not produced by this script.',
            'This manifests dataset allocation only; no retrieval/generation efficacy is implied.',
        ],
    }
    new_files[MANIFEST] = (json.dumps(metadata, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    return new_files, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Write only the six new JSONL files and their manifest; no ChromaDB writes')
    args = parser.parse_args()
    planned, manifest = prepare()
    print('=== Phase B2A — 1%/5% attack datasets (NO DATABASE ACCESS) ===')
    print('PASS: Frozen benchmark and all three 20-document source datasets validated')
    print(f'Fixed seed: {SEED}; common target order: {", ".join(manifest["selection_order"])}')
    for r in manifest['files']:
        print(f'{r["attack"]:4s} {r["target_ratio_pct"]:2d}% | {r["poison_chunks"]:2d} poisons + {r["clean_chunks"]} clean = {r["final_chunks"]} | achieved {r["achieved_ratio_pct"]:.2f}% | targets {r["target_coverage"]}')
    differing = [str(path) for path, content in planned.items() if path.exists() and path.read_bytes() != content]
    if differing:
        raise FileExistsError('Refusing to replace existing DIFFERENT output files:\n' + '\n'.join(differing))
    existing = sum(path.exists() for path in planned)
    if args.build:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for path, content in planned.items():
            if not path.exists():
                path.write_bytes(content)
                print('CREATED:', path.relative_to(ROOT).as_posix())
            else:
                print('VERIFIED existing:', path.relative_to(ROOT).as_posix())
        print('PASS: Six ratio datasets and provenance manifest created/verified; no frozen files modified.')
    else:
        print(f'PASS: Read-only preflight; {len(planned)} output files planned ({existing} already matching).')
        print('NEXT: After review, run this same script with --build.')
    print('SAFETY: No ChromaDB connection, embedding model, LLM run, or ingestion.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
