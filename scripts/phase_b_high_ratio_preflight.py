"""PoisonBench Phase B2B: READ-ONLY 10% / 20% high-ratio allocation preflight.

Usage (from repository root):
    python scripts/phase_b_high_ratio_preflight.py

This script does not create datasets, import embedding models, connect to
ChromaDB, or modify the frozen attack outputs. B2B variant generation is
intentionally deferred until this allocation and optimizer review passes.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'poisoned'
PHASE_B = BASE / 'phase_b'
BENCHMARK = ROOT / 'data' / 'benchmark' / 'real_trigger_queries_v1.jsonl'
SM = BASE / 'semantic_mimicry_real.jsonl'
OBEA = BASE / 'optimized_embedding_attack_real.jsonl'
RPI = BASE / 'retrieved_prompt_injection_real.jsonl'
SOURCES = {'SM': SM, 'OBEA': OBEA, 'RPI': RPI}
B2A_MANIFEST = PHASE_B / 'phase_b_1pct_5pct_allocation_v1.json'
OPTIMIZER = ROOT / 'scripts' / 'build_optimized_embedding_real.py'
CLEAN_CHUNKS = 316
SEED = 42
# Final-corpus ratios: P / (316 + P)
PLANS = {10: 35, 20: 79}
PREFIX = {'SM': 'POISON-SM-', 'OBEA': 'POISON-OBEA-', 'RPI': 'POISON-RPI-'}


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Required input not found: {path}')
    result = []
    for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if line.strip():
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f'Not a JSON object: {path} line {n}')
            result.append(obj)
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    print('=== Phase B2B — 10%/20% allocation preflight (READ ONLY) ===')
    b = jsonl(BENCHMARK)
    bench = {}
    for r in b:
        qid = str(r.get('query_id', '')).lower().strip()
        if not qid or qid in bench or not r.get('query') or not r.get('expected_doc_id'):
            raise ValueError(f'Missing/duplicate benchmark query: {qid}')
        bench[qid] = r
    if len(bench) != 20 or set(bench) != {f'q{i:02d}' for i in range(1, 21)}:
        raise ValueError('Expected exactly q01-q20 in frozen benchmark')

    sources = {}
    for name, path in SOURCES.items():
        index = {}
        text_seen = set()
        for r in jsonl(path):
            qid = str(r.get('query_id', '')).lower().strip()
            text = r.get('poison_text')
            pid = str(r.get('poison_doc_id', ''))
            if (qid not in bench or qid in index or not pid.startswith(PREFIX[name])
                    or str(r.get('legitimate_doc_id')) != str(bench[qid]['expected_doc_id'])
                    or r.get('is_poisoned') is not True or not isinstance(text, str)
                    or not text.strip() or len(text.split()) > 180):
                raise ValueError(f'{name}: invalid frozen source record: {qid}')
            if text in text_seen:
                raise ValueError(f'{name}: duplicate poison text in frozen source: {qid}')
            index[qid] = r
            text_seen.add(text)
        if set(index) != set(bench):
            raise ValueError(f'{name}: source must target all 20 benchmark queries')
        sources[name] = index
        print(f'PASS: {name:4s} frozen source: 20 targets / 20 distinct texts, single chunk each')

    if not B2A_MANIFEST.is_file():
        raise FileNotFoundError(f'First finish B2A: {B2A_MANIFEST}')
    prev = json.loads(B2A_MANIFEST.read_text(encoding='utf-8'))
    if (prev.get('selection_seed') != SEED or prev.get('clean_chunk_count') != CLEAN_CHUNKS
            or prev.get('benchmark_sha256') != sha256(BENCHMARK)):
        raise ValueError('B2A benchmark, clean chunk count or random seed differs')
    for name, path in SOURCES.items():
        if prev.get('source_dataset_sha256', {}).get(name) != sha256(path):
            raise ValueError(f'B2A input hash differs for frozen {name} source')

    order = sorted(bench)
    random.Random(SEED).shuffle(order)
    if order != prev.get('selection_order'):
        raise ValueError('Fixed seed-42 target order disagrees with B2A manifest')
    for row in prev.get('files', []):
        p = ROOT / row['output_dataset']
        if sha256(p) != row['output_sha256']:
            raise ValueError(f'B2A dataset modified: {p}')
    print('PASS: B2A manifest, all six datasets, frozen hashes and common seed-42 order match')

    if not OPTIMIZER.is_file():
        raise FileNotFoundError(f'Frozen optimizer source missing: {OPTIMIZER}')
    tree = ast.parse(OPTIMIZER.read_text(encoding='utf-8-sig'))
    fns = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant):
                constants[node.targets[0].id] = node.value.value
    expected_functions = {'optimize_query', 'generate_mutations', 'score_candidates', 'generate_query_phrases'}
    if not expected_functions.issubset(fns) or constants.get('ROUNDS') != 8 or constants.get('CANDIDATES_PER_ROUND') != 12:
        raise ValueError('Frozen OBEA optimizer interface or search budget differs from reviewed source')
    print('PASS: Frozen OBEA source exposes 8-round / 12-candidate optimization functions')

    extra_slots = {}
    for ratio, total in PLANS.items():
        n_extra = total - 20
        additions = [order[i % 20] for i in range(n_extra)]
        extra_slots[ratio] = additions
        counts = Counter(additions)
        targets = Counter({qid: 1 for qid in order}) + counts
        if len(targets) != 20 or max(targets.values()) - min(targets.values()) > 1:
            raise AssertionError('Variant distribution not balanced')
        assert len(additions) + 20 == total
        achieved = 100 * total / (CLEAN_CHUNKS + total)
        print(f'{ratio:2d}%: {total:2d} poison + {CLEAN_CHUNKS} clean = {CLEAN_CHUNKS+total} chunks | achieved {achieved:.2f}%')
        print(f'      20 frozen base poisons + {n_extra} NEW variants | per-query counts: {dict(sorted(Counter(targets.values()).items()))}')
        print(f'      Extra variant allocation: {", ".join(additions)}')
        for name in SOURCES:
            candidate = PHASE_B / f'phase_b_{name.lower()}_{ratio}pct_v1.jsonl'
            if candidate.exists():
                raise FileExistsError(f'High-ratio output already exists; inspect before proceeding: {candidate}')
    if extra_slots[10] != extra_slots[20][:15]:
        raise AssertionError('10% extra variants must be nested within 20% allocation')
    print('PASS: 10% variant allocation is nested within 20%, shared across all 3 attacks')
    print('PASS: 6 high-ratio dataset target paths currently free')
    print('SAFETY: Read only; no embeddings, optimizer execution, ChromaDB, LLM, or file writes.')
    print('NEXT: Validate one mechanism-preserving variant per attack before full B2B generation.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
