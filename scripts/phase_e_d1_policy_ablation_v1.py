"""PoisonBench Phase E read-only D1 hard-filter vs advisory policy ablation.

Place at scripts/phase_e_d1_policy_ablation_v1.py; run from repository root.

    python scripts/phase_e_d1_policy_ablation_v1.py           # no Chroma access
    python scripts/phase_e_d1_policy_ablation_v1.py --sweep   # Chroma read only

Policy comparisons on frozen B5 Top-5 candidates:
  A: D1 hard-block -> D4 RF/XGB frozen-reference classifier
  B: D1 advisory-only -> D4 RF/XGB frozen-reference classifier
  C: D4 alone (identical surviving chunks to B; D1 runs but does not filter)
D2/D3 intentionally not invoked: canonical provenance/trust metadata absent.
D4 is NOT a live feature provider: it looks up transductive Phase C benchmark
features. Counts include training-associated chunks, so are NOT independent
model generalization estimates. NO Chroma writes, training, Ollama, or files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.interfaces import RetrievedChunk
from defenses.defense4_random_forest.phase_e_defense4_adapter_v1 import (
    FrozenComparisonDefense, verify_artifacts,
)
from defenses.defense1_outlier_detection.filter import apply_outlier_filter

ATTACK_COLLECTIONS = {
    'sm': 'poisonbench_b_sm_20pct_v1',
    'obea': 'poisonbench_b_obea_20pct_v1',
    'rpi': 'poisonbench_b_rpi_20pct_v1',
}
B = ROOT / 'data' / 'results' / 'phase_b'
C = ROOT / 'data' / 'results' / 'phase_c'
BENCH = ROOT / 'data' / 'benchmark' / 'real_trigger_queries_v1.jsonl'
FORBIDDEN = {'is_poisoned', 'attack_type'}


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_from_chunk(cid):
    require('::chunk' in cid, f'Invalid chunk ID {cid}')
    return cid.rsplit('::chunk', 1)[0]


def preflight():
    columns, comparison = verify_artifacts()
    manifest = json.loads((C / 'split_manifest_v1.json').read_text(encoding='utf-8'))
    require(sha(BENCH) == manifest['benchmark_sha256'], 'Benchmark SHA mismatch')
    queries = {}
    for line in BENCH.read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            queries[r['query_id'].lower()] = r
    require(len(queries) == 20, 'Benchmark length mismatch')
    feats = pd.read_csv(C / 'combined_features_v1.csv')
    require(len(feats) == 553 and feats['chunk_id'].is_unique, 'Dataset shape invalid')
    by_doc = defaultdict(list)
    by_id = {}
    for r in feats.to_dict(orient='records'):
        by_doc[doc_from_chunk(str(r['chunk_id']))].append(r)
        by_id[r['chunk_id']] = r
    runs = {}
    for attack in ATTACK_COLLECTIONS:
        path = B / f'phase_b5_{attack}_20pct_top5_v1_results.csv'
        require(path.exists() and sha(path) == manifest['b5_csv_sha256'][attack],
                f'Frozen B5 SHA mismatch: {attack}')
        b5 = pd.read_csv(path)
        require(len(b5) == 20 and b5['query_id'].is_unique, 'B5 20-query schema mismatch')
        attack_runs = []
        for row in b5.to_dict(orient='records'):
            qid = row['query_id']
            require(qid in queries and row['query'] == queries[qid]['query'],
                    f'Benchmark query mismatch {attack}/{qid}')
            expected = queries[qid]['expected_doc_id']
            require(expected == row['legitimate_doc_id'], 'Legitimate doc mismatch')
            docs = [s.strip() for s in row['poisoned_retrieved_doc_ids'].split('|')]
            scores = [float(s.strip()) for s in row['poisoned_similarity_scores'].split('|')]
            require(len(docs) == len(scores) == 5 and len(set(docs)) == 5,
                    f'B5 top5 invalid {attack}/{qid}')
            chunks = []
            for d in docs:
                matches = by_doc.get(d, [])
                require(len(matches) == 1, f'Ambiguous B5 doc mapping: {attack}/{qid}/{d}')
                chunks.append(matches[0]['chunk_id'])
            attack_runs.append({'attack': attack, 'qid': qid, 'expected': expected,
                                'docs': docs, 'scores': scores, 'ids': chunks})
        runs[attack] = attack_runs
    print('PASS: Frozen Phase C/D model and dataset hashes, 12-feature contract')
    print('PASS: Three B5 hashes, 60 query runs, 300 unambiguous Top-5 positions')
    print('PASS: D1 policy comparison defined without changing Akilan thresholds')
    print('NOTE: D2/D3 skipped (required canonical metadata absent)')
    print('NOTE: D4 uses frozen benchmark-wide features, not live feature generation')
    return columns, by_id, manifest, runs


def query_rows(r, entries):
    rows = []
    for doc_id, score, chunk_id in zip(r['docs'], r['scores'], r['ids']):
        meta = dict(entries[chunk_id]['meta'])
        rows.append({
            'chunk_id': chunk_id, 'doc_id': doc_id,
            'text': entries[chunk_id]['text'],
            'metadata': meta,
            'embedding': entries[chunk_id]['embedding'],
            'source': str(meta.get('source', '')),
            'similarity': float(score), 'similarity_score': float(score),
        })
    return rows


def describe(rows, expected):
    return {
        'poison_positions': sum(r['doc_id'].startswith('POISON-') for r in rows),
        'clean_positions': sum(not r['doc_id'].startswith('POISON-') for r in rows),
        'legit_present': int(any(r['doc_id'] == expected for r in rows)),
        'any_poison': int(any(r['doc_id'].startswith('POISON-') for r in rows)),
        'empty': int(not rows),
    }


def sweep(columns, features_by_id, manifest, runs):
    import chromadb
    client = chromadb.HttpClient(host='localhost', port=8001)
    entries_by_attack = {}
    for attack, attack_runs in runs.items():
        collection = client.get_collection(ATTACK_COLLECTIONS[attack])
        require(collection.count() == 395, f'{attack}: collection count changed')
        ids = sorted({cid for r in attack_runs for cid in r['ids']})
        result = collection.get(ids=ids, include=['documents', 'metadatas', 'embeddings'])
        entries = {}
        for cid, txt, meta, emb in zip(result['ids'], result['documents'],
                                       result['metadatas'], result['embeddings']):
            require(len(emb) == 384 and isinstance(txt, str), f'{cid}: missing text/embedding')
            require(meta.get('doc_id') == doc_from_chunk(cid), f'{cid}: metadata mismatch')
            stripped = {k: v for k, v in dict(meta).items() if k not in FORBIDDEN}
            require(FORBIDDEN.isdisjoint(stripped), f'{cid}: ground truth leaked')
            entries[cid] = {'text': txt, 'meta': stripped,
                            'embedding': [float(v) for v in emb]}
        require(set(entries) == set(ids), f'{attack}: missing / extra Chroma chunks')
        entries_by_attack[attack] = entries
        print(f'PASS: {attack.upper()} read-only Chroma mapping ({len(ids)} unique chunks)')

    frozen_features = {cid: {'chunk_id': cid, **{f: float(r[f]) for f in columns}}
                       for cid, r in features_by_id.items()}
    def provider(chunks):
        return [frozen_features[c.chunk_id] for c in chunks]

    models = {model: FrozenComparisonDefense(model, provider)
              for model in ('random_forest', 'xgboost')}
    counters = defaultdict(Counter)
    affected = defaultdict(list)
    heldout = set(manifest['test_target_groups'])
    require(len(heldout) == 5, 'Heldout group count mismatch')
    for attack, attack_runs in runs.items():
        entries = entries_by_attack[attack]
        for r in attack_runs:
            rows = query_rows(r, entries)
            original_ids = [v['chunk_id'] for v in rows]
            passed, blocked = apply_outlier_filter(rows)
            require(len(passed) + len(blocked) == 5, 'D1 dropped or duplicated candidates')
            blocked_ids = {v['chunk_id'] for v in blocked}
            require(len(blocked_ids) == len(blocked), 'D1 duplicate block ids')
            if r['expected'] in {v['doc_id'] for v in blocked}:
                affected[attack].append(r['qid'])
            gt = {cid: bool(features_by_id[cid]['is_poisoned']) for cid in original_ids}
            for item in rows:
                require(gt[item['chunk_id']] == item['doc_id'].startswith('POISON-'),
                        f"Ground truth mismatch {attack}/{r['qid']}/{item['chunk_id']}")
            original = describe(rows, r['expected'])
            for key, value in original.items():
                counters[(attack, 'before')][key] += value
            for item in rows:
                truth = gt[item['chunk_id']]
                is_blocked = item['chunk_id'] in blocked_ids
                counters[(attack, 'd1')]['tp' if truth and is_blocked else
                                       'fp' if not truth and is_blocked else
                                       'fn' if truth else 'tn'] += 1
            for model_name, model in models.items():
                chunks = [RetrievedChunk(
                    chunk_id=item['chunk_id'], doc_id=item['doc_id'],
                    text=item['text'], similarity_score=item['similarity_score'],
                    source=item['source'], is_poisoned=False,
                    metadata=item['metadata']) for item in rows]
                verdicts = model.score(chunks)
                require([v.chunk_id for v in verdicts] == original_ids,
                        'D4 verdict identity/order mismatch')
                d4_blocked = {v.chunk_id for v in verdicts if v.flagged}
                policies = {
                    'd4_only_or_d1_advisory': d4_blocked,
                    'd1_hard_then_d4': blocked_ids | d4_blocked,
                }
                is_heldout = f"target:{r['qid']}" in heldout
                for policy, dropped in policies.items():
                    remaining = [item for item in rows if item['chunk_id'] not in dropped]
                    rec = describe(remaining, r['expected'])
                    for cohort in ('all', 'heldout') if is_heldout else ('all',):
                        c = counters[(attack, model_name, policy, cohort)]
                        c['runs'] += 1
                        for key, value in rec.items():
                            c[key] += value
                        c['legit_initially_present'] += original['legit_present']
                        c['poison_initially_present'] += original['any_poison']

    for attack in ATTACK_COLLECTIONS:
        b, d = counters[(attack, 'before')], counters[(attack, 'd1')]
        print(f'\n=== {attack.upper()} ===')
        print(f"D1 independent check: TP={d['tp']} FP={d['fp']} FN={d['fn']} TN={d['tn']}")
        print(f"D1 lost expected legitimate CVE in {len(affected[attack])} queries: {affected[attack]}")
        for model_name in models:
            for policy in ('d4_only_or_d1_advisory', 'd1_hard_then_d4'):
                c = counters[(attack, model_name, policy, 'all')]
                print(f"{model_name} | {policy} | all: runs={c['runs']}, "
                      f"poison_present_after={c['any_poison']}, "
                      f"legit_after={c['legit_present']}/{c['legit_initially_present']}, "
                      f"empty={c['empty']}")
                h = counters[(attack, model_name, policy, 'heldout')]
                print(f"  heldout target queries: runs={h['runs']}, "
                      f"poison_present_after={h['any_poison']}, "
                      f"legit_after={h['legit_present']}/{h['legit_initially_present']}, "
                      f"empty={h['empty']}")
    d1_total = Counter()
    for attack in ATTACK_COLLECTIONS:
        d1_total.update(counters[(attack, 'd1')])
    require((d1_total['tp'], d1_total['fp']) == (2, 61),
            'Previously observed D1 sweep not reproduced; inspect before use')
    print('\nPASS: D1 independent 60-query counts reproduced (TP=2, FP=61)')
    print('RESULT MODE: frozen Top-5, frozen-reference D4; not full D1-D4 live integration')
    print('RESTRICTIONS: D2/D3 absent; no top-k refill; no generation; train-associated chunks included')
    print('READ ONLY: No DB writes, file writes, retraining or Ollama calls')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', action='store_true', help='Chroma read-only policy ablation on 60 frozen queries')
    args = parser.parse_args()
    features, by_id, manifest, runs = preflight()
    if args.sweep:
        sweep(features, by_id, manifest, runs)
    else:
        print('PREFLIGHT PASS: READ ONLY; no Chroma connection, deserialization or file writes')
        print('NEXT: python scripts/phase_e_d1_policy_ablation_v1.py --sweep')


if __name__ == '__main__':
    main()
