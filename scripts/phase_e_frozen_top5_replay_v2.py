"""PoisonBench Phase E: frozen Top-5, DEFENSE-4-ONLY, OFFLINE filter replay.

Place at scripts/phase_e_frozen_top5_replay_v1.py, run from project root.

Default = READ-ONLY preflight. --run writes TWO NEW evaluation artifacts only.
NO ChromaDB, Ollama, live feature provider, defense 1-3, retraining or corpus mutation.

This is NOT an independent end-to-end mitigation test. Phase C's combined 553-row
feature table uses transductive geometry + benchmark-wide retrieval statistics.
228/300 retrieved *positions* were assigned to TRAIN, and 72 to TEST. The
results are partitioned and labeled explicitly, with held-out query groups
reported independently. Filtering never refills/re-ranks the frozen Top-5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.interfaces import RetrievedChunk  # noqa: E402
from defenses.defense4_random_forest.phase_e_defense4_adapter_v1 import (  # noqa: E402
    FrozenComparisonDefense, verify_artifacts,
)

C = ROOT / 'data' / 'results' / 'phase_c'
B = ROOT / 'data' / 'results' / 'phase_b'
OUT = ROOT / 'data' / 'results' / 'phase_e'
JSON_OUT = OUT / 'phase_e_defense4_frozen_top5_replay_v1.json'
CSV_OUT = OUT / 'phase_e_defense4_frozen_top5_replay_v1.csv'
ATTACKS = ('sm', 'obea', 'rpi')
MODELS = ('random_forest', 'xgboost')
DELIMITER = ' | '


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def split_field(value: object, *, allow_blank: bool = False) -> list[str]:
    if pd.isna(value):
        require(allow_blank, 'Missing ID field')
        return []
    s = str(value).strip()
    if not s:
        require(allow_blank, 'Blank ID field')
        return []
    return [v.strip() for v in s.split('|')]


def doc_id(chunk_id: str) -> str:
    require('::chunk' in chunk_id, f'Unexpected chunk identifier: {chunk_id}')
    return chunk_id.rsplit('::chunk', 1)[0]


def load_and_validate() -> tuple[list[str], pd.DataFrame, dict, list[dict], dict]:
    features, report = verify_artifacts()
    manifest = json.loads((C / 'split_manifest_v1.json').read_text(encoding='utf-8'))
    df = pd.read_csv(C / 'combined_features_v1.csv')
    require(len(df) == 553 and df['chunk_id'].is_unique, 'Incorrect feature dataset identities')
    require(df['is_poisoned'].value_counts().to_dict() == {0: 316, 1: 237}, 'Incorrect dataset labels')
    require(df['split'].value_counts().to_dict() == {'train': 416, 'test': 137}, 'Incorrect split counts')
    require(not df[features].isna().any().any(), 'Feature matrix contains missing values')
    require(all(pd.api.types.is_numeric_dtype(df[c]) for c in features), 'Nonnumeric features')
    require(df['group_id'].notna().all(), 'Missing split group ID')

    benchmark = ROOT / 'data' / 'benchmark' / 'real_trigger_queries_v1.jsonl'
    require(digest(benchmark) == manifest['benchmark_sha256'], 'Frozen benchmark digest differs')
    with benchmark.open(encoding='utf-8') as f:
        benchmark_by_q = {row['query_id'].lower(): row for s in f if s.strip()
                          for row in [json.loads(s)]}
    require(len(benchmark_by_q) == 20, 'Benchmark query count is not 20')

    by_doc: dict[str, list[dict]] = {}
    for row in df.to_dict(orient='records'):
        by_doc.setdefault(doc_id(str(row['chunk_id'])), []).append(row)
    # Multiple chunks per document are valid in the overall corpus.
    # Require unique identity only for the documents actually retrieved in B5.

    observations: list[dict] = []
    for attack in ATTACKS:
        path = B / f'phase_b5_{attack}_20pct_top5_v1_results.csv'
        require(path.is_file(), f'Missing B5 file: {path}')
        require(digest(path) == manifest['b5_csv_sha256'][attack], f'B5 checksum mismatch: {attack}')
        b5 = pd.read_csv(path)
        require(len(b5) == 20 and b5['query_id'].is_unique, f'Unexpected B5 rows: {attack}')
        require(set(b5['query_id']) == set(benchmark_by_q), f'Benchmark mismatch: {attack}')
        for rec in b5.to_dict(orient='records'):
            qid = str(rec['query_id'])
            require(str(rec['query']) == str(benchmark_by_q[qid]['query']),
                    f'Benchmark query text mismatch: {attack}/{qid}')
            expected_doc = str(benchmark_by_q[qid]['expected_doc_id'])
            require(str(rec['legitimate_doc_id']) == expected_doc,
                    f'Legitimate document mismatch: {attack}/{qid}')
            ids = split_field(rec['poisoned_retrieved_doc_ids'])
            scores = [float(v) for v in split_field(rec['poisoned_similarity_scores'])]
            require(len(ids) == len(scores) == 5, f'Not exactly 5 retrieved positions: {attack}/{qid}')
            require(len(set(ids)) == 5, f'Duplicate Top-5 document ID: {attack}/{qid}')
            require(all(math.isfinite(s) for s in scores), f'Invalid similarities: {attack}/{qid}')
            is_any_poison = any(v.startswith('POISON-') for v in ids)
            require(is_any_poison == bool(rec['any_poison_top5']),
                    f'B5 any-poison flag inconsistent: {attack}/{qid}')
            require((expected_doc in ids) == bool(rec['legitimate_top5']),
                    f'B5 legitimate flag inconsistent: {attack}/{qid}')
            for rank, (did, sim) in enumerate(zip(ids, scores), 1):
                require(did in by_doc, f'Unmapped B5 document ID: {attack}/{qid}/{did}')
                require(len(by_doc[did]) == 1,
                        f'Ambiguous RETRIEVED document-to-chunk mapping: {attack}/{qid}/{did}: {len(by_doc[did])} chunks')
                sample = by_doc[did][0]
                poison = int(did.startswith('POISON-'))
                require(int(sample['is_poisoned']) == poison,
                        f'Feature ground truth disagrees with ID: {attack}/{qid}/{did}')
                if poison:
                    require(did.startswith(f'POISON-{attack.upper()}-'),
                            f'Cross-collection poison observed: {attack}/{qid}/{did}')
                observations.append({
                    'attack_type': attack, 'query_id': qid, 'query': str(rec['query']),
                    'expected_doc_id': expected_doc, 'rank': rank, 'doc_id': did,
                    'chunk_id': str(sample['chunk_id']), 'similarity_score': sim,
                    'group_id': str(sample['group_id']), 'split': str(sample['split']),
                    'is_poisoned': poison,
                    'target_group_is_test': f'target:{qid}' in manifest['test_target_groups'],
                })
    require(len(observations) == 300, 'Expected 300 frozen retrieved positions')
    require(Counter(o['split'] for o in observations) == Counter({'train': 228, 'test': 72}),
            'Expected observed 228 training and 72 testing candidate positions')
    require(sum(o['target_group_is_test'] for o in observations) == 75,
            'Expected 75 retrieved positions from 15 held-out attack-query runs')
    return features, df, manifest, observations, report


def make_provider(df: pd.DataFrame, features: list[str]):
    # IMPORTANT: this is an OFFLINE FEATURE REPLAY ONLY. No production provider.
    # The scorer receives no ground-truth or split columns.
    lookup = {
        str(row['chunk_id']): {'chunk_id': str(row['chunk_id']),
                               **{feature: row[feature] for feature in features}}
        for row in df.to_dict(orient='records')
    }

    def provider(chunks):
        return [lookup[c.chunk_id] for c in chunks]
    return provider


def rate(num: int, denom: int):
    return num / denom if denom else None


def position_metrics(rows: list[dict]) -> dict:
    tp = sum(r['is_poisoned'] and r['flagged'] for r in rows)
    fp = sum(not r['is_poisoned'] and r['flagged'] for r in rows)
    fn = sum(r['is_poisoned'] and not r['flagged'] for r in rows)
    tn = sum(not r['is_poisoned'] and not r['flagged'] for r in rows)
    return {'retrieved_positions': len(rows), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'poison_detection_recall': rate(tp, tp + fn),
            'clean_false_positive_rate': rate(fp, fp + tn),
            'clean_retention_rate': rate(tn, fp + tn)}


def query_metrics(rows: list[dict]) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault((r['attack_type'], r['query_id']), []).append(r)
    for key, rr in groups.items():
        require(len(rr) == 5, f'Incomplete five-position query: {key}')
    before_any = sum(any(r['is_poisoned'] for r in rr) for rr in groups.values())
    after_any = sum(any(r['is_poisoned'] and not r['flagged'] for r in rr)
                    for rr in groups.values())
    before_legit = sum(any(r['doc_id'] == r['expected_doc_id'] for r in rr)
                       for rr in groups.values())
    after_legit = sum(any(r['doc_id'] == r['expected_doc_id'] and not r['flagged'] for r in rr)
                      for rr in groups.values())
    empty_after = sum(all(r['flagged'] for r in rr) for rr in groups.values())
    return {'queries': len(groups), 'queries_with_poison_before': before_any,
            'queries_with_poison_after_filter': after_any,
            'poison_present_query_rate_before': rate(before_any, len(groups)),
            'poison_present_query_rate_after': rate(after_any, len(groups)),
            'queries_with_legitimate_before': before_legit,
            'queries_with_legitimate_after_filter': after_legit,
            'legitimate_retention_given_initial_presence': rate(after_legit, before_legit),
            'queries_with_no_remaining_chunks': empty_after}


def build_results(features: list[str], df: pd.DataFrame, manifest: dict,
                  observations: list[dict], report: dict) -> tuple[dict, pd.DataFrame]:
    provider = make_provider(df, features)
    # Exact document/chunk IDs, other fields are NOT used by offline provider.
    # A chunk may be retrieved by MULTIPLE queries. Score each unique chunk
    # once and reuse its verdict across all frozen retrieved positions.
    unique_observations = {o['chunk_id']: o for o in observations}
    chunks = [RetrievedChunk(chunk_id=o['chunk_id'], doc_id=o['doc_id'],
                             text='', similarity_score=o['similarity_score'],
                             source='', is_poisoned=False, metadata={})
              for o in unique_observations.values()]
    results = {
        'schema': 'phase_e_defense4_frozen_top5_replay_v1',
        'scope': 'offline Phase C feature replay on frozen Phase B5 20%-poisoned Top-5; Defense 4 alone; threshold=0.5',
        'dataset_sha256': report['dataset_sha256'],
        'split_manifest_sha256': report['split_manifest_sha256'],
        'b5_csv_sha256': manifest['b5_csv_sha256'],
        'primary_feature_columns': features,
        'split_position_counts': dict(Counter(o['split'] for o in observations)),
        'heldout_target_query_ids': [g.removeprefix('target:') for g in manifest['test_target_groups']],
        'models': {},
        'limitations': [
            'OFFLINE replay of Phase C saved features: no live ChromaDB feature provider or real-time model deployment validated.',
            'Feature vectors use combined 553-row transductive embedding geometry and frozen benchmark-wide retrieval statistics.',
            '228 of 300 retrieved positions map to training rows. All-position metrics are DESCRIPTIVE and are not independent held-out evaluation.',
            'Five held-out target query groups per attack yield 75 candidate positions, some of which can still be training rows.',
            'Strict test-row position metrics count repeated retrieved positions and are not an independent sample-level classifier benchmark.',
            'Frozen Top-5 filtered in place: no refill, re-query, post-filter re-ranking, or subsequent retrieval validated.',
            'Defenses 1-3, end-to-end generation, attack text following, runtime feature overhead and production generalization not evaluated.',
            'No model selection, threshold tuning, classifier retraining, or label-based filtering performed here.',
        ],
    }
    all_rows = []
    for model_name in MODELS:
        model = FrozenComparisonDefense(model_name, provider, threshold=0.5)
        # Scores from feature table only, not labels. Ground truth is passed as false.
        verdicts = model.score(chunks)
        require(len(verdicts) == len(unique_observations), f'Incomplete verdicts for {model_name}')
        by_chunk_verdict = {v.chunk_id: v for v in verdicts}
        require(len(by_chunk_verdict) == len(verdicts), 'Duplicate model verdict IDs')
        rr = []
        for o in observations:
            verdict = by_chunk_verdict[o['chunk_id']]
            rr.append({**o, 'model': model_name, 'flagged': bool(verdict.flagged),
                       'poison_probability': float(verdict.score),
                       'retained_after_filter': not bool(verdict.flagged)})
        require(all(0 <= r['poison_probability'] <= 1 for r in rr), 'Invalid model probabilities')
        results['models'][model_name] = {
            'all_positions_descriptive': position_metrics(rr),
            'strict_test_positions_only': position_metrics([r for r in rr if r['split'] == 'test']),
            'strict_train_positions_only_in_sample': position_metrics([r for r in rr if r['split'] == 'train']),
            'all_queries_descriptive': query_metrics(rr),
            'heldout_target_queries_mixed_candidate_splits': query_metrics([r for r in rr if r['target_group_is_test']]),
            'by_attack': {
                attack: {
                    'all_positions_descriptive': position_metrics([r for r in rr if r['attack_type'] == attack]),
                    'strict_test_positions_only': position_metrics([r for r in rr if r['attack_type'] == attack and r['split'] == 'test']),
                    'all_queries_descriptive': query_metrics([r for r in rr if r['attack_type'] == attack]),
                    'heldout_target_queries_mixed_candidate_splits': query_metrics([r for r in rr if r['attack_type'] == attack and r['target_group_is_test']]),
                } for attack in ATTACKS
            },
        }
        all_rows.extend(rr)
    require(len(all_rows) == 600, 'Expected exactly 600 model×retrieved-position results')
    return results, pd.DataFrame(all_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='store_true', help='Create two NEW Phase E replay artifacts')
    args = ap.parse_args()
    features, df, manifest, observations, report = load_and_validate()
    print('PASS: 3 frozen B5 20% CSV checksums; 60 queries × 5 retrieved positions = 300')
    print('PASS: 300/300 unique document→chunk mappings; Phase C features are intact')
    print('PASS: Candidate split exposure: train=228, test=72')
    print('PASS: 15 held-out attack/query runs (75 positions), potentially mixed candidate splits')
    print('PASS: Frozen RF/XGBoost model and split checksums, 12 features')
    print('LIMITATION: This evaluates OFFLINE Defense 4 feature replay, not live model integration')
    if not args.run:
        print('PREFLIGHT PASS: READ ONLY, no model deserialization, fitting, or file writes')
        print(f'Planned new output: {JSON_OUT.relative_to(ROOT)}')
        print(f'Planned new output: {CSV_OUT.relative_to(ROOT)}')
        return
    require(not JSON_OUT.exists() and not CSV_OUT.exists(),
            'Refusing to overwrite existing frozen Phase E replay outputs')
    results, predictions = build_results(features, df, manifest, observations, report)
    require(not JSON_OUT.exists() and not CSV_OUT.exists(),
            'Output collision detected; refusing to overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    # Exclusive creation, never rewrite an existing report.
    with JSON_OUT.open('x', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    with CSV_OUT.open('x', encoding='utf-8', newline='') as f:
        predictions.to_csv(f, index=False)
    print('PASS: Two isolated offline Phase E replay outputs created')
    print('JSON SHA256:', digest(JSON_OUT))
    print('CSV SHA256:', digest(CSV_OUT))
    for name in MODELS:
        m = results['models'][name]
        print(name, 'heldout_target_query_stats:', m['heldout_target_queries_mixed_candidate_splits'])
        print(name, 'strict_test_position_stats:', m['strict_test_positions_only'])
    print('SAFETY: No ChromaDB/Ollama calls, model retraining, or changes to original files')


if __name__ == '__main__':
    main()
