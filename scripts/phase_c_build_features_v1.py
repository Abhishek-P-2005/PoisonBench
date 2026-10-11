"""PoisonBench Phase C: build an auditable, isolated ML comparison dataset.

Default: read-only preflight. --build: exclusive-create TWO new artifacts only.
Does not change any ChromaDB collection, frozen Phase B artifact, or teammate code.

Feature model deliberately reuses Defense 4's existing feature-engineering schema.
Its embedding/retrieval features are TRANSDUCTIVE corpus statistics; they do not
establish performance on genuinely unseen, independently collected corpora.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Direct execution as `python scripts/phase_c_build_features_v1.py` must find project modules.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.pipeline.embedding import embed_texts
from api.pipeline.vectorstore import get_client
from defenses.defense4_xgboost.feature_engineering import (
    FEATURE_COLUMNS, LABEL_COLUMN, build_feature_matrix,
)

BENCHMARK = ROOT / 'data/benchmark/real_trigger_queries_v1.jsonl'
B5_DIR = ROOT / 'data/results/phase_b'
OUT_DIR = ROOT / 'data/results/phase_c'
OUT_CSV = OUT_DIR / 'combined_features_v1.csv'
OUT_MANIFEST = OUT_DIR / 'split_manifest_v1.json'
CLEAN = 'poisonbench_rpi_clean_v1'
ATTACKS = ('sm', 'obea', 'rpi')
RATIO = 20
SEED = 42
TOP_K = 5
# Exclude direct source/label shortcut, and zero-variance timestamp feature.
PRIMARY_FEATURES = [x for x in FEATURE_COLUMNS
                    if x not in ('source_is_official', 'source_age_days')]
POISON_PATTERN = re.compile(r'^POISON-(SM|OBEA|RPI)-(q\d{2})(?:-v\d+)?::chunk0$')


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_benchmark():
    with BENCHMARK.open(encoding='utf-8') as f:
        rows = [json.loads(x) for x in f if x.strip()]
    require(len(rows) == 20, 'Expected 20 frozen queries')
    lookup = {r['query_id'].lower(): r for r in rows}
    require(len(lookup) == 20 and set(lookup) == {f'q{i:02d}' for i in range(1, 21)},
            'Unexpected frozen query identifiers')
    return lookup


def load_b5(attack, benchmark):
    path = B5_DIR / f'phase_b5_{attack}_20pct_top5_v1_results.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    lookup = {r['query_id']: r for r in rows}
    require(len(rows) == len(lookup) == 20 and set(lookup) == set(benchmark),
            f'B5 query cardinality failed: {path}')
    for qid, row in lookup.items():
        truth = benchmark[qid]
        require(row['query'] == truth['query'] and
                row['legitimate_doc_id'] == truth['expected_doc_id'],
                f'B5 benchmark mismatch: {path.name} {qid}')
    return lookup, path


def load_collection(collection):
    result = collection.get(include=['documents', 'metadatas', 'embeddings'])
    ids = result['ids']
    texts = result['documents']
    metas = result['metadatas']
    vectors = result['embeddings']
    require(len(ids) == len(texts) == len(metas) == len(vectors),
            f'{collection.name}: incomplete get()')
    output = {}
    for cid, text, meta, vec in zip(ids, texts, metas, vectors):
        require(cid not in output and bool(text.strip()), f'{collection.name}: invalid text/id {cid}')
        require(len(vec) == 384 and np.isfinite(vec).all(),
                f'{collection.name}: missing/nonfinite embedding {cid}')
        require(meta.get('doc_id') and
                cid == f"{meta['doc_id']}::chunk{meta.get('position')}",
                f'{collection.name}: malformed chunk id {cid}')
        require(type(meta.get('is_poisoned')) is bool,
                f'{collection.name}: missing truth label {cid}')
        output[cid] = dict(chunk_id=cid, text=text, metadata=meta,
                           embedding=np.asarray(vec, dtype=np.float64))
    return output


def check_retrieval(collection, expected_rows, qids, vectors, scores_by_chunk, collection_role):
    """Verify exact frozen B5 top-5 ordering, chunk IDs, and rounded similarities."""
    for qid, vector in zip(qids, vectors):
        result = collection.query(
            query_embeddings=[vector], n_results=TOP_K,
            include=['metadatas', 'distances'])
        cids, metas, distances = (result[k][0] for k in ('ids', 'metadatas', 'distances'))
        require(len(cids) == len(metas) == len(distances) == TOP_K and
                len(set(cids)) == TOP_K,
                f'{collection.name} {qid}: incomplete Top-5')
        row = expected_rows[qid]
        prefix = 'clean' if collection_role == 'clean' else 'poisoned'
        exp_ids = row[f'{prefix}_retrieved_doc_ids'].split(' | ')
        exp_scores = [float(x) for x in row[f'{prefix}_similarity_scores'].split(' | ')]
        actual_ids = [meta['doc_id'] for meta in metas]
        require(actual_ids == exp_ids,
                f'{collection.name} {qid}: frozen B5 Top-5 order mismatch')
        for cid, meta, dist, saved in zip(cids, metas, distances, exp_scores):
            actual = 1.0 - float(dist)
            require(math.isfinite(actual) and abs(actual - saved) < 1e-4,
                    f'{collection.name} {qid}: B5 score mismatch {cid}')
            require(cid == f"{meta['doc_id']}::chunk{meta['position']}",
                    f'{collection.name} {qid}: chunk metadata mismatch')
            if collection_role == 'clean':
                require(meta['source'] == 'NVD' and meta['is_poisoned'] is False,
                        f'{collection.name} {qid}: dirty clean result')
            # Record exact chunk-level scores instead of guessing ::chunk0
            scores_by_chunk[cid].append(actual)


def build():
    require(not OUT_CSV.exists() and not OUT_MANIFEST.exists(),
            'Phase C result file already exists; refusing to overwrite')
    benchmark = load_benchmark()
    qids = sorted(benchmark)
    client = get_client()
    clean_collection = client.get_collection(CLEAN)
    clean_rows = load_collection(clean_collection)
    require(len(clean_rows) == 316 and
            all(not r['metadata']['is_poisoned'] and
                r['metadata']['source'] == 'NVD'
                for r in clean_rows.values()),
            'Clean 316-chunk corpus invariant failed')

    b5 = {}
    b5paths = {}
    collections = {}
    for attack in ATTACKS:
        b5[attack], b5paths[attack] = load_b5(attack, benchmark)
        name = f'poisonbench_b_{attack}_20pct_v1'
        coll = client.get_collection(name)
        contents = load_collection(coll)
        require(len(contents) == 395, f'{name}: expected 395 chunks')
        clean_ids = {cid for cid, r in contents.items() if not r['metadata']['is_poisoned']}
        poisons = {cid: r for cid, r in contents.items() if r['metadata']['is_poisoned']}
        require(clean_ids == set(clean_rows) and len(poisons) == 79,
                f'{name}: wrong clean/poison composition')
        for cid in clean_ids:
            a, b = clean_rows[cid], contents[cid]
            require(a['text'] == b['text'] and a['metadata'] == b['metadata'] and
                    np.allclose(a['embedding'], b['embedding'], atol=1e-7, rtol=0),
                    f'{name}: clean reference drift at {cid}')
        group_counts = Counter()
        for cid, r in poisons.items():
            match = POISON_PATTERN.fullmatch(cid)
            require(match is not None and match.group(1).lower() == attack,
                    f'{name}: malformed poison chunk ID {cid}')
            require(r['metadata']['source'] != 'NVD', f'{name}: invalid poison source')
            group_counts[match.group(2)] += 1
        require(set(group_counts) == set(qids) and sorted(group_counts.values()) == [3] + [4]*19,
                f'{name}: variant grouping mismatch')
        collections[attack] = (coll, poisons)
    print('PASS: 316 shared clean chunks + 79 poisons per attack; frozen source text and vectors')

    # Frozen 20-query Top-5 evidence, independently resolved to exact chunk IDs.
    print('Embedding the 20 frozen benchmark queries (read only)...', flush=True)
    vectors = embed_texts([benchmark[qid]['query'] for qid in qids])
    require(len(vectors) == 20, 'Unexpected query embedding count')
    clean_scores = defaultdict(list)
    check_retrieval(clean_collection, b5['rpi'], qids, vectors, clean_scores, 'clean')
    attack_scores = {}
    for attack in ATTACKS:
        scores = defaultdict(list)
        check_retrieval(collections[attack][0], b5[attack], qids, vectors, scores, 'poisoned')
        attack_scores[attack] = scores
        print(f'PASS: {attack.upper()} 20% B5 Top-5 independently revalidated')

    # Build one deduplicated, transductive feature matrix.
    # Clean chunks use baseline CLEAN collection query appearances;
    # poison chunks use their attack-specific 20%-collection appearances.
    chunks = [clean_rows[cid] for cid in sorted(clean_rows)]
    scores_by_chunk = defaultdict(list)
    scores_by_chunk.update(clean_scores)
    attack_by_cid = {}
    group_by_cid = {}
    for chunk in chunks:
        attack_by_cid[chunk['chunk_id']] = 'clean'
        group_by_cid[chunk['chunk_id']] = 'nvd:' + chunk['metadata']['doc_id']
    for attack in ATTACKS:
        for cid, chunk in sorted(collections[attack][1].items()):
            chunks.append(chunk)
            scores_by_chunk[cid] = attack_scores[attack].get(cid, [])
            attack_by_cid[cid] = attack
            group_by_cid[cid] = 'target:' + POISON_PATTERN.fullmatch(cid).group(2)
    require(len(chunks) == 553 and len({c['chunk_id'] for c in chunks}) == 553,
            'Combined dataset size/unique IDs mismatch')

    print('Computing 14 Defense 4 features for 553 rows...', flush=True)
    frame = build_feature_matrix(chunks, scores_by_chunk)
    require(len(frame) == 553 and
            set(frame['chunk_id']) == set(attack_by_cid) and
            set(FEATURE_COLUMNS).issubset(frame),
            'Feature-matrix schema/count failure')
    frame['attack_type'] = frame['chunk_id'].map(attack_by_cid)
    frame['group_id'] = frame['chunk_id'].map(group_by_cid)
    require(frame[FEATURE_COLUMNS].notna().all().all(), 'Missing feature values')
    require(np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).all(),
            'Nonfinite numerical features')
    require(Counter(frame[LABEL_COLUMN]) == {0: 316, 1: 237},
            'Incorrect class distribution')
    require((frame['source_is_official'] == (1 - frame[LABEL_COLUMN])).all(),
            'Expected deterministic source leakage relationship changed: review design')
    require(frame['source_age_days'].nunique() == 1,
            'Expected zero-variance source-age feature changed: review design')

    # Deterministic, disjoint query-target groups held out across all attacks;
    # clean chunks split by original NVD doc_id (no same-doc split leakage).
    from sklearn.model_selection import train_test_split
    target_groups = [f'target:{x}' for x in qids]
    target_train, target_test = train_test_split(target_groups, test_size=0.25, random_state=SEED)
    clean_groups = sorted({group_by_cid[cid] for cid in clean_rows})
    # Also hold out the legitimate CVE documents associated with the test query
    # groups; otherwise a test poison's genuine advisory might appear in train.
    expected_doc_groups = {
        'target:' + qid: 'nvd:' + benchmark[qid]['expected_doc_id']
        for qid in qids
    }
    require(len(set(expected_doc_groups.values())) == len(qids),
            'Expected CVE IDs are not unique across query groups')
    require(set(expected_doc_groups.values()).issubset(set(clean_groups)),
            'Expected legitimate CVE not present in clean corpus')
    forced_test = {expected_doc_groups[g] for g in target_test}
    forced_train = {expected_doc_groups[g] for g in target_train}
    ordinary_groups = sorted(set(clean_groups) - forced_train - forced_test)
    desired_clean_test = round(len(clean_groups) * 0.25)
    n_extra_test = desired_clean_test - len(forced_test)
    require(0 < n_extra_test < len(ordinary_groups),
            'Cannot achieve requested clean train/test split')
    ordinary_train, ordinary_test = train_test_split(
        ordinary_groups, test_size=n_extra_test, random_state=SEED)
    clean_train = sorted(forced_train | set(ordinary_train))
    clean_test = sorted(forced_test | set(ordinary_test))
    train_groups = set(target_train) | set(clean_train)
    test_groups = set(target_test) | set(clean_test)
    require(not train_groups.intersection(test_groups), 'Group split leak')
    frame['split'] = frame['group_id'].map(lambda g: 'train' if g in train_groups else 'test')
    require(not frame['split'].isna().any(), 'Rows lacking assigned split')
    split_counts = {s: dict(Counter(frame.loc[frame['split'] == s, LABEL_COLUMN]))
                    for s in ('train', 'test')}
    require(all(split_counts[s].get(0, 0) >= 1 and split_counts[s].get(1, 0) >= 1
                for s in ('train', 'test')), 'Both classes required in both splits')
    for attack in ATTACKS:
        attack_splits = frame.loc[frame['attack_type'] == attack, 'split'].value_counts()
        require(attack_splits.get('train', 0) and attack_splits.get('test', 0),
                f'{attack}: no train/test coverage')
    frame = frame[['chunk_id'] + FEATURE_COLUMNS + [LABEL_COLUMN, 'attack_type', 'group_id', 'split']]
    manifest = {
        'schema': 'phase_c_rf_xgb_grouped_split_v1',
        'purpose': 'Shared RF vs XGBoost evaluation on frozen 20%-ratio SM/OBEA/RPI poisons',
        'seed': SEED,
        'clean_collection': CLEAN,
        'poisoned_collections': [f'poisonbench_b_{a}_20pct_v1' for a in ATTACKS],
        'benchmark': str(BENCHMARK.relative_to(ROOT)).replace('\\', '/'),
        'benchmark_sha256': sha(BENCHMARK),
        'b5_csv_sha256': {a: sha(b5paths[a]) for a in ATTACKS},
        'dataset_csv': str(OUT_CSV.relative_to(ROOT)).replace('\\', '/'),
        'dataset_row_count': len(frame),
        'class_counts': {str(k): int(v) for k, v in Counter(frame[LABEL_COLUMN]).items()},
        'attack_counts': {str(k): int(v) for k, v in Counter(frame['attack_type']).items()},
        'primary_feature_columns': PRIMARY_FEATURES,
        'excluded_features': {
            'source_is_official': 'Deterministically reveals NVD vs controlled poison source/label',
            'source_age_days': 'Constant 0.0 across this frozen experiment',
        },
        'split_counts': split_counts,
        'train_target_groups': sorted(target_train),
        'test_target_groups': sorted(target_test),
        'train_clean_doc_groups': sorted(clean_train),
        'test_clean_doc_groups': sorted(clean_test),
        'feature_model': 'Defense 4 14-column existing feature_engineering.build_feature_matrix',
        'retrieval_feature_protocol': 'Clean chunk B5 CLEAN Top-5 appearances; poison chunks respective attack B5 20% Top-5 appearances, exact chunk IDs by read-only retriever and cosine scores',
        'evaluation_limitations': [
            '553-row combined dataset aggregates 3 individual 20%-ratio experiments; it is not itself a single live 20%-poisoned collection',
            'Feature engineering uses transductive centroid and nearest-neighbor statistics over all 553 rows; test features are not strictly inductive',
            'Frozen query-benchmark retrieval scores are transductive; test target-query groups are nevertheless disjoint from train target groups',
            'Synthetic controlled poison documents have distinguishing writing styles and source annotations; avoid deployment-level generalization claims',
            'This does not evaluate 1%, 5%, 10% ratio-specific defense generalization',
        ],
    }
    print('PASS: 553 feature rows (316 clean, 237 poison), 12 primary features')
    print('PASS: source label shortcut and constant age excluded from PRIMARY feature set')
    print('PASS: grouped ~75/25 split; held-out target-query CVE documents also held out')
    print('Split class counts:', split_counts)
    print('Held-out poison query groups:', sorted(target_test))
    return frame, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true',
                        help='Explicitly create new, immutable Phase C CSV and manifest')
    args = parser.parse_args()
    require(not OUT_CSV.exists() and not OUT_MANIFEST.exists(),
            'Phase C output exists; refuse to overwrite/rebuild')
    frame, manifest = build()
    if not args.build:
        print('PREFLIGHT PASS: READ ONLY; no files written, no training performed')
        print('Planned outputs:', OUT_CSV.relative_to(ROOT), OUT_MANIFEST.relative_to(ROOT))
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Exclusive-create; never truncate existing artifacts.
    with OUT_CSV.open('x', newline='', encoding='utf-8') as f:
        frame.to_csv(f, index=False)
    manifest['dataset_sha256'] = sha(OUT_CSV)
    with OUT_MANIFEST.open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print('PASS: Phase C feature dataset created:', OUT_CSV.relative_to(ROOT))
    print('PASS: Phase C grouped split manifest created:', OUT_MANIFEST.relative_to(ROOT))
    print('SAFETY: No ChromaDB writes or frozen Phase B result changes')


if __name__ == '__main__':
    main()
