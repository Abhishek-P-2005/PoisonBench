"""PoisonBench Phase E: isolated read-only LIVE retrieval + six-text-feature RF smoke test.

Install: scripts/phase_e_live_rf_retrieval_probe_v1.py
From PoisonBench_Shared repo root:
  python scripts/phase_e_live_rf_retrieval_probe_v1.py
  python scripts/phase_e_live_rf_retrieval_probe_v1.py --probe --attack sm --query-id q01

Default: offline file preflight, no model load, Docker, Ollama, or writes.
--probe: compute a NEW query embedding using MiniLM; query an EXISTING named Chroma
20%-poison collection; obtain actual Top-5 texts/metadata/embeddings; run Akilan's
real D1 advisory-only; score the actual retrieved text with the separately trained
6-feature RF; print results. Never modifies original API, collections or files.

Limitations: a smoke test, NOT full 4-defense integration or independent benchmark
accuracy. Same corpus is used for model training and this probe; D2/D3 skipped due
to missing canonical ingestion metadata; text-feature model performance under
live retrieval not yet established; no top-k refill or LLM generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defenses.defense4_xgboost.feature_engineering import text_stat_features

E = ROOT / 'data/results/phase_e'
C = ROOT / 'data/results/phase_c'
BENCH = ROOT / 'data/benchmark/real_trigger_queries_v1.jsonl'
RF = E / 'random_forest_live_text_v1.joblib'
REPORT = E / 'live_text_models_v1.json'
MANIFEST = C / 'split_manifest_v1.json'
CSV = C / 'combined_features_v1.csv'
COLLECTIONS = {
    'sm': 'poisonbench_b_sm_20pct_v1',
    'obea': 'poisonbench_b_obea_20pct_v1',
    'rpi': 'poisonbench_b_rpi_20pct_v1',
}
FEATURES = (
    'text_length_chars', 'text_length_words', 'avg_word_length',
    'char_entropy', 'digit_ratio', 'punctuation_ratio',
)
FORBIDDEN = {'is_poisoned', 'attack_type'}


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    require(all(p.is_file() for p in (REPORT, RF, MANIFEST, CSV, BENCH)),
            'Missing experiment artifact; do not retrain or overwrite')
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    require(report.get('schema') == 'phase_e_live_text_models_v1',
            'Unexpected deployment-model report schema')
    require(tuple(report['feature_columns']) == FEATURES and report['threshold'] == 0.5,
            'Text feature order or prediction threshold changed')
    require(sha(RF) == report['sha256']['rf_model'], 'RF SHA256 mismatch')
    require(sha(CSV) == report['dataset_sha256'], 'Dataset SHA256 mismatch')
    require(sha(MANIFEST) == report['split_manifest_sha256'], 'Split SHA256 mismatch')
    require(sha(BENCH) == manifest['benchmark_sha256'], 'Benchmark SHA256 mismatch')
    require(set(text_stat_features('CVE-2026-54500 test.')) == set(FEATURES),
            'Runtime six-feature extractor changed')
    benchmark = {}
    for ln in BENCH.read_text(encoding='utf-8').splitlines():
        if ln.strip():
            obj = json.loads(ln)
            benchmark[obj['query_id'].lower()] = obj
    require(len(benchmark) == 20, 'Benchmark must contain exactly 20 queries')
    print('PASS: 6-feature RF model SHA256, frozen corpus/split and benchmark hashes')
    print('PASS: Live text feature names/order and fixed 0.5 threshold')
    print('PASS: Named existing 20%-poison collections only; D1 advisory, D2/D3 skipped')
    return report, benchmark


def probe(attack, query_id, report, benchmark):
    require(query_id in benchmark, f'Unknown query ID: {query_id}')
    import chromadb
    import joblib
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from defenses.defense1_outlier_detection.filter import apply_outlier_filter

    query_text = benchmark[query_id]['query']
    expected_doc = benchmark[query_id]['expected_doc_id']
    collection_name = COLLECTIONS[attack]
    print(f'PROBE: {attack.upper()} / {query_id}; expected legitimate={expected_doc}')
    print('Embedding live query with MiniLM-L6-v2 (may load/download a model cache) ...')
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    q_embedding = np.asarray(embedder.encode([query_text]), dtype=float)
    require(q_embedding.shape == (1, 384) and np.isfinite(q_embedding).all(),
            'Invalid query embedding shape or nonfinite values')

    client = chromadb.HttpClient(host='localhost', port=8001)
    collection = client.get_collection(collection_name)  # existing only
    require(collection.count() == 395,
            f'Unexpected collection size: {collection_name}; abort rather than change it')
    results = collection.query(
        query_embeddings=q_embedding.tolist(), n_results=5,
        include=['documents', 'metadatas', 'embeddings', 'distances'],
    )
    require(len(results['ids']) == 1 and len(results['ids'][0]) == 5,
            'Expected one live query with five results')

    rows = []
    for cid, txt, meta, emb, dist in zip(
        results['ids'][0], results['documents'][0],
        results['metadatas'][0], results['embeddings'][0],
        results['distances'][0],
    ):
        require(isinstance(txt, str) and len(emb) == 384, f'Bad Chroma chunk: {cid}')
        require(isinstance(meta, dict) and isinstance(meta.get('doc_id'), str),
                f'Missing doc_id in stored metadata for {cid}')
        require(math.isfinite(float(dist)), f'Invalid Chroma distance for {cid}')
        safe_meta = {key: value for key, value in meta.items() if key not in FORBIDDEN}
        require(FORBIDDEN.isdisjoint(safe_meta), 'Ground-truth label leaked')
        rows.append({
            'chunk_id': str(cid), 'doc_id': str(safe_meta['doc_id']),
            'text': txt, 'embedding': [float(v) for v in emb],
            'metadata': safe_meta, 'source': str(safe_meta.get('source', '')),
            'similarity': 1.0 - float(dist),
            'similarity_score': 1.0 - float(dist),
            'distance': float(dist),
        })
    require(len({r['chunk_id'] for r in rows}) == 5, 'Duplicate Top-5 chunks')

    # D1's scores are informative but never remove the retrieved context here.
    # D1 assumes cosine-like similarity and embeddings; inspect the reported
    # distance-space information if actual baseline differs.
    from copy import deepcopy
    d1_passed, d1_blocked = apply_outlier_filter(deepcopy(rows))
    d1_block_ids = {r['chunk_id'] for r in d1_blocked}
    require({r['chunk_id'] for r in d1_passed} | d1_block_ids ==
            {r['chunk_id'] for r in rows}, 'D1 returned inconsistent chunk IDs')
    require(len(d1_passed) + len(d1_blocked) == 5,
            'D1 duplicate or missing result')

    model = joblib.load(RF)  # locally trained, hash-validated artifact only
    require(getattr(model, 'n_features_in_', None) == 6,
            'RF model feature count incompatible')
    matrix = pd.DataFrame(
        [{name: float(text_stat_features(r['text'])[name]) for name in FEATURES}
         for r in rows], columns=list(FEATURES),
    )
    require(np.isfinite(matrix.to_numpy(dtype=float)).all(),
            'Nonfinite six-feature live text matrix')
    proba = model.predict_proba(matrix)[:, 1]
    require(len(proba) == 5 and np.isfinite(proba).all(),
            'Invalid probability vector')
    blocked = {row['chunk_id'] for row, p in zip(rows, proba)
               if float(p) >= report['threshold']}
    kept = [r for r in rows if r['chunk_id'] not in blocked]
    print('\nLIVE Chroma query — D1 advisory + RF live text classifier:')
    for idx, (row, p) in enumerate(zip(rows, proba), 1):
        print(f"{idx}. doc={row['doc_id']} dist={row['distance']:.4f} "
              f"RF_poison_prob={p:.4f} D1_would_block={row['chunk_id'] in d1_block_ids} "
              f"RF_block={row['chunk_id'] in blocked}")
    print(f'RF blocked {len(blocked)}/5; retained {len(kept)}/5')
    print(f'Expected legitimate CVE retrieved: {any(r["doc_id"] == expected_doc for r in rows)}')
    print(f'Expected legitimate CVE retained: {any(r["doc_id"] == expected_doc for r in kept)}')
    print('D1 advisory only; D2/D3 skipped; no generation or top-k refill')
    print('READ ONLY: No writes, no modifications to original endpoints or collections')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true', help='Run live Chroma/ML smoke test')
    parser.add_argument('--attack', choices=tuple(COLLECTIONS), default='sm')
    parser.add_argument('--query-id', default='q01')
    args = parser.parse_args()
    report, benchmark = preflight()
    if not args.probe:
        print('PREFLIGHT PASS: READ ONLY; no Chroma/model load or outputs written')
        print('NEXT: python scripts/phase_e_live_rf_retrieval_probe_v1.py --probe --attack sm --query-id q01')
        return
    probe(args.attack, args.query_id.lower(), report, benchmark)


if __name__ == '__main__':
    main()
