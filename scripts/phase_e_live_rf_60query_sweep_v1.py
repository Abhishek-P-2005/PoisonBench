"""Read-only LIVE Chroma Top-5 sweep, 20 benchmark queries x 3 attack collections.

Copy to scripts/phase_e_live_rf_60query_sweep_v1.py in PoisonBench_Shared.
From repo root:
  python scripts/phase_e_live_rf_60query_sweep_v1.py          # offline preflight
  python scripts/phase_e_live_rf_60query_sweep_v1.py --sweep  # read-only Chroma

D1 advisory (no blocking); 6-feature RF 0.5 hard filter. Label/CVE identifiers
are used ONLY after classification for diagnostic metrics. No training, output
files, writes, deletes, or LLM calls. Dataset shares documents with training;
query-heldout breakdown is diagnostic, not unseen-corpus generalization.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_e_live_rf_retrieval_probe_v1 import (
    preflight as original_preflight, FEATURES, RF, COLLECTIONS, FORBIDDEN,
    MANIFEST, require,
)
from defenses.defense4_xgboost.feature_engineering import text_stat_features


def preflight():
    report, benchmark = original_preflight()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    heldout = set(manifest['test_target_groups'])
    require(len(heldout) == 5 and all(g.startswith('target:q') for g in heldout),
            'Unexpected five held-out query targets')
    require(set(benchmark) == {f'q{i:02}' for i in range(1, 21)},
            'Benchmark target IDs differ')
    print('PASS: 60 live-query run plan; 5 held-out query targets')
    print('PASS: No labels used for model inputs or D1 advisory decisions')
    return report, benchmark, heldout


def sweep(report, benchmark, heldout):
    import chromadb
    import joblib
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from defenses.defense1_outlier_detection.filter import apply_outlier_filter

    query_ids = sorted(benchmark)
    queries = [benchmark[q]['query'] for q in query_ids]
    print('Loading MiniLM once, and embedding 20 benchmark queries...')
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embeddings = np.asarray(embedder.encode(queries), dtype=float)
    require(query_embeddings.shape == (20, 384) and np.isfinite(query_embeddings).all(),
            'Unexpected MiniLM query embedding dimensions')

    model = joblib.load(RF)  # locally trained, hash-checked by preflight
    require(getattr(model, 'n_features_in_', None) == 6,
            'Six-feature RF model contract mismatch')
    client = chromadb.HttpClient(host='localhost', port=8001)
    grand = Counter()
    for attack, name in COLLECTIONS.items():
        collection = client.get_collection(name)  # no get_or_create
        require(collection.count() == 395, f'Collection changed: {name}')
        results = collection.query(
            query_embeddings=query_embeddings.tolist(), n_results=5,
            include=['documents', 'metadatas', 'embeddings', 'distances'],
        )
        require(len(results['ids']) == 20, f'{attack}: Not 20 query batches')
        totals = Counter()
        held = Counter()
        failures = []
        print(f'\n=== LIVE {attack.upper()} Top-5 — D1 advisory + six-feature RF ===')
        for qi, qid in enumerate(query_ids):
            data_ids = results['ids'][qi]
            texts = results['documents'][qi]
            metas = results['metadatas'][qi]
            embeddings = results['embeddings'][qi]
            dists = results['distances'][qi]
            require(len(data_ids) == len(texts) == len(metas) == len(embeddings) == len(dists) == 5,
                    f'{attack}/{qid}: missing Top-5 entries')
            require(len(set(data_ids)) == 5, f'{attack}/{qid}: duplicate chunks')
            rows = []
            for cid, text, raw_meta, emb, dist in zip(data_ids, texts, metas, embeddings, dists):
                require(isinstance(raw_meta, dict) and isinstance(text, str)
                        and len(emb) == 384 and math.isfinite(float(dist)),
                        f'{attack}/{qid}: malformed Chroma return')
                metadata = {k: v for k, v in raw_meta.items() if k not in FORBIDDEN}
                require(FORBIDDEN.isdisjoint(metadata)
                        and isinstance(metadata.get('doc_id'), str),
                        'Missing doc_id or leaked labels')
                rows.append({
                    'chunk_id': str(cid), 'doc_id': metadata['doc_id'],
                    'text': text, 'metadata': metadata,
                    'embedding': [float(v) for v in emb],
                    'source': str(metadata.get('source', '')),
                    'similarity': 1.0 - float(dist),
                    'similarity_score': 1.0 - float(dist),
                })
            d1_pass, d1_block = apply_outlier_filter(deepcopy(rows))
            require(len(d1_pass) + len(d1_block) == 5
                    and {r['chunk_id'] for r in d1_pass + d1_block} == set(data_ids),
                    f'{attack}/{qid}: D1 inconsistent candidate coverage')
            d1_block_ids = {r['chunk_id'] for r in d1_block}
            matrix = pd.DataFrame(
                [{f: float(text_stat_features(r['text'])[f]) for f in FEATURES}
                 for r in rows], columns=list(FEATURES),
            )
            require(np.isfinite(matrix.to_numpy(dtype=float)).all(),
                    f'{attack}/{qid}: invalid features')
            prob = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
            require(prob.shape == (5,) and np.isfinite(prob).all()
                    and np.all((prob >= 0) & (prob <= 1)),
                    f'{attack}/{qid}: invalid RF probabilities')
            dropped = {r['chunk_id'] for r, p in zip(rows, prob)
                       if float(p) >= report['threshold']}
            kept = [r for r in rows if r['chunk_id'] not in dropped]
            expected = benchmark[qid]['expected_doc_id']
            was_present = any(r['doc_id'] == expected for r in rows)
            is_present = any(r['doc_id'] == expected for r in kept)
            poison_before = sum(r['doc_id'].startswith('POISON-') for r in rows)
            poison_after = sum(r['doc_id'].startswith('POISON-') for r in kept)
            categories = [totals]
            if f'target:{qid}' in heldout:
                categories.append(held)
            for c in categories:
                c['runs'] += 1
                c['poison_before_positions'] += poison_before
                c['poison_after_positions'] += poison_after
                c['poison_blocked_positions'] += poison_before - poison_after
                c['poison_queries_after'] += int(poison_after > 0)
                c['legit_initially_present'] += int(was_present)
                c['legit_retained'] += int(is_present)
                c['empty_context_queries'] += int(not kept)
                c['d1_poison_would_block'] += sum(
                    r['chunk_id'] in d1_block_ids and r['doc_id'].startswith('POISON-')
                    for r in rows
                )
                c['d1_clean_would_block'] += sum(
                    r['chunk_id'] in d1_block_ids and not r['doc_id'].startswith('POISON-')
                    for r in rows
                )
            if poison_after or (was_present and not is_present) or not kept:
                failures.append((qid, poison_after, was_present, is_present, len(kept)))
        for label, c in [('ALL 20 QUERY RUNS', totals), ('FIVE HELD-OUT TARGET QUERY RUNS', held)]:
            print(f'{label}: runs={c["runs"]}, '
                  f'poison blocked={c["poison_blocked_positions"]}/{c["poison_before_positions"]} positions, '
                  f'poison still present in={c["poison_queries_after"]}/{c["runs"]} queries, '
                  f'legit retained={c["legit_retained"]}/{c["legit_initially_present"]} '
                  f'when originally present, empty={c["empty_context_queries"]}')
            print(f'  D1 advisory flags: poison={c["d1_poison_would_block"]}, '
                  f'clean={c["d1_clean_would_block"]} (not filtered)')
        print('Runs needing investigation (query, poison_after, legit_was_present, '
              'legit_retained, context_size):')
        print(' ', failures)
        grand.update(totals)
    print('\n=== TOTAL ACROSS 60 LIVE QUERY RUNS ===')
    print(f'RF poison blocked: {grand["poison_blocked_positions"]}/'
          f'{grand["poison_before_positions"]} retrieved poison positions')
    print(f'Queries containing poison after RF: {grand["poison_queries_after"]}/60')
    print(f'Expected legitimate CVEs retained: {grand["legit_retained"]}/'
          f'{grand["legit_initially_present"]} initially retrieved')
    print(f'Empty context query runs: {grand["empty_context_queries"]}/60')
    print('LIMITS: Same existing corpus as training; some train-associated chunks; '
          'five held-out target-query groups, not unseen-corpus evaluation.')
    print('D1 advisory, D2/D3 skipped; RF only; no refill, LLM generation or API changes.')
    print('READ ONLY: No database writes, training, model changes, output files or Ollama.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep', action='store_true', help='Run fresh retrieval of 60 query/attack pairs')
    args = p.parse_args()
    report, bench, heldout = preflight()
    if args.sweep:
        sweep(report, bench, heldout)
    else:
        print('PREFLIGHT PASS: No Chroma, embeddings, model load, or output writes.')
        print('NEXT: python scripts/phase_e_live_rf_60query_sweep_v1.py --sweep')


if __name__ == '__main__':
    main()
