"""PoisonBench Phase E isolated, READ-ONLY defense bridge.

Install at: scripts/phase_e_isolated_defense_bridge_v1.py
Run from the repository root:
    python scripts/phase_e_isolated_defense_bridge_v1.py
    python scripts/phase_e_isolated_defense_bridge_v1.py --probe --attack sm --query-id q01

Default mode: local-file preflight (NO Chroma/Ollama/model deserialization).
--probe mode: READ-ONLY get_collection/get, uses saved B5 Top-5 identities and
scores and authentic Chroma stored text/embeddings. Runs real Defense 1 and an
EXPLICITLY OFFLINE-REFERENCE Defense 4 classifier. D2 and D3 are SKIPPED when
canonical metadata is absent; neither fabricated provenance nor computed-after-
ingestion hashes are accepted. Does NOT generate LLM answers, refill top-k,
create/change/delete collections, or write any result file.

This is an engineering integration probe, NOT a fully live RAG or valid full-
defense effectiveness estimate. In particular the D4 feature provider is a
look-up of frozen, transductive Phase C features (some training-associated).
Do not report its metrics as new independent generalization evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.interfaces import RetrievedChunk  # noqa: E402
from defenses.defense4_random_forest.phase_e_defense4_adapter_v1 import (  # noqa: E402
    FrozenComparisonDefense, verify_artifacts,
)

PHASE_C = ROOT / 'data' / 'results' / 'phase_c'
PHASE_B = ROOT / 'data' / 'results' / 'phase_b'
MODEL_NAMES = ('random_forest', 'xgboost')
COLLECTIONS = {
    'sm': 'poisonbench_b_sm_20pct_v1',
    'obea': 'poisonbench_b_obea_20pct_v1',
    'rpi': 'poisonbench_b_rpi_20pct_v1',
}
# Canonical D2 and D3 metadata required for a meaningful score; do NOT
# derive verified/hash from a source label or from current stored text.
TRUST_METADATA = ('doc_id', 'verified', 'published_date', 'modified_date')
PROVENANCE_METADATA = (
    'chunk_id', 'doc_id', 'source_id', 'source_type',
    'verified', 'published_date', 'modified_date', 'chunk_hash',
)
FORBIDDEN_GROUND_TRUTH = ('is_poisoned', 'attack_type')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_from_chunk(chunk_id: str) -> str:
    require('::chunk' in chunk_id, f'Unrecognized chunk id: {chunk_id!r}')
    return chunk_id.rsplit('::chunk', 1)[0]


def load_preflight(attack: str, query_id: str):
    features, comparison = verify_artifacts()  # file hashes, no model load
    manifest = json.loads((PHASE_C / 'split_manifest_v1.json').read_text(encoding='utf-8'))
    benchmark_path = ROOT / 'data' / 'benchmark' / 'real_trigger_queries_v1.jsonl'
    require(sha256(benchmark_path) == manifest['benchmark_sha256'],
            'Frozen benchmark SHA mismatch')
    with benchmark_path.open(encoding='utf-8') as handle:
        benchmarks = {r['query_id'].lower(): r
                      for line in handle if line.strip()
                      for r in [json.loads(line)]}
    require(len(benchmarks) == 20 and query_id in benchmarks,
            f'Unknown benchmark query ID: {query_id}')

    b5_path = PHASE_B / f'phase_b5_{attack}_20pct_top5_v1_results.csv'
    require(b5_path.is_file() and sha256(b5_path) == manifest['b5_csv_sha256'][attack],
            'Frozen B5 SHA mismatch')
    rows = pd.read_csv(b5_path)
    require(len(rows) == 20 and rows['query_id'].is_unique, 'Invalid B5 20-query file')
    selection = rows.loc[rows['query_id'].eq(query_id)]
    require(len(selection) == 1, 'Exactly one requested B5 query row required')
    selected = selection.iloc[0]
    require(selected['query'] == benchmarks[query_id]['query'], 'Benchmark query text mismatch')
    require(selected['legitimate_doc_id'] == benchmarks[query_id]['expected_doc_id'],
            'Expected legitimate doc ID mismatch')
    doc_ids = [s.strip() for s in selected['poisoned_retrieved_doc_ids'].split('|')]
    similarities = [float(s.strip()) for s in selected['poisoned_similarity_scores'].split('|')]
    require(len(doc_ids) == len(similarities) == 5 and len(set(doc_ids)) == 5,
            'Invalid saved Top-5 document IDs/scores')
    require(all(math.isfinite(v) for v in similarities), 'Invalid B5 scores')
    df = pd.read_csv(PHASE_C / 'combined_features_v1.csv')
    require(len(df) == 553 and df['chunk_id'].is_unique, 'Invalid Phase C feature rows')
    lookup = {}
    for row in df.to_dict(orient='records'):
        lookup.setdefault(doc_from_chunk(str(row['chunk_id'])), []).append(row)
    matched = []
    for doc_id in doc_ids:
        matches = lookup.get(doc_id, [])
        require(len(matches) == 1,
                f'B5 document must map to one chunk: {doc_id} -> {len(matches)}')
        matched.append(matches[0])
    require(len(set(row['chunk_id'] for row in matched)) == 5, 'Duplicate chunk IDs')
    return features, comparison, selected, doc_ids, similarities, matched


def metadata_gaps(rows: list[dict], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: sum(not isinstance(r['metadata'].get(key), (str, bool, int, float))
                     or r['metadata'].get(key) in ('', None) for r in rows)
            for key in keys
            if any(not isinstance(r['metadata'].get(key), (str, bool, int, float))
                   or r['metadata'].get(key) in ('', None) for r in rows)}


def fetch_frozen_top5(attack: str, doc_ids: list[str], similarities: list[float],
                      feature_rows: list[dict]) -> list[dict]:
    # Imported only in --probe so default preflight needs no running Docker.
    import chromadb

    client = chromadb.HttpClient(host='localhost', port=8001)
    collection = client.get_collection(COLLECTIONS[attack])  # NEVER get_or_create
    require(collection.count() == 395, 'Collection count differs from frozen 395 rows')
    ids = [str(r['chunk_id']) for r in feature_rows]
    found = collection.get(ids=ids, include=['documents', 'metadatas', 'embeddings'])
    by_id = {}
    for idx, chunk_id in enumerate(found['ids']):
        emb = found['embeddings'][idx]
        # Avoid numpy truthiness shortcuts for Chroma arrays.
        emb_values = list(emb) if emb is not None else []
        require(len(emb_values) == 384, f'Embedding missing or invalid: {chunk_id}')
        meta = dict(found['metadatas'][idx] or {})
        text = found['documents'][idx]
        require(isinstance(text, str), f'Invalid stored text: {chunk_id}')
        require(meta.get('doc_id') == doc_from_chunk(chunk_id),
                f'Metadata doc ID mismatch: {chunk_id}')
        safe_meta = {key: val for key, val in meta.items()
                     if key not in FORBIDDEN_GROUND_TRUTH}
        by_id[chunk_id] = {
            'chunk_id': chunk_id,
            'text': text,
            'metadata': safe_meta,
            'embedding': [float(x) for x in emb_values],
        }
    require(set(by_id) == set(ids), 'Chroma returned missing/extra chunk IDs')
    rows = []
    for doc_id, sim, c in zip(doc_ids, similarities, ids):
        record = dict(by_id[c])
        record['similarity'] = float(sim)
        record['similarity_score'] = float(sim)
        record['doc_id'] = doc_id
        record['source'] = str(record['metadata'].get('source', ''))
        rows.append(record)
    return rows


def probe(attack: str, query_id: str, model_name: str) -> None:
    features, _, selected, doc_ids, similarities, matched = load_preflight(attack, query_id)
    print(f'PROBE: attack={attack}, query={query_id}, model={model_name}')
    print('MODE: frozen B5 Top-5 + read-only Chroma metadata/embeddings, NOT a live query')
    rows = fetch_frozen_top5(attack, doc_ids, similarities, matched)
    print('PASS: Fetched exact 5 frozen retrieved chunks from named Chroma collection (read only)')
    print('Ground-truth metadata stripped before defense scoring')

    from defenses.defense1_outlier_detection.filter import apply_outlier_filter
    from defenses.defense2_trust_rerank.reranker import rerank_results
    from defenses.defense3_provenance.filter import apply_provenance_filter

    t0 = time.perf_counter()
    survivors, d1_blocked = apply_outlier_filter(rows)
    d1_ms = (time.perf_counter() - t0) * 1000
    require(len(survivors) + len(d1_blocked) == 5, 'D1 lost or duplicated chunks')
    print(f'D1 REAL: {len(d1_blocked)} blocked, {len(survivors)} passed, elapsed_ms={d1_ms:.2f}')

    d2_missing = metadata_gaps(survivors, TRUST_METADATA)
    if d2_missing:
        print('D2 SKIPPED: canonical verified/date fields absent; '
              f'missing_counts={d2_missing}')
    else:
        t0 = time.perf_counter()
        survivors = rerank_results(survivors)
        print(f'D2 REAL: {len(survivors)} re-ranked, '
              f'low_trust_flags={sum(bool(r["is_low_trust"]) for r in survivors)}, '
              f'elapsed_ms={(time.perf_counter()-t0)*1000:.2f}')

    d3_missing = metadata_gaps(survivors, PROVENANCE_METADATA)
    if d3_missing:
        print('D3 SKIPPED: provenance cannot be authenticated without original '
              f'ingestion metadata; missing_counts={d3_missing}')
    else:
        t0 = time.perf_counter()
        survivors, d3_blocked = apply_provenance_filter(survivors)
        require(len(d3_blocked) + len(survivors) <= 5, 'Invalid D3 output')
        print(f'D3 REAL: {len(d3_blocked)} blocked, {len(survivors)} passed, '
              f'elapsed_ms={(time.perf_counter()-t0)*1000:.2f}')

    # This provider is an audit/reference lookup of transductive Phase C
    # features, NOT a deployable feature-extraction algorithm. Its identity
    # mapping must not be misreported as an unseen/live classifier test.
    frozen_lookup = {
        str(record['chunk_id']): {
            'chunk_id': str(record['chunk_id']),
            **{f: float(record[f]) for f in features}
        } for record in matched
    }

    def frozen_reference_provider(chunks):
        return [frozen_lookup[c.chunk_id] for c in chunks]

    retrieved = [RetrievedChunk(
        chunk_id=r['chunk_id'], doc_id=r['doc_id'], text=r['text'],
        similarity_score=r['similarity_score'], source=r['source'],
        is_poisoned=False, metadata=r['metadata']) for r in survivors]
    d4 = FrozenComparisonDefense(model_name, frozen_reference_provider)
    verdicts = d4.score(retrieved)
    require([v.chunk_id for v in verdicts] == [r['chunk_id'] for r in survivors],
            'D4 verdict identity/order mismatch')
    retained = [r for r, verdict in zip(survivors, verdicts) if not verdict.flagged]
    print(f'D4 FROZEN-REFERENCE ONLY: {len(verdicts)-len(retained)} blocked, '
          f'{len(retained)} retained; includes source/preparation/model overhead')
    print('D4 SPOOF-SAFE INPUT: no is_poisoned or attack_type exposed to classifier')
    print('ATTENTION: D2/D3 missing metadata means FULL four-defense chain did NOT run')
    print('ATTENTION: D4 uses benchmark-wide/transductive saved features, NOT live feature generation')
    print('Final retained document IDs:', [r['doc_id'] for r in retained])
    print('OUTPUT: READ ONLY — no files, Chroma writes, model training or Ollama calls')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true',
                        help='Run read-only isolated single-query integration probe')
    parser.add_argument('--attack', choices=tuple(COLLECTIONS), default='sm')
    parser.add_argument('--query-id', choices=tuple(f'q{i:02}' for i in range(1, 21)), default='q01')
    parser.add_argument('--model', choices=MODEL_NAMES, default='random_forest')
    args = parser.parse_args()
    features, _, _, doc_ids, _, matched = load_preflight(args.attack, args.query_id)
    print(f'PASS: Frozen Phase B/C/D integrity and 12-feature order ({len(features)} features)')
    print(f'PASS: {args.attack.upper()} {args.query_id} Top-5 mapped to 5 unique chunk IDs')
    print('PASS: Chroma collection selected explicitly; never creates or mutates collections')
    if args.probe:
        probe(args.attack, args.query_id, args.model)
    else:
        print('PREFLIGHT PASS: READ ONLY — no Chroma connection/model deserialization/file writes')
        print('NEXT PROBE: python scripts/phase_e_isolated_defense_bridge_v1.py --probe --attack sm --query-id q01')
        print('LIMITATIONS: D2/D3 require ingestion metadata; D4 frozen-lookup is NOT live inference')


if __name__ == '__main__':
    main()
