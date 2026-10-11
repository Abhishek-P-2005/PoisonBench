"""Independent READ-ONLY audit of the first Phase B4 ChromaDB collection.

Run from PoisonBench_Shared repo root:
    python scripts/phase_b4_verify_sm_1pct_ingestion.py

Never creates/deletes/resets/upserts/adds collections, runs embeddings, or writes files.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET = 'poisonbench_b_sm_1pct_v1'
REF = 'poisonbench_rpi_clean_v1'
FROZEN = {
    'poisonbench_corpus': (336, 20, 'controlled_optimized_embedding'),
    'poisonbench_rpi_clean_v1': (316, 0, None),
    'poisonbench_rpi_poisoned_v1': (336, 20, 'controlled_retrieved_prompt_injection'),
}
SOURCE = 'controlled_semantic_mimicry'
DATASET = ROOT / 'data/poisoned/phase_b/phase_b_sm_1pct_v1.jsonl'
MANIFEST = ROOT / 'data/poisoned/phase_b/phase_b_1pct_5pct_allocation_v1.json'
BENCHMARK = ROOT / 'data/benchmark/real_trigger_queries_v1.jsonl'
EXPECTED_DIM = 384


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def rows(path):
    check(path.is_file(), f'Missing input: {path}')
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def fetch(collection):
    result = collection.get(include=['documents', 'metadatas', 'embeddings'])
    ids = result['ids']
    texts = result['documents']
    metas = result['metadatas']
    embeddings = result['embeddings']
    check(embeddings is not None and len(ids) == len(texts) == len(metas) == len(embeddings)
          == collection.count(), f'{collection.name}: inconsistent row counts')
    check(len(set(ids)) == len(ids), f'{collection.name}: repeated chunk IDs')
    out = {}
    for cid, text, metadata, embedding in zip(ids, texts, metas, embeddings):
        vec = list(embedding)
        check(isinstance(metadata, dict) and isinstance(text, str) and len(vec) == EXPECTED_DIM,
              f'{collection.name}: invalid record or embedding dimension {cid}')
        check(all(math.isfinite(float(x)) for x in vec), f'{collection.name}: non-finite vector {cid}')
        out[cid] = (text, metadata, vec)
    return out


def partition(items):
    return ({cid: (text, meta) for cid, (text, meta, _vec) in items.items()
             if meta.get('is_poisoned') is False},
            {cid: (text, meta) for cid, (text, meta, _vec) in items.items()
             if meta.get('is_poisoned') is True})


def main():
    print('=== Phase B4 — Independent SM 1% ingestion audit (READ ONLY) ===')
    from api.config import settings
    from api.pipeline.vectorstore import get_client
    from api.pipeline.chunking import chunk_text

    check(settings.chroma_collection == 'poisonbench_corpus', 'Unexpected default collection')
    check(settings.chunk_size_words == 180 and settings.chunk_overlap_words == 30,
          'Chunk configuration differs from frozen setup')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
    entry = next((item for item in manifest['files']
                  if item.get('attack') == 'SM' and item.get('target_ratio_pct') == 1), None)
    check(entry is not None, 'SM 1% manifest entry missing')
    check(hashlib.sha256(DATASET.read_bytes()).hexdigest() == entry['output_sha256'],
          'SM 1% dataset hash mismatch')
    records = rows(DATASET)
    check(len(records) == 3, 'Expected three SM poison records')
    lookup = {str(item['query_id']).strip().lower(): item['expected_doc_id'] for item in rows(BENCHMARK)}
    check(len(lookup) == 20, 'Expected 20 frozen benchmark queries')
    expected_poisons = {}
    for record in records:
        qid = record['query_id']
        doc_id = record['poison_doc_id']
        check(qid in lookup and record['legitimate_doc_id'] == lookup[qid],
              f'Incorrect benchmark mapping at {qid}')
        check(doc_id == f'POISON-SM-{qid}' and record['source'] == SOURCE
              and record['is_poisoned'] is True and record['attack_type'] == 'semantic_mimicry',
              f'Unexpected poison schema at {qid}')
        chunks = chunk_text(record['poison_text'], doc_id)
        check(len(chunks) == 1 and chunks[0]['position'] == 0,
              f'Non-single-chunk poison {doc_id}')
        cid = chunks[0]['chunk_id']
        check(cid not in expected_poisons, f'Duplicate poison ID {cid}')
        expected_poisons[cid] = (chunks[0]['text'], {
            'doc_id': doc_id, 'source': SOURCE, 'timestamp': '',
            'is_poisoned': True, 'position': 0,
        })
    print('PASS: Frozen benchmark, three poison documents, chunk mapping and dataset SHA-256')

    client = get_client()
    names = {item if isinstance(item, str) else item.name for item in client.list_collections()}
    check(TARGET in names and set(FROZEN).issubset(names), 'Pilot/frozen collection missing')
    expected_names = set(FROZEN) | {TARGET}
    check(names == expected_names,
          f'Unexpected collection state: expected {sorted(expected_names)}, got {sorted(names)}')
    print('PASS: Exactly 3 frozen + 1 isolated pilot collection exist; 11 other targets absent')

    frozen_records = {}
    for name, (expected_total, expected_pcount, expected_source) in FROZEN.items():
        c = client.get_collection(name=name)
        check(c.count() == expected_total, f'Frozen {name} count differs')
        contents = fetch(c)
        clean, poisons = partition(contents)
        check(len(clean) == 316 and len(poisons) == expected_pcount,
              f'Frozen {name}: clean/poison counts differ')
        check(all(m.get('source') == 'NVD' for _text, m in clean.values()),
              f'Frozen {name}: clean source differs')
        check(all(m.get('source') == expected_source for _text, m in poisons.values()),
              f'Frozen {name}: poison source differs')
        frozen_records[name] = contents
        print(f'PASS: Frozen {name}: 316 clean + {expected_pcount} poison, embeddings readable')
    ref_rows = frozen_records[REF]
    ref_clean, _ = partition(ref_rows)
    for name, contents in frozen_records.items():
        clean, _ = partition(contents)
        check(clean == ref_clean, f'Frozen {name}: NVD contents differ from reference')
    print('PASS: Frozen clean NVD chunk IDs, documents and metadata match across collections')

    target_coll = client.get_collection(name=TARGET)
    check(target_coll.count() == 319, 'Target expected 319 chunks')
    check(target_coll.metadata is not None and target_coll.metadata.get('hnsw:space') == 'cosine',
          'Target missing cosine-space collection metadata')
    target_rows = fetch(target_coll)
    target_clean, target_poison = partition(target_rows)
    check(len(target_clean) == 316 and len(target_poison) == 3, 'Wrong target clean/poison split')
    check(target_clean == ref_clean, 'Target clean chunk IDs/text/metadata differ from reference')
    check(target_poison == expected_poisons,
          'Target poison chunk IDs/text/metadata differ from frozen SM 1% dataset')
    check(Counter(meta.get('source') for _text, meta in target_poison.values()) == {SOURCE: 3},
          'Target poison source labels differ')
    print('PASS: Target 319 chunks: 316 exact clean + 3 exact SM poison; cosine configured')

    max_difference = 0.0
    for cid in ref_clean:
        before = ref_rows[cid][2]
        after = target_rows[cid][2]
        diff = max(abs(float(a)-float(b)) for a, b in zip(before, after))
        max_difference = max(max_difference, diff)
    check(max_difference <= 1e-6, f'Clean embeddings changed: max difference {max_difference}')
    for cid in expected_poisons:
        vector = target_rows[cid][2]
        norm = math.sqrt(sum(float(v)*float(v) for v in vector))
        check(norm > 0, f'Zero poison vector: {cid}')
    print(f'PASS: 316 clean embeddings preserved (max abs difference {max_difference:.3g})')
    print('PASS: Three poison embeddings exist, are 384-dimensional, finite and nonzero')
    print('PASS: PHASE B4 SM 1% INDEPENDENT VERIFICATION COMPLETE')
    print('SAFETY: READ ONLY — no Chroma writes, model inference, filesystem writes or resets')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
