"""Independent READ-ONLY audit for PoisonBench Phase B4 ChromaDB collections.

From the PoisonBench_Shared repository root:
    python scripts/phase_b4_verify_isolated_collections.py --attack OBEA --ratio 1

Verifies every existing Phase B collection (including the selected target) against
its frozen local JSONL, independent hash manifest, original clean NVD records and
stored embedding vectors. It NEVER creates/updates/deletes/resets any collection,
runs model inference, or writes files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN = {
    'poisonbench_corpus': (336, 20, 'controlled_optimized_embedding'),
    'poisonbench_rpi_clean_v1': (316, 0, None),
    'poisonbench_rpi_poisoned_v1': (336, 20, 'controlled_retrieved_prompt_injection'),
}
REFERENCE = 'poisonbench_rpi_clean_v1'
SOURCES = {'SM': 'controlled_semantic_mimicry',
           'OBEA': 'controlled_optimized_embedding',
           'RPI': 'controlled_retrieved_prompt_injection'}
ATTACK_TYPES = {'SM': 'semantic_mimicry',
                'OBEA': 'optimization_embedding',
                'RPI': 'retrieved_prompt_injection'}
RATIOS = {1: 3, 5: 17, 10: 35, 20: 79}
DIM = 384
CLEAN_COUNT = 316
MAX_VECTOR_DIFF = 1e-6
B_DIR = ROOT / 'data/poisoned/phase_b'
BENCHMARK = ROOT / 'data/benchmark/real_trigger_queries_v1.jsonl'


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    need(path.is_file(), f'Missing manifest {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    need(path.is_file(), f'Missing JSONL {path}')
    items = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
             if line.strip()]
    need(all(isinstance(item, dict) for item in items), f'Non-object JSONL record {path}')
    return items


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collection_name(attack, ratio):
    return f'poisonbench_b_{attack.lower()}_{ratio}pct_v1'


def expected_records(attack, ratio, lookup, chunk_text):
    dataset = B_DIR / f'phase_b_{attack.lower()}_{ratio}pct_v1.jsonl'
    if ratio in (1, 5):
        manifest = read_json(B_DIR / 'phase_b_1pct_5pct_allocation_v1.json')
        entries = [e for e in manifest['files']
                   if e.get('attack') == attack and e.get('target_ratio_pct') == ratio]
        need(len(entries) == 1, f'{attack} {ratio}%: missing/ambiguous manifest entry')
        expected_digest = entries[0]['output_sha256']
    else:
        manifest = read_json(B_DIR / f'phase_b_{ratio}pct_allocation_v1.json')
        expected_digest = manifest['datasets'][attack]['sha256']
    need(sha256(dataset) == expected_digest, f'{attack} {ratio}%: SHA-256 mismatch')
    data = read_jsonl(dataset)
    need(len(data) == RATIOS[ratio], f'{attack} {ratio}%: incorrect poison count')
    output = {}
    texts = set()
    per_query = Counter()
    for record in data:
        qid = str(record.get('query_id', '')).strip().lower()
        did = record.get('poison_doc_id')
        need(qid in lookup and record.get('legitimate_doc_id') == lookup[qid],
             f'{attack} {ratio}%: wrong query/CVE mapping {qid}')
        need(isinstance(did, str) and
             (did == f'POISON-{attack}-{qid}' or did.startswith(f'POISON-{attack}-{qid}-v')),
             f'{attack} {ratio}%: unexpected poison ID {did}')
        need(record.get('source') == SOURCES[attack] and
             record.get('attack_type') == ATTACK_TYPES[attack] and
             record.get('is_poisoned') is True,
             f'{attack} {ratio}%: unexpected poison record metadata {did}')
        text = record.get('poison_text')
        need(isinstance(text, str) and text.strip(), f'Empty poison text {did}')
        if attack == 'RPI':
            need('POISONBENCH_RPI_SUCCESS' in text, f'Missing RPI marker {did}')
        chunks = chunk_text(text, did)
        need(len(chunks) == 1 and chunks[0]['position'] == 0,
             f'{attack} {ratio}%: multi-chunk poison {did}')
        chunk = chunks[0]
        cid = chunk['chunk_id']
        need(cid == f'{did}::chunk0' and cid not in output,
             f'{attack} {ratio}%: duplicate/unexpected chunk ID {cid}')
        need(chunk['text'] not in texts, f'{attack} {ratio}%: duplicate poison text {did}')
        output[cid] = (chunk['text'], {
            'doc_id': did, 'source': SOURCES[attack], 'timestamp': '',
            'is_poisoned': True, 'position': 0,
        })
        texts.add(chunk['text'])
        per_query[qid] += 1
    expected_targets = {1: 3, 5: 17, 10: 20, 20: 20}[ratio]
    need(len(per_query) == expected_targets, f'{attack} {ratio}%: target coverage mismatch')
    return output


def get_all(collection):
    data = collection.get(include=['documents', 'metadatas', 'embeddings'])
    ids, docs, metas, embeddings = (data.get('ids'), data.get('documents'),
                                   data.get('metadatas'), data.get('embeddings'))
    count = collection.count()
    need(all(v is not None for v in (ids, docs, metas, embeddings)),
         f'{collection.name}: missing response fields')
    need(len(ids) == len(docs) == len(metas) == len(embeddings) == count,
         f'{collection.name}: inconsistent collection counts')
    need(len(set(ids)) == len(ids), f'{collection.name}: repeated chunk ID')
    output = {}
    for cid, doc, meta, vector in zip(ids, docs, metas, embeddings):
        need(isinstance(doc, str) and doc and isinstance(meta, dict),
             f'{collection.name}: invalid document/metadata {cid}')
        need(cid == f"{meta.get('doc_id')}::chunk{meta.get('position')}",
             f'{collection.name}: chunk ID does not match metadata {cid}')
        vec = [float(v) for v in vector]
        need(len(vec) == DIM and all(math.isfinite(v) for v in vec) and
             any(v != 0 for v in vec),
             f'{collection.name}: invalid stored vector {cid}')
        output[cid] = (doc, meta, vec)
    return output


def separate(rows):
    clean = {cid: (text, meta) for cid, (text, meta, _vec) in rows.items()
             if meta.get('is_poisoned') is False}
    poisons = {cid: (text, meta) for cid, (text, meta, _vec) in rows.items()
               if meta.get('is_poisoned') is True}
    need(len(clean) + len(poisons) == len(rows), 'Invalid poison flag in stored collection')
    return clean, poisons


def verify_clean_vectors(name, reference, target, clean_ids):
    largest = 0.0
    for cid in clean_ids:
        left, right = reference[cid][2], target[cid][2]
        largest = max(largest, max(abs(a - b) for a, b in zip(left, right)))
    need(largest <= MAX_VECTOR_DIFF,
         f'{name}: clean embeddings differ from reference (max={largest})')
    return largest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attack', choices=['SM', 'OBEA', 'RPI'], default='OBEA')
    parser.add_argument('--ratio', type=int, choices=[1, 5, 10, 20], default=1)
    args = parser.parse_args()
    selected = collection_name(args.attack, args.ratio)
    print('=== Phase B4 — Independent experimental collection audit (READ ONLY) ===')
    print(f'Selected: {args.attack} {args.ratio}% — {selected}')

    from api.config import settings
    from api.pipeline.chunking import chunk_text
    from api.pipeline.vectorstore import get_client

    need(settings.chroma_collection == 'poisonbench_corpus',
         'Configured default collection has changed')
    need(settings.chunk_size_words == 180 and settings.chunk_overlap_words == 30,
         'Frozen chunking configuration has changed')
    benchmark = read_jsonl(BENCHMARK)
    lookup = {str(item['query_id']).strip().lower(): item['expected_doc_id']
              for item in benchmark}
    need(len(benchmark) == len(lookup) == 20, 'Frozen benchmark mapping is ambiguous')
    print('PASS: Frozen 20-query benchmark and chunking configuration')

    client = get_client()
    names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    reserved = {collection_name(a, r) for a in SOURCES for r in RATIOS}
    need(set(FROZEN).issubset(names), 'One or more frozen collections are missing')
    need(selected in names, f'{selected}: not created yet; do not run verification as ingestion')
    need(collection_name('SM', 1) in names, 'Verified SM 1% pilot collection missing')
    need(not names - set(FROZEN) - reserved,
         f'Unexpected collection names: {sorted(names - set(FROZEN) - reserved)}')
    print(f'PASS: Three frozen collections plus {len(names & reserved)} existing Phase B collections')

    frozen = {}
    for name, (total, expected_poisons_count, poison_source) in FROZEN.items():
        contents = get_all(client.get_collection(name=name))
        clean, poisons = separate(contents)
        need(len(contents) == total and len(clean) == CLEAN_COUNT and
             len(poisons) == expected_poisons_count,
             f'Frozen {name}: count mismatch')
        need(all(meta.get('source') == 'NVD' for _t, meta in clean.values()),
             f'Frozen {name}: clean NVD source metadata differs')
        need(len({meta['doc_id'] for _t, meta in clean.values()}) == 300,
             f'Frozen {name}: clean CVE document coverage differs')
        need(all(meta.get('source') == poison_source for _t, meta in poisons.values()),
             f'Frozen {name}: poison source label differs')
        frozen[name] = contents
        print(f'PASS: Frozen {name}: 316 clean + {expected_poisons_count} poison')
    ref = frozen[REFERENCE]
    ref_clean, _ = separate(ref)
    for name, contents in frozen.items():
        clean, _ = separate(contents)
        need(clean == ref_clean, f'Frozen {name}: NVD IDs/text/metadata differ')
    print('PASS: Frozen clean IDs/text/metadata agree across all three collections')

    phase_b_existing = []
    for ratio in RATIOS:
        for attack in SOURCES:
            name = collection_name(attack, ratio)
            if name not in names:
                continue
            contents = get_all(client.get_collection(name=name))
            need(client.get_collection(name=name).metadata is not None and
                 client.get_collection(name=name).metadata.get('hnsw:space') == 'cosine',
                 f'{name}: cosine metric missing')
            clean, poisons = separate(contents)
            expected = expected_records(attack, ratio, lookup, chunk_text)
            need(len(clean) == CLEAN_COUNT and len(poisons) == RATIOS[ratio] and
                 len(contents) == CLEAN_COUNT + RATIOS[ratio],
                 f'{name}: unexpected clean/poison split')
            need(clean == ref_clean, f'{name}: stored clean text/metadata differs')
            need(poisons == expected, f'{name}: stored poison IDs/text/metadata differ')
            difference = verify_clean_vectors(name, ref, contents, ref_clean)
            print(f'PASS: {attack:4} {ratio:2}% — {len(contents)} exact chunks, cosine, '
                  f'poison dataset SHA-256 OK, clean vector max diff {difference:.3g}')
            phase_b_existing.append(name)
    need(selected in phase_b_existing, 'Selected collection missing from audit')
    print(f'PASS: Independently verified all {len(phase_b_existing)} existing Phase B collections')
    print('PASS: PHASE B4 INDEPENDENT VERIFICATION COMPLETE')
    print('SAFETY: READ ONLY — no DB writes, embedding inference, Ollama, or filesystem writes')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
