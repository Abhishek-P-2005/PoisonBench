"""Isolated, read-only PoisonBench Phase E defended retrieval API (NOT original API).

Install in PoisonBench_Shared/scripts/phase_e_defended_api_v1.py.
From repository root:
  python scripts/phase_e_defended_api_v1.py  # offline preflight, no Docker/model load
  python -m uvicorn scripts.phase_e_defended_api_v1:app --host 127.0.0.1 --port 8002

Then POST JSON to http://127.0.0.1:8002/api/v1/defended/retrieve
  {"query":"What is CVE-2026-54500?","attack":"sm"}

Isolated endpoint: MiniLM -> named, EXISTING 20%-poison Chroma Top-5 ->
Akilan D1 ADVISORY (record flags; no blocks) -> 6 text stats -> separate
Phase E RF hard filter (0.5). D2/D3 intentionally not activated, because
frozen collections lack authenticated provenance/trust metadata. Does NOT
invoke an LLM or modify original baseline endpoints. No retrieval refill.

Limitations: benchmark-associated training corpus, synthetic attacks; 5/60
benchmark query/attack runs retained poison under live sweep. Not full D1-D4
pipeline, not a substitute for defense-on/off generation experiments.
"""
from __future__ import annotations

import hashlib
import math
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_e_live_rf_retrieval_probe_v1 import (
    COLLECTIONS, FEATURES, FORBIDDEN, RF, REPORT, preflight as verify_frozen_artifacts,
    require,
)
from defenses.defense4_xgboost.feature_engineering import text_stat_features

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

API_NAME = 'PoisonBench Phase E — Experimental Defended Retrieval (isolated)'
app = FastAPI(title=API_NAME, version='1.0.0', docs_url='/docs', redoc_url=None)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=3000)
    attack: Literal['sm', 'obea', 'rpi'] = 'sm'


@lru_cache(maxsize=1)
def runtime():
    """Verify files before loading trusted locally-trained model and encoder."""
    import chromadb
    import joblib
    from sentence_transformers import SentenceTransformer

    report, _ = verify_frozen_artifacts()
    require(hashlib.sha256(RF.read_bytes()).hexdigest() == report['sha256']['rf_model'],
            'RF model checksum failed immediately before deserialization')
    model = joblib.load(RF)  # local hash-verified model only
    require(getattr(model, 'n_features_in_', None) == len(FEATURES),
            'RF runtime feature contract mismatch')
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    client = chromadb.HttpClient(host='localhost', port=8001)
    return report, model, embedder, client


@app.get('/api/v1/defended/health')
def health():
    """Lightweight process health, NOT a declaration of Chroma readiness."""
    return {
        'status': 'process_alive',
        'mode': 'experimental_isolated',
        'retrieval': 'existing_20pct_poison_collections_only',
        'd1': 'advisory_only',
        'd2': 'unavailable_missing_verified_metadata',
        'd3': 'unavailable_missing_provenance_metadata',
        'd4': 'live_six_text_feature_random_forest',
        'llm': 'not_connected',
    }


@app.post('/api/v1/defended/retrieve')
def defended_retrieve(req: RetrieveRequest):
    """Read-only Chroma retrieval + advisory D1 + 6-feature RF filtering."""
    import numpy as np
    import pandas as pd
    from defenses.defense1_outlier_detection.filter import apply_outlier_filter

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail='Query cannot be blank')
    report, model, embedder, client = runtime()
    collection_name = COLLECTIONS[req.attack]
    collection = client.get_collection(collection_name)  # never get_or_create
    require(collection.count() == 395,
            'Frozen Chroma collection count differs; refusing to evaluate')
    q_emb = np.asarray(embedder.encode([query]), dtype=float)
    require(q_emb.shape == (1, 384) and np.isfinite(q_emb).all(),
            'MiniLM returned invalid embedding')
    raw = collection.query(query_embeddings=q_emb.tolist(), n_results=5,
                           include=['documents', 'metadatas', 'embeddings', 'distances'])
    require(len(raw['ids']) == 1 and len(raw['ids'][0]) == 5,
            'Expected five live Chroma retrievals')
    rows = []
    for cid, content, meta, emb, dist in zip(
        raw['ids'][0], raw['documents'][0], raw['metadatas'][0],
        raw['embeddings'][0], raw['distances'][0],
    ):
        require(isinstance(content, str) and isinstance(meta, dict)
                and isinstance(meta.get('doc_id'), str) and len(emb) == 384
                and math.isfinite(float(dist)), 'Invalid retrieved chunk')
        # The classifier and response MUST NOT consume or expose label fields.
        safe_meta = {k: v for k, v in meta.items() if k not in FORBIDDEN}
        require(FORBIDDEN.isdisjoint(safe_meta), 'Ground-truth label exposure')
        rows.append({
            'chunk_id': str(cid), 'doc_id': safe_meta['doc_id'],
            'text': content, 'metadata': safe_meta,
            'source': str(safe_meta.get('source', '')),
            'embedding': [float(v) for v in emb],
            'distance': float(dist),
            'similarity': 1.0 - float(dist),
            'similarity_score': 1.0 - float(dist),
        })
    require(len({r['chunk_id'] for r in rows}) == len(rows),
            'Duplicate retrieved chunk IDs')

    d1_pass, d1_block = apply_outlier_filter(deepcopy(rows))
    require(len(d1_pass) + len(d1_block) == len(rows)
            and {r['chunk_id'] for r in d1_pass + d1_block}
            == {r['chunk_id'] for r in rows}, 'D1 did not score all retrieved chunks')
    d1_flagged_ids = {r['chunk_id'] for r in d1_block}

    feature_rows = [text_stat_features(r['text']) for r in rows]
    feature_matrix = pd.DataFrame(
        [{f: float(values[f]) for f in FEATURES} for values in feature_rows],
        columns=list(FEATURES),
    )
    require(np.isfinite(feature_matrix.to_numpy(dtype=float)).all(),
            'Nonfinite text-statistical feature values')
    probability = np.asarray(model.predict_proba(feature_matrix)[:, 1], dtype=float)
    require(probability.shape == (5,) and np.isfinite(probability).all()
            and np.all((probability >= 0) & (probability <= 1)),
            'RF probabilities are invalid')

    threshold = float(report['threshold'])
    passed, blocked, decisions = [], [], []
    for row, p in zip(rows, probability):
        flagged = bool(float(p) >= threshold)
        d = {
            'chunk_id': row['chunk_id'],
            'doc_id': row['doc_id'],
            'source': row['source'],
            'text': row['text'],
            'similarity_score': row['similarity_score'],
            'rf_poison_probability': float(p),
            'd1_advisory_flag': row['chunk_id'] in d1_flagged_ids,
            'rf_blocked': flagged,
        }
        decisions.append(d)
        (blocked if flagged else passed).append(d)
    return {
        'query': query,
        'attack': req.attack,
        'collection': collection_name,
        'retrieved_count': len(rows),
        'retained_count': len(passed),
        'blocked_count': len(blocked),
        'top_k_refilled': False,
        'defenses': {
            'd1': 'advisory_only',
            'd2': 'skipped_missing_canonical_metadata',
            'd3': 'skipped_missing_canonical_metadata',
            'd4': 'six_feature_rf_threshold_0.5',
        },
        'retained_chunks': passed,
        'blocked_chunks': blocked,
        'all_decisions': decisions,
        'warnings': [
            'Experimental benchmark-associated model; poisoned chunks may remain.',
            'D1 flags do not remove chunks; D2 and D3 are not operational.',
            'No Ollama generation, no Top-k refill.',
        ],
    }


def main():
    verify_frozen_artifacts()  # no file changes, Chroma calls, or model loading
    print('PASS: Isolated API routes: GET /api/v1/defended/health and POST /api/v1/defended/retrieve')
    print('PASS: Only existing SM/OBEA/RPI 20% collections; never creates or alters Chroma')
    print('PASS: D1 advisory; missing D2/D3 explicit; 6-feature model; no LLM')
    print('PREFLIGHT PASS: No Chroma or model load, no API server and no outputs written')
    print('NEXT: python -m uvicorn scripts.phase_e_defended_api_v1:app --host 127.0.0.1 --port 8002')


if __name__ == '__main__':
    main()
