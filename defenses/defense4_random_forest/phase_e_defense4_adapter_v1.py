"""PoisonBench Phase E: safe RF/XGBoost scoring adapter for frozen Phase C/D models.

Place this file at: defenses/defense4_random_forest/phase_e_defense4_adapter_v1.py

Default invocation is a READ-ONLY preflight. --audit loads the frozen models
and independently verifies the 137 held-out predictions, returning
DefenseVerdict objects through the shared integration.interfaces contract.
--timing measures loaded-model inference ONLY on the saved held-out matrix.

IMPORTANT DEPLOYMENT LIMITATION:
  The frozen 12-feature models use a TRANSDUCTIVE feature protocol: benchmark-
  wide retrieval statistics and embedding geometry over the *combined* 553-row
  experimental dataset. Do NOT substitute one live query's score or a different
  collection's centroid and claim it is identical to the audited experiment.
  A live feature provider is deliberately NOT supplied here. Integrate one only
  after verifying its feature construction and documenting evaluation limits.

This module never opens ChromaDB/Ollama, retrains, writes files, or reads
chunk.is_poisoned during inference. Do not load untrusted joblib models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.interfaces import Defense, DefenseVerdict, RetrievedChunk  # noqa: E402

OUT = ROOT / 'data' / 'results' / 'phase_c'
MANIFEST = OUT / 'split_manifest_v1.json'
REPORT = OUT / 'rf_xgb_comparison_v1.json'
DATA = OUT / 'combined_features_v1.csv'
PRED = OUT / 'rf_xgb_test_predictions_v1.csv'
MODELS = {
    'random_forest': OUT / 'random_forest_model_v1.joblib',
    'xgboost': OUT / 'xgboost_comparison_model_v1.joblib',
}
EXPECTED_DATASET_SHA = 'd4f0aeb73eb6236c9de15780b571ff3f088d2f7761b0af617f9f7b6f932fe0bf'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def verify_artifacts() -> tuple[list[str], dict]:
    """Read-only integrity preflight; model deserialization is NOT required."""
    for path in (MANIFEST, REPORT, DATA, PRED, *MODELS.values()):
        _require(path.is_file(), f'Missing artifact: {path}')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    _require(manifest.get('schema') == 'phase_c_rf_xgb_grouped_split_v1',
             'Unexpected split manifest')
    _require(report.get('schema') == 'phase_c_rf_xgb_comparison_v1',
             'Unexpected comparison report')
    _require(_sha(DATA) == EXPECTED_DATASET_SHA == manifest['dataset_sha256'] == report['dataset_sha256'],
             'Frozen dataset checksum mismatch')
    _require(_sha(MANIFEST) == report['split_manifest_sha256'],
             'Frozen manifest checksum mismatch')
    for name, key in (('random_forest', 'rf_model'), ('xgboost', 'xgb_model')):
        _require(_sha(MODELS[name]) == report['sha256'][key], f'{name} model checksum mismatch')
    _require(_sha(PRED) == report['sha256']['predictions_csv'],
             'Predictions checksum mismatch')
    features = manifest['primary_feature_columns']
    _require(len(features) == 12 and len(set(features)) == 12, 'Feature count/order invalid')
    _require(features == report['feature_columns'], 'Comparison report feature-order mismatch')
    _require(not {'source_is_official', 'source_age_days'}.intersection(features),
             'Excluded source metadata unexpectedly included')
    _require(report.get('test_rows_untouched') == 137, 'Unexpected held-out sample count')
    return features, report


FeatureProvider = Callable[[Sequence[RetrievedChunk]], Sequence[Mapping[str, object]]]


class FrozenComparisonDefense(Defense):
    """Classifier scored using externally constructed, EXACT-SCHEMA features.

    A provider must return one feature mapping per input RetrievedChunk in the
    same order. The provider MUST NOT use is_poisoned/attack_type/group_id/label
    and MUST match the frozen benchmark-wide/embedding reference protocol if
    numerical comparison to Phase C/D is intended.

    No implicit fallback to single-query similarity stats is permitted.
    """

    def __init__(self, model_name: str, feature_provider: FeatureProvider,
                 threshold: float = 0.5):
        _require(model_name in MODELS, f'Unknown model: {model_name}')
        _require(callable(feature_provider), 'A compatible feature provider is required')
        _require(0.0 <= threshold <= 1.0, 'Threshold must be between 0 and 1')
        self.name = model_name
        self.threshold = threshold
        self.feature_provider = feature_provider
        self.features, _ = verify_artifacts()
        # These are trusted, locally trained PoisonBench models only.
        self.model = joblib.load(MODELS[model_name])
        _require(getattr(self.model, 'n_features_in_', None) == len(self.features),
                 'Trained model expects a different number of features')
        model_features = getattr(self.model, 'feature_names_in_', None)
        if model_features is not None:
            _require(list(model_features) == self.features, 'Model feature names differ')
        _require(list(self.model.classes_) == [0, 1], 'Unexpected model class order')

    def score(self, chunks: list[RetrievedChunk]) -> list[DefenseVerdict]:
        if not chunks:
            return []
        ids = [c.chunk_id for c in chunks]
        _require(len(ids) == len(set(ids)), 'Duplicate requested chunk IDs')
        start = time.perf_counter()
        rows = list(self.feature_provider(chunks))
        _require(len(rows) == len(chunks), 'Feature provider returned incomplete rows')
        _require([r.get('chunk_id') for r in rows] == ids,
                 'Feature provider chunk IDs/order mismatch')
        _require(all(set(self.features).issubset(row.keys()) for row in rows),
                 'Feature provider missing required columns')
        matrix = pd.DataFrame(rows).loc[:, self.features]
        try:
            matrix = matrix.astype('float64')
        except (TypeError, ValueError) as exc:
            raise RuntimeError('Feature matrix contains nonnumerical values') from exc
        _require(bool(np.isfinite(matrix.to_numpy()).all()), 'NaN/Infinity in features')
        probabilities = self.model.predict_proba(matrix)[:, 1]
        _require(bool(np.isfinite(probabilities).all()), 'Invalid poison probabilities')
        _require(bool(((probabilities >= 0) & (probabilities <= 1)).all()),
                 'Probabilities outside [0, 1]')
        # Includes feature-provider overhead, preparation and model inference.
        elapsed_per_chunk = 1000 * (time.perf_counter() - start) / len(chunks)
        return [DefenseVerdict(chunk_id=chunk_id,
                               flagged=bool(prob >= self.threshold),
                               score=float(prob),
                               latency_ms=float(elapsed_per_chunk))
                for chunk_id, prob in zip(ids, probabilities)]


def _heldout_rows(features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA)
    saved = pd.read_csv(PRED)
    _require(len(data) == 553 and len(saved) == 137, 'Frozen row count mismatch')
    test = data.loc[data['split'].eq('test')].copy()
    _require(len(test) == 137 and test['chunk_id'].is_unique and saved['chunk_id'].is_unique,
             'Test identity or split mismatch')
    _require(set(test['chunk_id']) == set(saved['chunk_id']), 'Saved predictions do not match held-out IDs')
    test = test.set_index('chunk_id').loc[saved['chunk_id']].reset_index()
    _require(test['is_poisoned'].tolist() == saved['is_poisoned'].tolist(),
             'Saved true labels do not match dataset')
    _require(all(c in test.columns for c in features), 'Missing feature columns')
    return test, saved


def _offline_provider(features: list[str], heldout: pd.DataFrame) -> FeatureProvider:
    """AUDIT ONLY. Never use this provider as a live-production feature source."""
    lookup = {row['chunk_id']: {'chunk_id': row['chunk_id'],
              **{col: row[col] for col in features}}
              for row in heldout.to_dict(orient='records')}
    def provider(chunks: Sequence[RetrievedChunk]) -> list[dict]:
        return [lookup[c.chunk_id] for c in chunks]
    return provider


def audit() -> None:
    features, _ = verify_artifacts()
    heldout, saved = _heldout_rows(features)
    # Ground truth is intentionally excluded from chunks passed to classifier.
    chunks = [RetrievedChunk(chunk_id=cid, doc_id='', text='', similarity_score=0.0,
                             source='', is_poisoned=False, metadata={})
              for cid in saved['chunk_id']]
    provider = _offline_provider(features, heldout)
    for name, prefix in (('random_forest', 'rf'), ('xgboost', 'xgb')):
        classifier = FrozenComparisonDefense(name, provider)
        verdicts = classifier.score(chunks)
        probs = np.array([v.score for v in verdicts])
        flags = np.array([int(v.flagged) for v in verdicts])
        np.testing.assert_array_equal(flags, saved[f'{prefix}_prediction'].to_numpy())
        np.testing.assert_allclose(probs, saved[f'{prefix}_poison_probability'].to_numpy(),
                                   rtol=1e-5, atol=1e-6)
        cm = confusion_matrix(saved['is_poisoned'], flags, labels=[0, 1]).tolist()
        print(f'PASS: {name} verdict interface reproduced 137/137 saved predictions; CM={cm}')
    print('PASS: Interface exercised using frozen offline feature rows only')
    print('READ ONLY: no training, no ChromaDB/Ollama calls, no file writes')


def timing(repeats: int) -> None:
    """Optional offline microbenchmark. Does not include live feature construction."""
    features, _ = verify_artifacts()
    heldout, _ = _heldout_rows(features)
    X = heldout.loc[:, features]
    for name in MODELS:
        model = joblib.load(MODELS[name])
        for _ in range(5):
            model.predict_proba(X)
        samples_ms = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.predict_proba(X)
            samples_ms.append(1000 * (time.perf_counter() - t0))
        print(f'{name}: batch_n=137; median_model_only_ms={statistics.median(samples_ms):.3f}; '
              f'p95_model_only_ms={float(np.percentile(samples_ms, 95)):.3f}; '
              f'median_model_only_ms_per_chunk={statistics.median(samples_ms)/137:.5f}')
    print('READ ONLY: latency EXCLUDES feature construction, Chroma retrieval and LLM; not end-to-end defense overhead')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument('--audit', action='store_true', help='Replay 137 frozen predictions through shared verdict contract')
    exclusive.add_argument('--timing', action='store_true', help='Read-only loaded-model batch microbenchmark')
    parser.add_argument('--repeats', type=int, default=40, help='Timing batches (default 40)')
    args = parser.parse_args()
    _require(1 <= args.repeats <= 1000, '--repeats must be 1..1000')
    if args.audit:
        audit()
    elif args.timing:
        timing(args.repeats)
    else:
        features, _ = verify_artifacts()
        print('PASS: Frozen dataset, manifest, model and prediction checksums')
        print('PASS: Shared scoring contract: RetrievedChunk -> DefenseVerdict')
        print(f'PASS: {len(features)} feature columns, fixed order and source-shortcut exclusion')
        print('PRECHECK PASS: READ ONLY, no deserialization, fitting or file writes')
        print('LIMITATION: Live feature provider NOT implemented; offline benchmark-wide features must not be replaced by one-query scores')


if __name__ == '__main__':
    main()
