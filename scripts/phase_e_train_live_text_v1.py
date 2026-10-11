"""PoisonBench Phase E: train separately named, live-computable text-feature models.

Install: scripts/phase_e_train_live_text_v1.py
    python scripts/phase_e_train_live_text_v1.py           # read-only preflight
    python scripts/phase_e_train_live_text_v1.py --train   # create NEW output files

This is a deployment-candidate ABlation. It does not replace or modify frozen
Phase C/D 12-feature models/results. It uses exactly six features computable from
arbitrary retrieved chunk text by existing feature_engineering.text_stat_features,
not transductive centroid / frozen-benchmark retrieval statistics. Group-aware
train/test splits remain identical to Phase C. SMOTE is TRAIN-ONLY.

Caveats: dataset consists of controlled synthetic attack texts. Strong held-out
scores do not demonstrate generalization to unseen attackers/other corpora.
The threshold 0.5 is predeclared, not tuned on the held-out test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defenses.defense4_xgboost.feature_engineering import text_stat_features  # noqa: E402
from defenses.defense4_xgboost.smote_utils import apply_smote  # noqa: E402

BASE_C = ROOT / 'data/results/phase_c'
BASE_E = ROOT / 'data/results/phase_e'
CSV = BASE_C / 'combined_features_v1.csv'
MANIFEST = BASE_C / 'split_manifest_v1.json'
PHASE_CD = BASE_C / 'rf_xgb_comparison_v1.json'
RF = BASE_E / 'random_forest_live_text_v1.joblib'
XGB = BASE_E / 'xgboost_live_text_v1.joblib'
PRED = BASE_E / 'live_text_test_predictions_v1.csv'
REPORT = BASE_E / 'live_text_models_v1.json'
OUTPUTS = (RF, XGB, PRED, REPORT)
FEATURES = (
    'text_length_chars', 'text_length_words', 'avg_word_length',
    'char_entropy', 'digit_ratio', 'punctuation_ratio',
)
SEED = 42


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT)).replace('\\', '/')


def validate():
    require(all(p.is_file() for p in (CSV, MANIFEST, PHASE_CD)),
            'Frozen Phase C/D inputs missing')
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    r = json.loads(PHASE_CD.read_text(encoding='utf-8'))
    require(m.get('schema') == 'phase_c_rf_xgb_grouped_split_v1' and
            r.get('schema') == 'phase_c_rf_xgb_comparison_v1',
            'Unexpected frozen metadata schema')
    require(sha(CSV) == m['dataset_sha256'] == r['dataset_sha256'],
            'Frozen dataset SHA mismatch')
    require(sha(MANIFEST) == r['split_manifest_sha256'],
            'Frozen split manifest SHA mismatch')
    require(m.get('seed') == SEED, 'Group split seed mismatch')
    require(len(m['primary_feature_columns']) == 12 and
            set(FEATURES).issubset(m['primary_feature_columns']),
            'Six text features not drawn from established shared schema')
    require(set(text_stat_features('A test CVE-2026-11111.')) == set(FEATURES),
            'Current runtime text-stat function differs from training feature set')
    df = pd.read_csv(CSV)
    require(len(df) == 553 and df['chunk_id'].is_unique and
            set(df['split']) == {'train', 'test'}, 'Dataset identity/split invalid')
    require(Counter(df['is_poisoned']) == {0: 316, 1: 237}, 'Labels changed')
    train = df[df['split'] == 'train'].copy()
    test = df[df['split'] == 'test'].copy()
    require(len(train) == 416 and len(test) == 137, 'Split cardinality changed')
    require(Counter(train['is_poisoned']) == {0: 239, 1: 177} and
            Counter(test['is_poisoned']) == {0: 77, 1: 60}, 'Split labels changed')
    require(set(train['group_id']).isdisjoint(set(test['group_id'])),
            'Group leakage in train/test')
    require(set(train['chunk_id']).isdisjoint(set(test['chunk_id'])),
            'Chunk identity overlaps train/test')
    require(np.isfinite(df[list(FEATURES)].to_numpy(dtype=float)).all(),
            'Missing/invalid numeric text features')
    require(not any(p.exists() for p in OUTPUTS),
            'At least one LIVE TEXT output exists; do not overwrite or rerun')
    require(BASE_E.is_dir(), f'Expected Phase E directory missing: {BASE_E}')
    print('PASS: Frozen CSV/manifest SHA checks and original 416/137 group split')
    print('PASS: Exactly 6 text-only features reproducible from live chunk text')
    print('PASS: No centroid, retrieval-stat, source or ground-truth label features')
    print('PASS: Four NEW deployment-candidate output paths available')
    return df, train, test


def metrics(truth, pred, prob, attack_types):
    cm = confusion_matrix(truth, pred, labels=[0, 1]).tolist()
    per_attack = {}
    for attack in ('sm', 'obea', 'rpi'):
        mask = attack_types.eq(attack).to_numpy()
        require(int(mask.sum()) == 20, f'Unexpected heldout {attack} poison count')
        per_attack[attack] = {
            'detected': int(np.asarray(pred)[mask].sum()), 'total_poison': 20,
        }
    tn, fp = cm[0]
    fn, tp = cm[1]
    return {
        'precision': float(precision_score(truth, pred, zero_division=0)),
        'recall': float(recall_score(truth, pred, zero_division=0)),
        'f1': float(f1_score(truth, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(truth, prob)),
        'pr_auc_average_precision': float(average_precision_score(truth, prob)),
        'confusion_matrix': cm,
        'false_positive_rate': float(fp / (fp + tn)),
        'per_attack': per_attack,
    }


def train_models(train, test):
    feature_names = list(FEATURES)
    x_train, y_train = train[feature_names], train['is_poisoned'].astype(int)
    x_test, y_test = test[feature_names], test['is_poisoned'].astype(int)
    xb, yb = apply_smote(x_train, y_train, random_state=SEED)
    require(Counter(np.asarray(yb)) == {0: 239, 1: 239}, 'SMOTE balance wrong')
    require(list(xb.columns) == feature_names, 'SMOTE feature order changed')
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='logloss', random_state=SEED, n_jobs=-1)
    print('Fitting two 6-feature deployment-candidate models on train-only SMOTE...')
    rf.fit(xb, yb)
    xgb.fit(xb, yb)
    truth = y_test.to_numpy(dtype=int)
    predictions = test[['chunk_id', 'group_id', 'attack_type', 'is_poisoned']].copy()
    reports = {}
    for name, short, model in (('random_forest', 'rf', rf), ('xgboost', 'xgb', xgb)):
        require(getattr(model, 'n_features_in_', None) == len(feature_names),
                f'{name}: trained feature count mismatch')
        prob = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
        pred = (prob >= 0.5).astype(int)
        require(np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(),
                f'{name}: invalid prediction probabilities')
        predictions[f'{short}_prediction'] = pred
        predictions[f'{short}_poison_probability'] = prob
        reports[name] = metrics(truth, pred, prob, test['attack_type'])
        print(f"{name}: precision={reports[name]['precision']:.4f}, "
              f"recall={reports[name]['recall']:.4f}, "
              f"F1={reports[name]['f1']:.4f}, "
              f"CM={reports[name]['confusion_matrix']}")
    # Sanity: live extraction for arbitrary text must produce the exact schema.
    sample = text_stat_features('CVE-2026-54500 vulnerable Oj JSON parser.')
    require(set(sample) == set(feature_names) and
            all(np.isfinite(float(sample[f])) for f in feature_names),
            'Runtime extraction not compatible with training')
    # Exclusive-write only. If interrupted partway, stop and investigate;
    # never delete/rewrite a partial output without separate authorization.
    with RF.open('xb') as fh:
        joblib.dump(rf, fh)
    with XGB.open('xb') as fh:
        joblib.dump(xgb, fh)
    with PRED.open('x', encoding='utf-8', newline='') as fh:
        predictions.to_csv(fh, index=False)
    report = {
        'schema': 'phase_e_live_text_models_v1',
        'purpose': 'Separate deployment-compatible, six-text-feature models; frozen Phase C/D comparison untouched',
        'dataset_sha256': sha(CSV),
        'split_manifest_sha256': sha(MANIFEST),
        'seed': SEED,
        'feature_columns': feature_names,
        'feature_generator': 'defenses.defense4_xgboost.feature_engineering.text_stat_features',
        'split': {'train': 416, 'train_after_smote': 478, 'test': 137,
                  'train_clean': 239, 'train_poison': 177,
                  'test_clean': 77, 'test_poison': 60},
        'threshold': 0.5,
        'metrics': reports,
        'models': {'random_forest': rel(RF), 'xgboost': rel(XGB)},
        'predictions_csv': rel(PRED),
        'sha256': {'rf_model': sha(RF), 'xgb_model': sha(XGB), 'predictions': sha(PRED)},
        'limitations': [
            'Experimental separately trained models; not the original 12-feature Phase C/D models.',
            'Six numerical text features only; no corpus centroid, retrieval history, source metadata or labels used at inference.',
            'Training data from synthetic 20%-ratio SM/OBEA/RPI experiments; generalization beyond benchmark unknown.',
            'Potential text artifacts from synthetic attack generation can inflate held-out scores.',
            'No generation, live retrieval, or D1/D2/D3 filtering evaluated by this training run.',
            'Threshold fixed at 0.5 without held-out tuning.',
        ],
    }
    with REPORT.open('x', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
        fh.write('\n')
    print('PASS: Four isolated six-feature model, prediction and report artifacts created')
    print(f'RF SHA256: {sha(RF)}')
    print(f'XGB SHA256: {sha(XGB)}')
    print('SAFETY: No ChromaDB/Ollama calls; original Phase B/C/D artifacts untouched')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', action='store_true', help='Train and create four NEW files')
    args = ap.parse_args()
    _, train, test = validate()
    if args.train:
        train_models(train, test)
    else:
        for p in OUTPUTS:
            print('Planned new output:', rel(p))
        print('PREFLIGHT PASS: READ ONLY — no training, model loads, database calls or files written')


if __name__ == '__main__':
    main()
