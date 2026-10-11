"""PoisonBench Phase C/D: isolated, identical-split RF vs XGBoost evaluation.

Default: READ-ONLY preflight, no fitting/no file creation. --run: fit both
models on the same TRAIN split and same TRAIN-only SMOTE-resampled matrix;
evaluate the untouched TEST split; exclusively create NEW model/results files.

Never modifies existing XGBoost implementation, Phase B data, feature CSV,
ChromaDB, or team results. Original Defense-4 XGBoost hyperparameters are
mirrored here so that the comparison is reproducible. The 12-feature primary
configuration EXCLUDES deterministic source-label and constant-age fields.

Important limitations: the shared Phase C feature CSV includes transductive
whole-corpus embedding/retrieval features. Results are controlled in-corpus
classification results, not proven generalization to independent corpora or
proof of live pipeline defense effectiveness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from defenses.defense4_xgboost.feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS, LABEL_COLUMN,
)
from defenses.defense4_xgboost.smote_utils import apply_smote  # noqa: E402

BASE = ROOT / 'data/results/phase_c'
CSV_PATH = BASE / 'combined_features_v1.csv'
MANIFEST_PATH = BASE / 'split_manifest_v1.json'
RF_MODEL = BASE / 'random_forest_model_v1.joblib'
XGB_MODEL = BASE / 'xgboost_comparison_model_v1.joblib'
METRICS = BASE / 'rf_xgb_comparison_v1.json'
PREDICTIONS = BASE / 'rf_xgb_test_predictions_v1.csv'
OUTPUTS = (RF_MODEL, XGB_MODEL, METRICS, PREDICTIONS)
SEED = 42


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT)).replace('\\', '/')


def load_validate():
    require(CSV_PATH.is_file() and MANIFEST_PATH.is_file(),
            'Expected frozen Phase C CSV and split manifest are missing')
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    df = pd.read_csv(CSV_PATH)
    require(manifest.get('schema') == 'phase_c_rf_xgb_grouped_split_v1',
            'Unexpected split manifest schema')
    require(manifest.get('seed') == SEED and manifest.get('dataset_row_count') == 553,
            'Unexpected dataset manifest seed/size')
    require(sha(CSV_PATH) == manifest.get('dataset_sha256'), 'Dataset checksum mismatch')
    benchmark_path = ROOT / str(manifest['benchmark']).replace('\\', '/')
    require(benchmark_path.is_file() and sha(benchmark_path) == manifest['benchmark_sha256'],
            'Frozen benchmark checksum mismatch')
    for attack in ('sm', 'obea', 'rpi'):
        b5 = ROOT / f'data/results/phase_b/phase_b5_{attack}_20pct_top5_v1_results.csv'
        require(b5.is_file() and sha(b5) == manifest['b5_csv_sha256'][attack],
                f'Frozen {attack} B5 checksum mismatch')
    require(len(df) == 553 and df['chunk_id'].is_unique, 'Feature row identity/size mismatch')
    require(Counter(df[LABEL_COLUMN].tolist()) == {0: 316, 1: 237},
            'Class labels differ from frozen 316/237 balance')
    require(Counter(df['attack_type']) == {'clean': 316, 'sm': 79, 'obea': 79, 'rpi': 79},
            'Attack labels differ from frozen composition')
    require((df['attack_type'].eq('clean') == df[LABEL_COLUMN].eq(0)).all(),
            'Attack type / truth label inconsistency')
    for attack in ('sm', 'obea', 'rpi'):
        subset = df[df['attack_type'] == attack]
        require(subset['chunk_id'].str.startswith(f'POISON-{attack.upper()}-').all(),
                f'Malformed {attack} chunk identifiers')
    require(df[df['attack_type'] == 'clean']['chunk_id'].str.startswith('CVE-').all(),
            'Malformed clean chunk identifiers')
    require(set(df['split']) == {'train', 'test'}, 'Unexpected split labels')
    train, test = df[df['split'] == 'train'].copy(), df[df['split'] == 'test'].copy()
    require(len(train) == 416 and len(test) == 137, 'Unexpected train/test rows')
    require(Counter(train[LABEL_COLUMN]) == {0: 239, 1: 177} and
            Counter(test[LABEL_COLUMN]) == {0: 77, 1: 60},
            'Unexpected split class counts')
    require(set(train['chunk_id']).isdisjoint(set(test['chunk_id'])) and
            set(train['group_id']).isdisjoint(set(test['group_id'])),
            'Train/test groups overlap')
    target_train = set(manifest['train_target_groups'])
    target_test = set(manifest['test_target_groups'])
    require(len(target_train) == 15 and len(target_test) == 5 and
            target_train.isdisjoint(target_test), 'Target group split changed')
    require(set(train.loc[train['is_poisoned'] == 1, 'group_id']) == target_train and
            set(test.loc[test['is_poisoned'] == 1, 'group_id']) == target_test,
            'Poison targets misassigned to train/test')
    with benchmark_path.open(encoding='utf-8') as f:
        benchmark = {r['query_id'].lower(): r['expected_doc_id']
                     for line in f if line.strip() for r in [json.loads(line)]}
    require(len(benchmark) == 20, 'Frozen query cardinality differs')
    for group in target_test:
        qid = group.removeprefix('target:')
        linked = df[df['chunk_id'].str.startswith(benchmark[qid] + '::chunk')]
        require(len(linked) >= 1 and set(linked['split']) == {'test'},
                f'Held-out legitimate evidence leakage: {qid}')
    features = manifest['primary_feature_columns']
    require(isinstance(features, list) and len(features) == 12 and len(set(features)) == 12,
            'Unexpected feature count')
    require(all(col in FEATURE_COLUMNS and col in df.columns for col in features),
            'Feature schema mismatch')
    require('source_is_official' not in features and 'source_age_days' not in features,
            'Label shortcut or constant age accidentally included')
    require(not df.isna().any().any(), 'Missing feature values')
    require(np.isfinite(df[features].to_numpy(dtype=np.float64)).all(),
            'Nonfinite numeric input features')
    require(not any(path.exists() for path in OUTPUTS),
            'A comparison output already exists: refusing to overwrite or rerun')
    print('PASS: Phase C CSV SHA-256 matches split manifest')
    print('PASS: Frozen benchmark and all three 20% B5 checksums')
    print('PASS: 553 samples; train 416 (239 clean/177 poison); test 137 (77 clean/60 poison)')
    print('PASS: Group-disjoint split and five held-out target-query CVEs')
    print('PASS: 12 identical primary features; source-label leakage columns excluded')
    print('PASS: Four comparison output filenames are available')
    return df, manifest, train, test, features


def classify_metrics(model, x_test, y_test, meta):
    predictions = model.predict(x_test).astype(int)
    probabilities = model.predict_proba(x_test)[:, 1].astype(float)
    require(len(predictions) == len(y_test) and np.isfinite(probabilities).all(),
            'Invalid classifier predictions')
    require(((probabilities >= 0) & (probabilities <= 1)).all(),
            'Predicted probabilities outside [0, 1]')
    truth = y_test.to_numpy(dtype=int)
    cm = confusion_matrix(truth, predictions, labels=[0, 1]).tolist()
    per_attack = {}
    for attack in ('sm', 'obea', 'rpi'):
        mask = meta['attack_type'].eq(attack).to_numpy()
        require(mask.sum() == 20 and truth[mask].sum() == 20,
                f'Expected exactly 20 {attack} held-out poison examples')
        per_attack[attack] = {
            'poison_count': int(mask.sum()),
            'detected_as_poison': int(predictions[mask].sum()),
            'recall': float(recall_score(truth[mask], predictions[mask], zero_division=0)),
        }
    clean_mask = meta['attack_type'].eq('clean').to_numpy()
    require(int(clean_mask.sum()) == 77 and not truth[clean_mask].any(),
            'Held-out clean count mismatch')
    return {
        'precision': float(precision_score(truth, predictions, zero_division=0)),
        'recall': float(recall_score(truth, predictions, zero_division=0)),
        'f1': float(f1_score(truth, predictions, zero_division=0)),
        'accuracy_secondary': float(accuracy_score(truth, predictions)),
        'roc_auc': float(roc_auc_score(truth, probabilities)),
        'average_precision': float(average_precision_score(truth, probabilities)),
        'confusion_matrix_TN_FP_FN_TP': cm,
        'classification_report': classification_report(
            truth, predictions, output_dict=True, zero_division=0),
        'by_poison_attack': per_attack,
        'clean_false_positives': int(predictions[clean_mask].sum()),
        'clean_false_positive_rate': float(predictions[clean_mask].mean()),
    }, predictions, probabilities


def run(df, manifest, train, test, features):
    x_train, y_train = train[features], train[LABEL_COLUMN].astype(int)
    x_test, y_test = test[features], test[LABEL_COLUMN].astype(int)
    # Same training-only resampling, with exactly the same result for BOTH models.
    x_balanced, y_balanced = apply_smote(x_train, y_train, random_state=SEED)
    require(len(x_balanced) == len(y_balanced), 'SMOTE dimension mismatch')
    require(Counter(np.asarray(y_balanced).tolist()) == {0: 239, 1: 239},
            'Unexpected SMOTE class balance')
    require(x_balanced.columns.tolist() == features, 'SMOTE altered feature column order')
    print('Training RF and XGBoost using SAME SMOTE-resampled TRAIN set...')
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=SEED, n_jobs=-1,
    )
    rf.fit(x_balanced, y_balanced)
    xgb.fit(x_balanced, y_balanced)
    meta = test[['chunk_id', 'group_id', 'attack_type', LABEL_COLUMN]].copy()
    rf_results, rf_pred, rf_prob = classify_metrics(rf, x_test, y_test, meta)
    xgb_results, xgb_pred, xgb_prob = classify_metrics(xgb, x_test, y_test, meta)
    for name, results in (('Random Forest', rf_results), ('XGBoost', xgb_results)):
        print(f"{name}: precision={results['precision']:.4f} recall={results['recall']:.4f} "
              f"F1={results['f1']:.4f} PR-AUC={results['average_precision']:.4f} "
              f"ROC-AUC={results['roc_auc']:.4f} "
              f"CM={results['confusion_matrix_TN_FP_FN_TP']}")
    results = {
        'schema': 'phase_c_rf_xgb_comparison_v1',
        'comparison_protocol': 'identical frozen grouped train/test split, identical 12 numerical features, identical training-only SMOTE; original Subash XGBoost hyperparameters',
        'seed': SEED,
        'dataset_csv': relative(CSV_PATH),
        'dataset_sha256': sha(CSV_PATH),
        'split_manifest': relative(MANIFEST_PATH),
        'split_manifest_sha256': sha(MANIFEST_PATH),
        'train_rows_original': int(len(train)),
        'train_rows_after_smote': int(len(x_balanced)),
        'test_rows_untouched': int(len(test)),
        'training_class_counts_original': {str(k): int(v) for k, v in sorted(Counter(y_train).items())},
        'training_class_counts_after_smote': {str(k): int(v) for k, v in sorted(Counter(y_balanced).items())},
        'test_class_counts': {str(k): int(v) for k, v in sorted(Counter(y_test).items())},
        'feature_columns': features,
        'model_parameters': {
            'random_forest': {'n_estimators': 300, 'random_state': SEED, 'n_jobs': -1},
            'xgboost': {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
                        'subsample': 0.8, 'colsample_bytree': 0.8,
                        'eval_metric': 'logloss', 'random_state': SEED, 'n_jobs': -1},
        },
        'metrics': {'random_forest': rf_results, 'xgboost': xgb_results},
        'model_artifacts': {
            'random_forest': relative(RF_MODEL), 'xgboost': relative(XGB_MODEL),
        },
        'predictions_csv': relative(PREDICTIONS),
        'limitations': manifest['evaluation_limitations'] + [
            'Both models trained on synthetic 20%-ratio attack variants; test labels are based on constructed controlled corpus',
            'Random Forest / XGBoost classification scores do not themselves demonstrate live RAG pipeline mitigation',
            'Original XGBoost teammate training script and data are not modified by this isolated comparison',
        ],
    }
    predictions = meta.reset_index(drop=True)
    predictions['rf_prediction'] = rf_pred
    predictions['rf_poison_probability'] = rf_prob
    predictions['xgb_prediction'] = xgb_pred
    predictions['xgb_poison_probability'] = xgb_prob
    require(len(predictions) == 137 and predictions['chunk_id'].is_unique,
            'Wrong prediction CSV structure')
    # Exclusive-create outputs. If this process fails partway, stop for audit;
    # never rerun while any output exists.
    with RF_MODEL.open('xb') as f:
        joblib.dump(rf, f)
    with XGB_MODEL.open('xb') as f:
        joblib.dump(xgb, f)
    with PREDICTIONS.open('x', encoding='utf-8', newline='') as f:
        predictions.to_csv(f, index=False)
    results['sha256'] = {
        'rf_model': sha(RF_MODEL),
        'xgb_model': sha(XGB_MODEL),
        'predictions_csv': sha(PREDICTIONS),
    }
    with METRICS.open('x', encoding='utf-8') as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write('\n')
    print('PASS: Isolated RF and XGBoost model artifacts created')
    print('PASS: Comparison metrics and 137 held-out test predictions saved')
    print('SAFETY: Original Subash XGBoost files, frozen B5/B6, and ChromaDB untouched')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true',
                        help='EXPLICITLY train models and save four NEW artifacts')
    args = parser.parse_args()
    df, manifest, train, test, features = load_validate()
    if not args.run:
        print('PREFLIGHT PASS: READ ONLY — no model fitting or file writes')
        for path in OUTPUTS:
            print('Planned new output:', relative(path))
        return
    run(df, manifest, train, test, features)


if __name__ == '__main__':
    main()
