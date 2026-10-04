"""
Phase 3 core deliverable: trains the primary Defense 4 classifier (XGBoost)
on the labeled feature CSV produced by build_training_data.py.

Order of operations matters and is enforced here, not just documented:
  1. stratified train/test split FIRST (stratify=y, so the tiny poisoned
     class is represented proportionally in both splits)
  2. SMOTE applied to the TRAINING split only (see smote_utils.py docstring
     for why)
  3. train
  4. evaluate on the untouched, non-resampled test split
  5. metrics: precision/recall/F1 -- NOT accuracy (PRD 6.6 / 9), because
     accuracy on a 95/5 imbalanced set is misleading (predicting "clean"
     for everything scores ~95% accuracy while catching zero attacks).

Usage:
    python defense4_xgboost/train_xgboost.py --features defense4_xgboost/data/features.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from defenses.defense4_xgboost.feature_engineering import FEATURE_COLUMNS, LABEL_COLUMN  # noqa: E402
from defenses.defense4_xgboost.smote_utils import apply_smote                           # noqa: E402

MODEL_OUT = REPO_ROOT / "defenses" / "defense4_xgboost" / "trained_results" / "models" / "xgboost_model.joblib"
METRICS_OUT = REPO_ROOT / "defenses" / "defense4_xgboost" / "trained_results" / "results" / "xgboost_metrics.json"


def train_and_evaluate(df: pd.DataFrame, random_state: int = 42) -> dict:
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state=random_state)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "feature_importance": dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist())),
        "n_train": len(X_train_res),
        "n_test": len(X_test),
        "n_test_poisoned": int(y_test.sum()),
    }
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(REPO_ROOT / "defenses" / "defense4_xgboost" / "trained_results" / "data" / "features.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    if df[LABEL_COLUMN].nunique() < 2:
        print("ERROR: training data has only one class present (all clean or all poisoned). "
              "Run the attack module against the corpus first, then re-run build_training_data.py.")
        return

    model, metrics = train_and_evaluate(df)

    print("\n=== XGBoost Defense 4 -- Test Set Metrics ===")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1       : {metrics['f1']:.3f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {MODEL_OUT}")
    print(f"Metrics saved to {METRICS_OUT}")


if __name__ == "__main__":
    main()
