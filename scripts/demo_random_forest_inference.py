
"""PoisonBench — live Random Forest inference demonstration.

Loads the frozen Phase E six-feature Random Forest and evaluates
the existing held-out test split without retraining or writing files.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "data/results/phase_e/random_forest_live_text_v1.joblib"
DATA_PATH = ROOT / "data/results/phase_c/combined_features_v1.csv"

EXPECTED_FEATURES = [
    "text_length_chars",
    "text_length_words",
    "avg_word_length",
    "char_entropy",
    "digit_ratio",
    "punctuation_ratio",
]


def main():
    print("\n" + "=" * 65)
    print("       POISONBENCH — LIVE RANDOM FOREST DEFENSE")
    print("=" * 65, flush=True)

    print("\n[1] LOADING TRAINED RANDOM FOREST", flush=True)

    model = joblib.load(MODEL_PATH)

    assert list(model.feature_names_in_) == EXPECTED_FEATURES, (
        "Random Forest feature schema mismatch"
    )

    print("Model: RandomForestClassifier")
    print(f"Decision trees: {model.n_estimators}")
    print("Status: LOADED SUCCESSFULLY", flush=True)

    print("\n[2] LOADING HELD-OUT TEST DATA", flush=True)

    data = pd.read_csv(DATA_PATH)
    test = data.loc[data["split"] == "test"].copy()

    assert len(test) == 137, "Unexpected test-set size"
    assert test["chunk_id"].is_unique, "Duplicate test document IDs"

    X = test[EXPECTED_FEATURES]
    y_true = test["is_poisoned"].astype(int).to_numpy()

    assert np.isfinite(X.to_numpy(dtype=float)).all(), (
        "Invalid feature values"
    )

    print(f"Test documents: {len(test)}")
    print(f"Clean documents: {(y_true == 0).sum()}")
    print(f"Poisoned documents: {(y_true == 1).sum()}")
    print(f"Input features: {len(EXPECTED_FEATURES)}")

    print("\n[3] EXECUTING RANDOM FOREST INFERENCE", flush=True)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("PASS: Live model inference completed", flush=True)

    results = pd.DataFrame({
        "Document": test["chunk_id"].to_numpy(),
        "Actual": y_true,
        "Predicted": predictions,
        "Poison Probability": np.round(probabilities, 4),
    })

    print("\n[4] SAMPLE PREDICTIONS")
    print("Label 0 = CLEAN | Label 1 = POISONED")
    print(results.head(12).to_string(index=False))

    print("\n[5] RANDOM FOREST EVALUATION")
    print(f"Accuracy : {accuracy_score(y_true, predictions) * 100:.2f}%")
    print(f"Precision: {precision_score(y_true, predictions) * 100:.2f}%")
    print(f"Recall   : {recall_score(y_true, predictions) * 100:.2f}%")
    print(f"F1-score : {f1_score(y_true, predictions) * 100:.2f}%")

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    print("\n[6] CONFUSION MATRIX")
    print(f"True Negatives  (clean retained) : {tn}")
    print(f"False Positives (clean flagged)  : {fp}")
    print(f"False Negatives (poison missed)  : {fn}")
    print(f"True Positives  (poison detected): {tp}")

    print("\n" + "=" * 65)
    print("        RANDOM FOREST DEMONSTRATION COMPLETE")
    print("=" * 65)
    print("Read-only inference. No retraining or database modifications.")


if __name__ == "__main__":
    main()
