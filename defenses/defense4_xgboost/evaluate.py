"""
Loads the trained XGBoost model and exposes a single scoring function that
the rest of the system (integration/pipeline_orchestrator.py, the
evaluation harness) calls. This is the "Defense 4" plug-in referenced in
integration/interfaces.py -- keep its function signature stable, since
pipeline_orchestrator.py depends on it.
"""
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

from defenses.defense4_xgboost.feature_engineering import FEATURE_COLUMNS

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "trained_results"
    / "models"
    / "xgboost_model.joblib"
)

DEFAULT_THRESHOLD = 0.5  # probability threshold above which a chunk is flagged


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run train_xgboost.py first."
        )
    return joblib.load(MODEL_PATH)


def score_chunks(feature_rows: list[dict], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """
    feature_rows: list of dicts, each containing at least FEATURE_COLUMNS
    plus a 'chunk_id' key (this is exactly the row format
    feature_engineering.build_feature_matrix produces, minus the label).

    Returns list of {chunk_id, flagged: bool, score: float} -- 'score' is
    the model's raw poisoned-class probability, so the orchestrator/
    evaluation harness can log it even when using a non-default threshold.
    """
    model = _load_model()
    df = pd.DataFrame(feature_rows)
    X = df[FEATURE_COLUMNS]
    probs = model.predict_proba(X)[:, 1]

    return [
        {"chunk_id": row["chunk_id"], "flagged": bool(p >= threshold), "score": float(p)}
        for row, p in zip(feature_rows, probs)
    ]
