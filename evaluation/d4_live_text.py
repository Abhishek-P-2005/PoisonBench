"""
Defense 4 for the harness: the repo's own live-computable classifier.

data/results/phase_e/{random_forest,xgboost}_live_text_v1.joblib were trained
(scripts/phase_e_train_live_text_v1.py) on exactly six text statistics that can
be computed from any retrieved chunk, via
defenses.defense4_xgboost.feature_engineering.text_stat_features, with a
predeclared 0.5 threshold. This adapter plugs them into evaluation.stages.D4Stage
(`text_only = True`: D4Stage hands over raw chunk text, no upstream signals).

The frozen Phase C 12-feature models are NOT used: they need transductive,
benchmark-wide features and cannot score a live chunk.

Caveats that apply to every number produced with this model:
  * It was trained on this benchmark's controlled attack texts and shares
    documents with the benchmark queries. Report held-out queries separately
    (evaluation.real_source.heldout_query_ids) and do not read the full-benchmark
    figure as generalisation to unseen attackers or corpora.
  * Artifacts were built with scikit-learn 1.5.1; load them with the pinned
    requirements.txt versions (a newer version warns and may differ).
  * Only load joblib files you created yourself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "results" / "phase_e"
REPORT = MODEL_DIR / "live_text_models_v1.json"
FILES = {"random_forest": ("random_forest_live_text_v1.joblib", "rf_model"),
         "xgboost": ("xgboost_live_text_v1.joblib", "xgb_model")}
FEATURES = ("text_length_chars", "text_length_words", "avg_word_length",
            "char_entropy", "digit_ratio", "punctuation_ratio")


class LiveTextModel:
    text_only = True

    def __init__(self, estimator, threshold: float = 0.5, label: str = "live_text"):
        self.estimator = estimator
        self.threshold = threshold
        self.label = label

    def features(self, text: str) -> list[float]:
        from defenses.defense4_xgboost.feature_engineering import text_stat_features
        stats = text_stat_features(text)
        return [float(stats[name]) for name in FEATURES]

    def predict_proba(self, rows: list[dict]) -> list[float]:
        if not rows:
            return []
        import pandas as pd
        frame = pd.DataFrame([self.features(row["text"]) for row in rows], columns=list(FEATURES))
        return [float(p) for p in self.estimator.predict_proba(frame)[:, 1]]


def load_live_text_model(kind: str = "random_forest", model_dir: Path = MODEL_DIR) -> LiveTextModel:
    """Load and integrity-check a live-text model against live_text_models_v1.json."""
    if kind not in FILES:
        raise ValueError(f"kind must be one of {sorted(FILES)}")
    file_name, report_key = FILES[kind]
    path = Path(model_dir) / file_name
    report_path = Path(model_dir) / "live_text_models_v1.json"
    if not path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"missing {path} or {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if tuple(report["feature_columns"]) != FEATURES or report["threshold"] != 0.5:
        raise RuntimeError("live text feature order or threshold changed; refusing to load")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != report["sha256"][report_key]:
        raise RuntimeError(f"{file_name} checksum does not match live_text_models_v1.json")
    import joblib
    return LiveTextModel(joblib.load(path), report["threshold"], label=f"{kind}_live_text_v1")
