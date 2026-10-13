"""
evaluation/classifier.py
------------------------
Defense 4 interface for the harness.

Subash (XGBoost) and Abhishek (Random Forest) own the real classifiers. The
harness only needs one thing from a model:

    model.predict_proba(rows) -> list[float]     # P(chunk is poisoned)

where `rows` is a list of feature dicts with the keys in FEATURE_NAMES.
Any scikit-learn style estimator (XGBClassifier, RandomForestClassifier)
can be wrapped with SklearnProbaModel and plugged in unchanged.

The features are exactly the signals Defenses 1-3 already produce (see
FINAL_REVIEW_PLAN.md, "Feature contract") plus a few cheap text statistics.
`is_poisoned` and `attack_type` are never features.

IMPORTANT: train_standin_model() below exists only so the harness and
Module 5 can be tested end to end before the real models are plugged in.
Numbers produced with the stand-in are NOT Defense 4 results.
"""

from __future__ import annotations

import math

import numpy as np

from defenses.defense1_outlier_detection.config import MIN_NEIGHBORHOOD_SIZE
from defenses.defense1_outlier_detection.outlier_detector import OutlierDetector
from defenses.defense2_trust_rerank.trust_scorer import TrustScorer
from defenses.defense3_provenance.provenance_verifier import ProvenanceVerifier

FEATURE_NAMES = (
    "similarity",
    "similarity_zscore",
    "outlier_score",
    "trust_score",
    "provenance_score",
    "verified",
    "text_len",
    "injection_word_count",
)

INJECTION_WORDS = ("ignore", "disregard", "instead", "you must", "do not", "override")


# ── make sure the Defense 1-3 signals exist on each chunk ────────────────────

def ensure_signals(batch: list[dict]) -> list[dict]:
    """
    Return shallow copies of the chunks with outlier_score, trust_score and
    provenance_score filled in. Fields already added by earlier pipeline
    stages are reused, so inside the full pipeline this does no extra work.
    """
    chunks = [dict(chunk) for chunk in batch]
    if not chunks:
        return chunks

    if any("outlier_score" not in c for c in chunks):
        has_embeddings = all("embedding" in c for c in chunks)
        big_enough = len(chunks) - 1 >= MIN_NEIGHBORHOOD_SIZE
        if has_embeddings and big_enough:
            ids = [c["chunk_id"] for c in chunks]
            embeddings = [c["embedding"] for c in chunks]
            verdicts = OutlierDetector().score_batch(ids, embeddings)
            for chunk, verdict in zip(chunks, verdicts):
                chunk["outlier_score"] = verdict.outlier_score
        else:
            for chunk in chunks:
                chunk["outlier_score"] = 0.0   # same neutral value Defense 1 uses

    if any("trust_score" not in c for c in chunks):
        breakdowns = TrustScorer().score_batch([c["metadata"] for c in chunks])
        for chunk, breakdown in zip(chunks, breakdowns):
            chunk["trust_score"] = breakdown.final_score

    if any("provenance_score" not in c for c in chunks):
        verifier = ProvenanceVerifier()
        for chunk in chunks:
            verdict = verifier.verify(chunk["text"], chunk["metadata"])
            chunk["provenance_score"] = verdict.provenance_score

    return chunks


# ── feature rows ──────────────────────────────────────────────────────────────

def build_feature_rows(batch: list[dict], only_ids: set | None = None) -> list[dict]:
    """
    One feature dict per chunk. Batch-level statistics (similarity z-score)
    are always computed over the FULL batch, even when only_ids selects a
    subset of rows, so the router does not change what the model sees.
    """
    chunks = ensure_signals(batch)
    if not chunks:
        return []

    sims = [c["similarity"] for c in chunks]
    mean_sim = sum(sims) / len(sims)
    variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
    std_sim = max(math.sqrt(variance), 1e-6)

    rows = []
    for chunk in chunks:
        if only_ids is not None and chunk["chunk_id"] not in only_ids:
            continue
        text = chunk["text"]
        lower = text.lower()
        injection_count = 0
        for word in INJECTION_WORDS:
            if word in lower:
                injection_count += 1
        rows.append({
            "chunk_id":             chunk["chunk_id"],
            "similarity":           chunk["similarity"],
            "similarity_zscore":    (chunk["similarity"] - mean_sim) / std_sim,
            "outlier_score":        chunk["outlier_score"],
            "trust_score":          chunk["trust_score"],
            "provenance_score":     chunk["provenance_score"],
            "verified":             1.0 if chunk["metadata"].get("verified") else 0.0,
            "text_len":             float(len(text)),
            "injection_word_count": float(injection_count),
        })
    return rows


# ── model wrappers ────────────────────────────────────────────────────────────

class SklearnProbaModel:
    """Wrap any scikit-learn style classifier (XGBoost, Random Forest, ...)."""

    def __init__(self, estimator, feature_names=FEATURE_NAMES, label: str = "model"):
        self.estimator = estimator
        self.feature_names = tuple(feature_names)
        self.label = label

    def predict_proba(self, rows: list[dict]) -> list[float]:
        if not rows:
            return []
        matrix = np.array(
            [[row[name] for name in self.feature_names] for row in rows],
            dtype=float,
        )
        probabilities = self.estimator.predict_proba(matrix)
        classes = list(self.estimator.classes_)
        if 1 not in classes:
            return [0.0 for _ in rows]
        positive_index = classes.index(1)
        return [float(p) for p in probabilities[:, positive_index]]


def train_standin_model(rows: list[dict], labels: list[bool], seed: int = 0) -> SklearnProbaModel:
    """
    STAND-IN classifier for testing the harness. Random Forest with balanced
    class weights (SMOTE is not used here). Replace with Subash's XGBoost and
    Abhishek's Random Forest wrapped in SklearnProbaModel.
    """
    from sklearn.ensemble import RandomForestClassifier

    matrix = np.array(
        [[row[name] for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )
    y = np.array([1 if label else 0 for label in labels])
    estimator = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    estimator.fit(matrix, y)
    return SklearnProbaModel(estimator, FEATURE_NAMES, label="standin_random_forest")
