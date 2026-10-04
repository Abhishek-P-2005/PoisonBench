"""
Phase 3 deliverable: feature engineering for Defense 4 (supervised ML
classifier).

CONTRACT WITH ABHISHEK'S RANDOM FORest MODEL:
This module's output schema (the exact column names in the returned
DataFrame) is shared infrastructure. Per PRD 6.6 and the team responsibility
split, the Random Forest comparison classifier is trained on "the same
dataset, features, and evaluation protocol" as this XGBoost model. That
means: if you rename, add, or remove a feature column here, you MUST update
Abhishek's training script too, or the two-classifier comparison (the
project's strongest empirical claim, per PRD 6.6) becomes invalid because
the models weren't actually trained on the same features.

Feature groups (per PRD 6.6):
  1. embedding distance from cluster centroid
  2. text statistics
  3. source metadata
  4. retrieval score anomalies
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "embedding_dist_from_centroid",
    "embedding_dist_from_nearest_neighbor",
    "text_length_chars",
    "text_length_words",
    "avg_word_length",
    "char_entropy",
    "digit_ratio",
    "punctuation_ratio",
    "source_is_official",
    "source_age_days",
    "retrieval_score_mean",
    "retrieval_score_max",
    "retrieval_score_zscore",
    "retrieval_appearance_count",
]

LABEL_COLUMN = "is_poisoned"


# ---------------------------------------------------------------------------
# Group 1: embedding-space features
# ---------------------------------------------------------------------------

def compute_corpus_centroid(embeddings: np.ndarray) -> np.ndarray:
    return embeddings.mean(axis=0)


def embedding_features(embedding: np.ndarray, centroid: np.ndarray, all_embeddings: np.ndarray) -> dict:
    dist_from_centroid = float(np.linalg.norm(embedding - centroid))

    # nearest neighbor distance (excluding itself) -- optimization-based
    # attacks are specifically designed to sit CLOSE to legitimate
    # neighbors, so this feature is expected to matter more than centroid
    # distance for catching attack #2 in the PRD (embedding-optim attack).
    dists = np.linalg.norm(all_embeddings - embedding, axis=1)
    dists_sorted = np.sort(dists)
    nearest = dists_sorted[1] if len(dists_sorted) > 1 else dists_sorted[0]

    return {
        "embedding_dist_from_centroid": dist_from_centroid,
        "embedding_dist_from_nearest_neighbor": float(nearest),
    }


# ---------------------------------------------------------------------------
# Group 2: text statistics
# ---------------------------------------------------------------------------

def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def text_stat_features(text: str) -> dict:
    words = text.split()
    n_chars = len(text)
    n_words = len(words) or 1
    avg_word_len = sum(len(w) for w in words) / n_words
    digits = sum(1 for ch in text if ch.isdigit())
    punct = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())

    return {
        "text_length_chars": n_chars,
        "text_length_words": n_words,
        "avg_word_length": avg_word_len,
        "char_entropy": _char_entropy(text),
        "digit_ratio": digits / max(n_chars, 1),
        "punctuation_ratio": punct / max(n_chars, 1),
    }


# ---------------------------------------------------------------------------
# Group 3: source metadata
# ---------------------------------------------------------------------------

OFFICIAL_SOURCES = {"NVD"}


def source_features(source: str, timestamp_days_old: float) -> dict:
    return {
        "source_is_official": 1 if source in OFFICIAL_SOURCES else 0,
        "source_age_days": timestamp_days_old,
    }


# ---------------------------------------------------------------------------
# Group 4: retrieval score anomalies
# ---------------------------------------------------------------------------

def retrieval_score_features(scores_for_this_chunk: list[float], all_scores_in_corpus: list[float]) -> dict:
    """
    scores_for_this_chunk: similarity scores this chunk received across the
    fixed trigger-query benchmark set (see Abhishek's benchmark in
    scripts/known_answers.json-style trigger queries, run against every
    chunk). Optimization-based attacks push this artificially high for their
    target trigger query.
    """
    if not scores_for_this_chunk:
        return {
            "retrieval_score_mean": 0.0,
            "retrieval_score_max": 0.0,
            "retrieval_score_zscore": 0.0,
            "retrieval_appearance_count": 0,
        }

    mean_score = float(np.mean(scores_for_this_chunk))
    max_score = float(np.max(scores_for_this_chunk))

    corpus_mean = float(np.mean(all_scores_in_corpus)) if all_scores_in_corpus else 0.0
    corpus_std = float(np.std(all_scores_in_corpus)) if all_scores_in_corpus else 1.0
    zscore = (max_score - corpus_mean) / corpus_std if corpus_std > 0 else 0.0

    return {
        "retrieval_score_mean": mean_score,
        "retrieval_score_max": max_score,
        "retrieval_score_zscore": zscore,
        "retrieval_appearance_count": len(scores_for_this_chunk),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(chunks: list[dict], retrieval_scores_by_chunk: dict[str, list[float]]) -> pd.DataFrame:
    """
    chunks: output of vectorstore.get_all_chunks(), i.e. list of
        {chunk_id, text, metadata: {doc_id, source, timestamp, is_poisoned, ...}, embedding}
    retrieval_scores_by_chunk: {chunk_id: [scores across trigger-query benchmark]},
        built by running every trigger query against the corpus and recording
        the score each chunk got (0.0 if never retrieved for that query).

    Returns a DataFrame with FEATURE_COLUMNS + LABEL_COLUMN + 'chunk_id',
    one row per chunk. This exact schema is what train_xgboost.py and
    Abhishek's Random Forest script both consume.
    """
    embeddings = np.array([c["embedding"] for c in chunks])
    centroid = compute_corpus_centroid(embeddings)

    all_scores_flat = [s for scores in retrieval_scores_by_chunk.values() for s in scores]

    rows = []
    for c, emb in zip(chunks, embeddings):
        meta = c["metadata"]
        row = {}
        row.update(embedding_features(emb, centroid, embeddings))
        row.update(text_stat_features(c["text"]))
        row.update(source_features(meta.get("source", ""), meta.get("source_age_days", 0.0)))
        row.update(retrieval_score_features(
            retrieval_scores_by_chunk.get(c["chunk_id"], []),
            all_scores_flat,
        ))
        row["chunk_id"] = c["chunk_id"]
        row[LABEL_COLUMN] = int(bool(meta.get("is_poisoned", False)))
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered_cols = ["chunk_id"] + FEATURE_COLUMNS + [LABEL_COLUMN]
    return df[ordered_cols]
