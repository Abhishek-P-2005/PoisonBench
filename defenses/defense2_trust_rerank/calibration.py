"""
defense2_trust_rerank/calibration.py
---------------------------------------
Threshold calibration for Defense 2, per PRD Section 3.3.5
("Defense Analysis Framework"): thresholds for Defenses 1-3 are calibrated
using ROC and Precision-Recall analysis rather than picked arbitrarily.

This module cannot run until the attack module (Abhishek) has produced a
labeled set of poisoned and clean documents. Until then, LOW_TRUST_FLAG_THRESHOLD
in config.py is a placeholder. Call `calibrate_threshold()` once labeled data
exists to replace it with a data-driven value.

Usage (once labeled data is available)
---------------------------------------
    trust_scores = [scorer.score(m).final_score for m in metadata_list]
    labels       = [m["is_poisoned"] for m in metadata_list]   # ground truth
    result = calibrate_threshold(trust_scores, labels)
    print(result.best_threshold, result.best_f1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class CalibrationResult:
    roc_auc:          float
    best_threshold:   float   # threshold on trust_score that maximizes F1
    best_precision:   float
    best_recall:      float
    best_f1:          float
    fpr:              np.ndarray   # false positive rate at each ROC threshold
    tpr:              np.ndarray   # true positive rate at each ROC threshold
    roc_thresholds:   np.ndarray
    pr_precision:     np.ndarray
    pr_recall:        np.ndarray
    pr_thresholds:    np.ndarray


def calibrate_threshold(
    trust_scores: list[float],
    is_poisoned_labels: list[bool],
) -> CalibrationResult:
    """
    Given trust scores and ground-truth poisoned/clean labels for the SAME
    set of documents, find the trust-score threshold that best separates
    poisoned from clean documents.

    Note on polarity: trust_score is HIGH for trustworthy (clean) documents
    and LOW for suspicious (likely poisoned) documents. So "flag as poisoned"
    means trust_score < threshold. We invert trust_score to a "suspicion
    score" internally so sklearn's convention (higher score = positive class)
    holds, then invert the resulting threshold back.

    Parameters
    ----------
    trust_scores        : Defense 2 trust_score output, one per document
    is_poisoned_labels   : ground truth from the attack module, same order

    Returns
    -------
    CalibrationResult with ROC AUC, the F1-optimal threshold, and full
    curves for plotting in the analysis notebook (Section 3.3.5 deliverable).
    """
    if len(trust_scores) != len(is_poisoned_labels):
        raise ValueError("trust_scores and is_poisoned_labels must be the same length")
    if len(set(is_poisoned_labels)) < 2:
        raise ValueError(
            "Need both poisoned and clean examples to calibrate a threshold — "
            "got only one class in is_poisoned_labels"
        )

    suspicion_scores = [1.0 - t for t in trust_scores]
    y_true = [int(label) for label in is_poisoned_labels]

    fpr, tpr, roc_thresholds = roc_curve(y_true, suspicion_scores)
    auc = roc_auc_score(y_true, suspicion_scores)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, suspicion_scores)

    # F1 per PR threshold (precision/recall arrays are 1 longer than thresholds)
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )

    if len(f1_scores) == 0:
        best_idx = 0
        best_suspicion_threshold = 0.5
        best_p, best_r, best_f1 = 0.0, 0.0, 0.0
    else:
        best_idx = int(np.argmax(f1_scores))
        best_suspicion_threshold = pr_thresholds[best_idx]
        best_p = precision[best_idx]
        best_r = recall[best_idx]
        best_f1 = f1_scores[best_idx]

    # Convert back from suspicion-score threshold to trust-score threshold.
    best_trust_threshold = 1.0 - best_suspicion_threshold

    return CalibrationResult(
        roc_auc        = auc,
        best_threshold = best_trust_threshold,
        best_precision = float(best_p),
        best_recall    = float(best_r),
        best_f1        = float(best_f1),
        fpr            = fpr,
        tpr            = tpr,
        roc_thresholds = 1.0 - roc_thresholds,   # expressed in trust-score terms
        pr_precision   = precision,
        pr_recall      = recall,
        pr_thresholds  = 1.0 - pr_thresholds,
    )
