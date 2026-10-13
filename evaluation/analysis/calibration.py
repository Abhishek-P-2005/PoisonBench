"""
evaluation/analysis/calibration.py
----------------------------------
Module 1: Statistical Threshold Calibration (PRD v2.0 Reqs 1.1 - 1.3).

For each defense, take its continuous suspicion score (logged by
runner.score_pass), draw ROC and Precision-Recall curves, and pick the
decision threshold that catches the most poisoned chunks while keeping the
false positive rate on CLEAN BASELINE data at or below a ceiling (5%).

Suspicion scores (higher = more suspicious):
    d1  embedding distance z-score          (threshold in z units)
    d2  1 - trust_score                     (threshold in suspicion units)
    d3  1 - provenance_score                (only 5 distinct levels)
    d4  classifier probability of poison

Calibrate on seeds that are NOT used in the evaluation sweep, otherwise the
reported detection rates are optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from evaluation.config import FPR_CEILING

SCORE_COLUMNS = {
    "d1": "d1_z",
    "d2": "d2_suspicion",
    "d3": "d3_suspicion",
    "d4": "d4_prob",
}


@dataclass
class CalibrationReport:
    defense:             str
    n_clean:             int
    n_poison:            int
    auc:                 float
    average_precision:   float
    fpr_ceiling:         float
    threshold:           float    # flag when suspicion >= threshold
    fpr_at_threshold:    float
    recall_at_threshold: float
    f1_threshold:        float
    f1_best:             float
    fpr_curve:           np.ndarray = field(repr=False, default=None)
    tpr_curve:           np.ndarray = field(repr=False, default=None)
    pr_precision:        np.ndarray = field(repr=False, default=None)
    pr_recall:           np.ndarray = field(repr=False, default=None)

    def summary(self) -> dict:
        return {
            "defense": self.defense, "n_clean": self.n_clean, "n_poison": self.n_poison,
            "auc": self.auc, "average_precision": self.average_precision,
            "fpr_ceiling": self.fpr_ceiling, "threshold": self.threshold,
            "fpr_at_threshold": self.fpr_at_threshold,
            "recall_at_threshold": self.recall_at_threshold,
            "f1_threshold": self.f1_threshold, "f1_best": self.f1_best,
        }


def calibrate_defense(
    defense: str,
    clean_scores,
    poison_scores,
    fpr_ceiling: float = FPR_CEILING,
) -> CalibrationReport:
    clean = np.array([s for s in clean_scores if s is not None and not np.isnan(s)], dtype=float)
    poison = np.array([s for s in poison_scores if s is not None and not np.isnan(s)], dtype=float)
    if len(clean) == 0 or len(poison) == 0:
        raise ValueError(f"{defense}: need both clean and poisoned scores to calibrate")

    scores = np.concatenate([clean, poison])
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(poison))])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    allowed = np.where(fpr <= fpr_ceiling)[0]
    index = int(allowed[-1])
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = float(scores.max()) + 1e-9   # nothing can be flagged within the ceiling

    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    f1_values = []
    for p, r in zip(precision[:-1], recall[:-1]):
        f1_values.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    best = int(np.argmax(f1_values)) if f1_values else 0

    return CalibrationReport(
        defense=defense,
        n_clean=len(clean),
        n_poison=len(poison),
        auc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
        fpr_ceiling=fpr_ceiling,
        threshold=threshold,
        fpr_at_threshold=float(fpr[index]),
        recall_at_threshold=float(tpr[index]),
        f1_threshold=float(pr_thresholds[best]) if len(pr_thresholds) else threshold,
        f1_best=float(f1_values[best]) if f1_values else 0.0,
        fpr_curve=fpr,
        tpr_curve=tpr,
        pr_precision=precision,
        pr_recall=recall,
    )


def split_clean_and_poison(chunks_df, column: str, clean_source: str = "baseline", min_baseline: int = 100):
    """
    Clean scores come from the clean baseline runs (attack 'none') when there
    are enough of them, otherwise from all clean chunks. Poison scores come
    from every attacked run.
    """
    clean_all = chunks_df[chunks_df["is_poisoned"] == 0]
    baseline = clean_all[clean_all["poisoning_ratio"] == 0]
    if clean_source == "baseline" and len(baseline) >= min_baseline:
        clean_rows = baseline
    else:
        clean_rows = clean_all
    poison_rows = chunks_df[chunks_df["is_poisoned"] == 1]
    return clean_rows[column].tolist(), poison_rows[column].tolist()


def calibrate_all(
    chunks_df,
    fpr_ceiling: float = FPR_CEILING,
    block_fpr_ceiling: float = 0.01,
    clean_source: str = "baseline",
) -> dict:
    """Calibrate every defense that has scores. Returns {defense: CalibrationReport}."""
    reports = {}
    for defense, column in SCORE_COLUMNS.items():
        if column not in chunks_df.columns or chunks_df[column].notna().sum() == 0:
            continue
        clean, poison = split_clean_and_poison(chunks_df, column, clean_source)
        reports[defense] = calibrate_defense(defense, clean, poison, fpr_ceiling)
        if defense == "d1":
            # D1 also has a hard-block threshold, calibrated at a stricter ceiling.
            reports["d1_block"] = calibrate_defense("d1_block", clean, poison, block_fpr_ceiling)
    return reports


def thresholds_to_params(reports: dict) -> dict:
    """Convert calibrated suspicion thresholds into DefenseConfig params."""
    params = {}
    if "d1" in reports:
        soft = reports["d1"].threshold
        hard = soft
        if "d1_block" in reports:
            hard = max(reports["d1_block"].threshold, soft)
        params["d1_soft_z"] = soft
        params["d1_hard_z"] = hard
    if "d2" in reports:
        # flagged when trust < flag_threshold  <=>  suspicion > 1 - flag_threshold
        params["d2_flag_threshold"] = 1.0 - reports["d2"].threshold + 1e-9
    if "d3" in reports:
        params["d3_flag_below"] = 1.0 - reports["d3"].threshold + 1e-9
    if "d4" in reports:
        params["d4_threshold"] = reports["d4"].threshold
    return params


def plot_roc_pr(reports: dict, path: str, title: str = "Threshold calibration") -> None:
    """Deliverable 2: dual-panel ROC and Precision-Recall curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name in reports if name != "d1_block"]
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name in names:
        report = reports[name]
        ax_roc.plot(report.fpr_curve, report.tpr_curve, label=f"{name} (AUC {report.auc:.2f})")
        ax_roc.scatter([report.fpr_at_threshold], [report.recall_at_threshold], s=40, zorder=5)
        ax_pr.plot(report.pr_recall, report.pr_precision, label=f"{name} (AP {report.average_precision:.2f})")

    ax_roc.plot([0, 1], [0, 1], linestyle=":", color="grey")
    ax_roc.axvline(FPR_CEILING, linestyle="--", color="red", linewidth=1, label=f"FPR ceiling {FPR_CEILING:.0%}")
    ax_roc.set_xlabel("False positive rate (clean baseline)")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC (dots = calibrated operating point)")
    ax_roc.legend(loc="lower right")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall")
    ax_pr.legend(loc="lower left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
