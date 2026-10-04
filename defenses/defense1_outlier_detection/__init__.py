"""
defense1_outlier_detection
-----------------------------
Defense 1: Embedding Outlier Detection.

Flags documents whose embedding sits unusually far from the centroid of
the current retrieval batch, measured as a z-score of cosine distance.
Cheap and fast — the first stage of the defense pipeline (PRD §5) — but
documented to be weakest against optimization-based attacks that are
specifically crafted to blend into the embedding space.

Public API:
    from defenses.defense1_outlier_detection import apply_outlier_filter, OutlierDetector
"""

from defenses.defense1_outlier_detection.filter import apply_outlier_filter
from defenses.defense1_outlier_detection.outlier_detector import (
    OutlierDetector,
    OutlierVerdict,
)

__all__ = ["apply_outlier_filter", "OutlierDetector", "OutlierVerdict"]
