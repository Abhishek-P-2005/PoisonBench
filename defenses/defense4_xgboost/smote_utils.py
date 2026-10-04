"""
Phase 3: class imbalance handling.

Poisoned documents are always a small minority of the corpus (PRD 4 tech
stack table). Applied ONLY to the training split, never to the test split --
SMOTE synthesizes new minority-class samples by interpolating between real
ones, and if you apply it before the train/test split, synthetic points end
up leaking into the test set (a near-duplicate of a training point can end
up in "test", inflating every metric). This is the single most common SMOTE
mistake and reviewers/panels ask about it specifically -- keep the
train_xgboost.py split-then-SMOTE order intact.
"""
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Returns (X_resampled, y_resampled). Falls back to no-op if there aren't
    enough minority-class samples for SMOTE's k-neighbors (default k=5,
    needs at least 6 minority samples) -- this happens at very low
    poisoning ratios (e.g. 1%) on small corpora, and SMOTE will raise if you
    don't guard for it.
    """
    minority_count = int(y_train.value_counts().min())
    k_neighbors = min(5, max(1, minority_count - 1))

    if minority_count < 2:
        print(f"WARNING: only {minority_count} minority-class sample(s) in this "
              f"training split -- skipping SMOTE, training on raw imbalanced data.")
        return X_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE: {len(X_train)} -> {len(X_res)} training rows "
          f"(minority class {y_train.sum()} -> {y_res.sum()})")
    return X_res, y_res
