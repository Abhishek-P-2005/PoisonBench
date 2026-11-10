
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ieee_5000"
INPUT = BASE / "processed/defense_features_5300.csv"
MANIFEST = BASE / "processed/defense_features_manifest.json"
OUTPUT = BASE / "results/rf_xgb_comparison_5300.json"
SPLIT = BASE / "processed/defense_split_5300.csv"
MODEL_DIR = BASE / "models"

def main():
    model_paths = {
        "Random Forest": MODEL_DIR / "random_forest_ieee.joblib",
        "XGBoost": MODEL_DIR / "xgboost_ieee.joblib"
    }
    if any(p.exists() for p in [OUTPUT, SPLIT, *model_paths.values()]):
        raise FileExistsError("Existing IEEE defense output; refusing overwrite")

    df = pd.read_csv(INPUT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    features = manifest["features"]

    assert len(df) == 5300 and len(features) == 12
    assert df["is_poisoned"].sum() == 300
    assert df["chunk_id"].is_unique

    benchmark = [
        json.loads(x)
        for x in (BASE / "processed/benchmark_100.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    target_to_query = {q["expected_doc_id"]: q["query_id"] for q in benchmark}

    def group(row):
        doc_id = str(row["doc_id"])
        if row["attack_type"] != "clean":
            match = re.search(r"q\d{3}", doc_id)
            if not match:
                raise ValueError(f"Missing poison query ID: {doc_id}")
            return "target:" + match.group()
        if doc_id in target_to_query:
            return "target:" + target_to_query[doc_id]
        return "clean:" + doc_id

    df["group_id"] = df.apply(group, axis=1)

    target_groups = sorted(g for g in df["group_id"].unique()
                           if g.startswith("target:"))
    assert len(target_groups) == 100

    # Explicitly hold out 20 target groups, including all 3 attack types.
    rng = np.random.default_rng(42)
    held_out_targets = set(rng.choice(target_groups, size=20, replace=False))

    clean_groups = sorted(g for g in df["group_id"].unique()
                          if g.startswith("clean:"))
    held_out_clean = set(rng.choice(
        clean_groups, size=round(len(clean_groups) * 0.2), replace=False
    ))

    test_groups = held_out_targets | held_out_clean
    df["split"] = np.where(df["group_id"].isin(test_groups), "test", "train")

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    assert set(train["group_id"]).isdisjoint(set(test["group_id"]))
    assert test["is_poisoned"].sum() == 60
    assert all((test["attack_type"] == a).sum() == 20
               for a in ("sm", "obea", "rpi"))

    X_train = train[features].to_numpy(dtype=float)
    y_train = train["is_poisoned"].to_numpy(dtype=int)
    X_test = test[features].to_numpy(dtype=float)
    y_test = test["is_poisoned"].to_numpy(dtype=int)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1,
            class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss", random_state=42, n_jobs=-1,
            scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum())
        )
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(X_train, y_train)
        pred = model.predict(X_test).astype(int)
        prob = model.predict_proba(X_test)[:, 1]

        per_attack = {}
        for attack in ("sm", "obea", "rpi"):
            mask = test["attack_type"].eq(attack).to_numpy()
            per_attack[attack] = {
                "tested": int(mask.sum()),
                "detected": int(pred[mask].sum()),
                "recall": float(pred[mask].mean())
            }

        results[name] = {
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
            "per_attack": per_attack
        }
        print(f"{name} F1: {results[name]['f1']:.4f}", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df[["chunk_id", "doc_id", "attack_type", "group_id", "split"]].to_csv(SPLIT, index=False)
    for name, model in models.items():
        joblib.dump(model, model_paths[name])

    payload = {
        "dataset_rows": len(df),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_poison": int(y_train.sum()),
        "test_poison": int(y_test.sum()),
        "features": features,
        "seed": 42,
        "split": "group-disjoint 80/20 target-level holdout",
        "limitations": [
            "Transductive embedding and retrieval features",
            "Controlled, description-derived benchmark",
            "Single held-out split; no external-corpus validation"
        ],
        "models": results
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("PASS: RF vs XGBoost evaluation complete")
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
