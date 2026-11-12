"""
evaluation/ieee5000_d4.py
-------------------------
Defense 4 for the ieee_5000 corpus, in a form the harness can run on a LIVE chunk.

Why a new model: ieee_5000/models/*.joblib use 12 features, 6 of which (embedding position
and retrieval statistics) were computed over the whole benchmark ("transductive_features":
true in defense_features_manifest.json). They cannot be computed for one new chunk at query
time, so the harness cannot call them. This module trains the same two architectures on the
six text statistics that can be computed from the chunk alone, using the repo's own
group-disjoint train/test split (ieee_5000/processed/defense_split_5300.csv), then wraps the
result in evaluation.d4_live_text.LiveTextModel so D4Stage uses it unchanged.

Nothing in ieee_5000/ is written. Artifacts go to data/results/eval_ieee5000/models/ with a
SHA-256 manifest that the loader checks before unpickling (only load files you created).

Read the ablation table before quoting the F1. Poison chunks here are 2-4x longer than
clean chunks (the attacks repeat or append text), so `length_only` alone may already score
close to the full six-feature model. If it does, the classifier has learned "long chunk",
not "poison", and the number says little about attackers who match length.

    python -m evaluation.ieee5000_d4 --train
    python -m evaluation.ieee5000_d4 --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.d4_live_text import FEATURES, LiveTextModel          # noqa: E402
from evaluation.ieee5000_source import FEATURES_PATH, SPLIT_PATH      # noqa: E402

MODEL_DIR = ROOT / "data" / "results" / "eval_ieee5000" / "models"
MANIFEST = "ieee_text_models_v1.json"
KINDS = ("random_forest", "xgboost")
FILES = {"random_forest": "random_forest_ieee_text6_v1.joblib",
         "xgboost": "xgboost_ieee_text6_v1.joblib"}
FEATURE_SETS = {
    "text6": list(FEATURES),
    "text_no_length": ["avg_word_length", "char_entropy", "digit_ratio", "punctuation_ratio"],
    "length_only": ["text_length_chars", "text_length_words"],
}
THRESHOLD = 0.5
SEED = 42


def _make_model(kind: str, n_negative: int, n_positive: int):
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced")
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
                             random_state=SEED, n_jobs=-1,
                             scale_pos_weight=float(n_negative / max(n_positive, 1)))
    raise ValueError(f"kind must be one of {KINDS}")


def load_training_frame(features_path: Path = FEATURES_PATH, split_path: Path = SPLIT_PATH):
    import pandas as pd
    features = pd.read_csv(features_path)
    split = pd.read_csv(split_path)[["chunk_id", "group_id", "split"]]
    frame = features.merge(split, on="chunk_id", how="inner", validate="one_to_one")
    if len(frame) != len(features):
        raise RuntimeError("defense_features and defense_split do not cover the same chunks")
    return frame


def _metrics(model, frame, columns) -> dict:
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    y = frame["is_poisoned"].to_numpy(dtype=int)
    prob = model.predict_proba(frame[columns].to_numpy(dtype=float))[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    per_attack = {}
    for attack in ("sm", "obea", "rpi"):
        mask = (frame["attack_type"] == attack).to_numpy()
        per_attack[attack] = {"tested": int(mask.sum()), "detected": int(pred[mask].sum()),
                              "recall": float(pred[mask].mean()) if mask.any() else None}
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_attack": per_attack,
        "n_test": int(len(y)), "n_test_poison": int(y.sum()),
        "mean_prob_poison": float(np.mean(prob[y == 1])) if (y == 1).any() else None,
    }


def train(model_dir: Path = MODEL_DIR, kinds=KINDS, features_path: Path = FEATURES_PATH,
          split_path: Path = SPLIT_PATH, overwrite: bool = False, verbose: bool = True) -> dict:
    """Train on the repo's training split, evaluate on its held-out split, save and checksum."""
    import joblib

    model_dir = Path(model_dir)
    manifest_path = model_dir / MANIFEST
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass overwrite=True (--overwrite) to retrain")
    frame = load_training_frame(features_path, split_path)
    train_df, test_df = frame[frame["split"] == "train"], frame[frame["split"] == "test"]
    if set(train_df["group_id"]) & set(test_df["group_id"]):
        raise RuntimeError("train and test share a group; the split is not group-disjoint")
    y_train = train_df["is_poisoned"].to_numpy(dtype=int)
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())

    model_dir.mkdir(parents=True, exist_ok=True)
    report = {"features": list(FEATURES), "threshold": THRESHOLD, "seed": SEED,
              "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
              "train_poison": n_pos, "test_poison": int(test_df["is_poisoned"].sum()),
              "split": "repo group-disjoint split (ieee_5000/processed/defense_split_5300.csv)",
              "sha256": {}, "metrics": {}, "ablation": {},
              "limitations": ["Poison text is synthetic and written in one style per attack",
                              "Poison chunks are longer than clean chunks; see the length_only ablation",
                              "Single held-out split, no external-corpus validation"]}
    for kind in kinds:
        for set_name, columns in FEATURE_SETS.items():
            model = _make_model(kind, n_neg, n_pos)
            model.fit(train_df[columns].to_numpy(dtype=float), y_train)
            metrics = _metrics(model, test_df, columns)
            report["ablation"].setdefault(set_name, {})[kind] = {
                k: metrics[k] for k in ("precision", "recall", "f1", "fpr", "roc_auc")}
            if set_name == "text6":
                path = model_dir / FILES[kind]
                joblib.dump(model, path)
                report["sha256"][kind] = hashlib.sha256(path.read_bytes()).hexdigest()
                report["metrics"][kind] = metrics
                if verbose:
                    print(f"  {kind:14s} text6  P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"F1={metrics['f1']:.3f} FPR={metrics['fpr']:.3f}")
            elif verbose:
                print(f"  {kind:14s} {set_name:15s} F1={metrics['f1']:.3f} FPR={metrics['fpr']:.3f}")
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_ieee_text_model(kind: str = "random_forest", model_dir: Path = MODEL_DIR) -> LiveTextModel:
    """Load a model trained by train(), after checking its SHA-256 against the manifest."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    model_dir = Path(model_dir)
    path, manifest_path = model_dir / FILES[kind], model_dir / MANIFEST
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing {path} or {manifest_path}; run `python -m evaluation.ieee5000_d4 --train`")
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(report["features"]) != tuple(FEATURES) or report["threshold"] != THRESHOLD:
        raise RuntimeError("feature order or threshold changed; refusing to load")
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["sha256"][kind]:
        raise RuntimeError(f"{path.name} checksum does not match {MANIFEST}")
    import joblib
    return LiveTextModel(joblib.load(path), THRESHOLD, label=f"{kind}_ieee_text6_v1")


def load_transfer_model(kind: str = "random_forest") -> LiveTextModel:
    """
    The ORIGINAL repo model (trained on the 316-chunk corpus) applied unchanged to the new corpus.
    Useful as a cross-corpus transfer check: nothing about the 5,000-record data was seen in training.
    """
    from evaluation.d4_live_text import load_live_text_model
    model = load_live_text_model(kind)
    model.label = f"{kind}_phase_e_transfer"
    return model


def load_d4(name: str, model_dir: Path = MODEL_DIR):
    """'none' | 'random_forest' | 'xgboost' | 'transfer_random_forest' | 'transfer_xgboost'."""
    if name in ("", "none", None):
        return None
    if name.startswith("transfer_"):
        return load_transfer_model(name[len("transfer_"):])
    return load_ieee_text_model(name, model_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="train and save the six-feature models")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true", help="load saved models and print their held-out metrics")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    args = ap.parse_args(argv)
    if args.train:
        report = train(Path(args.model_dir), overwrite=args.overwrite)
        print(f"saved to {args.model_dir}  (train {report['train_rows']} rows, test {report['test_rows']} rows)")
        return 0
    if args.check:
        report = json.loads((Path(args.model_dir) / MANIFEST).read_text(encoding="utf-8"))
        for kind in KINDS:
            load_ieee_text_model(kind, Path(args.model_dir))
            m = report["metrics"][kind]
            print(f"{kind}: checksum OK  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['fpr']:.3f}")
        print("ablation (F1 by feature set):")
        for set_name, by_kind in report["ablation"].items():
            print("  ", set_name, {k: round(v["f1"], 3) for k, v in by_kind.items()})
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
