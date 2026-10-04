"""
Phase 3 support script: pulls the full corpus out of ChromaDB, runs the
trigger-query benchmark against it to compute retrieval-score-anomaly
features, and writes a labeled feature CSV that train_xgboost.py (and
Abhishek's Random Forest script) both read.

This only produces a MEANINGFUL training set once the corpus has actually
been poisoned (via Abhishek's attack module) -- on a 100% clean corpus,
is_poisoned is always 0 and there's nothing to learn. Until the attack
module exists, run this against a corpus you've manually seeded with a few
synthetic poisoned documents (see the __main__ block below for a quick
manual seeding helper) just to confirm the feature/training code path works
end-to-end -- do not treat those numbers as real results.

Usage (API/ChromaDB must be running):
    python defense4_xgboost/build_training_data.py --trigger-queries scripts/known_answers.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from api.pipeline.vectorstore import get_all_chunks, query as vs_query  # noqa: E402
from defenses.defense4_xgboost.feature_engineering import build_feature_matrix    # noqa: E402

OUT_DIR = REPO_ROOT / "defenses" / "defense4_xgboost" / "trained_results" / "data"


def load_trigger_questions(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    return [ex["question"] for ex in data["examples"]]


def compute_retrieval_scores_by_chunk(trigger_questions: list[str], top_k: int = 20) -> dict[str, list[float]]:
    scores_by_chunk: dict[str, list[float]] = {}
    for q in trigger_questions:
        results = vs_query(q, top_k=top_k)
        for r in results:
            scores_by_chunk.setdefault(r["chunk_id"], []).append(r["similarity_score"])
    return scores_by_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger-queries", default=str(REPO_ROOT / "scripts" / "known_answers.json"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default=str(OUT_DIR / "features.csv"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Pulling full corpus from ChromaDB...")
    chunks = get_all_chunks()
    print(f"  {len(chunks)} chunks in corpus")

    n_poisoned = sum(1 for c in chunks if c["metadata"].get("is_poisoned"))
    print(f"  {n_poisoned} labeled as poisoned ({n_poisoned / max(len(chunks),1):.1%})")
    if n_poisoned == 0:
        print("  WARNING: 0 poisoned chunks found. Either the attack module hasn't run "
              "yet, or you're pointed at a clean-only corpus. The resulting model will "
              "be meaningless (single-class training data) until this is non-zero.")

    print("Running trigger-query benchmark for retrieval-score features...")
    trigger_questions = load_trigger_questions(Path(args.trigger_queries))
    scores_by_chunk = compute_retrieval_scores_by_chunk(trigger_questions, top_k=args.top_k)

    print("Building feature matrix...")
    df = build_feature_matrix(chunks, scores_by_chunk)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {args.out}")


if __name__ == "__main__":
    main()
