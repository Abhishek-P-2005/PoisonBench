"""
Phase 2 deliverable: validates the CLEAN pipeline against a known-answer
query set BEFORE any attack or defense work starts (PRD 6.1). This is what
produces the "baseline is trustworthy" evidence you'll need to show the
panel -- without this, there's no way to later prove a bad result during
poisoning experiments is an attack effect and not a broken pipeline.

Checks two things per question, logged separately (do not conflate them,
per PRD 6.7):
  1. retrieval-level: did the expected doc_id show up in top-k at all?
  2. generation-level: does the LLM's final answer contain the expected
     keywords?

Usage (API must be running -- see README "How to run" section):
    python scripts/validate_baseline.py
    python scripts/validate_baseline.py --api-url http://localhost:8000 --top-k 5
"""
import argparse
import json
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
QA_PATH = REPO_ROOT / "scripts" / "known_answers.json"


def load_qa_set() -> list[dict]:
    with open(QA_PATH) as f:
        data = json.load(f)
    return data["examples"]


def run_one(api_url: str, top_k: int, qa: dict) -> dict:
    resp = requests.post(
        f"{api_url}/generate",
        json={"query": qa["question"], "top_k": top_k},
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()

    retrieved_doc_ids = [r["doc_id"] for r in result["retrieved"]]
    retrieval_hit = qa["expected_doc_id"] in retrieved_doc_ids

    answer_lower = result["answer"].lower()
    keyword_hits = [kw for kw in qa["expected_keywords"] if kw.lower() in answer_lower]
    generation_hit = len(keyword_hits) > 0

    return {
        "id": qa["id"],
        "question": qa["question"],
        "expected_doc_id": qa["expected_doc_id"],
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieval_hit": retrieval_hit,
        "answer": result["answer"],
        "keyword_hits": keyword_hits,
        "generation_hit": generation_hit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=str(REPO_ROOT / "results_baseline_validation.json"))
    args = parser.parse_args()

    qa_set = load_qa_set()
    if len(qa_set) < 5:
        print(f"WARNING: only {len(qa_set)} known-answer questions defined. "
              f"Add more to scripts/known_answers.json before trusting this number.")

    results = []
    for qa in qa_set:
        print(f"Running {qa['id']}: {qa['question'][:70]}...")
        try:
            r = run_one(args.api_url, args.top_k, qa)
        except Exception as e:
            r = {"id": qa["id"], "question": qa["question"], "error": str(e),
                 "retrieval_hit": False, "generation_hit": False}
        results.append(r)
        status = "OK" if r.get("generation_hit") else "MISS"
        print(f"  retrieval_hit={r.get('retrieval_hit')}  generation_hit={r.get('generation_hit')}  [{status}]")

    n = len(results)
    retrieval_acc = sum(1 for r in results if r.get("retrieval_hit")) / n if n else 0
    generation_acc = sum(1 for r in results if r.get("generation_hit")) / n if n else 0

    print("\n" + "=" * 60)
    print(f"Baseline retrieval accuracy : {retrieval_acc:.2%} ({sum(1 for r in results if r.get('retrieval_hit'))}/{n})")
    print(f"Baseline generation accuracy: {generation_acc:.2%} ({sum(1 for r in results if r.get('generation_hit'))}/{n})")
    print("=" * 60)

    with open(args.out, "w") as f:
        json.dump({
            "retrieval_accuracy": retrieval_acc,
            "generation_accuracy": generation_acc,
            "results": results,
        }, f, indent=2)
    print(f"\nFull results written to {args.out}")

    if generation_acc < 0.7:
        print("\nACCURACY BELOW 70% -- do not proceed to attack/defense work yet.")
        print("Debug the baseline pipeline first (chunking, retrieval top-k, prompt template).")
        sys.exit(1)


if __name__ == "__main__":
    main()
