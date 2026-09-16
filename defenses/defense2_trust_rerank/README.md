# Defense 2 — Trust-Scored Re-ranking

**Owner:** Akilan V S (Module 3) · **Status:** Implemented, tested, demo-ready

## What it does

Re-ranks retrieved documents by blending their raw retrieval similarity with
a **trust score** computed from three signals a real defense would legitimately
have access to:

| Signal | What it checks |
|---|---|
| Source reliability | Is the claimed `source_id` on the trusted allow-list? |
| Document age | Has it had time to be scrutinized / corroborated? |
| Corroboration | Do other verified chunks reference the same `doc_id`? |

```
final_score = ALPHA * normalized_similarity + (1 - ALPHA) * trust_score
```

**Important design constraint:** the scorer never reads `is_poisoned` or
`attack_type` from chunk metadata. Those are ground-truth labels the attack
module sets for evaluation only — using them would be scoring with the
answer key. Trust is computed only from source, timestamps, and corpus
co-occurrence.

## Files

- `config.py` — all weights and thresholds, tunable without touching logic
- `trust_scorer.py` — computes the three-component trust score
- `reranker.py` — blends similarity + trust, re-sorts, flags low-trust chunks
- `calibration.py` — ROC/PR threshold calibration (run once Abhishek's attack
  module provides labeled poisoned/clean data — see PRD §3.3.5)
- `test_smoke.py` — standalone test, no external services required

## Run it

```bash
export PYTHONPATH=.
python -m defenses.defense2_trust_rerank.test_smoke
```

## What to show the panel

1. Run the smoke test live — it prints a before/after ranking table.
2. Point out that the highest-similarity chunk (an attack simulating an
   optimization-based attack) gets demoted below two legitimate chunks
   purely on trust signals.
3. If asked "why is Defense 2 not enough by itself" — see the combined demo
   (`defenses/demo_review3.py`), which shows a case where Defense 2 alone
   is insufficient because the attacker reused a real CVE ID to inherit
   corroboration. That is the argument for layering Defense 3 on top.

## Known limitation (be upfront about this)

`RERANK_ALPHA = 0.7` in `config.py` is a placeholder that favors similarity
over trust. It has not been calibrated against real labeled data yet — that
requires poisoned/clean pairs from Abhishek's attack module, which is why
`calibration.py` exists but isn't run yet. This is an honest "not yet
calibrated" status, not a hidden gap — say so if asked.
