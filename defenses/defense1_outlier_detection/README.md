# Defense 1 — Embedding Outlier Detection

**Owner:** Akilan V S (Module 3) · **Status:** Implemented, tested, demo-ready

## What it does

Flags documents whose embedding sits unusually far from the centroid of the
current retrieval batch, measured as a **z-score of cosine distance** —
cheap, fast, and the first stage of the defense pipeline (PRD §6.3).

## Key design decision: leave-one-out scoring

Each chunk is scored against the centroid of **every other** chunk in the
batch, excluding itself. If the centroid were computed over the whole batch
including the candidate being scored, a single anomalous embedding drags
the centroid slightly toward itself and **dilutes its own z-score** — the
outlier partially hides itself just by being counted in its own reference
statistics. Leave-one-out avoids this.

## Key design decision: variance floor

When legitimate documents cluster extremely tightly, the standard deviation
of their distances can be tiny, and dividing by a near-zero std turns
negligible natural variation into a meaningless, huge z-score. `MIN_STD_FLOOR`
in `config.py` prevents this. This matters more for small synthetic test
batches than real 384-dim MiniLM embeddings (which won't cluster this
tightly), but costs nothing to have in production too.

## Documented limitation — say this upfront, don't wait to be asked

This defense is **expected to be weakest against the optimization-based
embedding attack** (PRD Attack 2), because that attack is specifically
designed to maximize similarity to legitimate content — i.e. to NOT be an
outlier. Defense 1 catches naive attacks cheaply; it is not expected to
catch sophisticated ones alone. **This is the argument for the layered
pipeline**, not a gap to be defensive about.

## Files

- `config.py` — thresholds, neighborhood size, variance floor
- `outlier_detector.py` — leave-one-out z-score computation
- `filter.py` — applies detection across a retrieval batch, ChromaDB adapter
- `test_smoke.py` — two cases: catches a naive off-topic attack, documents
  missing a blended-in optimization attack

## Run it

```bash
export PYTHONPATH=.
python -m defenses.defense1_outlier_detection.test_smoke
```

## What to show the panel

Run the smoke test — it deliberately shows both outcomes side by side: the
naive attack gets blocked (z-score off the charts), the optimization attack
passes through untouched (z-score near zero, indistinguishable from
legitimate chunks). Say plainly that the second outcome is expected and
documented, not a defect — then point to `defenses/demo_review3.py`, which
shows Defense 2 and Defense 3 catching exactly this attack afterward.

## Known limitation (be upfront about this)

`SOFT_FLAG_ZSCORE` / `HARD_BLOCK_ZSCORE` in `config.py` are placeholders,
not yet calibrated against real embeddings or labeled attack data — same
"not yet calibrated" status as Defense 2's `RERANK_ALPHA`. Calibration
requires real corpus embeddings and Abhishek's attack module output.
