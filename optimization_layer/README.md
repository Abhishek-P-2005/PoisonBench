# Optimization layer (Module 5): Adaptive Risk-Based Defense Routing

**Owner:** Akilan V S · **PRD:** v2.0, Module 5 (gated stretch goal), Reqs 5.1 to 5.4

**This is an optimization layer over the defense pipeline, not a fifth defense.** It never detects or blocks anything itself: every block still comes from Defenses 1 to 4. It only decides which chunks need the expensive Defense 4 check.

## What it does

Decides, per chunk, whether it needs the heavy inspection stage (the supervised
ML classifier, Defense 4) or can go straight to prompt-context construction.

```
retrieved chunks -> Defense 1 -> Defense 2 -> Defense 3 -> ROUTER -> Defense 4 (high risk only) -> context
                                                            |
                                          risk = f(provenance, centroid distance)
                                          risk <  threshold : fast path, skip Defense 4   (Req 5.2)
                                          risk >= threshold : Defense 4 inspects it        (Req 5.3)
```

| Req | Where |
|---|---|
| 5.1 fast risk score from provenance metadata + centroid distance | `risk_scorer.py` |
| 5.2 low-risk chunks bypass heavy inspection | `router.py::apply_adaptive_routing` |
| 5.3 high-risk chunks go to the full pipeline | `router.py` |
| 5.4 benchmark dynamic vs static | `evaluation/router_benchmark.py` |

## Design decisions

- **Signals are reused, not recomputed.** Inside the pipeline Defense 1 and 3 have
  already annotated every chunk (`outlier_score`, `provenance_score`); the router
  reads those. If they are missing it computes them with the same Defense 1 / 3 code.
- **Noisy-OR combination.** `risk = 1 - (1 - provenance_risk)(1 - centroid_risk)`: a
  chunk is risky if either signal fires. A weighted mean is available (`RISK_METHOD`).
- **Unknown is not safe.** A missing embedding, or a batch too small for Defense 1,
  gives the centroid signal the value `UNAVAILABLE_SIGNAL_RISK` (0.5), never 0.
- **The router is a pure gate.** The heavy inspector receives the full batch and a set
  of ids to inspect, so batch-level features are identical to the static pipeline.
  Consequence, tested: the router can never block a chunk the static pipeline would not.
- **No ground truth.** Nothing here reads `is_poisoned` or `attack_type`.

## Honest limitation (say it before the panel asks)

The router can only skip a chunk when the cheap signals say it is clean. A poisoned
chunk with valid-looking provenance whose embedding sits inside the legitimate cluster
(the optimization-based attack with a compromised store) scores risk ~0 and bypasses
Defense 4. The router trades detection of those chunks for latency. Whether that trade
is acceptable is an empirical question: `evaluation/router_benchmark.py` gives the
threshold sweep and the sensitivity to the share of clean-looking poisoned chunks.

## Run it

```bash
export PYTHONPATH=.
python -m optimization_layer.test_smoke
pytest tests/test_adaptive_routing.py -v
```

`RISK_THRESHOLD` in `config.py` (0.25) is a placeholder. Calibrate it with
`evaluation.router_benchmark.calibrate_from_scores`, which picks the highest threshold
that still sends at least 95% of poisoned chunks to the heavy path, and reports
`attained=False` (with threshold 0, everything inspected) when that is impossible.
