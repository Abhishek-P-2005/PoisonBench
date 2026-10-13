# Evaluation harness

Measures how well Defenses 1-4 (and the Module-5 adaptive router) protect the
RAG pipeline in this repo, with one scored, logged run per
(defense config x attack x poisoning ratio x seed). It is additive: nothing
outside `evaluation/`, `optimization_layer/` and the three `tests/` files was
changed. `optimization_layer/` is included because the harness's router stage and
risk scoring import it.

## What it runs on

| | |
|---|---|
| Synthetic source (`python -m evaluation.run_sweep --quick`) | Self-contained; checks the plumbing. Its numbers are **not results**. |
| Real source (`python -m evaluation.run_real`) | The repo's 20 trigger queries against the Phase-B collections `poisonbench_b_{sm,obea,rpi}_{1,5,10,20}pct_v1` and the clean `poisonbench_rpi_clean_v1`, retrieved with the API's own Chroma client and embedder. |

Attack names: `naive` = sm, `optimized` = obea, `injection` = rpi.

## Commands

```bash
export PYTHONPATH=.                                  # Windows: $env:PYTHONPATH="."
python -m unittest tests.test_evaluation_real -v     # no services needed
python -m evaluation.run_real --preflight            # which collections exist? (needs ChromaDB)
python -m evaluation.run_real --d4 random_forest     # full real sweep
python -m evaluation.run_real --attacks injection --ratios 0.20 --generate   # + LLM (RPI only)
```

Outputs: `eval_*` tables in `--db` (SQLite by default; any Postgres URL works) and
`data/results/eval_real/{runs,queries,summary_by_split}.csv`, `skipped_configs.json`, `report.md`.
Schema: `evaluation/sql/eval_schema.sql`.

## How it fits this pipeline (and the decisions that affect results)

1. **Defense 4** is the repo's live-computable model
   (`data/results/phase_e/random_forest_live_text_v1.joblib`, six text features,
   threshold 0.5), loaded by `evaluation/d4_live_text.py` after checking its
   SHA-256 against `live_text_models_v1.json`. It reproduces the stored held-out
   probabilities exactly (tested). The frozen 12-feature Phase C models need
   benchmark-wide features and are not used.
2. **Provenance (D2/D3).** The stored chunks carry only `doc_id, source,
   timestamp, is_poisoned, position`. Like the Phase E scripts, the harness
   defaults to **not** inventing provenance: on such collections configs with
   D2, D3 or the router are skipped and listed in `skipped_configs.json`. Pass
   `--provenance derived` for an exploratory run (labelled `chroma[derived]`;
   D3's integrity check then cannot fail, and chunks with an empty timestamp
   fail D3's temporal check whether or not they are poisoned). Collections
   ingested with the full `corpus.metadata` schema are detected automatically.
3. **Held-out queries.** D4 was trained on this benchmark, so `summary_by_split.csv`
   reports the five held-out queries (q01, q02, q09, q16, q18, from the Phase C
   split manifest) separately. Quote that split.
4. **Generation success** exists only for the RPI attack
   (`POISONBENCH_RPI_SUCCESS`); it is blank, not 0, for sm/obea.
5. Labels (`is_poisoned`, `attack_type`) are stripped before any defense runs
   and used only to score.
6. Real collections are static, so the real sweep uses a single seed.

## Layout

`config.py` run configs and parameters, `stages.py` D1-D4 + router stage,
`pipeline.py` ordered execution, `metrics.py` per-query metrics,
`runner.py` sweeps and logging, `store.py` SQLite/Postgres, `scenarios.py`
synthetic and Chroma sources, `real_source.py` / `run_real.py` the real benchmark,
`d4_live_text.py` the D4 model, `metadata_bridge.py` provenance handling,
`analysis/` calibration, ordering, interaction and Pareto analysis,
`router_benchmark.py` Module-5 benchmark.
