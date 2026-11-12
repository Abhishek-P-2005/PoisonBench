# Evaluating on the 5,000-record corpus (`ieee_5000/`)

Akilan's part (Defenses 1-3, evaluation harness, analysis modules, router) brought onto the new corpus.
Nothing under `ieee_5000/`, `scripts/`, `defenses/defense4_*`, `data/`, `corpus/` or `api/` (except one
file, below) is modified. New work lives in `evaluation/` and `tests/`.

## 1. Check it runs on your laptop (no Docker, no Ollama, no model download)

```powershell
# from the repo root, with the project's venv active
$env:PYTHONPATH = "."
python -m evaluation.smoke_ieee5000 --fast      # ~1 min: 400 docs, 12 queries
python -m evaluation.smoke_ieee5000             # ~5 min: the real 5,000 texts, 20 held-out queries
python -m pytest tests/test_ieee5000.py -q      # unit tests (+ the fast smoke)
python -m pytest tests -q --ignore=tests/test_integration_skeleton.py   # whole suite
```
The smoke test builds the four collections in an in-process ChromaDB from the real files in `ieee_5000/`
and embeds with a hash embedder instead of MiniLM. It proves the plumbing, the label-blindness of every
defense and the tier behaviour. **Its numbers are not results.**

`tests/test_integration_skeleton.py` already fails on the untouched repo (`corpus.ingestor` does not exist).
It is not part of this change.

## 2. Run it for real (Chroma + MiniLM up)

```powershell
docker compose up -d chromadb                   # collections already built by scripts/ingest_*_ieee.py
python -m evaluation.run_ieee5000 --preflight   # 4 collections, 5000 / 5100 chunks, 100 poison each
python -m evaluation.ieee5000_d4 --train        # once: six-feature D4 models for this corpus (seconds)
python -m evaluation.run_ieee5000 --max-eval-queries 5 --no-router-bench --tier forged   # quick look
python -m evaluation.run_ieee5000               # everything: 3 tiers, held-out queries
python -m evaluation.run_ieee5000 --d4 xgboost
python -m evaluation.run_ieee5000 --d4 transfer_random_forest   # old 316-chunk model, unchanged (transfer test)
python -m evaluation.run_ieee5000 --generate    # + Phi-3 for the RPI attack (needs Ollama)
```
Outputs go to `data/results/eval_ieee5000/`: `report.md` (read this first), `summary.csv`, and per tier
`<tier>/` with calibration, runs, ordering, interaction, Pareto, router benchmark and per-query outcomes.
Compare two runs: `python -m evaluation.ieee5000_report --compare outA/summary.csv outB/summary.csv`.
Chroma is read from `.env` (`CHROMA_HOST`/`CHROMA_PORT`, default localhost:8001) or `--chroma-host/--chroma-port`.

## 3. What changed and why

| Problem with the 316-chunk setup on the new data | What was added |
|---|---|
| `real_source.py` is hard-wired to 20 queries, `poisonbench_b_*_{1,5,10,20}pct_v1` collections and the Phase-C split | `ieee5000_source.py`: 100 queries, `poisonbench_ieee_5000_*` collections, the repo's 20 held-out targets, one fixed ratio (1.96%) |
| Stored metadata makes D2/D3 look good for the wrong reason (poison is named `controlled_*` and has an empty timestamp) | Explicit provenance tiers: `as_stored` (leaky, exploratory), `spoofed`, `forged` (valid identity: only content defenses can help) |
| The repo's IEEE D4 models use 12 transductive features and cannot score a live chunk | `ieee5000_d4.py`: six-feature text models on the repo's own group-disjoint split, SHA-256 manifest, feature-set ablation |
| No corpus-level reference; D1 is batch-level and was shown to invert | `reference.py`: `d1c` (corpus-level D1) and `ref` (one authoritative text per CVE) |
| Harness measured block rates but not whether the right answer survived | `ieee5000_report.py`: legit-CVE-kept, emptied context, Wilson/bootstrap CIs, AUC per defense, prevalence-adjusted precision |
| D2/D3/ordering/interaction/Pareto/router had never been run on real data | `run_ieee5000.py` runs the whole chain on the real collections |

Modified existing files (3): `evaluation/config.py` (two new stage names), `evaluation/stages.py` (builds them),
`api/pipeline/defended.py` (accepts `ieee_*` D4 kinds and the new stages; skips them cleanly if no reference).

## 4. Read before quoting numbers

* Poison is synthetic, one style per attack. One poison ratio only (1.96%).
* **D4's text classifier mostly learns length.** Poison chunks are 2-4x longer than clean ones. On the repo's
  held-out split: RF F1 0.875, XGBoost 0.911 with six features; `length_only` 0.839 / 0.866; without length
  0.526 / 0.600. Quote the ablation next to any D4 F1.
* D4 trained on 80 of 100 targets: quote the 20 held-out queries only. Thresholds for D1/D2/D3/router are
  calibrated on the other 80.
* `as_stored` D2/D3 results reflect naming artifacts. `forged` is the honest bound for metadata defenses.
* `ref` blocks poison that reuses a known CVE id; it does nothing against an invented id or a compromised
  registry. It is the baseline a classifier has to beat.
* Single seed (collections are static). Generation success exists only for RPI.

## 5. Defended API on the new collections

```powershell
$env:D4_MODEL = "ieee_random_forest"            # or put D4_MODEL=... in .env
# POST /defended/retrieve
# {"query": "...", "collection": "poisonbench_ieee_5000_sm", "stages": ["d1","d2","d3","d4"], "provenance": "derived"}
# add "d1c" / "ref" to stages to use the corpus-reference baselines (reference loaded lazily from poisonbench_ieee_5000_clean)
```
Models are read from `data/results/eval_ieee5000/models/`, which compose already mounts via `./data`.
