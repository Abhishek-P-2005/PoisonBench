# PoisonBench — Attack Engineering and RF Defense Handoff

## Scope

This branch contains Phase B poisoning-ratio experiments, Phase C Random Forest versus XGBoost evaluation, and Phase E experimental live-compatible RF defended retrieval.

It does not replace the team's existing API or pipeline orchestrator.

## Attack Engineering

Three attack families:
- Semantic Mimicry (SM)
- Optimization-Based Embedding Attack (OBEA)
- Retrieved Prompt Injection (RPI)

Phase B contains independently constructed 1%, 5%, 10%, and 20% poisoning-ratio datasets for all three attacks.

The existing frozen ChromaDB collections are not stored in Git. They must be prepared separately.

## Defense 4 — Random Forest

Two experimental model configurations exist:

1. Offline 12-feature RF/XGBoost comparison. These features include benchmark-dependent retrieval statistics and are not suitable for arbitrary live-query deployment.
2. Live-compatible six-text-feature RF/XGBoost comparison. The live RF model is the experimental deployment candidate.

Live RF model:
data/results/phase_e/random_forest_live_text_v1.joblib

Live evaluation report:
data/results/phase_e/live_text_models_v1.json

## Experimental Defended Retrieval API

Run from the repository root, with dependencies installed and ChromaDB available:

python -m uvicorn scripts.phase_e_defended_api_v1:app --host 127.0.0.1 --port 8002

Health:
GET http://127.0.0.1:8002/api/v1/defended/health

Retrieval:
POST http://127.0.0.1:8002/api/v1/defended/retrieve

Example JSON request:
{"query":"What is CVE-2026-54500?","attack":"sm"}

Supported attack values: sm, obea, rpi.

The endpoint retrieves from the existing isolated 20% poisoned ChromaDB collections.

## Important Limitations

- Defense 1 runs in advisory mode.
- Defense 2 is not operational because verified trust metadata is missing.
- Defense 3 is not operational because trusted provenance metadata is missing.
- Defense 4 uses the live-compatible six-text-feature RF model.
- No automatic Top-k refill is implemented.
- This endpoint does not call Ollama or generate an LLM answer.
- The frozen 20% ChromaDB collections are required; GitHub does not transfer the running Docker database.
- Phase B6 full LLM-generation evaluation is incomplete and should not be presented as finished.
- The original offline 12-feature evaluation and live six-feature evaluation must not be conflated.

## Team Integration

Subash: integrate the experimental defended retrieval endpoint with the frontend/backend and arrange reproducible ChromaDB initialization.

Akilan: address D2/D3 metadata compatibility and validate those defenses without changing the frozen experimental collections.

The full D1-D4 plus Ollama generation pipeline requires separate end-to-end integration and validation.
