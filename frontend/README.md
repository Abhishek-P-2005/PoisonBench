# PoisonBench demo site

Presentation site for the panel. Order: Subash, Abhishek, Akilan.
Fully offline. No API, database or LLM needed. All numbers come from the result files in `data/results/`.

## Place it here

    PoisonBench_Zip/
      frontend/        <-- this folder
      api/
      data/
      docker-compose.yml

## Run

    cd frontend
    npm install
    npm run dev          # http://localhost:3000

## Run with Docker Compose

In `docker-compose.yml`, uncomment the `frontend:` block at the bottom of `services:` (it already points to `./frontend`, port 3000), then:

    docker compose up -d --build frontend

## Controls

- Arrow Left / Arrow Right move between pages.
- Deep links work, for example `http://localhost:3000/#/akilan/chain`.

## Update the numbers

- `src/facts.js`: hand-copied metrics (source file named above each block).
- `src/queries.js`: per-query top-5 rankings at 1/5/10/20% from `data/results/phase_b/*_top5_v1_results.csv`, plus Defense 4 keep/remove decisions from `phase_e_defense4_frozen_top5_replay_v1.csv`.
- `src/generation.js`: Attack 3 clean vs poisoned answers from `real_retrieved_prompt_injection_generation_top5_v1_results.csv`.
