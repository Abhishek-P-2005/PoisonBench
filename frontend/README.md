# PoisonBench demo site

Presentation site for the panel. Order: Subash, Abhishek, Akilan, then Verdict.
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
- `src/queries.js`: per-query top-5 rankings from `data/results/phase_b/*_20pct_top5_v1_results.csv`.
