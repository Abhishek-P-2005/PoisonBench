"""
evaluation/runner.py
--------------------
The sweep: attack type x poisoning ratio x seed x defense configuration.

  run_sweep   runs every combination, logs per-query results, per-stage
              latency and drops, and clean-chunk false positives.
  score_pass  runs each defense on its own on the raw retrieval batch and
              logs the continuous score of every chunk. This is the data
              for Module 1 (ROC / PR threshold calibration) and Module 3
              (pairwise defense agreement).

Clean baseline runs (attack "none", ratio 0) are always included, because
false-positive rates cannot be recomputed later without re-running
everything (PRD v1 Section 5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from optimization_layer import RiskScorer
from evaluation.classifier import build_feature_rows
from evaluation.config import CLEAN_ATTACK, CONTEXT_SIZE, N_RETRIEVE
from evaluation.metrics import aggregate_queries, evaluate_query
from evaluation.pipeline import DefensePipeline, order_by_similarity, strip_labels
from evaluation.config import make_config
from evaluation.stages import D1Stage, D2Stage, D3Stage, D4Stage, build_stages


def run_id_for(group: str, config_name: str, attack: str, ratio: float, seed: int) -> str:
    return f"{group}|{config_name}|{attack}|{ratio:.3f}|{seed}"


def truth_map(candidates: list[dict]) -> dict:
    truth = {}
    for chunk in candidates:
        truth[chunk["chunk_id"]] = bool(chunk["metadata"].get("is_poisoned", False))
    return truth


@dataclass
class SweepResult:
    runs:            pd.DataFrame = field(default_factory=pd.DataFrame)
    queries:         pd.DataFrame = field(default_factory=pd.DataFrame)
    stages:          pd.DataFrame = field(default_factory=pd.DataFrame)
    false_positives: pd.DataFrame = field(default_factory=pd.DataFrame)
    chunks:          pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped:         list = field(default_factory=list)   # configs the source cannot support

    def merge(self, other: "SweepResult") -> "SweepResult":
        return SweepResult(
            runs=pd.concat([self.runs, other.runs], ignore_index=True),
            queries=pd.concat([self.queries, other.queries], ignore_index=True),
            stages=pd.concat([self.stages, other.stages], ignore_index=True),
            false_positives=pd.concat([self.false_positives, other.false_positives], ignore_index=True),
            chunks=pd.concat([self.chunks, other.chunks], ignore_index=True),
            skipped=[*self.skipped, *other.skipped],
        )


def load_sweep_result(store, group: str | None = None) -> SweepResult:
    """Rebuild a SweepResult from the database (analysis can run from logs alone)."""
    if group is None:
        runs = store.fetch("eval_runs")
    else:
        runs = store.fetch("eval_runs", "group_name = " + store.placeholder, (group,))
    run_ids = set(runs["run_id"])

    def pick(table: str) -> pd.DataFrame:
        frame = store.fetch(table)
        return frame[frame["run_id"].isin(run_ids)].reset_index(drop=True)

    return SweepResult(
        runs=runs,
        queries=pick("eval_query_results"),
        stages=pick("eval_stage_logs"),
        false_positives=pick("eval_false_positives"),
        chunks=pick("eval_chunk_scores"),
    )


def _scenarios(attacks, ratios, include_clean_baseline: bool) -> list:
    scenarios = []
    if include_clean_baseline:
        scenarios.append((CLEAN_ATTACK, 0.0))
    for attack in attacks:
        for ratio in ratios:
            scenarios.append((attack, ratio))
    return scenarios


def run_one(
    scenario_queries: list,
    config,
    attack: str,
    ratio: float,
    seed: int,
    group: str,
    source_name: str,
    d4_model=None,
    generator=None,
    context_size: int = CONTEXT_SIZE,
) -> dict:
    """Run one DefenseConfig on one scenario. Returns the rows to log."""
    stages = build_stages(config, d4_model=d4_model)
    pipeline = DefensePipeline(stages, context_size=context_size, name=config.name)
    run_id = run_id_for(group, config.name, attack, ratio, seed)

    # Warm-up call (not logged) so first-call overhead does not distort latency.
    if scenario_queries:
        pipeline.run(scenario_queries[0].candidates)

    query_rows, stage_rows, fp_rows = [], [], []
    for scenario_query in scenario_queries:
        result = pipeline.run(scenario_query.candidates)
        truth = truth_map(scenario_query.candidates)
        q_row, s_rows, f_rows = evaluate_query(
            scenario_query, result, context_size, truth, generator=generator,
        )
        query_rows.append(q_row)
        stage_rows.extend(s_rows)
        fp_rows.extend(f_rows)

    run_row = {
        "run_id": run_id,
        "group_name": group,
        "config_name": config.name,
        "config_json": config.to_json(),
        "attack_type": attack,
        "poisoning_ratio": ratio,
        "seed": seed,
        "source_name": source_name,
        "context_size": context_size,
        "n_retrieve": len(scenario_queries[0].candidates) if scenario_queries else 0,
        "generation_enabled": 1 if generator is not None else 0,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    run_row.update(aggregate_queries(query_rows, stage_rows))

    for row in query_rows:
        row["run_id"] = run_id
    for row in stage_rows:
        row["run_id"] = run_id
    for row in fp_rows:
        row["run_id"] = run_id
    return {"run": run_row, "queries": query_rows, "stages": stage_rows, "fps": fp_rows}


def run_sweep(
    source,
    configs: list,
    attacks,
    ratios,
    seeds,
    group: str = "main",
    store=None,
    generator=None,
    d4_model=None,
    context_size: int = CONTEXT_SIZE,
    include_clean_baseline: bool = True,
    verbose: bool = False,
) -> SweepResult:
    runs, queries, stages, fps, skipped = [], [], [], [], []
    started = time.perf_counter()
    scenarios = _scenarios(attacks, ratios, include_clean_baseline)
    total = len(scenarios) * len(seeds) * len(configs)
    done = 0

    for seed in seeds:
        for attack, ratio in scenarios:
            scenario_queries = source.iter_queries(attack, ratio, seed)
            for config in configs:
                supports = getattr(source, "supports", None)
                if supports is not None and not supports(config):
                    skipped.append({"config_name": config.name, "attack_type": attack,
                                    "poisoning_ratio": ratio, "reason": f"source {source.name} lacks the "
                                    "provenance metadata this config needs (see evaluation/metadata_bridge.py)"})
                    done += 1
                    continue
                logged = run_one(
                    scenario_queries, config, attack, ratio, seed, group,
                    source.name, d4_model=d4_model, generator=generator,
                    context_size=context_size,
                )
                if store is not None:
                    store.delete_run(logged["run"]["run_id"])
                    store.insert("eval_runs", [logged["run"]])
                    store.insert("eval_query_results", logged["queries"])
                    store.insert("eval_stage_logs", logged["stages"])
                    store.insert("eval_false_positives", logged["fps"])
                runs.append(logged["run"])
                queries.extend(logged["queries"])
                stages.extend(logged["stages"])
                fps.extend(logged["fps"])
                done += 1
            if verbose:
                elapsed = time.perf_counter() - started
                print(f"  [{group}] seed={seed} {attack}@{ratio:.2f}  {done}/{total} runs  {elapsed:.0f}s")

    return SweepResult(
        runs=pd.DataFrame(runs),
        queries=pd.DataFrame(queries),
        stages=pd.DataFrame(stages),
        false_positives=pd.DataFrame(fps),
        skipped=skipped,
    )


def score_pass(
    source,
    attacks,
    ratios,
    seeds,
    group: str = "scores",
    store=None,
    d4_model=None,
    include_clean_baseline: bool = True,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    Run each defense alone on every raw retrieval batch and log one row per
    chunk with its continuous suspicion score from each defense. Flags use
    the default thresholds.

    The Module 5 risk score is computed the way the router sees it in the
    pipeline: on the chunks that survive the cheap stages d1>d2>d3 run with
    `params`. Chunks blocked by those stages never reach the router, so
    their risk_score is left empty.
    """
    chunk_rows = []
    run_rows = []
    risk_scorer = RiskScorer()

    for seed in seeds:
        for attack, ratio in _scenarios(attacks, ratios, include_clean_baseline):
            run_id = f"{group}|scoring|{attack}|{ratio:.3f}|{seed}"
            scenario_chunk_rows = []

            for scenario_query in source.iter_queries(attack, ratio, seed):
                truth = truth_map(scenario_query.candidates)
                batch = order_by_similarity([strip_labels(c) for c in scenario_query.candidates])

                d1 = D1Stage().run(batch)
                d2 = D2Stage().run(batch)
                d3 = D3Stage().run(batch)
                d3_blocked_ids = {c["chunk_id"] for c in d3.blocked}
                d1_blocked_ids = {c["chunk_id"] for c in d1.blocked}

                d4_prob = {}
                if d4_model is not None:
                    d4_prob = D4Stage(d4_model).run(batch).suspicion

                survivors = batch
                for stage in build_stages(make_config(("d1", "d2", "d3"), params)):
                    survivors = stage.run(survivors).passed
                risk_by_id = {}
                for verdict in risk_scorer.score_batch(survivors):
                    risk_by_id[verdict.chunk_id] = verdict.risk_score

                for chunk in batch:
                    chunk_id = chunk["chunk_id"]
                    scenario_chunk_rows.append({
                        "run_id": run_id,
                        "query_id": scenario_query.query_id,
                        "chunk_id": chunk_id,
                        "is_poisoned": 1 if truth[chunk_id] else 0,
                        "attack_type": attack,
                        "poisoning_ratio": ratio,
                        "seed": seed,
                        "similarity": chunk["similarity"],
                        "d1_z": d1.suspicion.get(chunk_id),
                        "d1_flagged": 1 if chunk_id in d1.flagged_ids else 0,
                        "d1_blocked": 1 if chunk_id in d1_blocked_ids else 0,
                        "d2_suspicion": d2.suspicion.get(chunk_id),
                        "d2_flagged": 1 if chunk_id in d2.flagged_ids else 0,
                        "d3_suspicion": d3.suspicion.get(chunk_id),
                        "d3_flagged": 1 if chunk_id in d3.flagged_ids else 0,
                        "d3_blocked": 1 if chunk_id in d3_blocked_ids else 0,
                        "d4_prob": d4_prob.get(chunk_id),
                        "risk_score": risk_by_id.get(chunk_id),
                    })

            run_row = {
                "run_id": run_id, "group_name": group, "config_name": "scoring",
                "config_json": "{}", "attack_type": attack, "poisoning_ratio": ratio,
                "seed": seed, "source_name": source.name,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if store is not None:
                store.delete_run(run_id)
                store.insert("eval_runs", [run_row])
                store.insert("eval_chunk_scores", scenario_chunk_rows)
            run_rows.append(run_row)
            chunk_rows.extend(scenario_chunk_rows)

    return pd.DataFrame(chunk_rows)


def collect_training_rows(source, attacks, ratios, seeds, include_clean_baseline: bool = True):
    """
    Feature rows and labels from raw retrieval batches, for training the
    stand-in Defense 4 model. Use seeds that are NOT used for evaluation.
    """
    rows, labels = [], []
    for seed in seeds:
        for attack, ratio in _scenarios(attacks, ratios, include_clean_baseline):
            for scenario_query in source.iter_queries(attack, ratio, seed):
                truth = truth_map(scenario_query.candidates)
                batch = order_by_similarity([strip_labels(c) for c in scenario_query.candidates])
                for row in build_feature_rows(batch):
                    rows.append(row)
                    labels.append(truth[row["chunk_id"]])
    return rows, labels
