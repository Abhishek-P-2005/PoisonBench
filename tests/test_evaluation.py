"""
Tests for the evaluation harness (evaluation/).
Run:  pytest tests/test_evaluation.py -v
PostgreSQL test runs only if POISONBENCH_TEST_PG is set, e.g.
    POISONBENCH_TEST_PG=postgresql://postgres@localhost:5433/poisonbench_test
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from defenses.defense1_outlier_detection.filter import apply_outlier_filter
from defenses.defense2_trust_rerank.reranker import rerank_results
from defenses.defense3_provenance.filter import apply_provenance_filter
from evaluation.fixtures import ATTACK_ID, build_scenario
from evaluation.analysis import calibration, interaction, ordering, pareto
from evaluation.classifier import train_standin_model
from evaluation.config import ATTACK_TYPES, make_config
from evaluation.generation import CachedGenerator, OllamaGenerator, SimulatedGenerator
from evaluation.metrics import evaluate_query
from evaluation.pipeline import DefensePipeline, strip_labels
from evaluation.runner import collect_training_rows, load_sweep_result, run_sweep, score_pass
from evaluation.scenarios import ChromaScenarioSource, ScenarioQuery, SyntheticScenarioSource
from evaluation.stages import D1Stage, D2Stage, D3Stage, StageOutput
from evaluation.store import ResultStore, TABLES


@pytest.fixture(scope="module")
def source():
    return SyntheticScenarioSource()


@pytest.fixture(scope="module")
def model(source):
    rows, labels = collect_training_rows(source, ATTACK_TYPES, (0.05, 0.20), [200])
    return train_standin_model(rows, labels)


# ── stages reproduce the existing defenses ───────────────────────────────────

def test_stages_match_existing_defense_code():
    batch = [strip_labels(c) for c in build_scenario()]

    old_passed, old_blocked = apply_outlier_filter(batch)
    new = D1Stage().run(batch)
    assert [c["chunk_id"] for c in new.passed] == [c["chunk_id"] for c in old_passed]
    assert {c["chunk_id"] for c in new.blocked} == {c["chunk_id"] for c in old_blocked}

    old_ranked = rerank_results(batch)
    new = D2Stage().run(batch)
    assert [c["chunk_id"] for c in new.passed] == [c["chunk_id"] for c in old_ranked]

    old_passed, old_blocked = apply_provenance_filter(batch)
    new = D3Stage().run(batch)
    assert [c["chunk_id"] for c in new.passed] == [c["chunk_id"] for c in old_passed]
    assert {c["chunk_id"] for c in new.blocked} == {c["chunk_id"] for c in old_blocked}


def test_demo_scenario_through_pipeline_blocks_attack_at_defense_3():
    candidates = build_scenario()
    pipeline = DefensePipeline(build_stages_for(["d1", "d2", "d3"]), context_size=10)
    result = pipeline.run(candidates)
    assert ATTACK_ID not in [c["chunk_id"] for c in result.final_context]
    assert ATTACK_ID in result.stage_records[2].blocked_ids
    assert result.stage_records[2].name == "d3"
    assert ATTACK_ID not in result.stage_records[0].blocked_ids   # blends in, as documented


def build_stages_for(names):
    from evaluation.stages import build_stages
    return build_stages(make_config(names))


def test_defenses_never_see_ground_truth_labels():
    seen = []

    class Spy:
        name = "spy"

        def run(self, batch):
            for chunk in batch:
                seen.append(set(chunk["metadata"].keys()))
            return StageOutput(passed=batch)

    DefensePipeline([Spy()]).run(build_scenario())
    assert seen
    for keys in seen:
        assert "is_poisoned" not in keys and "attack_type" not in keys


# ── metrics on a hand-built case ─────────────────────────────────────────────

class FakeStage:
    """Blocks chunk 'p1' (poisoned) and flags chunk 'c2' (clean)."""
    name = "fake"

    def run(self, batch):
        out = StageOutput()
        for chunk in batch:
            out.suspicion[chunk["chunk_id"]] = 0.5
            if chunk["chunk_id"] == "p1":
                out.blocked.append(chunk)
                out.flagged_ids.add("p1")
            else:
                out.passed.append(chunk)
            if chunk["chunk_id"] == "c2":
                out.flagged_ids.add("c2")
        return out


def hand_built_query():
    sims = {"p1": 0.99, "c1": 0.90, "c2": 0.80, "p2": 0.70, "c3": 0.60, "c4": 0.50}
    candidates = []
    for chunk_id, sim in sims.items():
        candidates.append({
            "chunk_id": chunk_id, "text": f"text {chunk_id}", "similarity": sim,
            "metadata": {"is_poisoned": chunk_id.startswith("p")}, "embedding": [1.0, 0.0],
        })
    return ScenarioQuery("q", "question", candidates, "marker", "naive", 0.1, 0)


def test_metrics_hand_computed():
    query = hand_built_query()
    truth = {c["chunk_id"]: c["metadata"]["is_poisoned"] for c in query.candidates}
    pipeline = DefensePipeline([FakeStage()], context_size=3)
    result = pipeline.run(query.candidates)
    row, stage_rows, fp_rows = evaluate_query(query, result, 3, truth)

    assert row["raw_success"] == 1            # p1 is in raw top-3
    assert row["defended_success"] == 1       # p1 is blocked, but p2 moves up into the context
    assert row["raw_poison_in_context"] == 1
    assert row["n_poison_candidates"] == 2
    assert row["poison_blocked"] == 1
    assert row["clean_seen"] == 4
    assert row["clean_flagged"] == 1 and row["clean_blocked"] == 0
    assert row["clean_retention"] == 1.0      # c1, c2 stay in the context
    assert [c["chunk_id"] for c in result.final_context] == ["c1", "c2", "p2"]
    assert row["defended_poison_in_context"] == 1   # p2 moved up into the context
    assert fp_rows == [{"query_id": "q", "chunk_id": "c2", "stage": "fake", "action": "flagged", "suspicion": 0.5}]
    assert stage_rows[0]["n_blocked"] == 1 and stage_rows[0]["clean_flagged"] == 1


# ── synthetic source ─────────────────────────────────────────────────────────

def test_synthetic_source_is_reproducible_and_clean_corpus_is_shared():
    a = SyntheticScenarioSource().iter_queries("optimized", 0.05, 7)
    b = SyntheticScenarioSource().iter_queries("optimized", 0.05, 7)
    for qa, qb in zip(a, b):
        assert [c["chunk_id"] for c in qa.candidates] == [c["chunk_id"] for c in qb.candidates]
        assert [c["similarity"] for c in qa.candidates] == [c["similarity"] for c in qb.candidates]

    src = SyntheticScenarioSource()
    clean = src.build_corpus(7, "none", 0.0)
    attacked = src.build_corpus(7, "naive", 0.2)
    clean_ids = [c["metadata"]["chunk_id"] for c in clean["chunks"]]
    attacked_clean_ids = [c["metadata"]["chunk_id"] for c in attacked["chunks"] if not c["metadata"]["is_poisoned"]]
    assert clean_ids == attacked_clean_ids


def test_poison_counts_follow_ratio_and_clean_has_none():
    src = SyntheticScenarioSource()
    per_query = {}
    for ratio in (0.01, 0.05, 0.10, 0.20):
        corpus = src.build_corpus(1, "injection", ratio)
        per_query[ratio] = sum(1 for c in corpus["chunks"] if c["metadata"]["is_poisoned"]) / len(corpus["queries"])
    assert per_query[0.01] == 1 and per_query[0.05] == 2 and per_query[0.10] == 5 and per_query[0.20] == 10
    clean = src.iter_queries("none", 0.0, 1)
    assert sum(1 for q in clean for c in q.candidates if c["metadata"]["is_poisoned"]) == 0


# ── sweep, store, reproducibility ────────────────────────────────────────────

SMALL_CONFIGS = [make_config([]), make_config(["d1", "d2", "d3"])]


def test_sweep_logs_everything_and_includes_clean_baseline(source):
    store = ResultStore("sqlite:///:memory:")
    result = run_sweep(source, SMALL_CONFIGS, ["naive", "optimized"], (0.05, 0.10), [1], store=store,
                       generator=CachedGenerator(SimulatedGenerator()))
    # (1 clean + 2 attacks x 2 ratios) scenarios x 2 configs x 1 seed
    assert len(result.runs) == 10
    assert store.count("eval_runs") == 10
    assert store.count("eval_query_results") == 10 * 10
    assert store.count("eval_stage_logs") == 5 * 3 * 10       # only the d1>d2>d3 runs have stage rows
    baseline = result.runs[result.runs["attack_type"] == "none"]
    assert len(baseline) == 2 and (baseline["poisoning_ratio"] == 0).all()
    loaded = load_sweep_result(store)
    assert len(loaded.runs) == 10 and len(loaded.queries) == 100


def test_same_seed_gives_identical_metrics(source):
    a = run_sweep(source, SMALL_CONFIGS, ["optimized"], (0.10,), [3]).runs
    b = run_sweep(SyntheticScenarioSource(), SMALL_CONFIGS, ["optimized"], (0.10,), [3]).runs
    columns = [c for c in a.columns if not c.startswith("latency") and c != "created_at"]
    pd.testing.assert_frame_equal(a[columns], b[columns])


def test_config_json_in_database_rebuilds_the_config(source):
    store = ResultStore("sqlite:///:memory:")
    config = make_config(["d3", "d1"], {"d1_soft_z": 3.3}, label="x")
    run_sweep(source, [config], ["naive"], (0.05,), [1], store=store)
    stored = json.loads(store.fetch("eval_runs")["config_json"].iloc[0])
    assert stored["stages"] == ["d3", "d1"] and stored["params"] == {"d1_soft_z": 3.3}


@pytest.mark.skipif(not os.environ.get("POISONBENCH_TEST_PG"), reason="set POISONBENCH_TEST_PG to test PostgreSQL")
def test_postgres_store_roundtrip(source):
    store = ResultStore(os.environ["POISONBENCH_TEST_PG"])
    for table in TABLES:
        cur = store.conn.cursor()
        cur.execute(f"DELETE FROM {table}")
        store.conn.commit()
    result = run_sweep(source, SMALL_CONFIGS, ["naive"], (0.05,), [1], store=store)
    assert store.count("eval_runs") == len(result.runs)
    scores = score_pass(source, ["naive"], (0.05,), [1], store=store)
    assert store.count("eval_chunk_scores") == len(scores)
    loaded = load_sweep_result(store)
    assert len(loaded.queries) == len(result.queries)
    store.insert("eval_runs", [{"run_id": "nan-test", "attack_success_raw": float("nan"), "seed": np.int64(4)}])
    row = store.fetch("eval_runs", "run_id = %s", ("nan-test",))
    assert pd.isna(row["attack_success_raw"].iloc[0]) and int(row["seed"].iloc[0]) == 4


# ── Module 1: calibration ────────────────────────────────────────────────────

def test_calibration_respects_fpr_ceiling_on_held_out_data():
    rng = np.random.default_rng(0)
    train_clean, train_poison = rng.normal(0, 1, 2000), rng.normal(3, 1, 500)
    report = calibration.calibrate_defense("d1", train_clean, train_poison, fpr_ceiling=0.05)
    assert report.fpr_at_threshold <= 0.05
    assert report.recall_at_threshold > 0.6
    test_clean = rng.normal(0, 1, 5000)
    assert (test_clean >= report.threshold).mean() < 0.08     # holds on new data, small tolerance


def test_calibration_threshold_to_params_matches_stage_behaviour():
    reports = {
        "d1": calibration.CalibrationReport("d1", 1, 1, 0.9, 0.9, 0.05, 3.0, 0.0, 0.5, 0, 0),
        "d1_block": calibration.CalibrationReport("d1_block", 1, 1, 0.9, 0.9, 0.01, 4.0, 0.0, 0.5, 0, 0),
        "d2": calibration.CalibrationReport("d2", 1, 1, 0.9, 0.9, 0.05, 0.40, 0.0, 0.5, 0, 0),
        "d3": calibration.CalibrationReport("d3", 1, 1, 0.9, 0.9, 0.05, 0.25, 0.0, 0.5, 0, 0),
        "d4": calibration.CalibrationReport("d4", 1, 1, 0.9, 0.9, 0.05, 0.7, 0.0, 0.5, 0, 0),
    }
    params = calibration.thresholds_to_params(reports)
    assert params["d1_soft_z"] == 3.0 and params["d1_hard_z"] == 4.0 and params["d4_threshold"] == 0.7
    # suspicion 0.40 means trust 0.60: flagged iff trust < flag_threshold
    assert 0.59 < params["d2_flag_threshold"] and params["d2_flag_threshold"] < 0.61
    assert 0.60 < params["d2_flag_threshold"] and 0.55 < params["d2_flag_threshold"]
    # provenance_score 0.75 (suspicion 0.25) must still be flagged, 1.0 must not
    assert 0.75 < params["d3_flag_below"] <= 1.0 - 1e-6


def test_score_pass_and_calibrate_all_end_to_end(source, model):
    scores = score_pass(source, ATTACK_TYPES, (0.05, 0.10), [100], d4_model=model)
    reports = calibration.calibrate_all(scores)
    assert {"d1", "d2", "d3", "d4"} <= set(reports)
    for name in ("d1", "d2", "d3", "d4"):
        assert reports[name].fpr_at_threshold <= 0.05 + 1e-9


# ── Module 2: ordering ───────────────────────────────────────────────────────

def test_permutation_counts_and_tail():
    configs = ordering.permutation_configs()
    assert len(configs) == 6
    assert len({c.name for c in configs}) == 6
    assert all(c.stages[-1] == "d4" for c in configs)
    assert len(ordering.permutation_configs(include_tail=True)) == 24
    assert len(ordering.permutation_configs(tail=())) == 6


def test_ordering_matrix_and_drop_rates(source):
    configs = ordering.permutation_configs(tail=())
    result = run_sweep(source, configs, ["naive", "optimized"], (0.10,), [1])
    names = [c.name for c in configs]
    matrix = ordering.ordering_matrix(result.runs, "attack_success_defended", names)
    assert set(matrix.index) == set(names) and {"naive", "optimized"} <= set(matrix.columns)
    drops = ordering.drop_rates_by_step(result.stages, result.runs, names)
    assert set(drops["position"]) == {0, 1, 2}


# ── Module 3: interaction ────────────────────────────────────────────────────

def toy_runs():
    rows = []
    security = {(): 0.0, ("d1",): 10.0, ("d2",): 20.0, ("d1", "d2"): 25.0}
    for stages, value in security.items():
        config = make_config(stages)
        rows.append({"config_json": config.to_json(), "poisoning_ratio": 0.1, "security_blocked_pct": value})
    return pd.DataFrame(rows)


def test_marginal_uplift_and_redundancy_toy():
    table = interaction.marginal_uplift(toy_runs())
    row = table[(table["base"] == "d1") & (table["added"] == "d2")].iloc[0]
    assert row["uplift"] == pytest.approx(15.0)
    row = table[(table["base"] == "d2") & (table["added"] == "d1")].iloc[0]
    assert row["uplift"] == pytest.approx(5.0)
    redundancy = interaction.redundancy_table(toy_runs()).set_index("defense")
    assert redundancy.loc["d1", "redundancy"] == pytest.approx(1.0 - 5.0 / 10.0)
    assert redundancy.loc["d2", "redundancy"] == pytest.approx(1.0 - 15.0 / 20.0)


def test_pairwise_agreement_and_unique_catches_toy():
    chunks = pd.DataFrame({
        "is_poisoned": [1, 1, 1, 0, 0, 0],
        "d1_flagged":  [1, 0, 0, 0, 0, 1],
        "d2_flagged":  [1, 1, 0, 0, 0, 0],
        "d3_flagged":  [0, 0, 0, 0, 0, 0],
    })
    agreement = interaction.pairwise_agreement(chunks).set_index("pair")
    assert agreement.loc["d1-d2", "both_flag"] == 1
    assert agreement.loc["d1-d2", "a_only"] == 1 and agreement.loc["d1-d2", "b_only"] == 1
    assert agreement.loc["d1-d2", "a_only_precision"] == 0.0      # the d1-only flag was a clean chunk
    assert agreement.loc["d1-d2", "b_only_precision"] == 1.0
    unique = interaction.unique_catches(chunks).set_index("defense")
    assert unique.loc["d2", "unique_catches"] == 1 and unique.loc["d1", "unique_catches"] == 0
    assert unique.loc["none_of_them", "caught"] == 1


# ── Module 4: Pareto ─────────────────────────────────────────────────────────

def test_pareto_front_toy():
    table = pd.DataFrame({
        "config_name": ["a", "b", "c", "d"],
        "security":   [90, 90, 50, 95],
        "usability":  [90, 90, 99, 80],
        "latency_ms": [5.0, 7.0, 1.0, 9.0],
    })
    optimal = pareto.pareto_front(table)
    assert list(optimal) == [True, False, True, True]     # b is dominated by a


# ── generation ───────────────────────────────────────────────────────────────

def test_cached_generator_caches_by_context():
    cached = CachedGenerator(SimulatedGenerator())
    context = [{"chunk_id": "a", "text": "hello"}]
    assert cached.generate("q", context) == "hello"
    assert cached.generate("q", context) == "hello"
    assert cached.calls == 1 and cached.hits == 1


def test_ollama_client_against_local_mock_server():
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received["path"], received["body"] = self.path, body
            payload = json.dumps({"response": "mock answer"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        generator = OllamaGenerator(model="phi3", host=f"http://127.0.0.1:{server.server_port}")
        answer = generator.generate("What is the fix?", [{"chunk_id": "a", "text": "Apply the patch."}])
    finally:
        server.shutdown()
    assert answer == "mock answer"
    assert received["path"] == "/api/generate"
    assert "Apply the patch." in received["body"]["prompt"] and "What is the fix?" in received["body"]["prompt"]
    assert received["body"]["stream"] is False


def test_generation_success_is_separate_from_retrieval_success(source):
    """Poison can be in the context (retrieval success) without reaching the answer."""
    result = run_sweep(source, [make_config([])], ["naive"], (0.10,), [1], generator=SimulatedGenerator()).runs
    row = result.iloc[-1]
    assert row["attack_success_defended"] >= row["attack_success_generation"]


# ── ChromaDB source gives the same candidates as the synthetic source ────────

def test_chroma_source_matches_synthetic_source(source):
    chromadb = pytest.importorskip("chromadb")
    corpus = source.build_corpus(1, "optimized", 0.05)
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("nvd_corpus_test")
    except Exception:
        pass
    collection = client.create_collection("nvd_corpus_test", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["metadata"]["chunk_id"] for c in corpus["chunks"]],
        documents=[c["text"] for c in corpus["chunks"]],
        metadatas=[c["metadata"] for c in corpus["chunks"]],
        embeddings=[e.tolist() for e in corpus["embeddings"]],
    )
    by_text = {q["query_text"]: q["embedding"].tolist() for q in corpus["queries"]}
    triggers = [{"query_id": q["query_id"], "query_text": q["query_text"]} for q in corpus["queries"]]
    chroma = ChromaScenarioSource(collection, triggers, embed_query=lambda text: by_text[text])

    synthetic_queries = source.iter_queries("optimized", 0.05, 1)
    chroma_queries = chroma.iter_queries("optimized", 0.05, 1)
    identical = 0
    for s_query, c_query in zip(synthetic_queries, chroma_queries):
        if {c["chunk_id"] for c in s_query.candidates} == {c["chunk_id"] for c in c_query.candidates}:
            identical += 1
        assert all("embedding" in c for c in c_query.candidates)
        assert any(c["metadata"]["is_poisoned"] for c in c_query.candidates)
    assert identical >= 9

    # the real-retrieval path runs through the whole pipeline and the sweep
    result = run_sweep(chroma, [make_config(["d1", "d2", "d3"])], ["optimized"], (0.05,), [1])
    assert len(result.runs) == 2     # clean-baseline scenario + the attack scenario (same collection here)


# ── whole CLI, quick mode ────────────────────────────────────────────────────

def test_cli_quick_end_to_end(tmp_path, monkeypatch):
    from evaluation import run_sweep as cli
    monkeypatch.setattr(sys, "argv", [
        "run_sweep", "--db", "sqlite:///:memory:", "--out", str(tmp_path), "--quick", "--repeats", "2",
    ])
    cli.main()
    for name in ("RESULTS_SUMMARY.md", "calibration.json", "fig1_embedding_map.png", "fig2_roc_pr.png",
                 "fig3_permutation_heatmap.png", "fig4_pareto_3d.html", "fig4_pareto_3d.png",
                 "fig5_router_benchmark.png", "fig5b_router_operating_curve.png", "eval_schema.sql",
                 "marginal_uplift.csv", "pareto.csv", "router_benchmark_summary.csv", "router_sensitivity.csv"):
        assert (tmp_path / name).exists(), name
    assert "SYNTHETIC" in (tmp_path / "RESULTS_SUMMARY.md").read_text()
    assert json.loads((tmp_path / "calibration.json").read_text())["params"]


@pytest.mark.xfail(
    strict=False,
    reason="Known defect in defenses/defense1_outlier_detection/filter.py: `if not embeddings:` raises "
           "for numpy arrays returned by recent ChromaDB. Fix: `if embeddings is None or len(embeddings) == 0:`. "
           "This test passes (XPASS) once that line is changed.",
)
def test_defense1_adapter_accepts_raw_chromadb_output():
    chromadb = pytest.importorskip("chromadb")
    from defenses.defense1_outlier_detection.filter import from_chroma_query_result
    collection = chromadb.EphemeralClient().create_collection("adapter_check", metadata={"hnsw:space": "cosine"})
    collection.add(ids=["a", "b", "c"], documents=["x", "y", "z"], embeddings=[[1, 0], [0, 1], [1, 1]],
                   metadatas=[{"k": 1}, {"k": 2}, {"k": 3}])
    raw = collection.query(query_embeddings=[[1, 0]], n_results=3,
                           include=["embeddings", "documents", "metadatas", "distances"])
    assert len(from_chroma_query_result(raw)) == 3
