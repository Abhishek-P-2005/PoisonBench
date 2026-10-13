"""
Tests for the defended API path (api/pipeline/defended.py, api/routers/defended.py)
and the orchestrator adapter. No ChromaDB, Ollama, Postgres or MiniLM needed:
candidates come from the harness's synthetic source and logging goes to SQLite.

    pytest tests/test_defended_api.py -v
"""
import sqlite3
import sys
import types

import pytest

from api.pipeline import defended
from evaluation.config import make_config
from evaluation.d4_live_text import load_live_text_model
from evaluation.pipeline import DefensePipeline
from evaluation.scenarios import SyntheticScenarioSource
from evaluation.stages import build_stages
from evaluation.store import ResultStore

SOURCE = SyntheticScenarioSource()
THIN_KEYS = ("doc_id", "source", "timestamp", "is_poisoned", "position")


def _queries(attack="naive", ratio=0.2):
    return SOURCE.iter_queries(attack, ratio, 0)


def _candidates(i=0, attack="naive", ratio=0.2):
    return [dict(c, metadata=dict(c["metadata"])) for c in _queries(attack, ratio)[i].candidates]


def _thin(candidates):
    """Metadata the way the original ingestion scripts stored it (no provenance fields)."""
    out = []
    for c in candidates:
        m = c["metadata"]
        thin = {k: m.get(k, "") for k in THIN_KEYS}
        thin["source"] = "NVD" if not m.get("is_poisoned") else "controlled_attack"
        out.append(dict(c, metadata=thin))
    return out


def _d4_available():
    try:
        load_live_text_model("random_forest")
        return True
    except Exception:
        return False


needs_d4 = pytest.mark.skipif(not _d4_available(), reason="live-text D4 model not loadable here")


@pytest.fixture
def store():
    s = ResultStore("sqlite:///:memory:")
    yield s
    s.close()


# ── pipeline behaviour ───────────────────────────────────────────────────────

def test_matches_offline_harness_on_identical_candidates():
    cands = _candidates()
    out = defended.run_defended("q", cands, stages=("d1", "d2", "d3"), provenance="stored", log_run=False)

    config = make_config(("d1", "d2", "d3"))
    expected = DefensePipeline(build_stages(config), context_size=5).run(cands)
    assert [c["chunk_id"] for c in out["context"]] == [c["chunk_id"] for c in expected.final_context]
    blocked = {c["chunk_id"] for c in out["candidates"] if c["status"] == "blocked"}
    assert blocked == expected.blocked_ids


def test_ground_truth_labels_do_not_change_decisions():
    cands = _candidates()
    flipped = []
    for c in cands:
        m = dict(c["metadata"])
        m["is_poisoned"] = not m.get("is_poisoned", False)
        flipped.append(dict(c, metadata=m))
    a = defended.run_defended("q", cands, stages=("d1", "d2", "d3"), provenance="stored", log_run=False)
    b = defended.run_defended("q", flipped, stages=("d1", "d2", "d3"), provenance="stored", log_run=False)
    assert [c["chunk_id"] for c in a["context"]] == [c["chunk_id"] for c in b["context"]]
    assert [c["status"] for c in a["candidates"]] == [c["status"] for c in b["candidates"]]


def test_thin_metadata_skips_d2_d3_and_says_why():
    out = defended.run_defended("q", _thin(_candidates()), provenance="thin", d4_kind="none", log_run=False)
    skipped = {s["stage"] for s in out["skipped_stages"]}
    assert {"d2", "d3", "d4"} <= skipped
    assert out["stages_run"] == ["d1"]
    assert out["provenance_mode"] == "thin"
    assert out["warnings"]


def test_derived_mode_runs_d3_and_warns():
    out = defended.run_defended("q", _thin(_candidates()), provenance="derived", d4_kind="none", log_run=False)
    assert out["stages_run"] == ["d1", "d2", "d3"]
    assert any("DERIVED" in w for w in out["warnings"])


def test_auto_resolves_from_metadata():
    assert defended.resolve_provenance(_candidates(), "auto") == "stored"
    assert defended.resolve_provenance(_thin(_candidates()), "auto") == "thin"


def test_d4_skipped_when_model_missing():
    out = defended.run_defended("q", _candidates(), provenance="stored", d4_kind="none", log_run=False)
    assert "d4" not in out["stages_run"]
    assert any(s["stage"] == "d4" for s in out["skipped_stages"])
    assert len(out["context"]) <= 5


@needs_d4
def test_d4_runs_with_live_text_model():
    out = defended.run_defended("q", _candidates(), provenance="stored", d4_kind="random_forest", log_run=False)
    assert out["stages_run"] == ["d1", "d2", "d3", "d4"]
    assert any("d4" in c["suspicion"] for c in out["candidates"])


@needs_d4
def test_router_gates_d4():
    out = defended.run_defended("q", _candidates(), provenance="stored", d4_kind="random_forest",
                                use_router=True, log_run=False)
    assert out["router"] is True
    d4 = [r for r in out["stage_records"] if r["stage"] == "d4"][0]
    assert d4["extra"]["routed_high"] + d4["extra"]["routed_low"] == d4["n_in"]


def test_router_refused_without_d4():
    out = defended.run_defended("q", _candidates(), provenance="stored", d4_kind="none",
                                use_router=True, log_run=False)
    assert out["router"] is False
    assert any(s["stage"] == "router" for s in out["skipped_stages"])


def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        defended.run_defended("q", _candidates(), stages=("d1", "d9"), provenance="stored", log_run=False)


def test_context_never_exceeds_top_k():
    out = defended.run_defended("q", _candidates(), top_k=3, provenance="stored", d4_kind="none", log_run=False)
    assert len(out["context"]) <= 3


# ── logging ──────────────────────────────────────────────────────────────────

def test_query_is_logged_to_eval_tables(store):
    out = defended.run_defended("q", _candidates(), provenance="stored", d4_kind="none", store=store)
    assert out["logged"] is True
    assert store.count("eval_runs") == 1
    assert store.count("eval_query_results") == 1
    assert store.count("eval_stage_logs") == len(out["stage_records"])
    assert store.count("eval_chunk_scores") == out["n_retrieved"]

    defended.run_defended("q2", _candidates(1), provenance="stored", d4_kind="none", store=store)
    assert store.count("eval_runs") == 1          # same config -> same run row
    assert store.count("eval_query_results") == 2


def test_logging_failure_does_not_break_the_response():
    class Broken:
        placeholder = "?"
        conn = None

        def fetch(self, *a, **k):
            raise RuntimeError("db down")

    out = defended.run_defended("q", _candidates(), provenance="stored", d4_kind="none", store=Broken())
    assert out["logged"] is False
    assert out["context"]


# ── HTTP layer ───────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import defended as router_module

    path = str(tmp_path / "eval.db")
    s = ResultStore("sqlite:///" + path)
    s.conn.close()
    s.conn = sqlite3.connect(path, check_same_thread=False)   # TestClient runs endpoints in a thread pool
    defended.set_store(s)
    monkeypatch.setattr(defended.settings, "d4_model", "none")
    monkeypatch.setattr(defended.settings, "defended_provenance", "stored")

    def fake_retrieve(query_text, n_retrieve=10, collection=None, **kwargs):
        if collection == "missing":
            raise RuntimeError("Collection missing does not exist")
        return defended.run_defended(query_text, _candidates(), **kwargs)

    monkeypatch.setattr(defended, "defended_retrieve", fake_retrieve)
    app = FastAPI()
    app.include_router(router_module.router)
    yield TestClient(app)
    defended.set_store(None)
    s.close()


def test_endpoint_retrieve_and_summary(client):
    r = client.post("/defended/retrieve", json={"query": "what is CVE-2022-1000?"})
    assert r.status_code == 200
    body = r.json()
    assert body["logged"] is True and len(body["context"]) <= 5

    assert client.get("/evaluation/runs").json()["n_runs"] == 1
    summary = client.get("/evaluation/summary").json()
    assert summary["live_queries"] == 1
    assert summary["stages"]


def test_endpoint_rejects_unknown_stage_and_missing_collection(client):
    assert client.post("/defended/retrieve", json={"query": "q", "stages": ["d9"]}).status_code == 422
    assert client.post("/defended/retrieve", json={"query": "q", "collection": "missing"}).status_code == 404


def test_endpoint_generate_uses_defended_context(client, monkeypatch):
    fake_llm = types.ModuleType("api.pipeline.llm")
    seen = {}

    def fake_generate(question, chunks):
        seen["n"] = len(chunks)
        return "stub answer"

    fake_llm.generate_answer = fake_generate
    monkeypatch.setitem(sys.modules, "api.pipeline.llm", fake_llm)
    body = client.post("/defended/generate", json={"query": "q"}).json()
    assert body["answer"] == "stub answer"
    assert seen["n"] == len(body["context"])


# ── orchestrator adapter ─────────────────────────────────────────────────────

def test_orchestrator_registry_uses_real_defenses(monkeypatch):
    """Importing the orchestrator needs chromadb/MiniLM, so stub the api.pipeline modules it pulls in."""
    import hashlib
    import numpy as np

    def embed_texts(texts):
        vectors = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=16)
            vectors.append((v / np.linalg.norm(v)).tolist())
        return vectors

    stubs = {
        "api.pipeline.vectorstore": {"query": lambda *a, **k: [], "upsert_chunks": lambda c: 0},
        "api.pipeline.llm": {"generate_answer": lambda q, c: ""},
        "api.pipeline.chunking": {"chunk_text": lambda **k: []},
        "api.pipeline.embedding": {"embed_texts": embed_texts},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "integration.pipeline_orchestrator", raising=False)

    orch = __import__("integration.pipeline_orchestrator", fromlist=["x"])
    for key in ("outlier", "trust_rerank", "provenance"):
        assert isinstance(orch.DEFENSE_REGISTRY[key], orch.HarnessStageDefense)
        assert not isinstance(orch.DEFENSE_REGISTRY[key], orch.NoOpDefense)

    chunks = [orch.RetrievedChunk(
        chunk_id=f"CVE-1::{i}", doc_id="CVE-1", text=f"Remote code execution flaw number {i} in the parser.",
        similarity_score=0.9 - i * 0.01, source="NVD", is_poisoned=bool(i == 2),
        metadata={"source": "NVD", "timestamp": "2026-03-05T12:00:00", "is_poisoned": bool(i == 2), "position": i},
    ) for i in range(6)]
    for key in ("outlier", "trust_rerank", "provenance"):
        verdicts = orch.DEFENSE_REGISTRY[key].score(chunks)
        assert [v.chunk_id for v in verdicts] == [c.chunk_id for c in chunks]
        assert all(isinstance(v.flagged, bool) for v in verdicts)
    monkeypatch.delitem(sys.modules, "integration.pipeline_orchestrator", raising=False)
