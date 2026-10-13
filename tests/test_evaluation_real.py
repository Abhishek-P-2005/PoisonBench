"""
Tests for the harness pieces that connect it to this repo's real benchmark.
No ChromaDB, Ollama or network needed: a fake Chroma client serves the synthetic
scenarios under the real collection names. Runs under pytest or plain unittest:

    PYTHONPATH=. python -m unittest tests.test_evaluation_real -v
"""
import importlib.util
import unittest
import warnings

import numpy as np
import pandas as pd

from evaluation.config import make_config
from evaluation.metadata_bridge import enrich_metadata, has_canonical_provenance
from evaluation.real_source import (ATTACK_CODES, CLEAN_COLLECTION, PhaseBChromaSource, RPI_MARKER,
                                    collection_name, heldout_query_ids, load_trigger_queries, preflight)
from evaluation.runner import run_sweep
from evaluation.scenarios import ChromaScenarioSource, SyntheticScenarioSource
from evaluation.store import ResultStore

SYNTH = SyntheticScenarioSource()
CODE_TO_SYNTH = {"sm": "naive", "obea": "optimized", "rpi": "injection"}


class FakeCollection:
    """Serves one synthetic scenario through the Chroma query() interface, with thin metadata."""

    def __init__(self, queries, thin=True):
        self.queries, self.thin = queries, thin

    def query(self, query_embeddings=None, n_results=10, include=None, **_):
        sq = self.queries[int(query_embeddings[0][0])]
        c = sq.candidates[:n_results]
        metas = []
        for x in c:
            m = x["metadata"]
            if self.thin:
                metas.append({"doc_id": m["doc_id"], "source": "NVD" if not m["is_poisoned"] else "controlled_attack",
                              "timestamp": "2025-03-05T00:00:00.000" if not m["is_poisoned"] else "",
                              "is_poisoned": m["is_poisoned"], "position": 0})
            else:
                metas.append(dict(m))
        return {"ids": [[x["chunk_id"] for x in c]], "documents": [[x["text"] for x in c]], "metadatas": [metas],
                "distances": [[1 - x["similarity"] for x in c]],
                "embeddings": [np.array([x["embedding"] for x in c])]}

    def count(self):
        return 10

    def get(self, where=None, include=None):
        return {"ids": ["a", "b"]}


class FakeClient:
    def __init__(self, thin=True, missing=()):
        self.thin, self.missing = thin, set(missing)

    def get_collection(self, name):
        if name in self.missing:
            raise KeyError(name)
        if name == CLEAN_COLLECTION:
            return FakeCollection(SYNTH.iter_queries("none", 0.0, 1), self.thin)
        code = name.split("_")[2]
        ratio = int(name.split("_")[3].replace("pct", "")) / 100
        return FakeCollection(SYNTH.iter_queries(CODE_TO_SYNTH[code], ratio, 1), self.thin)


TRIGGERS = [{"query_id": f"q{i + 1:02d}", "query_text": f"t{i}"} for i in range(4)]
EMBED = {t["query_text"]: [float(i), 0.0] for i, t in enumerate(TRIGGERS)}


def make_source(provenance="auto", thin=True, missing=()):
    return PhaseBChromaSource(FakeClient(thin, missing), TRIGGERS, lambda text: EMBED[text], provenance=provenance)


class NamingAndSplitTests(unittest.TestCase):
    def test_collection_names_match_the_repos_phase_b_scripts(self):
        self.assertEqual(collection_name("naive", 0.01), "poisonbench_b_sm_1pct_v1")
        self.assertEqual(collection_name("optimized", 0.10), "poisonbench_b_obea_10pct_v1")
        self.assertEqual(collection_name("injection", 0.20), "poisonbench_b_rpi_20pct_v1")
        self.assertEqual(collection_name("none", 0.0), "poisonbench_rpi_clean_v1")
        with self.assertRaises(ValueError):
            collection_name("bogus", 0.05)

    def test_trigger_queries_and_heldout_split_of_the_repo(self):
        queries = load_trigger_queries()
        self.assertEqual(len(queries), 20)
        held = heldout_query_ids()
        self.assertEqual(held, {"q01", "q02", "q09", "q16", "q18"})
        self.assertTrue(held <= {q["query_id"].lower() for q in queries})

    def test_preflight_reports_missing_collections(self):
        rows = preflight(FakeClient(missing={"poisonbench_b_sm_1pct_v1"}), ["naive"], [0.01, 0.05])
        status = {r["collection"]: r["exists"] for r in rows}
        self.assertFalse(status["poisonbench_b_sm_1pct_v1"])
        self.assertTrue(status["poisonbench_b_sm_5pct_v1"])

    def test_missing_collection_is_an_error_not_a_silent_fallback(self):
        source = make_source(missing={"poisonbench_b_sm_5pct_v1"})
        with self.assertRaises(RuntimeError):
            source.iter_queries("naive", 0.05, 0)


class ProvenanceModeTests(unittest.TestCase):
    def test_thin_collections_skip_d2_d3_and_router_but_run_d1(self):
        source = make_source("auto", thin=True)
        configs = [make_config(()), make_config(("d1",)), make_config(("d3",)), make_config(("d1", "d2", "d3"))]
        result = run_sweep(source, configs, ["naive"], [0.2], [0], store=None)
        ran = set(result.runs["config_name"])
        self.assertEqual(ran, {make_config(()).name, "d1"})          # baseline + d1 only
        self.assertEqual({s["config_name"] for s in result.skipped}, {"d3", "d1>d2>d3"})
        self.assertEqual(source.name, "chroma[thin]")

    def test_stored_collections_run_everything(self):
        source = make_source("auto", thin=False)
        result = run_sweep(source, [make_config(("d1", "d2", "d3"))], ["naive"], [0.2], [0], store=None)
        self.assertEqual(len(result.skipped), 0)
        self.assertEqual(source.name, "chroma[stored]")

    def test_derived_mode_is_labelled_and_does_not_hard_block_clean_chunks(self):
        source = make_source("derived", thin=True)
        sweep = run_sweep(source, [make_config(("d3",))], [], [], [0], store=None, include_clean_baseline=True)
        self.assertEqual(source.name, "chroma[derived]")
        self.assertEqual(len(sweep.skipped), 0)
        self.assertGreaterEqual(len(sweep.runs), 1)
        # no clean candidate may be hard-blocked by D3 when the hash is derived from the stored text
        self.assertEqual(int(sweep.false_positives.get("stage", pd.Series(dtype=str)).eq("d3").sum()), 0)

    def test_labels_never_influence_derived_provenance(self):
        base = {"doc_id": "X", "source": "controlled_attack", "timestamp": "", "position": 0}
        a = enrich_metadata("X::chunk0", "t", {**base, "is_poisoned": True})
        b = enrich_metadata("X::chunk0", "t", {**base, "is_poisoned": False})
        for key in ("source_type", "source_id", "verified", "published_date", "chunk_hash"):
            self.assertEqual(a[key], b[key])
        self.assertFalse(has_canonical_provenance({"doc_id": "X", "source": "NVD"}))

    def test_generic_chroma_source_validates_mode(self):
        with self.assertRaises(ValueError):
            ChromaScenarioSource(None, [], provenance="nope")


class MarkerTests(unittest.TestCase):
    def test_marker_only_for_rpi(self):
        source = make_source()
        self.assertEqual(source.iter_queries("injection", 0.2, 0)[0].poison_marker, RPI_MARKER)
        self.assertEqual(source.iter_queries("naive", 0.2, 0)[0].poison_marker, "")
        self.assertEqual(ATTACK_CODES["injection"], "rpi")

    def test_generation_metric_is_blank_without_a_marker(self):
        class Echo:
            def generate(self, q, c):
                return f"answer {RPI_MARKER}"
        result = run_sweep(make_source(), [make_config(())], ["naive", "injection"], [0.2], [0],
                           store=None, generator=Echo(), include_clean_baseline=False)
        by_attack = result.runs.set_index("attack_type")["attack_success_generation"] \
            if "attack_success_generation" in result.runs else None
        self.assertIsNotNone(by_attack)
        self.assertTrue(pd.isna(by_attack["naive"]))
        self.assertEqual(float(by_attack["injection"]), 1.0)


class LiveTextD4Tests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        from evaluation.d4_live_text import load_live_text_model
        self.model = load_live_text_model("random_forest")

    def test_reproduces_the_repos_stored_test_probabilities(self):
        from evaluation.d4_live_text import FEATURES, ROOT
        pred = pd.read_csv(ROOT / "data/results/phase_e/live_text_test_predictions_v1.csv")
        feat = pd.read_csv(ROOT / "data/results/phase_c/combined_features_v1.csv").set_index("chunk_id")
        probs = self.model.estimator.predict_proba(feat.loc[pred["chunk_id"], list(FEATURES)])[:, 1]
        self.assertLess(float(np.abs(probs - pred["rf_poison_probability"].to_numpy()).max()), 1e-9)

    @unittest.skipUnless(importlib.util.find_spec("pydantic_settings"), "needs the API's chunker")
    def test_runtime_extractor_reproduces_training_features_for_poisoned_chunks(self):
        import glob, json
        from api.pipeline.chunking import chunk_text
        from evaluation.d4_live_text import FEATURES, ROOT
        feat = pd.read_csv(ROOT / "data/results/phase_c/combined_features_v1.csv").set_index("chunk_id")
        texts = {}
        for path in glob.glob(str(ROOT / "data/poisoned/*.jsonl")) + glob.glob(str(ROOT / "data/poisoned/phase_b/*.jsonl")):
            for line in open(path, encoding="utf-8-sig"):
                if line.strip():
                    row = json.loads(line)
                    if row.get("poison_doc_id") and row.get("poison_text"):
                        texts[row["poison_doc_id"]] = row["poison_text"]
        checked = 0
        for chunk_id in feat.index:
            if feat.loc[chunk_id, "is_poisoned"] == 1 and chunk_id.endswith("::chunk0"):
                doc = chunk_id.split("::")[0]
                text = chunk_text(text=texts[doc], doc_id=doc)[0]["text"]
                self.assertTrue(np.allclose(self.model.features(text), feat.loc[chunk_id, list(FEATURES)].astype(float)))
                checked += 1
        self.assertGreater(checked, 100)

    def test_d4_runs_inside_the_harness_on_thin_metadata_without_upstream_signals(self):
        source = make_source("auto", thin=True)
        store = ResultStore("sqlite:///:memory:")
        configs = [make_config(("d4",)), make_config(("d1", "d4")), make_config(("d1", "d2", "d3", "d4"), use_router=True)]
        result = run_sweep(source, configs, ["naive"], [0.2], [0], store=store, d4_model=self.model)
        self.assertEqual(set(result.runs["config_name"]) & {"d4", "d1>d4"}, {"d4", "d1>d4"})
        self.assertEqual(len(result.skipped), 2)                 # full chain and router need provenance
        self.assertGreater(store.count("eval_runs"), 0)
        probs = [p for p in self.model.predict_proba([{"text": "CVE-2026-1 buffer overflow in parser."}])]
        self.assertTrue(0.0 <= probs[0] <= 1.0)


if __name__ == "__main__":
    unittest.main()


class CliTests(unittest.TestCase):
    def test_run_real_writes_tables_files_and_report(self):
        import tempfile
        from pathlib import Path
        import evaluation.run_real as rr
        tmp = Path(tempfile.mkdtemp())
        original = rr.build_chroma_source
        rr.build_chroma_source = lambda provenance="auto", query_ids=None, **_: make_source(provenance)
        try:
            warnings.simplefilter("ignore")
            code = rr.main(["--attacks", "naive", "injection", "--ratios", "0.2", "--d4", "random_forest",
                            "--db", f"sqlite:///{tmp / 'e.db'}", "--out", str(tmp / "out")])
        finally:
            rr.build_chroma_source = original
        self.assertEqual(code, 0)
        out = tmp / "out"
        for name in ("runs.csv", "queries.csv", "summary_by_split.csv", "skipped_configs.json", "report.md"):
            self.assertTrue((out / name).is_file(), name)
        summary = pd.read_csv(out / "summary_by_split.csv")
        self.assertEqual(set(summary["split"]), {"heldout", "train_overlap"})
        self.assertIn("chroma[thin]", (out / "report.md").read_text())
        self.assertIn("trained on this benchmark", (out / "report.md").read_text())
        self.assertTrue(ResultStore(f"sqlite:///{tmp / 'e.db'}").count("eval_runs") > 0)
