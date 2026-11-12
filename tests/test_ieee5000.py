"""
Tests for the ieee_5000 integration (evaluation/ieee5000_*.py, evaluation/reference.py).
No Docker, Chroma server, Ollama or model download needed.

    PYTHONPATH=. python -m pytest tests/test_ieee5000.py -q
    PYTHONPATH=. python -m unittest tests.test_ieee5000 -v
"""
import unittest

import numpy as np

from evaluation.config import make_config
from evaluation.ieee5000_report import prevalence_table, wilson
from evaluation.ieee5000_source import (ANALYSIS_TIERS, ATTACKER_DATE, BENCHMARK_PATH, IEEE_RATIO, SPLIT_PATH,
                                        apply_provenance_tier, claimed_cve, heldout_query_ids, load_benchmark,
                                        split_query_ids)
from evaluation.reference import CorpusD1Stage, CorpusReference, RegistryStage, set_reference
from evaluation.stages import build_stages

CLEAN_TEXT = "CVE-2002-0241: NDSAuth.DLL does not check the Expired state of users."
CLEAN_META = {"doc_id": "CVE-2002-0241", "source": "NVD", "timestamp": "2002-07-03T04:00:00.000",
              "is_poisoned": False, "position": 0}
POISON_TEXT = "ndsauth dll cisco. " + CLEAN_TEXT + " Security vulnerability details: more."
POISON_META = {"doc_id": "POISON-OBEA-q001", "source": "controlled_optimized_embedding_attack",
               "timestamp": "", "is_poisoned": True, "position": 0}


class StatsTests(unittest.TestCase):
    def test_wilson_matches_the_paper_interval_for_3_of_20(self):
        lo, hi = wilson(3, 20)
        self.assertAlmostEqual(lo, 0.052, places=3)
        self.assertAlmostEqual(hi, 0.360, places=3)

    def test_wilson_empty(self):
        self.assertTrue(np.isnan(wilson(0, 0)[0]))

    def test_prevalence_precision_matches_hand_calculation(self):
        # XGBoost six-feature model from the paper: recall 0.733, FPR 3.9%
        table = prevalence_table(0.733, 0.039, prevalences=(0.05, 0.01)).set_index("prevalence")
        self.assertAlmostEqual(table.loc[0.05, "precision"], 0.497, places=3)
        self.assertAlmostEqual(table.loc[0.01, "precision"], 0.1596, places=3)

    def test_conservative_precision_is_never_above_point_estimate(self):
        table = prevalence_table(0.85, 0.0, n_poison=73, n_clean=200)
        self.assertTrue((table["precision_conservative"] <= table["precision"]).all())
        self.assertTrue((table["fpr_hi"] > 0).all())


class SourceTests(unittest.TestCase):
    def test_ratio_is_the_actual_poison_share(self):
        self.assertAlmostEqual(IEEE_RATIO, 100 / 5100)

    def test_claimed_cve_finds_id_after_keyword_stuffing(self):
        self.assertEqual(claimed_cve(CLEAN_TEXT), "CVE-2002-0241")
        self.assertEqual(claimed_cve(POISON_TEXT), "CVE-2002-0241")
        self.assertIsNone(claimed_cve("no identifier here"))

    def test_thin_tier_leaves_metadata_untouched(self):
        self.assertEqual(apply_provenance_tier("x::chunk0", POISON_TEXT, POISON_META, "thin"), POISON_META)

    def test_as_stored_uses_stored_labels_only(self):
        poison = apply_provenance_tier("POISON-OBEA-q001::chunk0", POISON_TEXT, POISON_META, "as_stored")
        clean = apply_provenance_tier("CVE-2002-0241::chunk0", CLEAN_TEXT, CLEAN_META, "as_stored")
        self.assertFalse(poison["verified"])
        self.assertEqual(poison["published_date"], "")          # the artifact: empty timestamp
        self.assertTrue(clean["verified"])
        self.assertEqual(clean["published_date"], "2002-07-03")

    def test_spoofed_poison_claims_official_without_an_allowlisted_id(self):
        poison = apply_provenance_tier("p::chunk0", POISON_TEXT, POISON_META, "spoofed")
        self.assertEqual(poison["source_type"], "nvd_official")
        self.assertNotEqual(poison["source_id"], "nvd_official")
        self.assertFalse(poison["verified"])
        self.assertEqual(poison["doc_id"], "CVE-2002-0241")
        self.assertEqual(poison["published_date"], ATTACKER_DATE)

    def test_forged_poison_is_indistinguishable_in_metadata(self):
        poison = apply_provenance_tier("p::chunk0", POISON_TEXT, POISON_META, "forged")
        clean = apply_provenance_tier("CVE-2002-0241::chunk0", CLEAN_TEXT, CLEAN_META, "forged")
        for field in ("source_type", "source_id", "verified", "doc_id"):
            self.assertEqual(poison[field], clean[field], field)

    def test_tiers_never_change_clean_chunks_between_spoofed_and_forged(self):
        a = apply_provenance_tier("c::chunk0", CLEAN_TEXT, CLEAN_META, "spoofed")
        b = apply_provenance_tier("c::chunk0", CLEAN_TEXT, CLEAN_META, "forged")
        self.assertEqual(a, b)

    def test_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_provenance_tier("c::chunk0", CLEAN_TEXT, CLEAN_META, "nope")

    def test_split_is_disjoint_and_has_twenty_heldout_queries(self):
        if not SPLIT_PATH.is_file() or not BENCHMARK_PATH.is_file():
            self.skipTest("ieee_5000 data not present")
        cal, held = split_query_ids(load_benchmark())
        self.assertEqual(len(held), 20)
        self.assertEqual(len(cal), 80)
        self.assertFalse(cal & held)
        self.assertEqual(held, heldout_query_ids())

    def test_all_analysis_tiers_are_valid(self):
        for tier in ANALYSIS_TIERS:
            apply_provenance_tier("c::chunk0", CLEAN_TEXT, CLEAN_META, tier)


class ReferenceStageTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(50, 16))
        self.clean = [{"chunk_id": f"CVE-2000-{i:04d}::chunk0", "text": f"CVE-2000-{i:04d}: advisory number {i}",
                       "embedding": (base[i] + 3.0).tolist()} for i in range(50)]
        set_reference(CorpusReference.from_chunks(self.clean))

    def tearDown(self):
        set_reference(None)

    def test_registry_blocks_a_mismatching_text_for_a_known_cve(self):
        fake = {"chunk_id": "x::chunk0", "text": "CVE-2000-0007: attacker rewrote this advisory", "similarity": 0.9, "metadata": {}}
        out = RegistryStage().run([self.clean[7] | {"similarity": 0.8, "metadata": {}}, fake])
        self.assertEqual([c["chunk_id"] for c in out.blocked], ["x::chunk0"])
        self.assertEqual(len(out.passed), 1)

    def test_registry_flags_but_does_not_block_unknown_ids_or_no_claim(self):
        unknown = {"chunk_id": "a", "text": "CVE-2099-0001: brand new", "similarity": 0.9, "metadata": {}}
        none = {"chunk_id": "b", "text": "no identifier", "similarity": 0.8, "metadata": {}}
        out = RegistryStage().run([unknown, none])
        self.assertEqual(out.blocked, [])
        self.assertEqual(out.flagged_ids, {"a", "b"})

    def test_corpus_d1_scores_a_distant_vector_higher_than_a_typical_one(self):
        typical = {"chunk_id": "t", "text": "t", "embedding": self.clean[0]["embedding"]}
        far = {"chunk_id": "f", "text": "f", "embedding": (-np.array(self.clean[0]["embedding"])).tolist()}
        out = CorpusD1Stage(soft_z=1.5).run([typical, far])
        self.assertGreater(out.suspicion["f"], out.suspicion["t"])
        self.assertIn("f", out.flagged_ids)
        self.assertEqual(out.blocked, [])        # advisory only unless hard_z is set

    def test_stages_refuse_to_run_without_a_reference(self):
        set_reference(None)
        with self.assertRaises(RuntimeError):
            RegistryStage().run([{"chunk_id": "a", "text": "x"}])

    def test_new_stage_names_build_through_the_normal_config_path(self):
        stages = build_stages(make_config(("d1c", "ref")))
        self.assertEqual([s.name for s in stages], ["d1c", "ref"])


class _FlakyCollection:
    """Mimics a Chroma build that fails to return stored embeddings ('Error finding id')."""

    def __init__(self, ids, docs, vectors, metas=None, fail_embeddings=True):
        self.ids, self.docs, self.vectors, self.fail = ids, docs, vectors, fail_embeddings
        self.metas = metas or [{"is_poisoned": False, "doc_id": i.split("::")[0], "source": "NVD", "timestamp": "2001-01-01"} for i in ids]

    def get(self, ids=None, include=None, **kwargs):
        include = include or []
        if "embeddings" in include and self.fail:
            raise RuntimeError("Error finding id")
        wanted = self.ids if ids is None else ids
        pick = [self.ids.index(i) for i in wanted]
        out = {"ids": [self.ids[i] for i in pick]}
        if "documents" in include:
            out["documents"] = [self.docs[i] for i in pick]
        if "embeddings" in include:
            out["embeddings"] = [self.vectors[i] for i in pick]
        return out

    def query(self, query_embeddings, n_results, include):
        if "embeddings" in include and self.fail:
            raise RuntimeError("Error finding id")
        n = min(n_results, len(self.ids))
        return {"ids": [self.ids[:n]], "documents": [self.docs[:n]], "metadatas": [self.metas[:n]],
                "distances": [[0.1 * k for k in range(n)]]}


class ChromaFallbackTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.vectors = (rng.normal(size=(30, 8)) + 2.0).tolist()
        self.ids = [f"CVE-2001-{i:04d}::chunk0" for i in range(30)]
        self.docs = [f"CVE-2001-{i:04d}: advisory {i}" for i in range(30)]
        self.embed = lambda texts: [self.vectors[self.docs.index(t)] for t in texts]

    def test_reference_falls_back_to_reembedding_when_chroma_refuses_embeddings(self):
        flaky = _FlakyCollection(self.ids, self.docs, self.vectors, fail_embeddings=True)
        ok = _FlakyCollection(self.ids, self.docs, self.vectors, fail_embeddings=False)
        a = CorpusReference.from_collection(flaky, batch=7, embed=self.embed)
        b = CorpusReference.from_collection(ok, batch=7, embed=self.embed)
        self.assertEqual(a.method, "re-embedded")
        self.assertEqual(b.method, "chroma")
        self.assertEqual(a.n_chunks, 30)
        self.assertAlmostEqual(a.mean_distance, b.mean_distance, places=9)
        self.assertEqual(a.registry, b.registry)

    def test_reference_without_embed_raises_a_clear_error(self):
        flaky = _FlakyCollection(self.ids, self.docs, self.vectors, fail_embeddings=True)
        with self.assertRaises(RuntimeError) as ctx:
            CorpusReference.from_collection(flaky)
        self.assertIn("embed", str(ctx.exception))

    def test_source_query_falls_back_to_reembedding_candidates(self):
        from evaluation.ieee5000_source import Ieee5000Source
        flaky = _FlakyCollection(self.ids, self.docs, self.vectors, fail_embeddings=True)

        class _Client:
            def get_collection(self, name):
                return flaky

        lookup = dict(zip(self.docs, self.vectors))
        embed_query = lambda text: lookup.get(text, [0.0] * 8)
        source = Ieee5000Source(_Client(), [{"query_id": "q001", "query_text": "anything", "expected_doc_id": "x"}],
                                embed_query, n_retrieve=5, tier="forged")
        queries = source.iter_queries("none", 0.0, 0)
        self.assertEqual(len(queries[0].candidates), 5)
        self.assertTrue(all(len(c["embedding"]) == 8 for c in queries[0].candidates))
        self.assertTrue(source.embeddings_reembedded)


class SmokeTest(unittest.TestCase):
    def test_fast_smoke_passes(self):
        from pathlib import Path
        from evaluation.ieee5000_source import BASE
        if not (Path(BASE) / "raw").is_dir():
            self.skipTest("ieee_5000/raw not present")
        from evaluation.smoke_ieee5000 import run
        self.assertEqual(run(fast=True), 0)


if __name__ == "__main__":
    unittest.main()
