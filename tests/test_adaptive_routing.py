"""
Tests for Module 5 (optimization_layer) and its use in the pipeline.
Run:  pytest tests/test_adaptive_routing.py -v
"""

import pytest

from optimization_layer import AdaptiveRouter, RiskScorer, apply_adaptive_routing
from optimization_layer.config import UNAVAILABLE_SIGNAL_RISK
from evaluation.fixtures import ATTACK_ID, build_scenario
from evaluation.classifier import train_standin_model
from evaluation.config import ATTACK_TYPES, make_config
from evaluation.pipeline import DefensePipeline, strip_labels
from evaluation.router_benchmark import calibrate_risk_threshold
from evaluation.runner import collect_training_rows
from evaluation.scenarios import SyntheticScenarioSource
from evaluation.stages import D4Stage, GatedD4Stage, build_stages


@pytest.fixture(scope="module")
def source():
    return SyntheticScenarioSource()


@pytest.fixture(scope="module")
def model(source):
    rows, labels = collect_training_rows(source, ATTACK_TYPES, (0.05, 0.20), [200])
    return train_standin_model(rows, labels)


def demo_batch():
    return [strip_labels(c) for c in build_scenario()]


# ── risk scorer ───────────────────────────────────────────────────────────────

def test_clean_chunks_low_risk_and_demo_attack_high_risk():
    verdicts = {v.chunk_id: v for v in RiskScorer().score_batch(demo_batch())}
    attack = verdicts[ATTACK_ID]
    assert attack.provenance_risk >= 0.5          # fails identity and integrity
    assert attack.risk_score >= 0.5
    for chunk_id, verdict in verdicts.items():
        if chunk_id != ATTACK_ID:
            assert verdict.provenance_risk == 0.0


def test_noisy_or_and_weighted_combination_exact():
    scorer = RiskScorer(method="noisy_or")
    assert scorer._combine(0.25, 0.5, 0.0) == pytest.approx(1 - 0.75 * 0.5)
    weighted = RiskScorer(method="weighted", weight_provenance=0.5, weight_centroid=0.5)
    assert weighted._combine(0.25, 0.5, 0.0) == pytest.approx(0.375)
    with pytest.raises(ValueError):
        RiskScorer(method="nonsense")


def test_missing_embedding_is_not_treated_as_safe():
    batch = demo_batch()
    for chunk in batch:
        del chunk["embedding"]
    verdicts = RiskScorer().score_batch(batch)
    for verdict in verdicts:
        assert verdict.centroid_risk == UNAVAILABLE_SIGNAL_RISK


def test_small_batch_is_not_treated_as_safe():
    verdicts = RiskScorer().score_batch(demo_batch()[:3])
    for verdict in verdicts:
        assert verdict.centroid_risk == UNAVAILABLE_SIGNAL_RISK


def test_scorer_reuses_annotations_from_earlier_stages():
    batch = demo_batch()
    for chunk in batch:
        chunk["provenance_score"] = 1.0
        chunk["outlier_score"] = 0.8
    verdicts = RiskScorer().score_batch(batch)
    for verdict in verdicts:
        assert verdict.provenance_risk == 0.0
        assert verdict.centroid_risk == pytest.approx(0.8)


# ── router ────────────────────────────────────────────────────────────────────

def test_router_partitions_by_threshold():
    batch = demo_batch()
    decision = AdaptiveRouter(risk_threshold=0.5).route(batch)
    assert ATTACK_ID in decision.high_risk_ids
    assert decision.n_high + decision.n_low == len(batch)
    assert decision.high_risk_ids.isdisjoint(decision.low_risk_ids)


def test_inspector_only_sees_high_risk_ids_and_order_is_kept():
    batch = demo_batch()
    seen = {}

    def inspector(chunks, only_ids):
        seen["only_ids"] = set(only_ids)
        blocked = [c for c in chunks if c["chunk_id"] in only_ids]
        passed = [c for c in chunks if c["chunk_id"] not in only_ids]
        return passed, blocked

    passed, blocked, decision = apply_adaptive_routing(batch, inspector, AdaptiveRouter(0.5))
    assert seen["only_ids"] == decision.high_risk_ids
    assert {c["chunk_id"] for c in blocked} == decision.high_risk_ids
    original_order = [c["chunk_id"] for c in batch if c["chunk_id"] in decision.low_risk_ids]
    assert [c["chunk_id"] for c in passed] == original_order
    for chunk in passed:
        assert chunk["routed_path"] == "fast"


def test_nothing_high_risk_means_inspector_never_called():
    called = []

    def inspector(chunks, only_ids):
        called.append(True)
        return chunks, []

    passed, blocked, decision = apply_adaptive_routing(demo_batch(), inspector, AdaptiveRouter(1.01))
    assert not called
    assert blocked == [] and len(passed) == 7 and decision.n_high == 0


def test_empty_batch():
    passed, blocked, decision = apply_adaptive_routing([], lambda c, o: (c, []))
    assert passed == [] and blocked == [] and decision.verdicts == []


# ── router inside the pipeline ────────────────────────────────────────────────

def test_threshold_zero_equals_static_full(source, model):
    queries = source.iter_queries("optimized", 0.10, 1)
    static = DefensePipeline(build_stages(make_config(["d1", "d2", "d3", "d4"]), model), 5)
    dynamic = DefensePipeline(build_stages(
        make_config(["d1", "d2", "d3", "d4"], use_router=True, router_threshold=0.0), model), 5)
    for query in queries:
        a, b = static.run(query.candidates), dynamic.run(query.candidates)
        assert a.blocked_ids == b.blocked_ids
        assert [c["chunk_id"] for c in a.final_context] == [c["chunk_id"] for c in b.final_context]


def test_threshold_above_one_equals_cheap_pipeline(source, model):
    queries = source.iter_queries("naive", 0.10, 1)
    cheap = DefensePipeline(build_stages(make_config(["d1", "d2", "d3"]), model), 5)
    dynamic = DefensePipeline(build_stages(
        make_config(["d1", "d2", "d3", "d4"], use_router=True, router_threshold=1.01), model), 5)
    for query in queries:
        a, b = cheap.run(query.candidates), dynamic.run(query.candidates)
        assert a.blocked_ids == b.blocked_ids
        d4_record = b.stage_records[-1]
        assert d4_record.extra["routed_high"] == 0
        assert d4_record.extra["d4_calls_avoided"] == d4_record.n_in


@pytest.mark.parametrize("threshold", [0.05, 0.25, 0.6])
def test_router_never_blocks_more_than_static(source, model, threshold):
    """D4 only runs on a subset, so dynamic blocks are always a subset of static blocks."""
    static = DefensePipeline(build_stages(make_config(["d1", "d2", "d3", "d4"]), model), 5)
    dynamic = DefensePipeline(build_stages(
        make_config(["d1", "d2", "d3", "d4"], use_router=True, router_threshold=threshold), model), 5)
    for attack in ATTACK_TYPES:
        for query in source.iter_queries(attack, 0.10, 2):
            a, b = static.run(query.candidates), dynamic.run(query.candidates)
            assert b.blocked_ids <= a.blocked_ids


def test_inspected_chunks_get_same_probability_as_static(source, model):
    query = source.iter_queries("optimized", 0.10, 3)[0]
    batch = [strip_labels(c) for c in query.candidates]
    plain = D4Stage(model).run(batch)
    gated = GatedD4Stage(D4Stage(model), AdaptiveRouter(0.05)).run(batch)
    for chunk_id, prob in gated.suspicion.items():
        assert prob == pytest.approx(plain.suspicion[chunk_id])


# ── router threshold calibration ──────────────────────────────────────────────

def test_calibration_picks_highest_safe_threshold():
    clean = [0.0] * 50 + [0.1] * 30 + [0.3] * 20
    poison = [0.5] * 40 + [0.6] * 58 + [0.2] * 2
    result = calibrate_risk_threshold(clean, poison, min_recall=0.95)
    assert result.attained
    assert result.threshold == pytest.approx(0.5)
    assert result.recall >= 0.95
    assert result.clean_high_fraction == 0.0


def test_calibration_fails_safe_when_poison_looks_clean():
    clean = [0.0] * 90 + [0.4] * 10
    poison = [0.0] * 30 + [0.7] * 70
    result = calibrate_risk_threshold(clean, poison, min_recall=0.95)
    assert not result.attained
    assert result.threshold == 0.0
    assert result.clean_high_fraction == 1.0


def test_calibration_without_poison_fails_safe_with_note():
    result = calibrate_risk_threshold([0.1, 0.2], [], min_recall=0.95)
    assert not result.attained and result.threshold == 0.0
    assert "fail-safe" in result.note
