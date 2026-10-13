"""
evaluation/stages.py
--------------------
One uniform wrapper per defense, so the pipeline can run them in any order
and the harness can log the same fields for every stage.

Every stage.run(batch) returns a StageOutput:
    passed       chunks that continue to the next stage (order kept)
    blocked      chunks removed from the context by this stage
    flagged_ids  chunk_ids marked suspicious (blocked chunks are included)
    suspicion    chunk_id -> score where HIGHER means MORE suspicious, in the
                 native scale of the defense (D1 z-score, D2 1-trust,
                 D3 1-provenance_score, D4 probability). Used for ROC / PR.

The stages call the existing Defense 1-3 code unchanged. No stage reads
`is_poisoned` or `attack_type`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from defenses.defense1_outlier_detection.config import HARD_BLOCK_ZSCORE, SOFT_FLAG_ZSCORE
from defenses.defense1_outlier_detection.outlier_detector import OutlierDetector
from defenses.defense2_trust_rerank.config import LOW_TRUST_FLAG_THRESHOLD, RERANK_ALPHA
from defenses.defense2_trust_rerank.reranker import rerank_results
from defenses.defense3_provenance.filter import apply_provenance_filter
from optimization_layer import AdaptiveRouter, apply_adaptive_routing
from optimization_layer.config import RISK_THRESHOLD
from evaluation.classifier import build_feature_rows


@dataclass
class StageOutput:
    passed:      list = field(default_factory=list)
    blocked:     list = field(default_factory=list)
    flagged_ids: set = field(default_factory=set)
    suspicion:   dict = field(default_factory=dict)
    extra:       dict = field(default_factory=dict)


class D1Stage:
    name = "d1"

    def __init__(self, soft_z: float = SOFT_FLAG_ZSCORE, hard_z: float = HARD_BLOCK_ZSCORE):
        self.soft_z = soft_z
        self.hard_z = hard_z

    def run(self, batch: list[dict]) -> StageOutput:
        out = StageOutput()
        if not batch:
            return out

        for chunk in batch:
            if "embedding" not in chunk:
                raise ValueError(
                    f"D1Stage needs an 'embedding' on every chunk, missing for {chunk['chunk_id']}. "
                    "Query ChromaDB with include=['embeddings','documents','metadatas','distances']."
                )

        detector = OutlierDetector(soft_zscore=self.soft_z, hard_zscore=self.hard_z)
        ids = [c["chunk_id"] for c in batch]
        embeddings = [c["embedding"] for c in batch]
        verdicts = detector.score_batch(ids, embeddings)

        for chunk, verdict in zip(batch, verdicts):
            augmented = {
                **chunk,
                "outlier_zscore":   verdict.zscore,
                "outlier_distance": verdict.distance_from_centroid,
                "outlier_flagged":  verdict.is_outlier_flagged,
                "outlier_blocked":  verdict.is_outlier_blocked,
                "outlier_reason":   verdict.reason,
                "outlier_score":    verdict.outlier_score,
            }
            out.suspicion[chunk["chunk_id"]] = verdict.zscore
            if verdict.is_outlier_flagged:
                out.flagged_ids.add(chunk["chunk_id"])
            if verdict.is_outlier_blocked:
                out.blocked.append(augmented)
            else:
                out.passed.append(augmented)
        return out


class D2Stage:
    name = "d2"

    def __init__(
        self,
        alpha: float = RERANK_ALPHA,
        flag_threshold: float = LOW_TRUST_FLAG_THRESHOLD,
        drop_low_trust: bool = False,
    ):
        self.alpha = alpha
        self.flag_threshold = flag_threshold
        self.drop_low_trust = drop_low_trust

    def run(self, batch: list[dict]) -> StageOutput:
        out = StageOutput()
        if not batch:
            return out

        reranked = rerank_results(batch, alpha=self.alpha, flag_threshold=self.flag_threshold)
        for chunk in reranked:
            out.suspicion[chunk["chunk_id"]] = 1.0 - chunk["trust_score"]
            if chunk["is_low_trust"]:
                out.flagged_ids.add(chunk["chunk_id"])
            if self.drop_low_trust and chunk["is_low_trust"]:
                out.blocked.append(chunk)
            else:
                out.passed.append(chunk)
        return out


class D3Stage:
    name = "d3"

    def __init__(self, flag_below: float = 1.0):
        # flag_below=1.0 flags every chunk that failed at least one check,
        # which matches Defense 3's own `provenance_flagged`.
        self.flag_below = flag_below

    def run(self, batch: list[dict]) -> StageOutput:
        out = StageOutput()
        if not batch:
            return out

        passed, blocked = apply_provenance_filter(batch)
        for chunk in passed + blocked:
            out.suspicion[chunk["chunk_id"]] = 1.0 - chunk["provenance_score"]
            if chunk["provenance_blocked"] or chunk["provenance_score"] < self.flag_below:
                out.flagged_ids.add(chunk["chunk_id"])
        out.passed = passed
        out.blocked = blocked
        return out


class D4Stage:
    """Supervised classifier gate. `model` needs predict_proba(rows)."""
    name = "d4"

    def __init__(self, model, threshold: float = 0.5):
        if model is None:
            raise ValueError("D4Stage needs a model with predict_proba(rows)")
        self.model = model
        self.threshold = threshold

    def inspect(self, batch: list[dict], only_ids: set | None = None):
        """Classify chunks in only_ids (all if None). Others pass unchanged."""
        out = StageOutput()
        if getattr(self.model, "text_only", False):
            # Models that score raw chunk text (e.g. the repo's live 6-feature RF)
            # need neither upstream signals nor the 8-feature contract.
            rows = [{"chunk_id": c["chunk_id"], "text": c["text"]}
                    for c in batch if only_ids is None or c["chunk_id"] in only_ids]
        else:
            rows = build_feature_rows(batch, only_ids=only_ids)
        probabilities = self.model.predict_proba(rows)
        prob_by_id = {}
        for row, prob in zip(rows, probabilities):
            prob_by_id[row["chunk_id"]] = prob

        for chunk in batch:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in prob_by_id:
                out.passed.append(chunk)
                continue
            prob = prob_by_id[chunk_id]
            augmented = {**chunk, "d4_prob": prob, "d4_blocked": prob >= self.threshold}
            out.suspicion[chunk_id] = prob
            if prob >= self.threshold:
                out.flagged_ids.add(chunk_id)
                out.blocked.append(augmented)
            else:
                out.passed.append(augmented)
        return out

    def run(self, batch: list[dict]) -> StageOutput:
        return self.inspect(batch, None)


class GatedD4Stage:
    """
    Module 5: the d4 stage behind the adaptive router. Only chunks the router
    marks high-risk reach the classifier; low-risk chunks skip it.
    """
    name = "d4"

    def __init__(self, d4_stage: D4Stage, router: AdaptiveRouter):
        self.d4_stage = d4_stage
        self.router = router

    def run(self, batch: list[dict]) -> StageOutput:
        out = StageOutput()
        if not batch:
            return out

        collected = {}

        def inspector(chunks, only_ids):
            result = self.d4_stage.inspect(chunks, only_ids)
            collected["result"] = result
            return result.passed, result.blocked

        passed, blocked, decision = apply_adaptive_routing(batch, inspector, self.router)
        inner = collected.get("result")
        if inner is not None:
            out.suspicion = inner.suspicion
            out.flagged_ids = inner.flagged_ids
        out.passed = passed
        out.blocked = blocked
        out.extra = {
            "routed_high": decision.n_high,
            "routed_low": decision.n_low,
            "d4_calls_avoided": decision.n_low,
        }
        return out


def build_stages(config, d4_model=None) -> list:
    """Create the stage objects for a DefenseConfig, in its order."""
    params = config.params
    stages = []
    for name in config.stages:
        if name == "d1":
            stages.append(D1Stage(
                soft_z=params.get("d1_soft_z", SOFT_FLAG_ZSCORE),
                hard_z=params.get("d1_hard_z", HARD_BLOCK_ZSCORE),
            ))
        elif name == "d2":
            stages.append(D2Stage(
                alpha=params.get("d2_alpha", RERANK_ALPHA),
                flag_threshold=params.get("d2_flag_threshold", LOW_TRUST_FLAG_THRESHOLD),
                drop_low_trust=params.get("d2_drop_low_trust", False),
            ))
        elif name == "d3":
            stages.append(D3Stage(flag_below=params.get("d3_flag_below", 1.0)))
        elif name == "d4":
            d4 = D4Stage(d4_model, threshold=params.get("d4_threshold", 0.5))
            if config.use_router:
                threshold = config.router_threshold
                if threshold is None:
                    threshold = RISK_THRESHOLD
                stages.append(GatedD4Stage(d4, AdaptiveRouter(risk_threshold=threshold)))
            else:
                stages.append(d4)
    return stages
