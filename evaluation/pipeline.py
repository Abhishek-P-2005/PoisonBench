"""
evaluation/pipeline.py
----------------------
Runs a retrieval batch through an ordered list of defense stages and records,
for every stage: how many chunks went in, how many were blocked or flagged,
and how long it took (PRD v2.0 Req 2.3). Ground-truth labels are removed
before the defenses see the chunks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from evaluation.config import CONTEXT_SIZE


@dataclass
class StageRecord:
    name:         str
    position:     int
    n_in:         int
    n_blocked:    int
    n_flagged:    int
    latency_ms:   float
    blocked_ids:  set = field(default_factory=set)
    flagged_ids:  set = field(default_factory=set)
    suspicion:    dict = field(default_factory=dict)
    extra:        dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    final_context:    list
    stage_records:    list
    total_latency_ms: float
    blocked_ids:      set
    flagged_ids:      set


def strip_labels(chunk: dict) -> dict:
    """Copy of a chunk without the evaluation-only ground-truth fields."""
    metadata = {}
    for key, value in chunk["metadata"].items():
        if key not in ("is_poisoned", "attack_type"):
            metadata[key] = value
    return {**chunk, "metadata": metadata}


def order_by_similarity(batch: list[dict]) -> list[dict]:
    return sorted(batch, key=lambda c: c["similarity"], reverse=True)


class DefensePipeline:
    def __init__(self, stages: list, context_size: int = CONTEXT_SIZE, name: str = ""):
        self.stages = stages
        self.context_size = context_size
        self.name = name

    def run(self, candidates: list[dict]) -> PipelineResult:
        batch = order_by_similarity([strip_labels(c) for c in candidates])

        records = []
        blocked_ids = set()
        flagged_ids = set()
        started = time.perf_counter()

        for position, stage in enumerate(self.stages):
            n_in = len(batch)
            stage_start = time.perf_counter()
            output = stage.run(batch)
            latency_ms = (time.perf_counter() - stage_start) * 1000.0

            stage_blocked = {c["chunk_id"] for c in output.blocked}
            records.append(StageRecord(
                name=stage.name,
                position=position,
                n_in=n_in,
                n_blocked=len(stage_blocked),
                n_flagged=len(output.flagged_ids),
                latency_ms=latency_ms,
                blocked_ids=stage_blocked,
                flagged_ids=set(output.flagged_ids),
                suspicion=dict(output.suspicion),
                extra=dict(output.extra),
            ))
            blocked_ids |= stage_blocked
            flagged_ids |= set(output.flagged_ids)
            batch = output.passed

        total_latency_ms = (time.perf_counter() - started) * 1000.0
        return PipelineResult(
            final_context=batch[: self.context_size],
            stage_records=records,
            total_latency_ms=total_latency_ms,
            blocked_ids=blocked_ids,
            flagged_ids=flagged_ids,
        )
