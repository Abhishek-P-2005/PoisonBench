"""
evaluation/config.py
--------------------
Shared settings and the DefenseConfig description used by the whole harness.
A DefenseConfig fully describes one pipeline, so a run can be reproduced
from the config JSON stored in the database (PRD NFR-2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Attack names used by the synthetic source. Abhishek's real attack module
# can use any strings; the harness only passes them through.
ATTACK_TYPES = ("naive", "optimized", "injection")
CLEAN_ATTACK = "none"

POISONING_RATIOS = (0.01, 0.05, 0.10, 0.20)

N_RETRIEVE = 10        # candidates returned by the retriever (top-k)
CONTEXT_SIZE = 5       # chunks that finally go into the LLM prompt
FPR_CEILING = 0.05     # PRD v2.0 Req 1.3

# Static overhead limit from PRD v2.0 NFRs (milliseconds per query batch).
STATIC_OVERHEAD_LIMIT_MS = 150.0

STAGE_NAMES = ("d1", "d2", "d3", "d4")


@dataclass
class DefenseConfig:
    """
    stages          : stage names in execution order, e.g. ("d1", "d2", "d3")
    params          : optional overrides, keys used by the stages:
                        d1_soft_z, d1_hard_z, d2_alpha, d2_flag_threshold,
                        d2_drop_low_trust, d3_flag_below, d4_threshold
    use_router      : gate the d4 stage with the Module 5 adaptive router
    router_threshold: risk threshold for the router (None = config default)
    """
    name:             str
    stages:           tuple
    params:           dict = field(default_factory=dict)
    use_router:       bool = False
    router_threshold: float | None = None

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "stages": list(self.stages),
            "params": self.params,
            "use_router": self.use_router,
            "router_threshold": self.router_threshold,
        }, sort_keys=True)


def make_config(
    stages,
    params: dict | None = None,
    use_router: bool = False,
    router_threshold: float | None = None,
    label: str = "",
) -> DefenseConfig:
    """Build a DefenseConfig with an automatic readable name like 'd1>d2>d3'."""
    stages = tuple(stages)
    for stage in stages:
        if stage not in STAGE_NAMES:
            raise ValueError(f"unknown stage '{stage}', expected one of {STAGE_NAMES}")
    name = ">".join(stages) if stages else "none"
    if use_router:
        name = name + "+router"
    if label:
        name = f"{name}[{label}]"
    return DefenseConfig(
        name=name,
        stages=stages,
        params=dict(params or {}),
        use_router=use_router,
        router_threshold=router_threshold,
    )
