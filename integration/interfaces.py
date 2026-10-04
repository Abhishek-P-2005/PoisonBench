"""
Phase 4 deliverable: integration contracts.

This is the file that makes "plug the attack module, all four defenses, and
the evaluation harness into one running pipeline" (PRD Phase 4 / section 10,
Subash's integration ownership) actually enforceable instead of just a goal.
Akilan (Defenses 1-3, evaluation harness) and Abhishek (attack module,
Defense 4 comparison) build concrete implementations of these interfaces;
pipeline_orchestrator.py only ever talks to the interface, never to a
specific teammate's internal code. That's what "plug in without conflicts"
means in practice: nobody's module can silently change its own inputs or
outputs and break someone else's code, because everyone is coding against
the same contract checked in one place.

If a defense or the attack module doesn't fit these shapes, the fix is to
update THIS file (as a team, on purpose) and update every implementer --
not to quietly special-case it in the orchestrator.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Shared data shapes
# ---------------------------------------------------------------------------

@dataclass
class CorpusDocument:
    doc_id: str
    text: str
    source: str
    timestamp: str
    is_poisoned: bool = False   # ground truth, set only by the attack module


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    similarity_score: float
    source: str
    is_poisoned: bool           # ground truth, carried through for evaluation only
    metadata: dict = field(default_factory=dict)


@dataclass
class DefenseVerdict:
    chunk_id: str
    flagged: bool
    score: float                # higher = more suspicious; scale is defense-specific
    latency_ms: float


# ---------------------------------------------------------------------------
# Attack module contract (owned by Abhishek)
# ---------------------------------------------------------------------------

class AttackModule(ABC):
    """
    One implementation per attack strategy (naive_mimicry, embedding_optim,
    prompt_injection). Each implementation must be able to generate poisoned
    documents targeting a specific trigger query, at a given "dose" -- the
    orchestrator controls HOW MANY get injected (the poisoning ratio sweep),
    the attack module controls WHAT they look like.
    """

    name: str  # e.g. "naive_mimicry" -- must match PRD 6.2 attack names exactly,
               # this string is what gets written into runs.attack_type in Postgres

    @abstractmethod
    def generate_poisoned_documents(
        self, trigger_query: str, target_claim: str, n_documents: int
    ) -> list[CorpusDocument]:
        """
        Returns `n_documents` CorpusDocument objects, each with
        is_poisoned=True, crafted to surface for `trigger_query` and push
        `target_claim` (the false information) into the model's answer.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Defense contract (Defenses 1-3 owned by Akilan, Defense 4 owned by Subash
# with Abhishek's Random Forest as a second implementation of the same
# contract for the head-to-head comparison)
# ---------------------------------------------------------------------------

class Defense(ABC):
    """
    Every defense takes a batch of retrieved chunks and returns a verdict
    per chunk. Defenses must NOT read chunk.is_poisoned -- that field exists
    purely for the evaluation harness to score the defense afterwards, not
    for the defense to cheat with.
    """

    name: str  # e.g. "outlier", "trust_rerank", "provenance", "xgboost", "random_forest"

    @abstractmethod
    def score(self, chunks: list[RetrievedChunk]) -> list[DefenseVerdict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Evaluation harness contract (owned by Akilan)
# ---------------------------------------------------------------------------

class EvaluationHarness(ABC):
    """
    Consumes one full run's worth of retrieval results, defense verdicts,
    and generation output, and writes rows into the Postgres tables defined
    in postgres/init.sql (retrieval_logs, generation_logs, defense_flags).
    Must log EVERY run, including defense-off / attack-off baselines --
    those are needed to compute false-positive rates and cannot be
    regenerated later (PRD 5, "Log everything at every stage").
    """

    @abstractmethod
    def log_retrieval(self, run_id: int, query: str, results: list[RetrievedChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_generation(self, run_id: int, query: str, answer: str, poisoned_content_detected: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_defense_verdicts(self, run_id: int, defense_name: str, verdicts: list[DefenseVerdict],
                              ground_truth: dict[str, bool]) -> None:
        raise NotImplementedError
