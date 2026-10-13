"""
Phase 4 deliverable: the actual "one running pipeline" the PRD asks for.

Defense order is fixed here, matching PRD Section 5: outlier detection ->
trust re-rank -> provenance -> ML classifier (as a final gate). This is a
DELIBERATE design choice, not arbitrary: cheap statistical checks run first
to cut the candidate set down before the more expensive trust/provenance
lookups and the classifier inference pass see it. Changing this order
changes results (a chunk downstream never even reaches a later defense if
an earlier one already dropped it in "hard filter" mode) -- if the team
decides to test a different order, change DEFENSE_ORDER here and note it
explicitly in the results section, don't silently reorder.

Defenses 1-3 are Akilan's real implementations, wrapped by
HarnessStageDefense around evaluation/stages.py (the same code the evaluation
harness and the /defended/* API endpoints run). Defense 4 is Subash's XGBoost
wrapper. NoOpDefense is kept only as a stand-in for tests.

For the full defended path (10 candidates with embeddings, router, logging) use
api/pipeline/defended.py; this orchestrator is the simpler top-k path.
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from api.pipeline.vectorstore import query as vs_query, upsert_chunks   # noqa: E402
from api.pipeline.llm import generate_answer                             # noqa: E402
from api.pipeline.chunking import chunk_text                             # noqa: E402
from integration.interfaces import Defense, RetrievedChunk, DefenseVerdict  # noqa: E402
from evaluation.stages import D1Stage, D2Stage, D3Stage                    # noqa: E402

DEFENSE_ORDER = ["outlier", "trust_rerank", "provenance", "xgboost"]


# ---------------------------------------------------------------------------
# Placeholder defense -- replace with Akilan's real Defense 1/2/3 classes.
# Keeping a NoOp implementation here means the orchestrator is runnable and
# testable end-to-end today, instead of blocking on teammates' modules.
# ---------------------------------------------------------------------------

class NoOpDefense(Defense):
    """Flags nothing. Stand-in for a defense that hasn't been implemented yet."""

    def __init__(self, name: str):
        self.name = name

    def score(self, chunks: list[RetrievedChunk]) -> list[DefenseVerdict]:
        return [
            DefenseVerdict(chunk_id=c.chunk_id, flagged=False, score=0.0, latency_ms=0.0)
            for c in chunks
        ]


class HarnessStageDefense(Defense):
    """
    Runs one real stage from evaluation/stages.py behind the Defense interface.

    vs_query() returns no embeddings or provenance fields, so the adapter
    embeds the chunk texts and, for provenance-based stages, derives the
    missing fields with evaluation.metadata_bridge.enrich_metadata (labelled
    "derived": see that module for what it does and does not prove).
    Ground-truth labels are removed before the stage sees a chunk.

    flag_on="blocked": a chunk is flagged (and so removed by this orchestrator)
    only if the stage hard-blocks it. flag_on="flagged": any flag removes it.
    """

    def __init__(self, name: str, stage_factory, needs_provenance: bool = False, flag_on: str = "blocked"):
        self.name = name
        self.stage_factory = stage_factory
        self.needs_provenance = needs_provenance
        self.flag_on = flag_on

    def score(self, chunks: list[RetrievedChunk]) -> list[DefenseVerdict]:
        if not chunks:
            return []
        from api.pipeline.embedding import embed_texts
        from evaluation.metadata_bridge import enrich_metadata

        t0 = time.time()
        vectors = embed_texts([c.text for c in chunks])
        batch = []
        for c, vec in zip(chunks, vectors):
            meta = {k: v for k, v in c.metadata.items() if k not in ("is_poisoned", "attack_type")}
            if self.needs_provenance:
                meta = enrich_metadata(c.chunk_id, c.text, meta)
            batch.append({"chunk_id": c.chunk_id, "text": c.text, "similarity": c.similarity_score,
                          "metadata": meta, "embedding": vec})

        out = self.stage_factory().run(batch)
        removed = {b["chunk_id"] for b in out.blocked} if self.flag_on == "blocked" else set(out.flagged_ids)
        ms = (time.time() - t0) * 1000 / len(chunks)
        return [
            DefenseVerdict(chunk_id=c.chunk_id, flagged=c.chunk_id in removed,
                           score=float(out.suspicion.get(c.chunk_id, 0.0)), latency_ms=ms)
            for c in chunks
        ]


class XGBoostDefense(Defense):
    """Real Defense 4 wrapper. Requires a trained model (see train_xgboost.py)."""

    name = "xgboost"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, chunks: list[RetrievedChunk]) -> list[DefenseVerdict]:
        from defenses.defense4_xgboost.evaluate import score_chunks
        from defenses.defense4_xgboost.feature_engineering import (
            embedding_features, text_stat_features, source_features,
            retrieval_score_features, compute_corpus_centroid,
        )
        import numpy as np
        from api.pipeline.vectorstore import get_all_chunks

        t0 = time.time()

        # Build features for exactly these chunks against the current corpus
        # centroid. In a full ablation run this is expensive to redo per
        # query; the evaluation harness / ablation script should batch this
        # once per run rather than calling XGBoostDefense.score() per query
        # in a tight loop.
        all_chunks = get_all_chunks()
        embeddings = np.array([c["embedding"] for c in all_chunks])
        centroid = compute_corpus_centroid(embeddings)
        by_id = {c["chunk_id"]: c for c in all_chunks}

        feature_rows = []
        for c in chunks:
            full = by_id.get(c.chunk_id)
            if full is None:
                continue
            row = {"chunk_id": c.chunk_id}
            row.update(embedding_features(np.array(full["embedding"]), centroid, embeddings))
            row.update(text_stat_features(c.text))
            row.update(source_features(c.source, 0.0))
            row.update(retrieval_score_features([c.similarity_score], [c.similarity_score]))
            feature_rows.append(row)

        if not feature_rows:
            return []

        scored = score_chunks(feature_rows, threshold=self.threshold)
        elapsed_ms = (time.time() - t0) * 1000 / max(len(feature_rows), 1)

        return [
            DefenseVerdict(chunk_id=s["chunk_id"], flagged=s["flagged"], score=s["score"], latency_ms=elapsed_ms)
            for s in scored
        ]


DEFENSE_REGISTRY: dict[str, Defense] = {
    "outlier": HarnessStageDefense("outlier", D1Stage),                                        # Defense 1
    "trust_rerank": HarnessStageDefense("trust_rerank", D2Stage, True, flag_on="flagged"),     # Defense 2
    "provenance": HarnessStageDefense("provenance", D3Stage, True),                            # Defense 3
    "xgboost": XGBoostDefense(),                 # real, Subash
}


def _to_retrieved_chunk(r: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"],
        similarity_score=r["similarity_score"], source=r["source"],
        is_poisoned=r["is_poisoned"], metadata=r["metadata"],
    )


def run_query_through_defenses(
    query_text: str, top_k: int = 5, defenses_enabled: list[str] | None = None,
) -> dict:
    """
    Runs one query through: retrieve -> defense chain (in DEFENSE_ORDER) ->
    filter -> generate. Returns everything the evaluation harness needs to
    log for this query.
    """
    enabled = defenses_enabled if defenses_enabled is not None else DEFENSE_ORDER

    raw_results = vs_query(query_text, top_k=top_k)
    chunks = [_to_retrieved_chunk(r) for r in raw_results]

    all_verdicts: dict[str, list[DefenseVerdict]] = {}
    surviving_chunks = chunks

    for defense_name in DEFENSE_ORDER:
        if defense_name not in enabled:
            continue
        defense = DEFENSE_REGISTRY[defense_name]
        verdicts = defense.score(surviving_chunks)
        all_verdicts[defense_name] = verdicts

        flagged_ids = {v.chunk_id for v in verdicts if v.flagged}
        surviving_chunks = [c for c in surviving_chunks if c.chunk_id not in flagged_ids]

    retrieved_for_generation = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
         "similarity_score": c.similarity_score, "source": c.source,
         "is_poisoned": c.is_poisoned, "metadata": c.metadata}
        for c in surviving_chunks
    ]
    answer = generate_answer(query_text, retrieved_for_generation) if retrieved_for_generation else \
        "No context passed the defense layer -- unable to answer."

    return {
        "query": query_text,
        "pre_defense_chunks": chunks,
        "post_defense_chunks": surviving_chunks,
        "defense_verdicts": all_verdicts,
        "answer": answer,
    }


def inject_poisoned_documents(poisoned_docs: list) -> int:
    """
    Takes CorpusDocument objects from an AttackModule implementation, chunks
    and stores them exactly like the clean ingestion path (same chunking
    function, same vectorstore), so poisoned chunks are indistinguishable in
    format from clean ones -- only metadata.is_poisoned differs.
    """
    all_chunks = []
    for doc in poisoned_docs:
        doc_chunks = chunk_text(text=doc.text, doc_id=doc.doc_id)
        for c in doc_chunks:
            c["doc_id"] = doc.doc_id
            c["source"] = doc.source
            c["timestamp"] = doc.timestamp
            c["is_poisoned"] = True
        all_chunks.extend(doc_chunks)
    return upsert_chunks(all_chunks)


if __name__ == "__main__":
    # Smoke test: run a single query through the full defense chain against
    # whatever corpus is currently loaded. Requires the API stack to be up
    # and at least one document ingested.
    result = run_query_through_defenses(
        "What is a remote code execution vulnerability?", top_k=5,
        defenses_enabled=["outlier", "trust_rerank", "provenance"],  # xgboost needs a trained model
    )
    print(f"Query: {result['query']}")
    print(f"Chunks before defense: {len(result['pre_defense_chunks'])}")
    print(f"Chunks after defense : {len(result['post_defense_chunks'])}")
    print(f"Answer: {result['answer'][:300]}")
