"""
Defended retrieval path: retrieve -> D1 outlier -> D2 trust -> D3 provenance
-> D4 classifier (optionally behind the adaptive router) -> top-k context.

This module only composes code that already exists:
  * stages and pipeline   evaluation/stages.py, evaluation/pipeline.py
  * metadata bridge       evaluation/metadata_bridge.py
  * D4 live-text model    evaluation/d4_live_text.py
  * logging tables        evaluation/store.py (eval_* tables)
so the live API and the offline harness run the same defense code.

Ground-truth labels (is_poisoned) are never given to a defense:
DefensePipeline strips them before any stage runs. They are read only to
fill the evaluation logs and the "is_poisoned" field of the response.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from api.config import settings
from evaluation.config import CONTEXT_SIZE, N_RETRIEVE, STAGE_NAMES, make_config
from evaluation.metadata_bridge import enrich_metadata, has_canonical_provenance
from evaluation.metrics import evaluate_query
from evaluation.pipeline import DefensePipeline
from evaluation.runner import truth_map
from evaluation.scenarios import ScenarioQuery
from evaluation.stages import build_stages
from evaluation.store import ResultStore

log = logging.getLogger("poisonbench.defended")

DEFAULT_STAGES = ("d1", "d2", "d3", "d4")
PROVENANCE_MODES = ("auto", "thin", "stored", "derived")


# ---------------------------------------------------------------------------
# Defense 4 model
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_d4(kind: str):
    """Returns (model_or_None, reason_if_none). Cached, so the joblib loads once."""
    if kind in ("", "none"):
        return None, "D4_MODEL is 'none'"
    try:
        from evaluation.d4_live_text import load_live_text_model
        return load_live_text_model(kind), None
    except Exception as exc:  # missing file, checksum mismatch, missing xgboost, ...
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Result store (Postgres in Docker, any URL the ResultStore accepts)
# ---------------------------------------------------------------------------

_store = None
_store_retry_at = 0.0
_store_lock = threading.Lock()
_write_lock = threading.Lock()


def set_store(store) -> None:
    """Inject a store (tests use a SQLite one)."""
    global _store
    _store = store


def get_store():
    """The shared ResultStore, or None if logging is off or the DB is unreachable."""
    global _store, _store_retry_at
    if not settings.defended_log_enabled:
        return None
    with _store_lock:
        if _store is not None:
            return _store
        if time.time() < _store_retry_at:
            return None
        try:
            _store = ResultStore(settings.eval_db_dsn)
        except Exception as exc:
            log.warning("eval logging disabled for 60s, cannot open result store: %s", exc)
            _store_retry_at = time.time() + 60
            return None
        return _store


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def resolve_provenance(candidates: list[dict], requested: str) -> str:
    if requested not in PROVENANCE_MODES:
        raise ValueError(f"provenance must be one of {PROVENANCE_MODES}")
    if requested != "auto":
        return requested
    if candidates and has_canonical_provenance(candidates[0]["metadata"]):
        return "stored"
    return "thin"


def plan_pipeline(stages, use_router: bool, mode: str, d4_kind: str):
    """Decide which stages can really run. Returns (config, d4_model, skipped)."""
    requested = list(stages) if stages else list(DEFAULT_STAGES)
    for name in requested:
        if name not in STAGE_NAMES:
            raise ValueError(f"unknown stage '{name}', expected a subset of {STAGE_NAMES}")

    d4_model, d4_reason = load_d4(d4_kind)
    skipped, runnable = [], []
    for name in requested:
        if name in ("d2", "d3") and mode == "thin":
            skipped.append({"stage": name, "reason": "collection has no provenance metadata and provenance mode is 'thin'"})
        elif name == "d4" and d4_model is None:
            skipped.append({"stage": name, "reason": f"no Defense 4 model available ({d4_reason})"})
        else:
            runnable.append(name)

    router = False
    if use_router:
        if "d4" not in runnable:
            skipped.append({"stage": "router", "reason": "router gates D4, which is not running"})
        elif mode == "thin":
            skipped.append({"stage": "router", "reason": "router needs provenance metadata (mode is 'thin')"})
        else:
            router = True
    return make_config(runnable, use_router=router), d4_model, skipped


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _context_chunk(candidate: dict) -> dict:
    """Context chunk in the same shape as the baseline /retrieve results."""
    meta = candidate["metadata"]
    return {
        "chunk_id": candidate["chunk_id"],
        "doc_id": meta.get("doc_id", ""),
        "text": candidate["text"],
        "similarity_score": candidate["similarity"],
        "source": meta.get("source", ""),
        "is_poisoned": bool(meta.get("is_poisoned", False)),
        "metadata": meta,
    }


def run_defended(
    query_text: str,
    candidates: list[dict],
    top_k: int = CONTEXT_SIZE,
    stages=None,
    use_router: bool = False,
    provenance: str | None = None,
    d4_kind: str | None = None,
    log_run: bool = True,
    store=None,
) -> dict:
    """
    Run already-retrieved candidates ({chunk_id, text, similarity, metadata,
    embedding}) through the defense pipeline and return the full response dict.
    """
    mode = resolve_provenance(candidates, provenance or settings.defended_provenance)
    working = []
    for c in candidates:
        meta = c["metadata"]
        if mode == "derived":
            meta = enrich_metadata(c["chunk_id"], c["text"], meta)
        working.append({**c, "metadata": meta})

    config, d4_model, skipped = plan_pipeline(
        stages, use_router, mode, d4_kind if d4_kind is not None else settings.d4_model)
    pipeline = DefensePipeline(build_stages(config, d4_model=d4_model), context_size=top_k, name=config.name)
    result = pipeline.run(working)

    final_ids = [c["chunk_id"] for c in result.final_context]
    blocked_by = {}
    for record in result.stage_records:
        for chunk_id in record.blocked_ids:
            blocked_by[chunk_id] = record.name

    ordered = sorted(working, key=lambda c: c["similarity"], reverse=True)
    out_candidates = []
    for c in ordered:
        cid = c["chunk_id"]
        if cid in blocked_by:
            status = "blocked"
        elif cid in final_ids:
            status = "kept"
        else:
            status = "not_selected"
        out_candidates.append({
            "chunk_id": cid,
            "doc_id": c["metadata"].get("doc_id", ""),
            "text": c["text"],
            "similarity": c["similarity"],
            "source": c["metadata"].get("source", ""),
            "is_poisoned": bool(c["metadata"].get("is_poisoned", False)),  # label for evaluation only
            "status": status,
            "flagged": cid in result.flagged_ids,
            "blocked_by": blocked_by.get(cid),
            "suspicion": {r.name: r.suspicion[cid] for r in result.stage_records if cid in r.suspicion},
        })

    warnings = []
    if mode == "derived":
        warnings.append("Provenance is DERIVED from each chunk's source and timestamp, not authentic. D3's integrity check "
                        "cannot fail, and chunks with an empty timestamp fail D3's date check regardless of poisoning.")
    if mode == "thin":
        warnings.append("Collection has no provenance metadata: D2, D3 and the router were skipped.")
    if any(s["stage"] == "d4" for s in skipped):
        warnings.append("Defense 4 did not run; see skipped_stages.")

    response = {
        "query": query_text,
        "config": config.name,
        "provenance_mode": mode,
        "stages_run": list(config.stages),
        "skipped_stages": skipped,
        "router": config.use_router,
        "n_retrieved": len(working),
        "context": [_context_chunk(c) for c in result.final_context],
        "candidates": out_candidates,
        "stage_records": [
            {"stage": r.name, "position": r.position, "n_in": r.n_in, "n_blocked": r.n_blocked,
             "n_flagged": r.n_flagged, "latency_ms": round(r.latency_ms, 3), "extra": r.extra}
            for r in result.stage_records
        ],
        "total_latency_ms": round(result.total_latency_ms, 3),
        "warnings": warnings,
        "logged": False,
    }

    if log_run:
        target = store if store is not None else get_store()
        if target is not None:
            response["logged"] = _log(target, query_text, working, result, config, mode, top_k)
    return response


def defended_retrieve(
    query_text: str,
    n_retrieve: int = N_RETRIEVE,
    collection: str | None = None,
    **kwargs,
) -> dict:
    """Retrieve from Chroma (with embeddings), then run_defended()."""
    from api.pipeline.vectorstore import query_candidates   # lazy: needs chromadb + MiniLM
    candidates = query_candidates(query_text, n_results=n_retrieve, collection_name=collection)
    return run_defended(query_text, candidates, **kwargs)


# ---------------------------------------------------------------------------
# Logging into the eval_* tables
# ---------------------------------------------------------------------------

_SUSPICION_COLUMN = {"d1": "d1_z", "d2": "d2_suspicion", "d3": "d3_suspicion", "d4": "d4_prob"}


def _log(store, query_text, candidates, result, config, mode, context_size) -> bool:
    try:
        query_id = uuid.uuid4().hex[:12]
        scenario = ScenarioQuery(query_id=query_id, query_text=query_text, candidates=candidates,
                                 poison_marker="", attack_type="live", poisoning_ratio=0.0, seed=0)
        q_row, s_rows, f_rows = evaluate_query(scenario, result, context_size, truth_map(candidates))

        run_id = f"live|{config.name}|{mode}"
        chunk_rows = []
        for c in candidates:
            row = {"run_id": run_id, "query_id": query_id, "chunk_id": c["chunk_id"],
                   "is_poisoned": 1 if c["metadata"].get("is_poisoned") else 0,
                   "attack_type": "live", "poisoning_ratio": 0.0, "seed": 0, "similarity": c["similarity"]}
            for record in result.stage_records:
                column = _SUSPICION_COLUMN.get(record.name)
                if column and c["chunk_id"] in record.suspicion:
                    row[column] = record.suspicion[c["chunk_id"]]
            chunk_rows.append(row)
        for rows in (s_rows, f_rows):
            for row in rows:
                row["run_id"] = run_id
        q_row["run_id"] = run_id

        with _write_lock:
            existing = store.fetch("eval_runs", "run_id = " + store.placeholder, (run_id,))
            if existing.empty:
                store.insert("eval_runs", [{
                    "run_id": run_id, "group_name": "live", "config_name": config.name,
                    "config_json": config.to_json(), "attack_type": "live", "poisoning_ratio": 0.0,
                    "seed": 0, "source_name": f"api[{mode}]", "context_size": context_size,
                    "n_retrieve": len(candidates), "generation_enabled": 0,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }])
            store.insert("eval_query_results", [q_row])
            store.insert("eval_stage_logs", s_rows)
            store.insert("eval_chunk_scores", chunk_rows)
            store.insert("eval_false_positives", f_rows)
        return True
    except Exception as exc:
        log.warning("could not log defended query: %s", exc)
        try:
            store.conn.rollback()
        except Exception:
            pass
        return False
