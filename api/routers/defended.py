"""
Defended endpoints and read-only views of the evaluation logs.

    POST /defended/retrieve    retrieve N candidates, run D1->D2->D3->D4, return top-k context
    POST /defended/generate    same, then answer from the surviving context (needs Ollama)
    GET  /evaluation/runs      logged run rows (eval_runs)
    GET  /evaluation/summary   per-run and per-stage aggregates of the logged live queries

The baseline /retrieve and /generate are unchanged and stay defense-free.
"""
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.pipeline import defended

router = APIRouter()


class DefendedRequest(BaseModel):
    query: str
    n_retrieve: int = Field(10, ge=1, le=50, description="candidates fetched before the defenses")
    top_k: int = Field(5, ge=1, le=50, description="chunks kept for the context")
    collection: Optional[str] = Field(None, description="existing Chroma collection; default is the configured one")
    stages: Optional[list[str]] = Field(None, description="subset of d1,d2,d3,d4 in run order; default all")
    use_router: bool = Field(False, description="gate D4 behind the adaptive router")
    provenance: Optional[str] = Field(None, description="auto|thin|stored|derived; default from settings")
    log: bool = True


def _run(req: DefendedRequest) -> dict:
    try:
        return defended.defended_retrieve(
            req.query, n_retrieve=req.n_retrieve, collection=req.collection,
            top_k=req.top_k, stages=req.stages, use_router=req.use_router,
            provenance=req.provenance, log_run=req.log,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        if req.collection and "does not exist" in str(exc).lower():
            raise HTTPException(status_code=404, detail=f"collection '{req.collection}' not found")
        raise HTTPException(status_code=503, detail=f"retrieval failed: {exc}")


@router.post("/defended/retrieve")
def defended_retrieve_endpoint(req: DefendedRequest):
    return _run(req)


@router.post("/defended/generate")
def defended_generate_endpoint(req: DefendedRequest):
    from api.pipeline.llm import generate_answer
    out = _run(req)
    if out["context"]:
        try:
            out["answer"] = generate_answer(req.query, out["context"])
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"generation failed: {exc}")
    else:
        out["answer"] = "No context passed the defense layer -- unable to answer."
    return out


# ── evaluation views ─────────────────────────────────────────────────────────

def _store_or_503():
    store = defended.get_store()
    if store is None:
        raise HTTPException(status_code=503, detail="evaluation store unavailable (see DEFENDED_LOG_ENABLED / EVAL_DB_URL)")
    return store


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notnull(frame), None).to_dict(orient="records")


@router.get("/evaluation/runs")
def evaluation_runs(limit: int = Query(50, ge=1, le=500), group: Optional[str] = None):
    store = _store_or_503()
    if group:
        runs = store.fetch("eval_runs", "group_name = " + store.placeholder, (group,))
    else:
        runs = store.fetch("eval_runs")
    if not runs.empty and "created_at" in runs.columns:
        runs = runs.sort_values("created_at", ascending=False, na_position="last")
    return {"n_runs": len(runs), "runs": _records(runs.head(limit))}


@router.get("/evaluation/summary")
def evaluation_summary():
    store = _store_or_503()
    queries = store.fetch("eval_query_results", "run_id LIKE " + store.placeholder, ("live|%",))
    stages = store.fetch("eval_stage_logs", "run_id LIKE " + store.placeholder, ("live|%",))
    if queries.empty:
        return {"live_queries": 0, "runs": [], "stages": []}

    per_run = (queries.groupby("run_id")
               .agg(n_queries=("query_id", "count"),
                    mean_latency_ms=("total_latency_ms", "mean"),
                    mean_clean_retention=("clean_retention", "mean"),
                    poison_in_raw_context=("raw_poison_in_context", "sum"),
                    poison_in_defended_context=("defended_poison_in_context", "sum"),
                    poison_blocked=("poison_blocked", "sum"),
                    clean_blocked=("clean_blocked", "sum"))
               .reset_index())
    per_stage = (stages.groupby(["run_id", "stage"])
                 .agg(n_in=("n_in", "sum"), n_blocked=("n_blocked", "sum"),
                      n_flagged=("n_flagged", "sum"), mean_latency_ms=("latency_ms", "mean"))
                 .reset_index()) if not stages.empty else pd.DataFrame()
    return {
        "live_queries": int(len(queries)),
        "note": ("poison counts use the is_poisoned label stored in the collection; for real traffic "
                 "with no labels they are zero and only latency, blocks and flags are meaningful"),
        "runs": _records(per_run),
        "stages": _records(per_stage),
    }
