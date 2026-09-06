"""
Baseline retrieval endpoint: plain top-k cosine similarity search, no
defense layer applied. This is intentional for Phase 1/2 -- the PRD requires
validating the CLEAN pipeline before any attack or defense work starts
(section 6.1), so this endpoint must stay "dumb" during baseline validation.

The defense-wrapped retrieval path (outlier -> trust re-rank -> provenance ->
classifier) is assembled later in integration/pipeline_orchestrator.py, which
calls vectorstore.query() and then layers defenses on top -- it does not
replace this endpoint, it composes with it.
"""
from fastapi import APIRouter
from api.pipeline.schemas import RetrieveRequest, RetrieveResponse
from api.pipeline.vectorstore import query as vs_query

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    results = vs_query(req.query, top_k=req.top_k)
    return RetrieveResponse(query=req.query, results=results)
