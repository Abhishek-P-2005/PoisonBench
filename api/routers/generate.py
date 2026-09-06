"""
Full RAG answer endpoint: retrieve top-k chunks, then generate an answer
with the local LLM. Same "no defense layer" caveat as retrieve.py applies --
this is the baseline path used for clean-pipeline validation in Phase 2.
"""
from fastapi import APIRouter
from api.pipeline.schemas import GenerateRequest, GenerateResponse
from api.pipeline.vectorstore import query as vs_query
from api.pipeline.llm import generate_answer

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    retrieved = vs_query(req.query, top_k=req.top_k)
    answer = generate_answer(req.query, retrieved)
    return GenerateResponse(query=req.query, answer=answer, retrieved=retrieved)
