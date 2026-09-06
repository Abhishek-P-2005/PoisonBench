"""
Shared request/response schemas.
Kept separate from routers so that the attack module and defense modules
(built by Abhishek / Akilan) can import these types without importing
FastAPI route code, avoiding circular imports.
"""
from typing import Optional
from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    doc_id: str
    text: str
    source: str = Field(..., description="e.g. 'NVD', 'unverified_upload'")
    timestamp: Optional[str] = None
    is_poisoned: bool = False   # ground-truth label, set by the attack module only


class ChunkRequest(BaseModel):
    text: str
    chunk_size_words: Optional[int] = None
    chunk_overlap_words: Optional[int] = None


class ChunkOut(BaseModel):
    chunk_id: str
    text: str
    position: int


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedOut(BaseModel):
    vectors: list[list[float]]
    dims: int


class IngestRequest(BaseModel):
    documents: list[DocumentIn]


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_stored: int


class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    similarity_score: float
    source: str
    is_poisoned: bool
    metadata: dict


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


class GenerateRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class GenerateResponse(BaseModel):
    query: str
    answer: str
    retrieved: list[RetrievedChunk]
