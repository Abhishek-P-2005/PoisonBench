"""
Exposes ingestion pipeline stages as independently callable endpoints.
Per PRD 4.1: "exposing pipeline stages as testable, independently callable
endpoints" is the whole reason we're using FastAPI instead of a plain script.

/chunk  -> test chunking alone, no side effects
/embed  -> test embedding alone, no side effects
/ingest -> full pipeline: chunk -> embed -> store, for each document
"""
from fastapi import APIRouter
from api.pipeline.schemas import (
    ChunkRequest, ChunkOut, EmbedRequest, EmbedOut,
    IngestRequest, IngestResponse,
)
from api.pipeline.chunking import chunk_text
from api.pipeline.embedding import embed_texts, embedding_dims
from api.pipeline.vectorstore import upsert_chunks

router = APIRouter()


@router.post("/chunk", response_model=list[ChunkOut])
def chunk_endpoint(req: ChunkRequest):
    chunks = chunk_text(
        text=req.text,
        doc_id="preview",
        chunk_size_words=req.chunk_size_words,
        chunk_overlap_words=req.chunk_overlap_words,
    )
    return chunks


@router.post("/embed", response_model=EmbedOut)
def embed_endpoint(req: EmbedRequest):
    vectors = embed_texts(req.texts)
    return EmbedOut(vectors=vectors, dims=embedding_dims())


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest):
    all_chunks = []
    for doc in req.documents:
        doc_chunks = chunk_text(text=doc.text, doc_id=doc.doc_id)
        for c in doc_chunks:
            c["doc_id"] = doc.doc_id
            c["source"] = doc.source
            c["timestamp"] = doc.timestamp
            c["is_poisoned"] = doc.is_poisoned
        all_chunks.extend(doc_chunks)

    stored = upsert_chunks(all_chunks)
    return IngestResponse(documents_ingested=len(req.documents), chunks_stored=stored)
