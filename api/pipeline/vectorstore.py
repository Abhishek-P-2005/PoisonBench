"""
ChromaDB access layer. Every other module talks to Chroma through this
file only -- nobody else should import chromadb directly. That keeps the
storage backend swappable and gives us one place to enforce the metadata
schema every chunk must carry.

Metadata schema (per PRD 4/6.1 -- source, timestamp, doc ID -- plus fields
the defense layer needs):
    doc_id        : str   -- parent document id
    source        : str   -- e.g. "NVD", "unverified_upload"
    timestamp     : str   -- ISO date string
    is_poisoned   : bool  -- GROUND TRUTH label. Only ever set by the attack
                             module when it injects a document. Defenses must
                             never read this at decision time (that would be
                             cheating); it exists purely so the evaluation
                             harness can compute true/false positive rates.
    position      : int   -- chunk position within the parent document
"""
import chromadb
from functools import lru_cache
from api.config import settings
from api.pipeline.embedding import embed_texts


@lru_cache(maxsize=1)
def get_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: list[dict]) -> int:
    """
    chunks: list of {chunk_id, text, doc_id, source, timestamp, is_poisoned, position}
    Embeds and stores in one batched call. Returns number of chunks stored.
    """
    if not chunks:
        return 0

    collection = get_collection()
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "source": c["source"],
            "timestamp": c.get("timestamp") or "",
            "is_poisoned": bool(c.get("is_poisoned", False)),
            "position": c.get("position", 0),
        }
        for c in chunks
    ]

    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    return len(ids)


def query(query_text: str, top_k: int | None = None) -> list[dict]:
    """
    Returns a list of {chunk_id, doc_id, text, similarity_score, source,
    is_poisoned, metadata}, ranked best-first.

    NOTE on the score: Chroma returns a cosine *distance* (0 = identical).
    We convert to a similarity score (1 - distance) so every downstream
    consumer (defenses, evaluation harness) works with "higher = more
    similar", matching the PRD's "similarity" language throughout.
    """
    k = top_k or settings.default_top_k
    collection = get_collection()
    query_vec = embed_texts([query_text])[0]

    res = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
        results.append({
            "chunk_id": chunk_id,
            "doc_id": meta.get("doc_id", ""),
            "text": text,
            "similarity_score": 1.0 - dist,
            "source": meta.get("source", ""),
            "is_poisoned": meta.get("is_poisoned", False),
            "metadata": meta,
        })
    return results


def get_all_chunks() -> list[dict]:
    """
    Pull the entire collection back out (embeddings included). Used by
    Defense 1 (outlier detection needs the full embedding neighborhood) and
    by defense4_xgboost/feature_engineering.py (needs the full corpus to
    compute centroid distances). Fine at this project's corpus scale
    (thousands of CVE chunks); would need pagination at real production scale.
    """
    collection = get_collection()
    res = collection.get(include=["documents", "metadatas", "embeddings"])
    out = []
    for chunk_id, text, meta, emb in zip(
        res["ids"], res["documents"], res["metadatas"], res["embeddings"]
    ):
        out.append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": meta,
            "embedding": emb,
        })
    return out


def reset_collection():
    """Deletes and recreates the collection. Used between clean ablation runs."""
    client = get_client()
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass
    return get_collection()
