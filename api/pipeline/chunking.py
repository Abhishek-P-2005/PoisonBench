"""
Chunking stage.

DESIGN DECISION (be ready to defend this in the viva):
We use fixed-size, word-count-based chunking with overlap, NOT semantic
chunking. Reasoning:

1. CVE/NVD advisories are already short, single-topic documents (a
   description paragraph + references). There usually isn't a long
   multi-topic document where semantic boundary detection would meaningfully
   change what gets grouped together.
2. Fixed-size chunking is fully deterministic. Given the same document, the
   same chunk boundaries appear on every run. Semantic chunking (e.g.
   embedding-similarity-based splitting) is not fully deterministic, or
   at minimum introduces its own model dependency and hyperparameters --
   this project already has a lot of moving parts and doesn't need a second
   embedding model in the loop just to cut text.
3. It keeps the attack surface simpler to reason about: the poisoning
   attacks operate on whole synthetic documents, so chunk-boundary
   sophistication isn't where the research value is.

Trade-off we accept: fixed-size chunking can occasionally split a sentence
mid-thought at a chunk boundary. Overlap (default 30 words) mitigates this.
"""
from api.config import settings


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size_words: int | None = None,
    chunk_overlap_words: int | None = None,
) -> list[dict]:
    """
    Split `text` into overlapping fixed-size word chunks.

    Returns a list of {chunk_id, text, position} dicts.
    chunk_id format: "{doc_id}::chunk{position}" -- deterministic and
    reversible, which the defense/evaluation layers rely on to trace a
    chunk back to its parent document.
    """
    size = chunk_size_words or settings.chunk_size_words
    overlap = chunk_overlap_words if chunk_overlap_words is not None else settings.chunk_overlap_words

    if overlap >= size:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    words = text.split()
    if not words:
        return []

    chunks = []
    position = 0
    start = 0
    step = size - overlap

    while start < len(words):
        end = min(start + size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append({
            "chunk_id": f"{doc_id}::chunk{position}",
            "text": chunk_str,
            "position": position,
        })
        position += 1
        if end == len(words):
            break
        start += step

    return chunks
