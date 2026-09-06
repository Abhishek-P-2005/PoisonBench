"""
Embedding stage.

Loads sentence-transformers/all-MiniLM-L6-v2 ONCE as a module-level
singleton. Reloading the model per-request (or per-document during bulk
ingestion) is the single easiest way to make this pipeline needlessly slow --
the PRD explicitly flags "unbatched embedding calls that are needlessly slow"
as a likely bug class, so this module also always batches.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from api.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch-embed a list of texts. Always call this with a list, even for a
    single text -- do not loop and call it one text at a time; that defeats
    the point of batching and will make bulk corpus ingestion painfully slow.
    """
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity == dot product after this
    )
    return vectors.tolist()


def embedding_dims() -> int:
    return get_embedding_model().get_sentence_embedding_dimension()
