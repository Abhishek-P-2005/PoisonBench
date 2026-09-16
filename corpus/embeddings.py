from __future__ import annotations

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]