"""
Central configuration for the whole pipeline.
Every stage (chunking, embedding, vectorstore, llm) reads its settings from
here instead of hardcoding host/port pairs, so the same code works whether
it's running inside Docker (service DNS names) or being called from a host
script for local testing (localhost + mapped ports).
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "poisonbench_corpus"

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "poisonbench"
    postgres_user: str = "poisonbench"
    postgres_password: str = "poisonbench"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Ollama (local LLM)
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_model: str = "phi3:mini"

    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Chunking (fixed-size, see api/pipeline/chunking.py for rationale)
    chunk_size_words: int = 180
    chunk_overlap_words: int = 30

    # Retrieval
    default_top_k: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"


settings = Settings()
