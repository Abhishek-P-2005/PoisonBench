from fastapi import FastAPI
from api.routers import health, ingest, retrieve, generate

app = FastAPI(
    title="PoisonBench API",
    description="Baseline RAG pipeline: ingest -> chunk -> embed -> store -> retrieve -> generate.",
    version="0.1.0",
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(retrieve.router, tags=["retrieve"])
app.include_router(generate.router, tags=["generate"])


@app.get("/")
def root():
    return {
        "service": "PoisonBench API",
        "endpoints": ["/health", "/chunk", "/embed", "/ingest", "/retrieve", "/generate", "/docs"],
    }
