from fastapi import APIRouter
from api.pipeline.vectorstore import get_collection
from api.pipeline.embedding import get_embedding_model

router = APIRouter()


@router.get("/health")
def health():
    status = {"api": "ok"}

    try:
        col = get_collection()
        status["chromadb"] = f"ok ({col.count()} chunks stored)"
    except Exception as e:
        status["chromadb"] = f"error: {e}"

    try:
        get_embedding_model()
        status["embedding_model"] = "ok"
    except Exception as e:
        status["embedding_model"] = f"error: {e}"

    return status
