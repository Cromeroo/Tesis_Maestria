"""rag-svc: microservicio RAG. POST /search -> documentos + fuentes."""
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from src.config import get_settings
from src.utils.logging import setup_logging

setup_logging()
log = logging.getLogger(__name__)
app = FastAPI(title="rag-svc")
_ret = None


class Query(BaseModel):
    query: str
    k: int = 5


class NullRetriever:
    def search(self, query, k=5):
        return [], []

    def count(self):
        return 0


def get_ret():
    global _ret
    if _ret is None:
        try:
            from src.rag.retriever import Retriever
            s = get_settings()
            _ret = Retriever(chroma_path=s.CHROMA_PATH, collection=s.CHROMA_COLLECTION,
                             embedding_model=s.EMBEDDING_MODEL)
        except Exception as e:
            log.warning("Chroma no disponible (%s); retrieval vacío.", e)
            _ret = NullRetriever()
    return _ret


@app.get("/health")
def health():
    return {"docs": get_ret().count()}


@app.post("/search")
def search(q: Query):
    docs, sources = get_ret().search(q.query, k=q.k)
    return {"documents": docs, "sources": sources}
