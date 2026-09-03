"""Retriever: usa la MISMA embedding function que la ingesta (v1 las mezclaba)."""
from src.rag.ingest import get_embedding_function


class Retriever:
    def __init__(self, chroma_path: str = "./chroma_db", collection: str = "tizon_tardio",
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=chroma_path, settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = client.get_or_create_collection(
            name=collection, embedding_function=get_embedding_function(embedding_model))

    def search(self, query: str, k: int = 5) -> tuple[list[str], list[str]]:
        res = self.collection.query(query_texts=[query], n_results=k,
                                    include=["documents", "metadatas"])
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        return docs, [m.get("archivo", "?") for m in metas]

    def count(self) -> int:
        return self.collection.count()
