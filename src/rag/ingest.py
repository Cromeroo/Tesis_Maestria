"""Ingesta unificada (fusiona pipeline_rag.py + limpieza.py de v1)."""
import time
from pathlib import Path

from src.rag.preprocessing import extract_text_pdf, detect_lang, clean_text


def get_embedding_function(model_name: str):
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


class IngestPipeline:
    """Procesa PDFs -> ChromaDB. `llm` opcional: solo se usa para traducir EN->ES."""

    def __init__(self, docs_dir: str | Path, chroma_path: str = "./chroma_db",
                 collection: str = "tizon_tardio",
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 llm=None):
        self.docs_dir = Path(docs_dir)
        self.chroma_path = chroma_path
        self.collection_name = collection
        self.embedding_model = embedding_model
        self.llm = llm
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def translate_if_needed(self, text: str, lang: str) -> str:
        if lang == "es" or self.llm is None:
            return text
        out, size = [], 1500
        for i in range(0, len(text), size):
            chunk = text[i:i + size]
            try:
                resp = self.llm.invoke(f"Traduce al español conservando terminología técnica: {chunk}")
                out.append(resp.content if hasattr(resp, "content") else str(resp))
            except Exception as e:
                print(f"[ingest] fallo traducción chunk: {e}")
                out.append(chunk)
            time.sleep(1)
        return " ".join(out)

    def run(self, recreate: bool = True) -> dict:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=self.chroma_path, settings=ChromaSettings(anonymized_telemetry=False))
        if recreate:
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
        col = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Tizón tardío en tomate"},
            embedding_function=get_embedding_function(self.embedding_model),
        )
        pdfs = sorted(self.docs_dir.glob("*.pdf"))
        n_chunks = 0
        for pdf in pdfs:
            raw = extract_text_pdf(pdf)
            if not raw.strip():
                continue
            lang = detect_lang(raw)
            text = clean_text(self.translate_if_needed(raw, lang))
            chunks = self.splitter.split_text(text)
            for i, ch in enumerate(chunks):
                col.add(
                    documents=[ch],
                    metadatas=[{"archivo": pdf.name, "idioma_origen": lang,
                                "chunk_id": i, "total_chunks": len(chunks)}],
                    ids=[f"{pdf.name}_chunk_{i}"],
                )
            n_chunks += len(chunks)
            print(f"[ingest] {pdf.name} ({lang}): {len(chunks)} chunks")
        return {"pdfs": len(pdfs), "chunks": n_chunks, "collection": self.collection_name}
