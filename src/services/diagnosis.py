"""Fachada: UI y API hablan aquí, nunca con el grafo directo."""
import logging
import os
import uuid
from src.config import get_settings
from src.models.classifier import LeafClassifier
from src.rag.retriever import Retriever
from src.graph.builder import build_graph
from src.utils.device import device_info
from src.utils.logging import setup_logging

log = logging.getLogger(__name__)


def _build_llm(settings):
    if os.getenv("TESIS_OFFLINE", "0") == "1":
        return None
    try:
        from langchain_google_vertexai import ChatVertexAI
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                              settings.GOOGLE_APPLICATION_CREDENTIALS)
        return ChatVertexAI(model_name=settings.GEMINI_MODEL,
                            temperature=0.7, max_output_tokens=2048,
                            location=settings.GEMINI_LOCATION)
    except Exception as e:
        log.warning("LLM no disponible (%s); modo degradado.", e)
        return None


def _build_checkpointer(path: str):
    """SqliteSaver persistente; MemorySaver si el paquete falta.

    En langgraph-checkpoint-sqlite v3, from_conn_string() es un context manager:
    se entra UNA vez y se conserva abierto durante la vida del servicio.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        ctx = SqliteSaver.from_conn_string(path)
        cp = ctx.__enter__()
        log.info("checkpointer sqlite en %s", path)
        return ctx, cp, "sqlite"  # ctx se conserva para no cerrar la conexión
    except Exception as e:
        from langgraph.checkpoint.memory import MemorySaver
        log.warning("sqlite no disponible (%s); MemorySaver en RAM.", e)
        return None, MemorySaver(), "memory"


class NullRetriever:
    def search(self, query, k=5):
        return [], []

    def count(self):
        return 0


class DiagnosisService:
    """Si VISION_URL/RAG_URL están definidos, orquesta los microservicios
    (clientes remotos con la misma interfaz); si no, corre in-process."""

    def __init__(self, classifier=None, retriever=None):
        setup_logging()
        self.settings = get_settings()
        self.classifier = classifier or self._build_classifier()
        self.retriever = retriever or self._build_retriever()
        self.llm = _build_llm(self.settings)
        self._cp_ctx, checkpointer, self.checkpointer_kind = _build_checkpointer(
            self.settings.CHECKPOINT_PATH)
        self.graph = build_graph(self.classifier, self.retriever or NullRetriever(),
                                 self.llm, self.settings,
                                 checkpointer=checkpointer)

    def _build_classifier(self):
        if self.settings.VISION_URL:
            from src.tools.remote import RemoteClassifier
            log.info("visión remota: %s", self.settings.VISION_URL)
            return RemoteClassifier(self.settings.VISION_URL)
        return LeafClassifier(weights_path=self.settings.CNN_WEIGHTS_PATH,
                              backbone=self.settings.CNN_BACKBONE,
                              device=self.settings.DEVICE)

    def _build_retriever(self):
        if self.settings.RAG_URL:
            from src.tools.remote import RemoteRetriever
            log.info("RAG remoto: %s", self.settings.RAG_URL)
            return RemoteRetriever(self.settings.RAG_URL)
        try:
            return Retriever(chroma_path=self.settings.CHROMA_PATH,
                             collection=self.settings.CHROMA_COLLECTION,
                             embedding_model=self.settings.EMBEDDING_MODEL)
        except Exception as e:
            log.warning("Chroma no disponible (%s); retrieval vacío.", e)
            return None

    def diagnose(self, image_path: str, user_question: str = "¿Qué debo hacer?",
                 thread_id: str | None = None, lot_id: str | None = None) -> dict:
        """Diagnostica. Sin thread_id crea un hilo nuevo; con thread_id reanuda
        (misma memoria: history conserva turnos previos). Con lot_id, registra
        el turno en el timeline del lote."""
        thread_id = thread_id or uuid.uuid4().hex[:12]
        out = self.graph.invoke(
            {"image_path": image_path, "user_question": user_question,
             "in_scope": True, "needs_input": False, "clarification": "",
             "contexts": [], "history": [], "warnings": []},
            config={"configurable": {"thread_id": thread_id}})
        out["thread_id"] = thread_id
        if lot_id and out.get("diagnosis"):
            try:
                from src.services.lots import LotStore
                LotStore().record(lot_id, out["diagnosis"],
                                  out.get("needs_input", False),
                                  user_question, thread_id)
                out["lot_id"] = lot_id
            except Exception as e:
                log.warning("no se pudo registrar lote (%s)", e)
        return out

    def thread_history(self, thread_id: str) -> list[dict]:
        """Lee la memoria (history) de un hilo sin invocar el grafo."""
        try:
            state = self.graph.get_state(config={"configurable": {"thread_id": thread_id}})
            return (state.values.get("history", []) if state.values else [])
        except Exception:
            return []

    def lot(self, lot_id: str) -> dict:
        """Timeline + resumen de un lote para la UI."""
        from src.services.lots import LotStore
        store = LotStore()
        return {"lot_id": lot_id, "summary": store.summary(lot_id),
                "timeline": store.timeline(lot_id)}

    def health(self) -> dict:
        device = device_info(self.classifier.device) if hasattr(self.classifier, "device") \
            else {"device": "remote", "via": self.settings.VISION_URL}
        return {"cnn_loaded": self.classifier.loaded if hasattr(self.classifier, "loaded") else None,
                "device": device,
                "chroma_docs": self.retriever.count() if self.retriever else 0,
                "llm": self.llm is not None,
                "checkpointer": self.checkpointer_kind,
                "mode": "microservices" if (self.settings.VISION_URL or self.settings.RAG_URL) else "in-process"}
