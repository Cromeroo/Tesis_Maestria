"""Nodo retrieve: fan-out sobre planned_queries. Delega en tools/rag_tool."""
import logging
from src.tools.rag_tool import search_knowledge

log = logging.getLogger(__name__)


def run(state: dict, *, retriever, k: int = 5) -> dict:
    contexts = []
    for q in state.get("planned_queries", []):
        out = search_knowledge(q, retriever, k=k)
        if out.get("warning"):
            log.warning("%s", out["warning"])
            continue
        contexts.append({"query": q, "documents": out["documents"], "sources": out["sources"]})
    if not contexts:
        return {"warnings": ["Base de conocimiento no disponible; respuesta solo visual."]}
    return {"contexts": contexts}
