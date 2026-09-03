"""Nodo plan: genera 1-2 queries dinámicas (v1 hacía 3-4 fijas = 4 llamadas LLM)."""
from src.graph import prompts

_FALLBACK = {
    "Tizon_tardio": ["síntomas tizón tardío tomate", "tratamiento y prevención tizón tardío"],
    "Otras_enfermedades": ["diferenciar tizón tardío de otras enfermedades tomate"],
    "Sana": ["buenas prácticas para mantener tomate sano y prevenir tizón"],
}


def run(state: dict, *, llm, settings) -> dict:
    if not state.get("in_scope", True):
        return {"planned_queries": []}  # fuera de dominio: no gastar retrieval
    d = state["diagnosis"]
    k = settings.MAX_QUERIES
    if llm is None:  # offline / tests
        return {"planned_queries": _FALLBACK.get(d["label"], _FALLBACK["Sana"])[:k]}
    resp = llm.invoke(prompts.QUERY_PLANNER.format(
        label=d["label"], confidence=d["confidence"], k=k))
    text = resp.content if hasattr(resp, "content") else str(resp)
    queries = [q.strip("- •1234567890. ") for q in text.strip().split("\n") if q.strip()][:k]
    return {"planned_queries": queries or ["tizón tardío tomate síntomas tratamiento"]}
