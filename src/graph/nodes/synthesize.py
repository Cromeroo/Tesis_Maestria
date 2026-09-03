"""Nodo synthesize: UNA sola llamada LLM con todo el contexto (v1 hacía 4)."""
from src.graph import prompts

ABSTAIN = ("Soy un asistente de sanidad del tomate: solo respondo sobre tizón tardío, "
           "síntomas, tratamiento y prevención en tomate. Reformula tu pregunta en ese "
           "ámbito o sube una foto de una hoja.")


def run(state: dict, *, llm) -> dict:
    d = state["diagnosis"]
    if not state.get("in_scope", True):
        return {"final_answer": ABSTAIN,
                "history": [{"event": "abstain",
                             "question": state.get("user_question", "")[:200]}]}
    blocks = []
    for c in state.get("contexts", []):
        blocks.append(f"## {c['query']}\nFuentes: {', '.join(set(c['sources']))}\n" +
                      "\n\n".join(c["documents"][:3]))
    context = "\n\n".join(blocks) if blocks else "(sin contexto recuperado)"
    if llm is None:  # offline / tests
        return {"final_answer": (
            f"Diagnóstico visual: {d['label']} ({d['confidence']:.0%}, confianza {d['level']}).\n"
            f"Contexto: {context[:800]}"),
            "history": [{"event": "diagnosis", "label": d["label"],
                         "confidence": round(d["confidence"], 3)}]}
    resp = llm.invoke(prompts.SYNTHESIZE.format(
        label=d["label"], level=d["level"], confidence=d["confidence"],
        severity=d.get("severity", 0.0), severity_level=d.get("severity_level", "?"),
        question=state.get("user_question", "¿Qué debo hacer?"), context=context))
    return {"final_answer": resp.content if hasattr(resp, "content") else str(resp),
            "history": [{"event": "diagnosis", "label": d["label"],
                         "confidence": round(d["confidence"], 3),
                         "severity": d.get("severity", 0.0)}]}
