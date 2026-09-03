"""Nodo triage (guardrail): decide si la pregunta es del dominio.

Fuera de dominio (p.ej. "cuéntame un chiste") -> se abstiene sin gastar RAG/LLM caro.
Offline/tests: heurística por palabras clave, sin LLM.
"""
_IN_SCOPE_HINTS = ("tomate", "tomat", "tizón", "tizon", "hoja", "planta", "cultivo",
                   "hongo", "phytophthora", "plaga", "enfermedad", "hacer", "hago", "hace",
                   "diagnost", "tratamiento", "prevenir", "síntoma", "sintoma",
                   "fungicida", "riego", "invernadero")

TRIAGE_PROMPT = """¿La siguiente pregunta es sobre tomate, tizón tardío, sanidad vegetal
o qué hacer con una planta? Responde SOLO 'SI' o 'NO'.

Pregunta: {question}"""


def run(state: dict, *, llm) -> dict:
    q = (state.get("user_question") or "").strip()
    if not q or q == "¿Qué debo hacer?":
        return {"in_scope": True}
    if llm is None:
        ql = q.lower()
        return {"in_scope": any(h in ql for h in _IN_SCOPE_HINTS)}
    resp = llm.invoke(TRIAGE_PROMPT.format(question=q))
    text = (resp.content if hasattr(resp, "content") else str(resp)).strip().upper()
    return {"in_scope": text.startswith("SI")}
