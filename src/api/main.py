"""API REST: escala más allá de Streamlit (v1 solo tenía UI).

Endurecida: rate-limit en memoria, API key opcional y /metrics.
"""
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from src.services.diagnosis import DiagnosisService
from src.utils.logging import setup_logging

setup_logging()
app = FastAPI(title="Tizón tardío — Agente LangGraph")
_service: DiagnosisService | None = None

_hits: dict[str, list[float]] = defaultdict(list)  # rate-limit por IP
_metrics = {"requests_total": 0, "diagnoses_total": 0, "by_label": defaultdict(int)}


def get_service() -> DiagnosisService:
    global _service
    if _service is None:
        _service = DiagnosisService()
    return _service


def require_api_key(x_api_key: str | None = Header(default=None)):
    want = get_service().settings.API_KEY
    if want and x_api_key != want:
        raise HTTPException(status_code=401, detail="X-API-Key inválida o ausente")


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    limit = get_service().settings.RATE_LIMIT_PER_MIN
    _metrics["requests_total"] += 1
    if limit > 0 and request.url.path == "/diagnose":
        now = time.time()
        ip = request.client.host if request.client else "?"
        window = [t for t in _hits[ip] if now - t < 60]
        _hits[ip] = window
        if len(window) >= limit:
            return JSONResponse(status_code=429,
                                content={"detail": f"límite de {limit}/min excedido"})
        window.append(now)
    return await call_next(request)


@app.get("/health")
def health():
    return get_service().health()


@app.get("/metrics")
def metrics():
    return {"requests_total": _metrics["requests_total"],
            "diagnoses_total": _metrics["diagnoses_total"],
            "by_label": dict(_metrics["by_label"])}


@app.post("/diagnose", dependencies=[Depends(require_api_key)])
async def diagnose(file: UploadFile = File(...),
                   question: str = Form("¿Qué debo hacer?"),
                   thread_id: str | None = Form(default=None)):
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    out = get_service().diagnose(path, question, thread_id=thread_id)
    d = out["diagnosis"]
    _metrics["diagnoses_total"] += 1
    _metrics["by_label"][d["label"]] += 1
    return {"label": d["label"], "confidence": d["confidence"], "level": d["level"],
            "top3": d.get("probs", []), "thread_id": out["thread_id"],
            "in_scope": out.get("in_scope", True),
            "needs_input": out.get("needs_input", False),
            "clarification": out.get("clarification", ""),
            "answer": out["final_answer"], "warnings": out.get("warnings", []),
            "sources": [s for c in out.get("contexts", []) for s in c["sources"]]}


@app.get("/threads/{thread_id}/history")
def thread_history(thread_id: str):
    return {"thread_id": thread_id,
            "history": get_service().thread_history(thread_id)}
