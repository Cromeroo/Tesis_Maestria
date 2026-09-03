"""API agent: contratos, métricas, rate-limit, auth y lote (TestClient, offline)."""
import io
import os
os.environ["TESIS_OFFLINE"] = "1"

from fastapi.testclient import TestClient
from PIL import Image

from src.api import main as api


def _img() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "green").save(buf, format="JPEG")
    return buf.getvalue()


def _client() -> TestClient:
    return TestClient(api.app)


def _post(c: TestClient, **data) -> dict:
    r = c.post("/diagnose", files={"file": ("h.jpg", _img())}, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_health_shape():
    body = _client().get("/health").json()
    assert set(body) >= {"device", "chroma_docs", "llm", "checkpointer", "mode"}


def test_diagnose_contract_and_metrics():
    c = _client()
    before = c.get("/metrics").json()["diagnoses_total"]
    body = _post(c, question="tizón en mi tomate?", lot_id="test-lote")
    assert set(body) >= {"label", "top3", "severity", "thread_id", "needs_input",
                         "clarification", "answer", "warnings", "sources"}
    after = c.get("/metrics").json()
    assert after["diagnoses_total"] == before + 1
    lot = c.get("/lots/test-lote").json()
    assert lot["summary"]["records"] >= 1  # quedó registrado en el timeline


def test_rate_limit_429():
    c = _client()
    api._hits.clear()
    svc = api.get_service()
    old = svc.settings.RATE_LIMIT_PER_MIN
    svc.settings.RATE_LIMIT_PER_MIN = 1
    try:
        _post(c, question="tizón?")
        r = c.post("/diagnose", files={"file": ("h.jpg", _img())},
                   data={"question": "tizón?"})
        assert r.status_code == 429
    finally:
        svc.settings.RATE_LIMIT_PER_MIN = old


def test_api_key_401_then_200():
    c = _client()
    svc = api.get_service()
    old = svc.settings.API_KEY
    svc.settings.API_KEY = "secret"
    try:
        r = c.post("/diagnose", files={"file": ("h.jpg", _img())},
                   data={"question": "tizón?"})
        assert r.status_code == 401
        r = c.post("/diagnose", files={"file": ("h.jpg", _img())},
                   data={"question": "tizón?"}, headers={"X-API-Key": "secret"})
        assert r.status_code == 200
    finally:
        svc.settings.API_KEY = old
