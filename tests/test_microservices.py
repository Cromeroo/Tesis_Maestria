"""Microservicios: contratos HTTP y clientes remotos (sin levantar contenedores)."""
import io
import os
os.environ["TESIS_OFFLINE"] = "1"

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from src.tools.remote import RemoteClassifier, RemoteRetriever

CLASSIFY_PAYLOAD = {
    "label": "Tizon_tardio", "confidence": 0.82, "level": "moderada",
    "probs": [{"label": "Tizon_tardio", "confidence": 0.82},
              {"label": "Sana", "confidence": 0.10},
              {"label": "Otras_enfermedades", "confidence": 0.08}],
    "severity": 0.12, "severity_level": "moderada",
}


def _img_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "green").save(buf, format="JPEG")
    return buf.getvalue()


def test_remote_classifier_parses_contract():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/classify"
        return httpx.Response(200, json=CLASSIFY_PAYLOAD)
    clf = RemoteClassifier("http://vision:8001",
                           client=httpx.Client(base_url="http://vision:8001",
                                               transport=httpx.MockTransport(handler)))
    out = clf.analyze(Image.new("RGB", (64, 64), "green"))
    assert out["severity"] == 0.12 and out["severity_level"] == "moderada"
    assert clf.predict_proba(Image.new("RGB", (64, 64), "green"))[0] == ("Tizon_tardio", 0.82)


def test_remote_retriever_parses_contract_and_count():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search":
            return httpx.Response(200, json={"documents": ["doc1"], "sources": ["p.pdf"]})
        return httpx.Response(200, json={"docs": 42})
    ret = RemoteRetriever("http://rag:8002",
                          client=httpx.Client(base_url="http://rag:8002",
                                              transport=httpx.MockTransport(handler)))
    assert ret.search("tizón") == (["doc1"], ["p.pdf"])
    assert ret.count() == 42


def test_vision_svc_contract():
    from src.vision_svc.app import app
    c = TestClient(app)
    assert "device" in c.get("/health").json()
    r = c.post("/classify", files={"file": ("h.jpg", _img_bytes())})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"label", "confidence", "probs", "severity", "severity_level"}


def test_rag_svc_degrades_without_chroma():
    from src.rag_svc.app import app
    c = TestClient(app)
    assert c.get("/health").json() == {"docs": 0}
    r = c.post("/search", json={"query": "tizón", "k": 3})
    assert r.json() == {"documents": [], "sources": []}


def test_service_accepts_injected_doubles(tmp_path):
    from src.services.diagnosis import DiagnosisService
    from src.models.classifier import LeafClassifier

    class FakeRet:
        def search(self, q, k=5):
            return (["docs"], ["p.pdf"])

        def count(self):
            return 1

    svc = DiagnosisService(classifier=LeafClassifier("./no_existe.pth", device="cpu"),
                           retriever=FakeRet())
    p = str(tmp_path / "inj.jpg")
    Image.new("RGB", (64, 64), "green").save(p)
    out = svc.diagnose(p, "tizón en mi tomate?")
    assert out["diagnosis"]["severity"] >= 0.0 and svc.health()["mode"] == "in-process"
