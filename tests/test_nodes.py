"""Nodos y bordes en aislamiento: matriz de routing del grafo."""
import os
os.environ["TESIS_OFFLINE"] = "1"

from types import SimpleNamespace

from src.graph.nodes import triage, plan, retrieve, synthesize, request_info, classify
from src.graph import edges


def _settings():
    return SimpleNamespace(CONFIDENCE_LOW=0.6, CONFIDENCE_HIGH=0.9, MAX_QUERIES=2)


def _diag(label="Sana", conf=0.95, level="alta"):
    return {"label": label, "confidence": conf, "level": level, "probs": [],
            "severity": 0.01, "severity_level": "leve"}


def test_edges_routing_matrix():
    assert edges.route_after_triage({"in_scope": True}) == "plan"
    assert edges.route_after_triage({"in_scope": False}) == "synthesize"
    assert edges.route_after_plan({"diagnosis": _diag(level="baja")}) == "request_info"
    assert edges.route_after_plan({"diagnosis": _diag(level="alta")}) == "retrieve"


def test_plan_skips_when_out_of_scope():
    out = plan.run({"diagnosis": _diag(), "in_scope": False}, llm=None,
                   settings=_settings())
    assert out == {"planned_queries": []}


def test_retrieve_overwrites_contexts_and_warns_when_empty():
    class Empty:
        def search(self, q, k=5):
            return [], []
    out = retrieve.run({"planned_queries": ["tizón"]}, retriever=Empty())
    # Vacío real (no excepción): contexts=[] equivale a "sin resultados"
    assert out == {"contexts": [{"query": "tizón", "documents": [], "sources": []}]}

    class Broken:
        def search(self, q, k=5):
            raise RuntimeError("db caída")
    out = retrieve.run({"planned_queries": ["tizón"]}, retriever=Broken())
    assert "warnings" in out


def test_synthesize_offline_includes_severity_and_history():
    state = {"diagnosis": _diag("Tizon_tardio", 0.9), "user_question": "¿Qué hago?",
             "contexts": [{"query": "q", "documents": ["Phytophthora"], "sources": ["p.pdf"]}]}
    out = synthesize.run(state, llm=None)
    assert "90%" in out["final_answer"]
    assert out["history"][0]["label"] == "Tizon_tardio"


def test_classify_node_passes_severity(tmp_path):
    from PIL import Image
    from src.models.classifier import LeafClassifier
    p = tmp_path / "h.jpg"
    Image.new("RGB", (64, 64), "green").save(p)
    clf = LeafClassifier("./no_existe.pth", device="cpu")
    out = classify.run({"image_path": str(p)}, classifier=clf, settings=_settings())
    assert out["diagnosis"]["severity"] >= 0.0
