import os
os.environ["TESIS_OFFLINE"] = "1"

from PIL import Image
from types import SimpleNamespace

from src.models.classifier import LeafClassifier
from src.rag.preprocessing import clean_text
from src.graph.builder import build_graph
from src.graph.nodes import triage, request_info
from src.tools.vision_tool import classify_leaf
from src.tools.rag_tool import search_knowledge


def _settings():
    return SimpleNamespace(CONFIDENCE_LOW=0.6, CONFIDENCE_HIGH=0.9, MAX_QUERIES=2)


def test_clean_keeps_accents():
    assert "tizón" in clean_text("tizón tardío  Page 12")


def test_classifier_runs_without_weights():
    c = LeafClassifier(weights_path="./no_existe.pth", device="cpu")
    img = Image.new("RGB", (200, 200), "green")
    label, conf = c.predict(img)
    assert label in ["Sana", "Tizon_tardio", "Otras_enfermedades"]
    assert 0 <= conf <= 1


def test_vision_tool_pure_function(tmp_path):
    img_path = tmp_path / "hoja.jpg"
    Image.new("RGB", (200, 200), "green").save(img_path)
    clf = LeafClassifier(weights_path="./no_existe.pth", device="cpu")
    out = classify_leaf(str(img_path), clf)
    assert out["level"] in {"baja", "moderada", "alta"}
    assert len(out["probs"]) == 3
    assert abs(sum(p["confidence"] for p in out["probs"]) - 1.0) < 1e-5


def test_rag_tool_never_raises():
    class Broken:
        def search(self, q, k=5):
            raise RuntimeError("db caída")
    out = search_knowledge("tizón", Broken())
    assert out["documents"] == [] and "warning" in out


def test_graph_offline_end_to_end(tmp_path):
    img_path = tmp_path / "hoja.jpg"
    Image.new("RGB", (200, 200), "green").save(img_path)
    clf = LeafClassifier(weights_path="./no_existe.pth", device="cpu")

    class FakeRet:
        def search(self, q, k=5):
            return (["El tizón tardío lo causa Phytophthora infestans."], ["paper.pdf"])

    g = build_graph(clf, FakeRet(), None, _settings())
    out = g.invoke({"image_path": str(img_path), "user_question": "¿Qué hago?",
                    "contexts": [], "warnings": []})
    assert out["diagnosis"]["label"]
    if out.get("needs_input"):
        assert "luz natural" in out["final_answer"]  # human-in-the-loop
    else:
        assert "tiz" in out["final_answer"].lower() or "visual" in out["final_answer"].lower()


def test_triage_offline_guardrail():
    assert triage.run({"user_question": "¿Qué hago con el tizón del tomate?"}, llm=None) == {"in_scope": True}
    assert triage.run({"user_question": "¿Qué debo hacer?"}, llm=None) == {"in_scope": True}
    assert triage.run({"user_question": "cuéntame un chiste"}, llm=None) == {"in_scope": False}


def test_request_info_human_in_the_loop():
    out = request_info.run({"diagnosis": {"label": "Sana", "confidence": 0.34,
                                          "level": "baja", "probs": []}})
    assert out["needs_input"] is True
    assert "luz natural" in out["clarification"]


def test_out_of_scope_abstains_without_rag(tmp_path):
    img_path = tmp_path / "hoja.jpg"
    Image.new("RGB", (200, 200), "green").save(img_path)
    clf = LeafClassifier(weights_path="./no_existe.pth", device="cpu")

    class ExplodingRet:
        def search(self, q, k=5):
            raise AssertionError("no debería llegar a retrieval")

    g = build_graph(clf, ExplodingRet(), None, _settings())
    out = g.invoke({"image_path": str(img_path), "user_question": "cuéntame un chiste",
                    "contexts": [], "warnings": []})
    assert out["in_scope"] is False
    assert "tizón tardío" in out["final_answer"]


def test_memory_accumulates_history_per_thread(tmp_path):
    from langgraph.checkpoint.memory import MemorySaver
    img_path = tmp_path / "hoja.jpg"
    Image.new("RGB", (200, 200), "green").save(img_path)
    clf = LeafClassifier(weights_path="./no_existe.pth", device="cpu")
    # confianza alta determinista -> garantiza ruta retrieve (no request_info)
    clf.predict_proba = lambda img: [("Sana", 0.95), ("Tizon_tardio", 0.03),
                                     ("Otras_enfermedades", 0.02)]

    class FakeRet:
        def search(self, q, k=5):
            return (["docs"], ["paper.pdf"])

    g = build_graph(clf, FakeRet(), None, _settings(), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    base = {"image_path": str(img_path), "user_question": "¿Qué hago con mi tomate?",
            "in_scope": True, "needs_input": False, "clarification": "",
            "contexts": [], "history": [], "warnings": []}
    g.invoke(dict(base), config=cfg)
    out2 = g.invoke(dict(base), config=cfg)
    assert len(out2["history"]) == 2  # un evento por turno, memoria del hilo
    assert out2["contexts"]  # contextos frescos del turno (sin acumular basura)
