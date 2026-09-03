"""Tool de visión: clasifica una hoja. Función pura + wrapper @tool de LangChain."""
from PIL import Image


def classify_leaf(image_path: str, classifier) -> dict:
    """Clasifica + severidad. Compatible con clasificadores remotos (microservicio)."""
    img = Image.open(image_path).convert("RGB")
    if hasattr(classifier, "analyze"):
        return classifier.analyze(img)
    ranked = classifier.predict_proba(img)
    label, confidence = ranked[0]
    return {"label": label, "confidence": confidence,
            "level": classifier.confidence_level(confidence),
            "probs": [{"label": l, "confidence": c} for l, c in ranked],
            "severity": 0.0, "severity_level": "indeterminada"}


def make_vision_tool(classifier):
    """Devuelve un StructuredTool `classify_leaf` bindeado al clasificador.

    Uso futuro: `ToolNode([make_vision_tool(clf)])` si el grafo se vuelve agéntico
    (loop ReAct en vez de flujo fijo).
    """
    from langchain_core.tools import tool

    @tool("classify_leaf")
    def _classify_leaf(image_path: str) -> str:
        """Clasifica una hoja de tomate. Input: ruta a la imagen. Clases: Sana, Tizon_tardio, Otras_enfermedades."""
        out = classify_leaf(image_path, classifier)
        return f"{out['label']} ({out['confidence']:.0%}, confianza {out['level']})"

    return _classify_leaf
