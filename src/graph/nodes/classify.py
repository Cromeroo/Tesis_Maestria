"""Nodo classify: visión. Delega en tools/vision_tool ( Testeable sin grafo )."""
import logging
from src.tools.vision_tool import classify_leaf

log = logging.getLogger(__name__)


def run(state: dict, *, classifier, settings) -> dict:
    out = classify_leaf(state["image_path"], classifier)
    log.info("visión=%s conf=%.2f", out["label"], out["confidence"])
    warnings = []
    if out["level"] == "baja":
        warnings.append("Confianza baja: se sugiere subir una foto más nítida con luz natural.")
    return {"diagnosis": {"label": out["label"], "confidence": out["confidence"],
                          "level": out["level"], "probs": out["probs"],
                          "severity": out.get("severity", 0.0),
                          "severity_level": out.get("severity_level", "indeterminada")},
            "warnings": warnings}
