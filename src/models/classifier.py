"""Clasificador de hojas: único punto de entrada a visión.

Device-aware: en tu 9060 XT pasa `device="cuda"` y la inferencia corre en GPU.
"""
from pathlib import Path
import torch
from torchvision import transforms as T
from PIL import Image

from src.models.cnn import build_model
from src.models.severity import estimate_severity
from src.utils.device import get_device

LABELS = ["Sana", "Tizon_tardio", "Otras_enfermedades"]


class LeafClassifier:
    """El grafo no sabe qué red hay detrás; solo `predict()` + `confidence_level()`."""

    def __init__(self, weights_path: str = "./best_model_3class.pth",
                 backbone: str = "simple_cnn", image_size: int = 128,
                 device: str = "auto"):
        self.backbone = backbone
        self.weights_path = Path(weights_path)
        self.device = get_device(device)
        self.model = build_model(backbone).to(self.device)
        self.loaded = False
        if self.weights_path.exists():
            try:
                state = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.loaded = True
            except Exception as e:
                print(f"[classifier] no se pudo cargar {self.weights_path}: {e}")
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image) -> tuple[str, float]:
        probs = self.predict_proba(image)
        return probs[0]

    def analyze(self, image: Image.Image) -> dict:
        """Diagnóstico visual completo: clase + distribución + severidad foliar."""
        img = image.convert("RGB")
        ranked = self.predict_proba(img)
        label, confidence = ranked[0]
        sev = estimate_severity(img)
        return {"label": label, "confidence": confidence,
                "level": self.confidence_level(confidence),
                "probs": [{"label": l, "confidence": c} for l, c in ranked],
                "severity": sev["fraction"], "severity_level": sev["level"]}

    def predict_proba(self, image: Image.Image) -> list[tuple[str, float]]:
        """Distribución completa ordenada desc. La UI la usa para las barras por clase."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        ranked = sorted(((l, float(probs[i])) for i, l in enumerate(LABELS)),
                        key=lambda t: t[1], reverse=True)
        return ranked

    @staticmethod
    def confidence_level(conf: float, low: float = 0.60, high: float = 0.90) -> str:
        if conf < low:
            return "baja"
        if conf < high:
            return "moderada"
        return "alta"
