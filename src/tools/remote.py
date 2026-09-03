"""Clientes remotos: MISMA interfaz que los locales (predict_proba/analyze,
search/count) para que los nodos no sepan si hablan con un microservicio.

Testeables con httpx.MockTransport, sin levantar contenedores.
"""
import io
import httpx
from PIL import Image


class RemoteClassifier:
    def __init__(self, base_url: str, timeout: float = 60.0, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def analyze(self, image: Image.Image) -> dict:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        r = self.client.post("/classify", files={"file": ("img.jpg", buf.getvalue())})
        r.raise_for_status()
        return r.json()

    def predict_proba(self, image: Image.Image) -> list[tuple[str, float]]:
        out = self.analyze(image)
        return [(p["label"], p["confidence"]) for p in out["probs"]]

    @staticmethod
    def confidence_level(conf: float, low: float = 0.60, high: float = 0.90) -> str:
        if conf < low:
            return "baja"
        if conf < high:
            return "moderada"
        return "alta"

    @property
    def loaded(self) -> bool:
        try:
            return bool(self.client.get("/health").json().get("loaded"))
        except Exception:
            return False


class RemoteRetriever:
    def __init__(self, base_url: str, timeout: float = 30.0, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def search(self, query: str, k: int = 5) -> tuple[list[str], list[str]]:
        r = self.client.post("/search", json={"query": query, "k": k})
        r.raise_for_status()
        out = r.json()
        return out.get("documents", []), out.get("sources", [])

    def count(self) -> int:
        try:
            return int(self.client.get("/health").json().get("docs", 0))
        except Exception:
            return 0
