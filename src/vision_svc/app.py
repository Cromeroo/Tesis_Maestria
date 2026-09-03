"""vision-svc: microservicio de visión. POST /classify -> clase + top3 + severidad."""
import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from src.config import get_settings
from src.models.classifier import LeafClassifier
from src.utils.device import device_info
from src.utils.logging import setup_logging

setup_logging()
app = FastAPI(title="vision-svc")
_clf: LeafClassifier | None = None


def get_clf() -> LeafClassifier:
    global _clf
    if _clf is None:
        s = get_settings()
        _clf = LeafClassifier(weights_path=s.CNN_WEIGHTS_PATH,
                              backbone=s.CNN_BACKBONE, device="cpu")
    return _clf


@app.get("/health")
def health():
    clf = get_clf()
    return {"loaded": clf.loaded, "device": device_info(clf.device)}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    clf = get_clf()
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return clf.analyze(img)  # JSON-safe: label, confidence, level, probs, severity
