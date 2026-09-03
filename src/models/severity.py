"""Severidad foliar: % de área afectada por lesiones (el diferencial del producto).

Sin modelo nuevo: segmentación HSV (marrones/necróticos + halo amarillento)
sobre la máscara de hoja (verde + lesión). Funciona en CPU, testeable con
imágenes sintéticas.
"""
import numpy as np
from PIL import Image


def _masks(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    hsv = np.asarray(img.convert("RGB").convert("HSV"), dtype=np.int32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    leaf = ((h >= 25) & (h <= 90) & (s > 35))                       # tejido verde
    necrotic = (v < 70) & (s > 20)                                  # tejido muerto oscuro
    brown = (h >= 5) & (h <= 30) & (s > 60) & (v >= 70) & (v < 200)  # lesión marrón
    yellow = (h >= 30) & (h <= 45) & (s > 60) & (v > 120)            # halo amarillento
    lesion = necrotic | brown | yellow
    return (leaf | lesion), lesion


def estimate_severity(img: Image.Image) -> dict:
    """Devuelve fraction 0..1, level y overlay PIL con lesiones en rojo."""
    leaf_mask, lesion_mask = _masks(img)
    leaf_px = int(leaf_mask.sum())
    if leaf_px == 0:
        return {"fraction": 0.0, "level": "indeterminada", "overlay": img.copy()}
    frac = float(lesion_mask.sum() / leaf_px)
    level = "leve" if frac < 0.05 else ("moderada" if frac < 0.15 else "severa")
    arr = np.asarray(img.convert("RGB")).copy()
    arr[lesion_mask] = (arr[lesion_mask] * 0.35 + np.array([255, 0, 0]) * 0.65).astype("uint8")
    return {"fraction": round(frac, 4), "level": level, "overlay": Image.fromarray(arr)}
