"""Eval mínima: precisión visión + hit-rate RAG. Base para el capítulo de tesis."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # root del proyecto

from src.services.diagnosis import DiagnosisService


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="./eval_images")
    args = ap.parse_args()
    svc = DiagnosisService()
    ok = total = 0
    for img in sorted(Path(args.images).glob("*.*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        total += 1
        out = svc.diagnose(str(img))
        d = out["diagnosis"]
        print(f"{img.name}: {d['label']} {d['confidence']:.0%} | fuentes={len(out.get('contexts', []))}")
        ok += d["confidence"] >= 0.6
    print(f"confianza>=0.6: {ok}/{total}" if total else "sin imágenes en ./eval_images")


if __name__ == "__main__":
    main()
