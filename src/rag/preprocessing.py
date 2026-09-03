"""Preprocesamiento de PDFs. Imports perezosos: este módulo importa sin GPU ni fitz."""
import re
from pathlib import Path


def extract_text_pdf(path: str | Path) -> str:
    import fitz  # PyMuPDF (lazy: solo se necesita en ingesta)
    doc = fitz.open(path)
    parts = []
    for page in doc:
        t = page.get_text()
        t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)  # palabras pegadas del PDF
        t = re.sub(r"\s+", " ", t)
        parts.append(t)
    doc.close()
    return "\n".join(parts)


def detect_lang(text: str) -> str:
    try:
        from langdetect import detect
        sample = re.sub(r"[^\w\s]", "", text[:2000])
        sample = re.sub(r"\s+", " ", sample).strip()
        if len(sample) < 50:
            return "en"
        return detect(sample)
    except Exception:
        return "en"


def clean_text(text: str) -> str:
    """Conservadora: NO quita tildes (v1 usaba unidecode y destruía el español)."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+|Página \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return text.strip()
