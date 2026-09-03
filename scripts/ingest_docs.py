"""Ingesta: python scripts/ingest_docs.py --docs ./Documentos"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # root del proyecto

from src.config import get_settings
from src.rag.ingest import IngestPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="./Documentos")
    ap.add_argument("--no-recreate", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    pipe = IngestPipeline(docs_dir=args.docs, chroma_path=s.CHROMA_PATH,
                          collection=s.CHROMA_COLLECTION,
                          embedding_model=s.EMBEDDING_MODEL)
    print(pipe.run(recreate=not args.no_recreate))


if __name__ == "__main__":
    main()
