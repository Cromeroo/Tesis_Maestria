"""Configuración centralizada. Adiós paths hardcodeados tipo C:\\Users\\danil..."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GOOGLE_APPLICATION_CREDENTIALS: str = "./credentials.json"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_LOCATION: str = "us-central1"
    CHROMA_PATH: str = "./chroma_db"
    CHROMA_COLLECTION: str = "tizon_tardio"
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CNN_WEIGHTS_PATH: str = "./best_model_3class.pth"
    CNN_BACKBONE: str = "simple_cnn"
    CONFIDENCE_LOW: float = 0.60
    CONFIDENCE_HIGH: float = 0.90
    MAX_QUERIES: int = 2
    DEVICE: str = "auto"  # auto | cuda | cpu (cuda = ROCm en tu 9060 XT)
    CHECKPOINT_PATH: str = "./checkpoints.db"  # memoria conversacional (sqlite)
    LOT_DB_PATH: str = "./lots.db"  # timeline por lote (en compose: /app/data/...)
    API_KEY: str = ""  # si se define, /diagnose exige header X-API-Key
    RATE_LIMIT_PER_MIN: int = 30  # 0 = sin límite
    VISION_URL: str = ""  # ej. http://vision:8001 (vacío = in-process)
    RAG_URL: str = ""  # ej. http://rag:8002 (vacío = in-process)


@lru_cache
def get_settings() -> Settings:
    return Settings()
