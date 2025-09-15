import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Settings:
    """Configuración centralizada del sistema RAG"""
    
    # Rutas base
    BASE_DIR = Path(__file__).parent.parent
    DOCUMENTS_DIR = Path(r"C:\Users\danil\OneDrive\Escritorio\Tesis\Documentos")
    OUTPUT_DIR = DOCUMENTS_DIR / "textos_procesados"
    TRANSLATIONS_DIR = DOCUMENTS_DIR / "textos_traducidos"
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    
    # Configuración de Google Cloud / Vertex AI
    GOOGLE_CREDENTIALS_PATH = r"C:\Users\danil\OneDrive\Escritorio\Tesis\credentials.json"
    GOOGLE_PROJECT_ID = "stately-moon-451804-a9"
    GOOGLE_LOCATION = "us-central1"
    
    # Configuración del modelo LLM (usando Vertex AI)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
    LLM_MODEL_NAME = "gemini-2.5-flash"  # ← Modelo corregido que funciona
    LLM_TEMPERATURE = 0.1
    LLM_MAX_OUTPUT_TOKENS = 8192
    LLM_MAX_RETRIES = 3
    
    # Configuración de procesamiento de texto
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CHUNK_SIZE_FOR_TRANSLATION = 1500
    
    # Configuración de ChromaDB
    COLLECTION_NAME = "tizon_tardio"
    COLLECTION_DESCRIPTION = "Base de conocimiento sobre tizón tardío en tomate"
    
    # Configuración de embeddings
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Configuración de pausas
    TRANSLATION_DELAY = 5  # segundos entre traducciones
    
    @classmethod
    def create_directories(cls):
        """Crear directorios necesarios"""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.TRANSLATIONS_DIR.mkdir(exist_ok=True)
        cls.CHROMA_DB_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_google_credentials(cls):
        """Obtener ruta de credenciales de Google"""
        if os.path.exists(cls.GOOGLE_CREDENTIALS_PATH):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cls.GOOGLE_CREDENTIALS_PATH
            return True
        return False 