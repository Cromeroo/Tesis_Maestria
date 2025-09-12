# Sistema RAG Refactorizado para Procesamiento de Documentos Científicos

## 🎯 Descripción

Este sistema RAG (Retrieval-Augmented Generation) ha sido completamente refactorizado para procesar documentos PDF científicos, traducirlos al español si es necesario, limpiarlos y filtrarlos, y crear una base de conocimiento vectorial para búsquedas semánticas.

## 🏗️ Arquitectura Refactorizada

### Estructura de Directorios

```
Tesis/
├── config/                 # Configuración centralizada
│   └── settings.py        # Configuración del sistema
├── components/            # Componentes reutilizables
│   ├── pdf_extractor.py   # Extracción de texto PDF
│   ├── language_detector.py # Detección de idioma
│   ├── text_translator.py # Traducción de texto
│   ├── text_cleaner.py    # Limpieza y filtrado
│   ├── text_chunker.py    # División en chunks
│   └── vector_store.py    # Manejo de ChromaDB
├── models/                # Modelos de datos
│   └── document.py        # Modelos de documentos
├── services/              # Servicios de negocio
│   ├── document_processor.py # Procesamiento de documentos
│   └── rag_pipeline.py    # Pipeline principal RAG
├── utils/                 # Utilidades comunes
│   └── file_utils.py      # Manejo de archivos
├── examples/              # Ejemplos de uso
│   └── ejemplo_uso.py     # Ejemplos programáticos
├── main.py                # Punto de entrada principal
├── pipeline_automatico.py # Pipeline automático completo
└── requirements.txt       # Dependencias
```

## 🚀 Uso del Sistema

### 🎯 Pipeline Automático (RECOMENDADO)

El **pipeline automático** ejecuta todos los pasos sin necesidad de correr cada uno manualmente:

```bash
# Ejecutar pipeline completo automáticamente
python pipeline_automatico.py

# Ejecutar con directorio específico de PDFs
python pipeline_automatico.py --pdf-dir /ruta/a/documentos

# Ejecutar en modo prueba
python pipeline_automatico.py --test

# Ejecutar sin confirmaciones
python pipeline_automatico.py --force

# Ver configuración actual
python pipeline_automatico.py --config
```

### Línea de Comandos Manual

```bash
# Mostrar ayuda
python main.py --help

# Procesar documentos
python main.py --process

# Procesar desde directorio específico
python main.py --process --pdf-dir /ruta/a/documentos

# Buscar en la base de conocimiento
python main.py --search "tizón tardío"

# Buscar con número específico de resultados
python main.py --search "enfermedad tomate" --results 10

# Mostrar información del sistema
python main.py --info

# Reiniciar base de conocimiento
python main.py --reset
```

### Uso Programático

```python
from services.rag_pipeline import RAGPipeline

# Crear pipeline
pipeline = RAGPipeline()

# Ejecutar pipeline completo
success = pipeline.execute_full_pipeline()

# Buscar en la base de conocimiento
results = pipeline.search_knowledge_base("consulta", n_results=5)

# Obtener información del sistema
info = pipeline.get_knowledge_base_info()
```

## 🔧 Componentes Principales

### 1. **PDFExtractor** (`components/pdf_extractor.py`)
- Extrae texto de archivos PDF usando PyMuPDF
- Limpia texto extraído (separa palabras pegadas)
- Obtiene metadatos del PDF

### 2. **LanguageDetector** (`components/language_detector.py`)
- Detecta automáticamente el idioma del texto
- Soporte para múltiples idiomas
- Manejo robusto de errores

### 3. **TextTranslator** (`components/text_translator.py`)
- Traduce texto usando Google Gemini
- Manejo de chunks para textos largos
- Pausas automáticas para evitar rate limits

### 4. **TextCleaner** (`components/text_cleaner.py`)
- Filtra secciones científicas no deseadas
- Limpia y normaliza texto
- Patrones configurables para diferentes idiomas

### 5. **TextChunker** (`components/text_chunker.py`)
- Divide texto en chunks para procesamiento
- Configuración flexible de tamaño y solapamiento
- Estadísticas de chunks

### 6. **VectorStore** (`components/vector_store.py`)
- Manejo completo de ChromaDB
- Embeddings multilingües
- Operaciones CRUD para colecciones

### 7. **DocumentProcessor** (`services/document_processor.py`)
- Orquesta el procesamiento completo de documentos
- Manejo de traducciones y limpieza
- Generación de metadatos y estadísticas

### 8. **RAGPipeline** (`services/rag_pipeline.py`)
- Pipeline principal del sistema
- Coordina todos los componentes
- Manejo de errores y reportes

## ⚙️ Configuración

### Archivo de Configuración (`config/settings.py`)

```python
class Settings:
    # Rutas base
    BASE_DIR = Path(__file__).parent.parent
    DOCUMENTS_DIR = Path("ruta/a/documentos")
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Configuración de Google Cloud
    GOOGLE_CREDENTIALS_PATH = "ruta/a/credentials.json"
    GOOGLE_LOCATION = "us-central1"
    
    # Configuración del modelo LLM
    LLM_MODEL_NAME = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.1
    
    # Configuración de procesamiento
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
```

### Variables de Entorno

```bash
# Archivo .env
GOOGLE_APPLICATION_CREDENTIALS=ruta/a/credentials.json
```

## 📊 Flujo de Procesamiento

1. **Extracción**: Se extrae texto de archivos PDF
2. **Detección de Idioma**: Se identifica automáticamente el idioma
3. **Filtrado**: Se eliminan secciones científicas no deseadas
4. **Traducción**: Se traduce al español si es necesario
5. **Limpieza**: Se normaliza y limpia el texto final
6. **Chunking**: Se divide en chunks para procesamiento
7. **Embeddings**: Se crean vectores semánticos
8. **Almacenamiento**: Se guarda en ChromaDB

## 🔍 Búsquedas Semánticas

```python
# Búsqueda básica
results = pipeline.search_knowledge_base("síntomas del tizón tardío")

# Búsqueda con filtros
results = pipeline.search_knowledge_base(
    "control biológico", 
    n_results=10
)

# Los resultados incluyen:
# - Texto del chunk
# - Metadatos del documento
# - Distancia semántica
# - Información del archivo original
```

## 📈 Estadísticas y Reportes

El sistema genera automáticamente:

- **Metadatos de procesamiento**: Información completa de cada documento
- **Estadísticas por idioma**: Distribución y métricas
- **Ratio de compresión**: Eficiencia del filtrado
- **Información de chunks**: Tamaños y distribución
- **Estado de la base de conocimiento**: Colecciones y contenido

## 🛠️ Instalación y Dependencias

### Requisitos

```bash
pip install -r requirements.txt
```

### Dependencias Principales

- `PyMuPDF`: Extracción de PDF
- `langchain-google-vertexai`: Integración con Gemini
- `chromadb`: Base de datos vectorial
- `langdetect`: Detección de idioma
- `langchain-text-splitters`: División de texto
- `sentence-transformers`: Embeddings

## 🔧 Personalización

### Agregar Nuevos Patrones de Filtrado

```python
# En text_cleaner.py
additional_patterns = [
    r'\bNueva Sección\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
    # Agregar más patrones según necesidad
]
```

### Configurar Nuevos Idiomas

```python
# En language_detector.py
language_names = {
    'es': 'Español',
    'en': 'Inglés',
    'fr': 'Francés',
    'pt': 'Portugués',
    # Agregar más idiomas
}
```

### Ajustar Parámetros de Chunking

```python
# En settings.py
CHUNK_SIZE = 800        # Tamaño de chunk
CHUNK_OVERLAP = 150     # Solapamiento entre chunks
```

## 🚨 Manejo de Errores

El sistema incluye manejo robusto de errores:

- **Reintentos automáticos** para operaciones de red
- **Fallbacks** cuando fallan servicios externos
- **Logging detallado** para debugging
- **Validación** de datos en cada paso
- **Recuperación** de errores sin perder progreso

## 📝 Logs y Monitoreo

```python
# El sistema genera logs detallados:
print("🔄 Procesando documento...")
print("✅ Documento procesado exitosamente")
print("⚠️ Advertencia: texto muy corto")
print("❌ Error: no se pudo conectar a Gemini")
```

## 🔄 Mantenimiento

### Limpieza de Archivos Antiguos

```python
from utils.file_utils import FileUtils

# Limpiar archivos de más de 30 días
FileUtils.cleanup_old_files(
    directory="output", 
    days_old=30, 
    extensions=['.txt', '.json']
)
```

### Respaldo de Base de Conocimiento

```python
# Crear respaldo antes de cambios importantes
FileUtils.backup_file(
    file_path="chroma_db/collection_info.json",
    backup_dir="backups"
)
```

## 🤝 Contribución

Para contribuir al proyecto:

1. **Fork** el repositorio
2. **Crea** una rama para tu feature
3. **Implementa** los cambios siguiendo la arquitectura
4. **Prueba** con los ejemplos incluidos
5. **Envía** un pull request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

## 🆘 Soporte

Para obtener ayuda:

1. Revisa los **ejemplos** en `examples/ejemplo_uso.py`
2. Consulta la **documentación** de cada componente
3. Verifica la **configuración** en `config/settings.py`
4. Ejecuta `python main.py --info` para diagnóstico

## 🎉 Beneficios de la Refactorización

- **Modularidad**: Cada componente tiene una responsabilidad específica
- **Reutilización**: Los componentes se pueden usar independientemente
- **Mantenibilidad**: Código más fácil de entender y modificar
- **Testabilidad**: Cada componente se puede probar por separado
- **Escalabilidad**: Fácil agregar nuevas funcionalidades
- **Configurabilidad**: Parámetros centralizados y ajustables
- **Robustez**: Mejor manejo de errores y excepciones
- **Automatización**: Pipeline completo sin intervención manual

## 🗑️ Archivos Eliminados

Los siguientes archivos han sido eliminados por estar obsoletos después de la refactorización:

- `limpieza.py` - Funcionalidad migrada a componentes individuales
- `pipeline_rag.py` - Reemplazado por el nuevo sistema modular
- `conexión.py` - Integrado en `text_translator.py`
- `clasificador.py` - Funcionalidad no utilizada
- `consulta_rag.py` - Reemplazado por `main.py`
- `verificar_chromadb.py` - Integrado en `vector_store.py`
- `borrar_chromadb.py` - Funcionalidad disponible en `main.py --reset`
- `agente_multimodal.py` - No utilizado en el sistema actual

## 🚀 Inicio Rápido

### Opción 1: Script Interactivo (RECOMENDADO)

```bash
python inicio_rapido.py
```

Este script te guía paso a paso a través de:
- ✅ Verificación de credenciales
- 📁 Configuración de directorios
- 🚀 Ejecución del pipeline automático
- 🔍 Realización de búsquedas de ejemplo
- 📊 Menú interactivo con todas las opciones

### Opción 2: Comandos Directos

1. **Configurar credenciales** de Google Cloud
2. **Colocar archivos PDF** en el directorio `Documentos/`
3. **Ejecutar pipeline automático**:
   ```bash
   python pipeline_automatico.py --force
   ```
4. **Realizar consultas**:
   ```bash
   python main.py --search "tu consulta"
   ``` 