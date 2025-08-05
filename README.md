# 🤖 Agente Multimodal para Diagnóstico de Tizón Tardío en Tomate

Este proyecto implementa un sistema multimodal que combina **visión por computadora** y **RAG (Retrieval Augmented Generation)** para diagnosticar y proporcionar recomendaciones sobre el tizón tardío en tomate.

## 🏗️ Arquitectura del Sistema

### Componentes Principales

1. **🔍 Clasificador CNN** - Modelo de visión por computadora entrenado para detectar:
   - 🟢 Hojas sanas
   - 🔴 Tizón tardío
   - 🟡 Otras enfermedades

2. **📚 Sistema RAG** - Base de conocimiento especializada que incluye:
   - Detección automática de idioma
   - Traducción con Gemini 2.5
   - Almacenamiento en ChromaDB
   - Generación de respuestas contextuales

3. **🤖 Agente Multimodal** - Integración completa con interfaz Streamlit

## 📁 Estructura del Proyecto

```
Tesis/
├── pipeline_rag.py          # Pipeline completo de procesamiento RAG
├── consulta_rag.py          # Sistema de consulta a la base de conocimiento
├── agente_multimodal.py     # Agente multimodal principal
├── clasificador.py          # Clasificador CNN original
├── conexión.py              # Configuración de Gemini
├── limpieza.py              # Script de limpieza original
├── Documentos/              # Carpeta con PDFs de investigación
├── chroma_db/               # Base de datos vectorial (se crea automáticamente)
└── README.md               # Este archivo
```

## 🚀 Instalación y Configuración

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

O instalar manualmente:

```bash
pip install torch torchvision streamlit langchain-google-vertexai chromadb langdetect unidecode PyMuPDF python-dotenv sentence-transformers
```

### 2. Configurar Credenciales

1. Crear archivo `credentials.json` con tus credenciales de Google Cloud
2. Configurar variables de entorno en `.env`:

```env
GOOGLE_APPLICATION_CREDENTIALS=C:\Users\danil\OneDrive\Escritorio\Tesis\credentials.json
```

### 3. Preparar el Modelo CNN

Asegúrate de tener el archivo `best_model_3class.pth` en el directorio raíz.

## 📋 Uso del Sistema

### Paso 1: Procesar Documentos (Pipeline RAG)

```bash
python pipeline_rag.py
```

Este script:
- ✅ Extrae texto de todos los PDFs en la carpeta `Documentos/`
- 🌍 Detecta automáticamente el idioma
- 🔄 Traduce documentos en inglés usando Gemini 2.5
- 🧹 Limpia y normaliza el texto
- 📚 Carga la información en ChromaDB
- 💾 Guarda metadatos del procesamiento

### Paso 2: Probar Consultas RAG

```bash
python consulta_rag.py
```

Prueba consultas como:
- "¿Cuáles son los síntomas del tizón tardío?"
- "¿Cómo prevenir el tizón tardío?"
- "¿Qué tratamientos son efectivos?"

### Paso 3: Ejecutar Agente Multimodal

```bash
streamlit run agente_multimodal.py
```

## 🎯 Funcionalidades del Agente Multimodal

### Análisis Visual
- **Clasificación automática** de imágenes de hojas
- **Nivel de confianza** para cada predicción
- **Interfaz intuitiva** con Streamlit

### Recomendaciones Inteligentes
- **Diagnóstico específico** basado en la clasificación
- **Recomendaciones contextuales** del sistema RAG
- **Información especializada** sobre tratamientos y prevención

### Base de Conocimiento
- **Documentos científicos** procesados y traducidos
- **Búsqueda semántica** en ChromaDB
- **Respuestas generadas** con Gemini 2.5

## 🔧 Configuración Avanzada

### Ajustar Parámetros del Pipeline

En `pipeline_rag.py`:

```python
# Tamaño de chunks para ChromaDB
chunk_size=1000
chunk_overlap=200

# Configuración de Gemini
temperature=0.3  # Para traducciones más consistentes
max_output_tokens=2048
```

### Personalizar Consultas RAG

En `agente_multimodal.py`, modifica las consultas según tus necesidades:

```python
consultas = [
    "¿Cuáles son los síntomas específicos del tizón tardío?",
    "¿Qué tratamientos químicos son efectivos?",
    # Agregar más consultas...
]
```

## 📊 Monitoreo y Logs

El sistema proporciona logs detallados:

- ✅ Progreso de procesamiento de documentos
- 🌍 Detección de idiomas
- 🔄 Estado de traducciones
- 📚 Carga en ChromaDB
- 🤖 Generación de respuestas

## 🛠️ Solución de Problemas

### Error de Credenciales
```
❌ Error al configurar Gemini: [Error de autenticación]
```
**Solución:** Verificar que `credentials.json` esté en la ruta correcta y tenga permisos válidos.

### Error de ChromaDB
```
❌ Error al cargar en ChromaDB: [Error de conexión]
```
**Solución:** Verificar que ChromaDB esté instalado correctamente y no haya conflictos de versiones.

### Error de Modelo CNN
```
❌ Error al cargar modelo: [Archivo no encontrado]
```
**Solución:** Asegurar que `best_model_3class.pth` esté en el directorio raíz.

## 🎓 Casos de Uso

### Para Investigadores
- **Análisis rápido** de muestras de campo
- **Acceso a literatura científica** especializada
- **Recomendaciones basadas en evidencia**

### Para Agricultores
- **Diagnóstico temprano** de enfermedades
- **Guías de tratamiento** específicas
- **Medidas preventivas** personalizadas

### Para Estudiantes
- **Herramienta educativa** interactiva
- **Base de conocimiento** estructurada
- **Ejemplos prácticos** de aplicación

## 🔮 Próximas Mejoras

- [ ] **Integración con LangGraph** para flujos más complejos
- [ ] **Análisis de múltiples imágenes** simultáneamente
- [ ] **Historial de diagnósticos** y seguimiento
- [ ] **API REST** para integración con otros sistemas
- [ ] **Análisis temporal** de progresión de enfermedades

## 📞 Soporte

Para preguntas o problemas:
1. Revisar los logs de error
2. Verificar la configuración de credenciales
3. Consultar la documentación de las librerías utilizadas

---

**Desarrollado para:** Tesis sobre tizón tardío en tomate  
**Tecnologías:** PyTorch, LangChain, Gemini, ChromaDB, Streamlit  
**Licencia:** Académica 