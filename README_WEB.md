# 🍅 Interfaz Web - Sistema de Diagnóstico de Tomate

## 🚀 Cómo usar la interfaz web

### 1. **Iniciar la aplicación**

```bash
# Opción 1: Usar el script batch
iniciar_web.bat

# Opción 2: Comando directo
streamlit run web_interface.py --server.port 8501
```

### 2. **Acceder a la aplicación**

- **URL local**: http://localhost:8501
- Se abrirá automáticamente en tu navegador

## 📱 Funcionalidades de la Interfaz

### 🖼️ **Subida de Imagen**

- Formatos soportados: JPG, JPEG, PNG
- Previsualización instantánea
- Información del archivo

### 💬 **Tipos de Consulta**

1. **🚨 Urgencia Básica**: Diagnóstico y tratamiento rápido
2. **🔬 Diagnóstico Técnico**: Analysis fitopatológico completo
3. **🌾 Contexto Específico**: Con condiciones ambientales
4. **💰 Enfoque Económico**: Tratamientos costo-efectivos
5. **🌱 Prevención Futura**: Medidas preventivas
6. **⚡ Solo Imagen**: Análisis visual únicamente
7. **Personalizada**: Tu propia pregunta

### 🤖 **Resultados**

- **Clasificación**: Enfermedad detectada + confianza
- **Recomendaciones**: Plan de manejo completo
- **Información técnica**: Detalles del sistema

## 🔧 Arquitectura del Sistema

```
Imagen + Pregunta
       ↓
   CNN (PyTorch)
       ↓
   LangGraph (Flujo condicional)
       ↓
   RAG (ChromaDB) ← [Solo si es Tizón Tardío]
       ↓
   Gemini 2.0 Flash
       ↓
   Respuesta del Agrónomo
```

## 📊 Tecnologías

- **Frontend**: Streamlit
- **CNN**: PyTorch (HuggingFace)
- **Flujo**: LangGraph
- **RAG**: ChromaDB + sentence-transformers
- **LLM**: Google Gemini 2.0 Flash
- **Cloud**: Google Vertex AI

## 🎯 Casos de Uso

### **Agricultor Básico**

- Sube foto de planta enferma
- Pregunta: "¿Qué enfermedad es y cómo tratarla?"
- Recibe: Plan de acción simple y directo

### **Ingeniero Agrónomo**

- Sube imagen con síntomas específicos
- Pregunta: "Diagnóstico fitopatológico completo con MIP"
- Recibe: Análisis técnico y plan integrado

### **Productor Comercial**

- Sube foto de cultivo afectado
- Pregunta: "Tratamiento más económico para 10 hectáreas"
- Recibe: Recomendaciones costo-efectivas

## 🛠️ Resolución de Problemas

### **Error de inicio**

```bash
# Verificar que el entorno virtual esté activado
env\Scripts\activate

# Instalar dependencias
pip install streamlit pillow
```

### **Error de importación**

- Verificar que `langgraph_system_simple.py` esté en la misma carpeta
- Comprobar que todas las dependencias estén instaladas

### **Error de modelo**

- El modelo se descarga automáticamente la primera vez
- Verificar conexión a internet para HuggingFace

## 📈 Métricas de Rendimiento

- **Precisión CNN**: 74.1%
- **Tiempo respuesta**: 10-30 segundos
- **Formatos soportados**: JPG, JPEG, PNG
- **Tamaño máximo**: 200MB (Streamlit default)

## 🔒 Seguridad

- Imágenes procesadas temporalmente (se eliminan tras análisis)
- Credenciales Google Cloud desde archivo local
- No se almacenan consultas del usuario

---

**🍅 Sistema completo y listo para producción**
