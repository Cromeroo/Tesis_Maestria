# Sistema Integrado RAG + Clasificador de Imágenes de Tomate 🍅🤖

## Descripción

Sistema completo que combina:

- **Clasificación de imágenes** de tomate (Sana, Tizón tardío, Otras enfermedades)
- **Sistema RAG** para consultas automáticas en base de conocimiento científico
- **Recomendaciones automáticas** basadas en la clasificación

## Flujo del Sistema

```
📸 Imagen → 🤖 Clasificación → 🔍 Consulta RAG → 📚 Recomendaciones
```

1. **Subir imagen** de tomate
2. **Modelo clasifica** automáticamente (descarga desde Hugging Face)
3. **RAG consulta** automáticamente documentos científicos
4. **Obtener diagnóstico** + recomendaciones específicas

## Instalación Rápida

```bash
# Instalar dependencias
pip install -r requirements.txt

# Verificar estado del sistema
python demo_rag_classifier.py --status
```

## Uso Básico

### 1. Clasificar una sola imagen

```bash
python demo_rag_classifier.py --image path/to/tomato_image.jpg
```

### 2. Ver estado del sistema

```bash
python demo_rag_classifier.py --status
```

### 3. Procesar múltiples imágenes

```bash
python demo_rag_classifier.py --batch path/to/images_folder/
```

### 4. Probar solo clasificación

```bash
python demo_rag_classifier.py --test-classification
```

## Ejemplo de Salida

```
📊 RESULTADO DE CLASIFICACIÓN
==================================================
🏷️ Clase predicha: Tizon_tardio
🎯 Confianza: 94.3%
💻 Dispositivo usado: cpu

📚 RECOMENDACIONES DE LA BASE DE CONOCIMIENTO
============================================================
🔍 Consulta 1: tratamiento tizón tardío tomate Phytophthora infestans
📄 Resultados encontrados: 3

📋 RESUMEN DEL DIAGNÓSTICO
========================================
🩺 Diagnóstico: 🔴 Tizón tardío detectado - ACCIÓN REQUERIDA
🎯 Nivel de confianza: Alta (0.943)
⚠️ Prioridad: Alta
📚 Recomendaciones encontradas: 9

🎯 ACCIONES SUGERIDAS
==============================
   🚨 ACCIÓN INMEDIATA: Implementar tratamiento fungicida
   🔬 Confirmar diagnóstico con especialista si es posible
   🚫 Aislar plantas afectadas para prevenir propagación
   ...
```

## Uso Programático

```python
from services.integrated_rag_classifier import diagnose_tomato_from_image

# Diagnóstico rápido
result = diagnose_tomato_from_image("mi_tomate.jpg")

if result["success"]:
    classification = result["classification"]
    print(f"Clase: {classification['class_name']}")
    print(f"Confianza: {classification['confidence']:.1%}")

    # Recomendaciones automáticas
    for rec in result["rag_recommendations"]:
        print(f"Consulta: {rec['query']}")
        print(f"Resultados: {len(rec['results'])}")
```

## Uso Avanzado

```python
from services.integrated_rag_classifier import IntegratedRAGClassifier

# Servicio completo
service = IntegratedRAGClassifier()

# Diagnóstico completo
result = service.diagnose_tomato_image("imagen.jpg", n_rag_results=5)

# Procesar lote
results = service.batch_diagnose_images(["img1.jpg", "img2.jpg"])

# Guardar reporte
service.save_diagnosis_report(result, "reporte.json")
```

## Configuración del Modelo

El modelo se descarga automáticamente desde:

- **Repositorio**: `DaniloR2011/Tomato_accuracy`
- **Archivo**: `best_model_3class.pth`
- **Clases**: Sana, Tizón tardío, Otras enfermedades

### Usar modelo personalizado:

```python
service = IntegratedRAGClassifier(model_repo="tu_usuario/tu_modelo")
```

## Archivos Principales

```
📁 components/
   └── image_classifier.py          # Clasificador de imágenes
📁 services/
   └── integrated_rag_classifier.py # Servicio integrado
📄 demo_rag_classifier.py          # Script de demostración
📄 requirements.txt                 # Dependencias
```

## Dependencias Clave

- `torch` + `torchvision`: Modelo de clasificación
- `huggingface-hub`: Descarga automática del modelo
- `Pillow`: Procesamiento de imágenes
- `chromadb`: Base de datos vectorial
- `langchain`: Sistema RAG

## Troubleshooting

### Modelo no se descarga

```bash
# Verificar conexión y token HF
hf auth login
```

### RAG no funciona

```bash
# Construir base de conocimiento
python main.py --process
```

### Errores de memoria

El modelo usa CPU por defecto. Para GPU:

```python
classifier = TomatoImageClassifier(device=torch.device("cuda"))
```

## Próximas Mejoras

- [ ] Interfaz web con Streamlit
- [ ] API REST con FastAPI
- [ ] Soporte para más tipos de cultivos
- [ ] Integración con cámaras en tiempo real
- [ ] Notificaciones automáticas por email/SMS

## Contacto

- **Modelo**: https://huggingface.co/DaniloR2011/Tomato_accuracy
- **Repositorio**: https://github.com/Cromeroo/Tesis_Maestria

---

¡Sistema listo para diagnosticar tus tomates! 🍅✨
