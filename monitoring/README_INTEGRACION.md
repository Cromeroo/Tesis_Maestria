# EJEMPLO DE INTEGRACIÓN DEL SISTEMA DE MÉTRICAS

## Cómo usar el sistema de métricas avanzadas con tu código existente

### 1. Integración Básica (Sin modificar código existente)

```python
# Al inicio de tu archivo web_langgraph_demo.py, añadir:
from monitoring.advanced_metrics import metrics_collector

# En la función principal donde procesas consultas:
def process_query_with_metrics(query, classification=None):
    # Iniciar sesión de métricas
    session_id = metrics_collector.start_session()

    try:
        # Tu código existente aquí...
        resultado = st.session_state.langgraph_system.procesar_consulta(query, classification)

        # Finalizar métricas
        metrics_collector.finalize_session_metrics(
            final_response=resultado.get('respuesta_final', ''),
            context_detected=resultado.get('contexto_detectado', 'unknown'),
            success=True
        )

        return resultado

    except Exception as e:
        metrics_collector.finalize_session_metrics(
            final_response='',
            context_detected='error',
            success=False,
            error_count=1
        )
        raise
```

### 2. Integración Avanzada (Decoradores)

```python
from monitoring.advanced_metrics import track_node_execution

# Para trackear automáticamente nodos de LangGraph:
@track_node_execution("detectar_contexto")
def _detectar_contexto_node(self, state: RAGState) -> RAGState:
    # Tu código existente sin cambios
    pass

@track_node_execution("buscar_documentos")
def _buscar_documentos_node(self, state: RAGState) -> RAGState:
    # Tu código existente sin cambios
    pass
```

### 3. Métricas de CNN (Para image_analyzer.py)

```python
from monitoring.advanced_metrics import metrics_collector

def classify_image_with_metrics(self, image_path):
    start_time = time.time()

    # Preprocesamiento
    preprocess_start = time.time()
    # ... tu código de preprocesamiento ...
    preprocess_time = time.time() - preprocess_start

    # Inferencia
    inference_start = time.time()
    # ... tu código de inferencia ...
    inference_time = time.time() - inference_start

    # Registrar métricas
    metrics_collector.record_cnn_metrics(
        image_path=image_path,
        prediction=predicted_class,
        confidence=confidence,
        preprocessing_time=preprocess_time,
        inference_time=inference_time
    )

    return result
```

### 4. Panel de Métricas en Streamlit

```python
# Añadir a tu web_langgraph_demo.py:
def show_metrics_dashboard():
    st.subheader("📊 Métricas del Sistema")

    if st.button("🔍 Generar Reporte"):
        report = metrics_collector.generate_performance_report()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Tiempo Total (ms)",
                report['session_summary']['total_execution_time_ms']
            )

        with col2:
            st.metric(
                "Tasa de Éxito (%)",
                report['session_summary']['success_rate']
            )

        with col3:
            st.metric(
                "Consultas Procesadas",
                report['global_stats']['total_queries_processed']
            )

        # Gráfico de rendimiento por nodo
        st.subheader("⚡ Rendimiento por Nodo")
        node_data = report['node_performance']

        if node_data:
            import pandas as pd
            df = pd.DataFrame([
                {
                    'Nodo': name,
                    'Tiempo Promedio (ms)': data['avg_time_ms'],
                    'Tasa Éxito (%)': data['success_rate']
                }
                for name, data in node_data.items()
            ])
            st.dataframe(df)
```

### 5. Archivo de Configuración de Métricas

```json
{
  "metrics_config": {
    "enabled": true,
    "log_level": "INFO",
    "save_to_file": true,
    "show_console": true,
    "performance_thresholds": {
      "node_execution_warning_ms": 1000,
      "total_execution_warning_ms": 5000,
      "rag_retrieval_warning_ms": 500
    },
    "retention_days": 30
  }
}
```

## VENTAJAS DE ESTE SISTEMA:

✅ **No modifica código existente**: Se integra como capa adicional
✅ **Métricas detalladas**: Tiempo por nodo, calidad RAG, rendimiento CNN
✅ **Persistencia**: Guarda métricas en archivos JSON
✅ **Reportes automáticos**: Análisis de rendimiento y estadísticas
✅ **Debugging avanzado**: Trazabilidad completa del flujo
✅ **Dashboard visual**: Panel de métricas en Streamlit
✅ **Configuración flexible**: Sistema habilitado/deshabilitado fácilmente

## MÉTRICAS CAPTURADAS:

📊 **Por Nodo**: Tiempo de ejecución, éxito/fallo, tamaño entrada/salida
🔍 **RAG**: Documentos recuperados, scores de similitud, tiempo de búsqueda
🖼️ **CNN**: Confianza, tiempo de inferencia, preprocesamiento
💬 **Conversacional**: Seguimiento de casos, cambios de contexto
🌍 **Sistema**: Métricas globales, distribución de contextos, errores
