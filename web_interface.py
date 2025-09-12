import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import tempfile
import os

# Configurar página
st.set_page_config(
    page_title="🍅 Diagnóstico de Enfermedades en Tomate",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar path del sistema
sys.path.append(str(Path(__file__).parent))

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🍅 Sistema de Diagnóstico de Enfermedades en Tomate</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Sistema Inteligente con LangGraph + CNN + RAG + Gemini 2.0 Flash")
    
    # Sidebar con información
    with st.sidebar:
        st.header("📋 Información del Sistema")
        st.markdown("""
        **🔬 Componentes:**
        - **CNN**: Clasificación de imágenes (74.1% precisión)
        - **LangGraph**: Flujo condicional inteligente  
        - **RAG**: Búsqueda en documentos especializados
        - **Gemini 2.0 Flash**: Respuestas de agrónomo experto
        
        **🎯 Flujo:**
        1. Sube imagen + escribe pregunta
        2. CNN detecta enfermedad
        3. Si es tizón tardío → activa RAG
        4. LLM genera plan de manejo
        """)
        
        st.header("💡 Tipos de Preguntas")
        st.markdown("""
        - **Básica**: "¿Qué enfermedad es y cómo tratarla?"
        - **Técnica**: "Diagnóstico fitopatológico completo"  
        - **Contextual**: "Mi invernadero tiene alta humedad..."
        - **Económica**: "Tratamiento más económico"
        - **Preventiva**: "¿Cómo prevenir en futuras siembras?"
        """)
    
    # Área principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Sube tu Imagen")
        
        # Subida de imagen
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de la planta:",
            type=['jpg', 'jpeg', 'png'],
            help="Formatos soportados: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Mostrar imagen
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            # Información de la imagen
            st.info(f"📊 **Archivo:** {uploaded_file.name} | **Tamaño:** {image.size}")
        
        st.header("💬 Escribe tu Pregunta")
        
        # Preguntas predefinidas
        pregunta_tipo = st.selectbox(
            "🎯 Selecciona tipo de consulta:",
            [
                "Personalizada",
                "🚨 Urgencia Básica", 
                "🔬 Diagnóstico Técnico",
                "🌾 Contexto Específico", 
                "💰 Enfoque Económico",
                "🌱 Prevención Futura",
                "⚡ Solo Imagen"
            ]
        )
        
        # Preguntas predefinidas
        preguntas_default = {
            "🚨 Urgencia Básica": "¿Qué enfermedad tiene mi planta y cómo la trato urgentemente?",
            "🔬 Diagnóstico Técnico": "Necesito un diagnóstico fitopatológico completo con plan de manejo integrado para esta sintomatología",
            "🌾 Contexto Específico": "Mi cultivo de tomate bajo invernadero presenta estos síntomas. Tengo alta humedad y temperatura de 22°C. ¿Qué aplicar?",
            "💰 Enfoque Económico": "¿Cuál es el tratamiento más costo-efectivo para esta enfermedad? Tengo presupuesto limitado",
            "🌱 Prevención Futura": "¿Cómo puedo prevenir que esta enfermedad vuelva a aparecer en próximas siembras?",
            "⚡ Solo Imagen": ""
        }
        
        # Campo de texto
        if pregunta_tipo == "Personalizada":
            user_question = st.text_area(
                "Escribe tu pregunta personalizada:",
                height=100,
                placeholder="Ejemplo: Mi cultivo muestra estos síntomas después de lluvia..."
            )
        else:
            user_question = st.text_area(
                f"Pregunta {pregunta_tipo}:",
                value=preguntas_default.get(pregunta_tipo, ""),
                height=100
            )
    
    with col2:
        st.header("🤖 Resultado del Diagnóstico")
        
        if uploaded_file is not None:
            # Botón de análisis
            if st.button("🚀 Analizar Imagen", type="primary", use_container_width=True):
                
                with st.spinner("🔄 Procesando con sistema LangGraph..."):
                    try:
                        # Guardar imagen temporal
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                            image.save(tmp_file.name)
                            temp_image_path = tmp_file.name
                        
                        # Importar y ejecutar sistema
                        from langgraph_system_simple import TomatoDiseaseGraph
                        
                        # Inicializar sistema
                        graph_system = TomatoDiseaseGraph()
                        
                        # Procesar
                        result = graph_system.process_input(
                            image_path=temp_image_path,
                            user_text=user_question or ""
                        )
                        
                        # Limpiar archivo temporal
                        os.unlink(temp_image_path)
                        
                        # Mostrar resultados
                        if result["success"]:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success("✅ Análisis completado exitosamente")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Clasificación
                            if result.get("disease_prediction"):
                                pred = result["disease_prediction"]
                                st.markdown("### 🔬 Clasificación")
                                
                                col_a, col_b = st.columns([1, 1])
                                with col_a:
                                    st.metric("Enfermedad Detectada", pred.get('class_name', 'N/A'))
                                with col_b:
                                    confidence = pred.get('confidence', 0)
                                    st.metric("Nivel de Confianza", f"{confidence:.1%}")
                                
                                # Barra de confianza
                                st.progress(confidence)
                            
                            # Respuesta del agente
                            if result.get("final_response"):
                                st.markdown("### 🤖 Recomendaciones del Agrónomo")
                                
                                # Respuesta en un contenedor con scroll
                                with st.container():
                                    st.markdown(result["final_response"])
                            
                            # Advertencias si las hay
                            if result.get("error"):
                                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                                st.warning(f"⚠️ Información adicional: {result['error']}")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                        else:
                            st.markdown('<div class="error-box">', unsafe_allow_html=True)
                            st.error(f"❌ Error en el análisis: {result.get('error', 'Error desconocido')}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error(f"❌ Error del sistema: {str(e)}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Información técnica
            with st.expander("🔧 Información Técnica del Sistema"):
                st.markdown("""
                **🏗️ Arquitectura:**
                - **Modelo CNN**: PyTorch en HuggingFace (DaniloR2011/Tomato_accuracy)
                - **Framework de flujo**: LangGraph para decisiones condicionales
                - **Base de conocimiento**: ChromaDB con sentence-transformers
                - **LLM**: Google Gemini 2.0 Flash Experimental
                - **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2
                
                **📊 Clases detectables:**
                - Sana
                - Tizón Tardío (Late Blight)
                - Otras Enfermedades
                
                **🎯 Solo se activa RAG completo para Tizón Tardío**
                """)
                
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.info("👆 Sube una imagen para comenzar el análisis")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🍅 Sistema de Diagnóstico de Enfermedades en Tomate | 
        Tecnología: LangGraph + CNN + RAG + Gemini 2.0 Flash
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
