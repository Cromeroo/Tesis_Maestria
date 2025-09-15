#!/usr/bin/env python3
"""
🍅 ASISTENTE AGRÓNOMO INTELIGENTE
Interfaz conversacional moderna similar a Claude/ChatGPT
Integra CNN + RAG + LLM para diagnóstico completo de tomate
"""

import streamlit as st
import os
import sys
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import time

# Configurar paths
sys.path.append(str(Path(__file__).parent))

# Importar componentes
from components.image_classifier import TomatoImageClassifier
from sistema_final import setup_system, generar_respuesta_completa

def configure_page():
    """Configurar la página con estilo Claude/ChatGPT"""
    st.set_page_config(
        page_title="🍅 Asistente Agrónomo",
        page_icon="🍅",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_css():
    """Cargar CSS personalizado para interfaz moderna"""
    st.markdown("""
    <style>
    /* Importar fuentes modernas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS */
    :root {
        --primary-green: #16a085;
        --light-green: #52c41a;
        --warm-yellow: #faad14;
        --text-dark: #2c3e50;
        --text-light: #7f8c8d;
        --bg-light: #f8f9fa;
        --bg-chat: #ffffff;
        --border-light: #e8ecef;
        --shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Fuente global */
    .main .block-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        max-width: 1200px;
        padding-top: 2rem;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, var(--primary-green), var(--light-green));
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Contenedor de chat */
    .chat-container {
        background: var(--bg-chat);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-light);
    }
    
    /* Mensajes de chat */
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: var(--bg-light);
        color: var(--text-dark);
        padding: 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 1rem 0;
        max-width: 85%;
        border-left: 4px solid var(--primary-green);
        box-shadow: var(--shadow);
    }
    
    /* Input área */
    .input-container {
        background: white;
        border-radius: 24px;
        padding: 1rem;
        box-shadow: var(--shadow);
        border: 2px solid var(--border-light);
        margin: 1rem 0;
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-green), var(--light-green));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(22, 160, 133, 0.4);
    }
    
    /* Sidebar */
    .sidebar .block-container {
        background: var(--bg-light);
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* Upload área */
    .upload-area {
        border: 2px dashed var(--primary-green);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(22, 160, 133, 0.05);
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-healthy { background: rgba(82, 196, 26, 0.1); color: #52c41a; }
    .status-disease { background: rgba(245, 34, 45, 0.1); color: #f5222d; }
    .status-other { background: rgba(250, 173, 20, 0.1); color: #faad14; }
    
    /* Animaciones */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .user-message, .assistant-message { max-width: 95%; }
    }
    
    /* Hiding streamlit elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {display: none;}
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Crear header principal"""
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🍅 Asistente Agrónomo Inteligente</h1>
        <p>Diagnóstico visual + Conocimiento especializado en tizón tardío</p>
    </div>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Inicializar estado de la sesión"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False

def load_systems():
    """Cargar sistemas CNN y RAG"""
    if not st.session_state.system_ready:
        with st.spinner("🔧 Cargando sistemas de IA..."):
            try:
                # Cargar clasificador CNN
                st.write("📥 Cargando modelo CNN...")
                st.session_state.classifier = TomatoImageClassifier()
                st.write("✅ CNN cargado")
                
                # Cargar sistema RAG
                st.write("📚 Inicializando sistema RAG...")
                collection, llm = setup_system()
                st.session_state.rag_system = (collection, llm)
                st.write("✅ RAG inicializado")
                
                st.session_state.system_ready = True
                st.success("✅ Todos los sistemas cargados correctamente")
                
            except Exception as e:
                st.error(f"❌ Error cargando sistemas: {e}")
                st.error(f"Detalle del error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def classify_image(image):
    """Clasificar imagen usando CNN"""
    try:
        result = st.session_state.classifier.classify_image(image)
        
        if "error" in result:
            st.error(f"Error en clasificación: {result['error']}")
            return None, None, None
        
        prediction = result["class_name"]
        confidence = result["confidence"] * 100  # Convertir a porcentaje
        probabilities = result["probabilities"]
        
        return prediction, confidence, probabilities
    except Exception as e:
        st.error(f"Error en clasificación: {e}")
        return None, None, None

def get_rag_response(query, classification):
    """Obtener respuesta del sistema RAG"""
    try:
        collection, llm = st.session_state.rag_system
        if collection and llm:
            # Adaptar query según clasificación
            if classification == "Tizon_tardio":
                enhanced_query = f"tizón tardío Phytophthora infestans tomate {query}"
            else:
                enhanced_query = query
                
            response, context = generar_respuesta_completa(enhanced_query, collection, llm)
            return response
        return "Sistema RAG no disponible"
    except Exception as e:
        return f"Error en RAG: {e}"

def render_chat_message(role, content, image=None, classification=None):
    """Renderizar mensaje de chat"""
    if role == "user":
        if image:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Imagen enviada", use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class="user-message fade-in">
                    <strong>🧑‍🌾 Tú:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="user-message fade-in">
                <strong>🧑‍🌾 Tú:</strong><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
    
    elif role == "assistant":
        # Determinar estilo según clasificación
        if classification:
            if classification == "Sana":
                status_class = "status-healthy"
                icon = "🌱"
            elif classification == "Tizon_tardio":
                status_class = "status-disease"
                icon = "🍄"
            else:
                status_class = "status-other"
                icon = "⚠️"
            
            st.markdown(f"""
            <div class="assistant-message fade-in">
                <div style="margin-bottom: 1rem;">
                    <span class="status-indicator {status_class}">
                        {icon} Diagnóstico: {classification.replace('_', ' ').title()}
                    </span>
                </div>
                <strong>🤖 Agrónomo Virtual:</strong><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message fade-in">
                <strong>🤖 Agrónomo Virtual:</strong><br>
                {content}
            </div>
            """, unsafe_allow_html=True)

def sidebar_controls():
    """Controles en la sidebar"""
    with st.sidebar:
        st.markdown("### 🎛️ Controles")
        
        # Upload de imagen
        st.markdown("#### 📸 Subir Imagen")
        uploaded_file = st.file_uploader(
            "Sube una foto de tu planta de tomate",
            type=['png', 'jpg', 'jpeg'],
            help="Formatos soportados: PNG, JPG, JPEG"
        )
        
        # Información del sistema
        st.markdown("#### ℹ️ Estado del Sistema")
        if st.session_state.system_ready:
            st.success("✅ CNN + RAG Activos")
        else:
            st.warning("⏳ Cargando sistemas...")
        
        # Estadísticas de chat
        st.markdown("#### 📊 Estadísticas")
        st.metric("Mensajes en chat", len(st.session_state.chat_history))
        
        # Limpiar chat
        if st.button("🗑️ Limpiar Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Información adicional
        st.markdown("---")
        st.markdown("""
        #### 🧠 Cómo funciona:
        1. **📤 Sube imagen** → CNN analiza
        2. **🔍 Si es tizón tardío** → Activa RAG especializado
        3. **💬 Pregunta libre** → Respuesta experta
        4. **📊 Resultado completo** → Diagnóstico + Recomendaciones
        """)
        
        return uploaded_file

def handle_image_upload(uploaded_file):
    """Manejar subida de imagen"""
    if uploaded_file and st.session_state.system_ready:
        try:
            # Mostrar imagen
            image = Image.open(uploaded_file)
            st.write(f"📷 Imagen cargada: {image.size} píxeles")
            
            with st.spinner("🔍 Analizando imagen..."):
                # Clasificar imagen
                prediction, confidence, probabilities = classify_image(image)
                
                if prediction:
                    st.write(f"🎯 Clasificación: {prediction} (Confianza: {confidence:.1f}%)")
                    
                    # Agregar a historial
                    user_message = "He subido una imagen de mi planta de tomate para diagnóstico."
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_message,
                        "image": image,
                        "timestamp": time.time()
                    })
                    
                    # Generar respuesta según clasificación
                    if prediction == "Tizon_tardio":
                        rag_query = "diagnóstico tizón tardío tratamiento recomendaciones manejo"
                        rag_response = get_rag_response(rag_query, prediction)
                        
                        response = f"""**Diagnóstico Visual Confirmado: TIZÓN TARDÍO**

Confianza: {confidence:.1f}%

{rag_response}

**⚠️ Recomendación:** Este es un caso confirmado de tizón tardío. Las medidas deben aplicarse inmediatamente para evitar propagación."""
                    
                    elif prediction == "Sana":
                        response = f"""**¡Excelente! Tu planta se ve SANA**

Confianza: {confidence:.1f}%

🌱 **Estado actual:** La planta muestra signos saludables sin evidencia de tizón tardío.

🛡️ **Mantén la prevención:**
- Continúa con riego en la base, evitando mojar follaje
- Asegura buena ventilación entre plantas
- Aplica fungicidas preventivos cada 7-10 días
- Monitorea diariamente, especialmente después de lluvia

💡 **Tip:** Una planta sana es la mejor defensa contra enfermedades."""
                    
                    else:  # Otras enfermedades
                        rag_query = "enfermedades tomate diagnóstico tratamiento control"
                        rag_response = get_rag_response(rag_query, prediction)
                        
                        response = f"""**Detectada: OTRA ENFERMEDAD (No Tizón Tardío)**

Confianza: {confidence:.1f}%

{rag_response}

**💡 Sugerencia:** Para un diagnóstico más específico, comparte más detalles sobre los síntomas visibles."""
                    
                    # Agregar respuesta al historial
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "classification": prediction,
                        "confidence": confidence,
                        "timestamp": time.time()
                    })
                    
                    return True
                else:
                    st.error("❌ No se pudo clasificar la imagen")
                    
        except Exception as e:
            st.error(f"❌ Error procesando imagen: {e}")
            import traceback
            st.code(traceback.format_exc())
            
    return False

def main():
    """Función principal"""
    # Configurar página
    configure_page()
    load_css()
    
    # Inicializar estado
    initialize_session_state()
    
    # Cargar sistemas
    load_systems()
    
    # Header
    create_header()
    
    # Sidebar
    uploaded_file = sidebar_controls()
    
    # Manejar upload de imagen
    if uploaded_file:
        handle_image_upload(uploaded_file)
    
    # Área principal de chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Mostrar historial de chat
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            render_chat_message(
                message["role"],
                message["content"],
                message.get("image"),
                message.get("classification")
            )
    else:
        # Mensaje de bienvenida
        st.markdown("""
        <div class="assistant-message fade-in">
            <strong>🤖 Agrónomo Virtual:</strong><br>
            ¡Hola! Soy tu asistente especializado en tomate y tizón tardío. 
            
            Puedes:
            📸 **Subir una imagen** de tu planta para diagnóstico automático
            💬 **Preguntar directamente** sobre manejo, tratamientos o prevención
            
            ¿En qué puedo ayudarte hoy?
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input de texto
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "💬 Escribe tu pregunta...",
                placeholder="Ej: ¿Cómo puedo prevenir el tizón tardío? o ¿Qué fungicida recomiendas?",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("📤 Enviar", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Procesar input de texto
    if (send_button or user_input) and user_input.strip() and st.session_state.system_ready:
        # Agregar mensaje del usuario
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # Generar respuesta RAG
        with st.spinner("🤔 Consultando base de conocimiento..."):
            response = get_rag_response(user_input, None)
        
        # Agregar respuesta del asistente
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time()
        })
        
        # Rerun para mostrar nuevos mensajes
        st.rerun()

if __name__ == "__main__":
    main()