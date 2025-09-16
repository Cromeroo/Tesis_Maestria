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
                st.session_state.classifier = TomatoImageClassifier()
                st.write("✅ Clasificador CNN cargado")
                
                # Cargar sistema RAG
                collection, llm = setup_system()
                if collection is None or llm is None:
                    raise Exception("No se pudo inicializar el sistema RAG. Verifica ChromaDB y las credenciales de Google.")
                
                st.session_state.rag_system = (collection, llm)
                st.write("✅ Sistema RAG cargado")
                
                st.session_state.system_ready = True
                st.success("✅ Sistemas cargados correctamente")
            except Exception as e:
                st.error(f"❌ Error cargando sistemas: {e}")
                st.write("Detalle del error:", str(e))
                # Mostrar información de diagnóstico
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

def get_rag_response(query, classification, chat_history=None):
    """Obtener respuesta del sistema RAG con memoria conversacional"""
    try:
        collection, llm = st.session_state.rag_system
        if collection and llm:
            # Verificar si hay una imagen analizada recientemente
            image_context = ""
            if "temp_image_display" in st.session_state:
                img_pred = st.session_state.temp_image_display['prediction']
                img_conf = st.session_state.temp_image_display['confidence']
                image_context = f"\n\nCONTEXTO DE IMAGEN RECIENTE: Se analizó una imagen con resultado '{img_pred}' (confianza: {img_conf:.1f}%). "
            
            # Construir contexto conversacional
            conversation_context = ""
            if chat_history:
                # Tomar las últimas 6 interacciones (3 pares pregunta-respuesta)
                recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
                conversation_pairs = []
                for i in range(0, len(recent_history)-1, 2):
                    if i+1 < len(recent_history):
                        user_msg = recent_history[i].get("content", "")
                        assistant_msg = recent_history[i+1].get("content", "")
                        conversation_pairs.append(f"Usuario: {user_msg}\nAsistente: {assistant_msg}")
                
                if conversation_pairs:
                    conversation_context = f"\n\nContexto de conversación previa:\n" + "\n---\n".join(conversation_pairs[-2:])  # Últimas 2 interacciones
            
            # Adaptar query según clasificación o imagen analizada
            if classification == "Tizon_tardio" or (image_context and "Tizon_tardio" in image_context):
                enhanced_query = f"tizón tardío Phytophthora infestans tomate {query}{image_context}{conversation_context}"
            else:
                enhanced_query = f"{query}{image_context}{conversation_context}"
                
            response, context = generar_respuesta_completa(enhanced_query, collection, llm)
            return response
        return "Sistema RAG no disponible"
    except Exception as e:
        return f"Error en RAG: {e}"

def render_chat_message(role, content, image=None, classification=None, image_id=None):
    """Renderizar mensaje de chat"""
    if role == "user":
        if image and image_id:
            # Para imágenes, solo mostrar una referencia textual después de la primera vez
            if "image_analysis_shown" not in st.session_state:
                st.session_state.image_analysis_shown = set()
            
            if image_id not in st.session_state.image_analysis_shown:
                # Primera vez que aparece esta imagen en el chat
                st.session_state.image_analysis_shown.add(image_id)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption="📸 Imagen para análisis", use_container_width=True)
                with col2:
                    st.markdown(f"""
                    <div class="user-message fade-in">
                        <strong>🧑‍🌾 Tú:</strong><br>
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Referencias posteriores a la misma imagen - solo texto
                st.markdown(f"""
                <div class="user-message fade-in">
                    <strong>🧑‍🌾 Tú:</strong><br>
                    � {content}
                </div>
                """, unsafe_allow_html=True)
        else:
            # Mensaje de texto normal
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
            # Limpiar imagen temporal
            if "temp_image_display" in st.session_state:
                del st.session_state["temp_image_display"]
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
        # Mostrar imagen inmediatamente (fuera del historial)
        image = Image.open(uploaded_file)
        
        with st.spinner("🔍 Analizando imagen..."):
            # Clasificar imagen
            prediction, confidence, probabilities = classify_image(image)
            
            if prediction:
                # Guardar imagen y resultados para mostrar temporalmente
                st.session_state.temp_image_display = {
                    "image": image,
                    "prediction": prediction,
                    "confidence": confidence
                }
                
                # NO agregar nada al historial automáticamente
                # El usuario puede hacer preguntas después si quiere
                
                return True
            else:
                st.error("❌ No se pudo procesar la imagen. Intenta con otra imagen.")
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
    
    # Contenedor para mostrar temporalmente las imágenes subidas
    if "temp_image_display" in st.session_state:
        with st.container():
            st.markdown("### 📸 Resultado del Análisis")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.image(st.session_state.temp_image_display["image"], 
                        caption="Imagen analizada", use_container_width=True)
            with col2:
                pred = st.session_state.temp_image_display['prediction']
                conf = st.session_state.temp_image_display['confidence']
                
                if pred == "Tizon_tardio":
                    st.error(f"🍄 **TIZÓN TARDÍO DETECTADO**")
                    st.warning(f"⚠️ Confianza: {conf:.1f}%")
                    st.info("💬 **Haz preguntas** sobre tratamientos, prevención o manejo")
                elif pred == "Sana":
                    st.success(f"🌱 **PLANTA SANA**")
                    st.info(f"✅ Confianza: {conf:.1f}%")
                    st.info("💬 **Haz preguntas** sobre cuidados preventivos")
                else:
                    st.warning(f"⚠️ **OTRA ENFERMEDAD**")
                    st.info(f"🔍 Confianza: {conf:.1f}%")
                    st.info("💬 **Haz preguntas** para más detalles")
                
            with col3:
                if st.button("❌ Ocultar", key="hide_temp_image"):
                    del st.session_state.temp_image_display
                    st.rerun()
        st.markdown("---")
    
    # Mostrar historial de chat
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            render_chat_message(
                message["role"],
                message["content"],
                message.get("image"),
                message.get("classification"),
                message.get("image_id")
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
    
    # Input de texto con form para evitar loops
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "💬 Escribe tu pregunta...",
                    placeholder="Ej: ¿Cómo puedo prevenir el tizón tardío? o ¿Qué fungicida recomiendas?",
                    label_visibility="collapsed",
                    key="user_input"
                )
            
            with col2:
                send_button = st.form_submit_button("📤 Enviar", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Procesar input de texto
    if send_button and user_input and user_input.strip() and st.session_state.system_ready:
        # Agregar mensaje del usuario
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip(),
            "timestamp": time.time()
        })
        
        # Generar respuesta RAG
        with st.spinner("🤔 Consultando base de conocimiento..."):
            response = get_rag_response(user_input.strip(), None, st.session_state.chat_history)
        
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