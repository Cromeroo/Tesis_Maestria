#!/usr/bin/env python3
"""
🍅 ASISTENTE AGRÓNOMO LANGGRAPH
Interfaz web para demostrar el sistema RAG con detección de contexto funcional
Muestra el flujo de nodos de LangGraph y la personalización por contexto
"""

import streamlit as st
import os
import sys
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import time
from datetime import datetime

# Configurar credenciales desde archivo JSON
try:
    import json
    with open('credentials.json', 'r') as f:
        credentials = json.load(f)
    if 'huggingface_token' in credentials:
        os.environ["HF_TOKEN"] = credentials['huggingface_token']
        print("✅ Token HF cargado desde credentials.json")
except Exception as e:
    print(f"⚠️ No se pudo cargar token desde credentials.json: {e}")

# Configurar paths
sys.path.append(str(Path(__file__).parent))

# Importar componentes
from components.image_classifier import TomatoImageClassifier
try:
    from sistema_langgraph_demo import SistemaRAGLangGraph
except ImportError:
    from sistema_rag_simple import SistemaRAGSimple as SistemaRAGLangGraph

def configure_page():
    """Configurar la página con estilo moderno"""
    st.set_page_config(
        page_title="🍅 LangGraph RAG Demo",
        page_icon="🍅",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_css():
    """CSS moderno estilo ChatGPT/Claude - Legible y profesional"""
    st.markdown("""
    <style>
    /* Importar fuentes modernas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Variables CSS - Paleta más contrastante */
    :root {
        --primary-green: #10b981;
        --primary-blue: #3b82f6;
        --primary-orange: #f59e0b;
        --primary-red: #ef4444;
        --text-dark: #111827;
        --text-medium: #374151;
        --text-light: #6b7280;
        --bg-main: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-accent: #f1f5f9;
        --border: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --radius: 12px;
    }
    
    /* Reset y base */
    * {
        box-sizing: border-box;
    }
    
    .main .block-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        max-width: 1200px;
        width: 100%;
        padding: 1rem 2rem;
        background: var(--bg-main);
    }
    
    /* Header moderno */
    .main-header {
        background: linear-gradient(135deg, var(--primary-green), var(--primary-blue));
        color: white;
        padding: 2.5rem 2rem;
        border-radius: var(--radius);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border);
        width: 100%;
        max-width: 100%;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.025em;
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 400;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    

    
    /* Badges e indicadores */
    .context-indicator, .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.75rem 1.25rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 0.5rem 0.5rem 0.5rem 0;
        box-shadow: var(--shadow-sm);
        border: none;
        letter-spacing: 0.025em;
    }
    
    .context-campesino { 
        background: #10b981; 
        color: white;
    }
    .context-tecnico { 
        background: #3b82f6; 
        color: white;
    }
    .context-urgencia { 
        background: #ef4444; 
        color: white;
    }
    .context-general { 
        background: #6b7280; 
        color: white;
    }
    
    .status-healthy { 
        background: #10b981; 
        color: white;
    }
    .status-disease { 
        background: #ef4444; 
        color: white;
    }
    .status-other { 
        background: #f59e0b; 
        color: white;
    }
    
    /* Mensajes de chat estilo ChatGPT */
    .user-message {
        background: var(--primary-blue);
        color: white;
        padding: 1.25rem 1.75rem;
        border-radius: var(--radius);
        margin: 1.5rem 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: var(--shadow);
        font-size: 1rem;
        line-height: 1.6;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .assistant-message {
        background: var(--bg-main);
        color: var(--text-dark);
        padding: 1.75rem 2rem;
        border-radius: var(--radius);
        margin: 1.5rem 0;
        width: 100%;
        max-width: 100%;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        font-size: 1rem;
        line-height: 1.7;
        font-family: 'Inter', sans-serif;
    }
    
    .assistant-message strong {
        color: var(--primary-green);
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .warning-box {
        background: #fef3c7;
        border: 2px solid #f59e0b;
        color: #7c2d12;
        padding: 1.75rem;
        border-radius: var(--radius);
        margin: 1.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: var(--shadow);
        text-align: center;
        line-height: 1.6;
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
    }
    
    /* Botones modernos */
    .stButton > button {
        background: var(--primary-green);
        color: white;
        border: none;
        border-radius: var(--radius);
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow);
        min-height: 48px;
    }
    
    .stButton > button:hover {
        background: #059669;
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }
    
    .stButton > button:disabled {
        background: #d1d5db;
        color: #9ca3af;
        transform: none;
        box-shadow: var(--shadow-sm);
    }
    
    /* Input personalizado */
    .stTextInput > div > div > input {
        border-radius: var(--radius);
        border: 2px solid var(--border);
        padding: 0.875rem 1rem;
        font-size: 1rem;
        transition: all 0.2s ease;
        background: var(--bg-main);
        color: #2c3e50 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-green);
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
        color: #2c3e50 !important;
    }
    
    /* Métricas mejoradas */
    .metric-container {
        background: var(--bg-main);
        padding: 1.5rem;
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
        text-align: center;
        margin: 0.75rem 0;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-container .metric-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-green);
        margin-bottom: 0.5rem;
    }
    
    .metric-container .metric-label {
        font-size: 0.875rem;
        color: var(--text-light);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Métricas económicas */
    .economic-metric-critical {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1.5rem;
        border-radius: var(--radius);
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-lg);
    }
    
    .economic-metric-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1.5rem;
        border-radius: var(--radius);
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-lg);
    }
    
    .economic-metric-success {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1.5rem;
        border-radius: var(--radius);
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-lg);
    }
    
    .economic-metric h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .economic-metric .metric-value {
        margin: 0.5rem 0 0 0;
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    .economic-metric .metric-subtitle {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Alertas de urgencia */
    .urgency-alert-critical {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
        padding: 1.5rem;
        border-radius: var(--radius);
        margin: 1rem 0;
        border-left: 5px solid #fca5a5;
        box-shadow: var(--shadow-lg);
        animation: pulse 2s infinite;
    }
    
    .urgency-alert-high {
        background: linear-gradient(135deg, #d97706, #b45309);
        color: white;
        padding: 1.5rem;
        border-radius: var(--radius);
        margin: 1rem 0;
        border-left: 5px solid #fcd34d;
        box-shadow: var(--shadow-lg);
    }
    
    .urgency-alert-moderate {
        background: linear-gradient(135deg, #059669, #047857);
        color: white;
        padding: 1.5rem;
        border-radius: var(--radius);
        margin: 1rem 0;
        border-left: 5px solid #6ee7b7;
        box-shadow: var(--shadow);
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
    
    .status-healthy { background: rgba(16, 185, 129, 0.1); color: #10b981; }
    .status-disease { background: rgba(239, 68, 68, 0.1); color: #ef4444; }
    .status-other { background: rgba(245, 158, 11, 0.1); color: #f59e0b; }
    
    /* Animaciones */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Estilos adicionales para chat mejorado */
    .chat-message {
        margin: 1.5rem 0;
        animation: fadeIn 0.3s ease-out;
    }
    
    .response-container {
        background: var(--bg-main);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem;
        margin: 1.5rem 0;
        min-height: 200px;
        position: relative;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    
    .response-container::before {
        content: '🤖 Asistente Agrícola';
        position: absolute;
        top: 1rem;
        right: 1.5rem;
        font-size: 0.875rem;
        color: var(--text-light);
        font-weight: 500;
    }
    
    .loading-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-light);
        font-style: italic;
    }
    
    .loading-dots::after {
        content: '...';
        animation: loadingDots 1.5s infinite;
    }
    
    @keyframes loadingDots {
        0%, 20% { color: transparent; }
        40% { color: var(--text-light); }
        100% { color: transparent; }
    }
    
    /* Responsividad mejorada */
    @media (max-width: 768px) {
        .user-message, .assistant-message {
            max-width: 100%;
            margin-left: 0;
            margin-right: 0;
        }
        
        .metric-container {
            margin: 0.5rem 0;
            min-height: 100px;
        }
        
        .response-container {
            padding: 1.5rem;
            font-size: 1rem;
        }
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
        <h1>🔗 LangGraph RAG + Contexto Funcional</h1>
        <p>Demostración de flujo de nodos con detección de contexto que realmente funciona</p>
    </div>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Inicializar estado de la sesión"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'langgraph_system' not in st.session_state:
        st.session_state.langgraph_system = None
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'last_execution_flow' not in st.session_state:
        st.session_state.last_execution_flow = None
    
    # 🧠 NUEVA: Memoria conversacional
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = {
            'history': [],  # Lista de conversaciones previas
            'user_profile': {
                'detected_context': None,  # último contexto detectado
                'preference_style': None,  # técnico/campesino/general
                'previous_cases': [],  # casos tratados anteriormente
                'last_diagnosis': None,  # último diagnóstico
                'follow_up_pending': False  # si hay seguimiento pendiente
            },
            'session_count': 0  # número de conversación en esta sesión
        }


def load_systems():
    """Cargar sistemas CNN y LangGraph"""
    if not st.session_state.system_ready:
        with st.spinner("🔧 Cargando sistemas LangGraph..."):
            try:
                # Cargar clasificador de imágenes desde HuggingFace
                st.session_state.classifier = TomatoImageClassifier()
                st.write("✅ Clasificador de imágenes cargado desde HuggingFace")
                
                # Cargar sistema LangGraph
                st.session_state.langgraph_system = SistemaRAGLangGraph()
                st.write("✅ Sistema LangGraph cargado")
                
                st.session_state.system_ready = True
                st.success("🎉 Sistemas cargados correctamente")
            except Exception as e:
                st.error(f"❌ Error cargando sistemas: {e}")
                st.write("Detalle del error:", str(e))
                import traceback
                st.code(traceback.format_exc())

def classify_image(image):
    """Clasificar imagen usando CNN con análisis económico integrado"""
    try:
        result = st.session_state.classifier.classify_image(image)
        if "error" in result:
            st.error(f"Error en clasificación: {result['error']}")
            return None, None, None
        
        prediction = result["class_name"]
        confidence = result["confidence"] * 100
        probabilities = result["probabilities"]
        
        return prediction, confidence, probabilities
    except Exception as e:
        st.error(f"Error en clasificación: {e}")
        return None, None, None

def analyze_single_image_with_economics(image):
    """
    Analizar imagen individual con análisis de severidad y económico completo
    
    Args:
        image: PIL Image object
    
    Returns:
        dict: Resultado completo con análisis económico
    """
    prediction, confidence, probabilities = classify_image(image)
    
    if not prediction:
        return None
    
    result = {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': probabilities,
        'severity_analysis': None,
        'economic_analysis': None
    }
    
    # Solo hacer análisis económico si hay enfermedad
    if prediction == "Tizon_tardio":
        # Obtener query del usuario y contexto
        user_query = st.session_state.get('current_user_query', '')
        user_profile = st.session_state.conversation_memory.get('user_profile', {})
        user_context = user_profile.get('detected_context') or 'general'  # Asegurar que no sea None
        
        # Calcular severidad (sin consenso, solo CNN)
        severity_level, severity_score, severity_details, severity_color = calculate_severity_score(
            cnn_confidence=confidence,
            consensus_data=None,  # No hay consenso en imagen individual
            user_query=user_query,
            conversation_history=st.session_state.conversation_memory.get('history', [])
        )
        
        result['severity_analysis'] = {
            'level': severity_level,
            'score': severity_score,
            'details': severity_details,
            'color': severity_color
        }
        
        # Calcular impacto económico
        result['economic_analysis'] = calculate_economic_impact_cop(
            severity_level=severity_level,
            classification="Tizon_tardio",
            user_context=user_context,
            num_plants=100
        )
    
    return result

def analyze_multiple_images(images_with_labels):
    """
    Analizar múltiples imágenes y generar consenso inteligente con análisis económico
    
    Args:
        images_with_labels: Lista de tuplas (image, label) donde label puede ser None
    
    Returns:
        dict con resultado consensuado + análisis de severidad y económico
    """
    if not images_with_labels:
        return None
    
    results = []
    
    # Analizar cada imagen individualmente
    for i, (image, label) in enumerate(images_with_labels):
        prediction, confidence, probabilities = classify_image(image)
        
        if prediction:
            results.append({
                'image_index': i,
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities
            })
    
    if not results:
        return None
    
    # Algoritmo de consenso inteligente
    consensus_result = generate_consensus_diagnosis(results)
    
    # 🆕 ANÁLISIS DE SEVERIDAD INTELIGENTE
    severity_analysis = None
    
    if consensus_result and consensus_result['final_prediction'] == "Tizon_tardio":
        # Calcular severidad
        cnn_confidence = consensus_result['consensus_confidence']
        user_query = st.session_state.get('current_user_query', '')  # Obtener query actual
        
        severity_level, severity_score, severity_details, severity_color = calculate_severity_score(
            cnn_confidence=cnn_confidence,
            consensus_data=consensus_result,
            user_query=user_query,
            conversation_history=st.session_state.conversation_memory.get('history', [])
        )
        
        severity_analysis = {
            'level': severity_level,
            'score': severity_score,
            'details': severity_details,
            'color': severity_color
        }
    
    return {
        'individual_results': results,
        'consensus': consensus_result,
        'total_images': len(results),
        'analysis_type': 'multi-image',
        'severity_analysis': severity_analysis  # 🆕 NUEVO - Solo severidad, economía será interactiva
    }

def generate_consensus_diagnosis(results):
    """Generar diagnóstico consensuado basado en múltiples análisis"""
    if not results:
        return None
    
    # Contar predicciones por clase
    class_votes = {}
    confidence_scores = {}
    
    for result in results:
        pred = result['prediction']
        conf = result['confidence']
        
        if pred not in class_votes:
            class_votes[pred] = 0
            confidence_scores[pred] = []
        
        class_votes[pred] += 1
        confidence_scores[pred].append(conf)
    
    # Calcular scores ponderados (votos * confianza promedio)
    weighted_scores = {}
    for class_name, votes in class_votes.items():
        avg_confidence = sum(confidence_scores[class_name]) / len(confidence_scores[class_name])
        weighted_scores[class_name] = votes * (avg_confidence / 100)
    
    # Determinar resultado consensuado
    winning_class = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
    winning_votes = class_votes[winning_class]
    total_images = len(results)
    consensus_confidence = sum(confidence_scores[winning_class]) / len(confidence_scores[winning_class])
    
    # Generar interpretación inteligente
    interpretation = generate_consensus_interpretation(
        winning_class, winning_votes, total_images, consensus_confidence, class_votes
    )
    
    return {
        'final_prediction': winning_class,
        'consensus_confidence': consensus_confidence,
        'votes': f"{winning_votes}/{total_images}",
        'agreement_level': winning_votes / total_images,
        'interpretation': interpretation,
        'class_distribution': class_votes
    }

def generate_consensus_interpretation(winning_class, winning_votes, total_images, confidence, all_votes):
    """Generar interpretación textual del consenso"""
    agreement_pct = (winning_votes / total_images) * 100
    
    if winning_class == "Tizon_tardio":
        if agreement_pct == 100:
            return f"🚨 TIZÓN TARDÍO CONFIRMADO - Todas las imágenes muestran síntomas claros (confianza promedio: {confidence:.1f}%)"
        elif agreement_pct >= 60:
            return f"⚠️ TIZÓN TARDÍO DETECTADO - {winning_votes} de {total_images} imágenes muestran síntomas. Posible infección localizada o en desarrollo."
        else:
            return f"🤔 POSIBLE TIZÓN TARDÍO - Solo {winning_votes} de {total_images} imágenes sugieren infección. Monitorear de cerca."
    
    elif winning_class == "Sana":
        if agreement_pct == 100:
            return f"✅ PLANTA SANA - Todas las imágenes confirman estado saludable (confianza promedio: {confidence:.1f}%)"
        elif agreement_pct >= 60:
            return f"🌱 MAYORMENTE SANA - {winning_votes} de {total_images} imágenes muestran estado saludable. Algunas áreas podrían requerir atención."
        else:
            return f"⚠️ ESTADO MIXTO - Resultados variables. Se recomienda análisis más detallado."
    
    else:  # Otra enfermedad
        if agreement_pct >= 60:
            return f"🔍 OTRA CONDICIÓN DETECTADA - {winning_votes} de {total_images} imágenes sugieren {winning_class}. Sistema especializado en tizón tardío tiene limitaciones para este diagnóstico."
        else:
            return f"❓ DIAGNÓSTICO INCIERTO - Resultados mixtos. Se recomienda consulta con especialista."

def calculate_severity_score(cnn_confidence, consensus_data=None, user_query="", conversation_history=None):
    """
    Calcular score de severidad inteligente (0-100) basado en múltiples factores
    
    Args:
        cnn_confidence: Confianza del CNN (0-100)
        consensus_data: Datos de consenso multi-imagen (opcional)
        user_query: Pregunta del usuario para detectar urgencia
        conversation_history: Historial para detectar progresión
    
    Returns:
        tuple: (nivel_severidad, score, detalles)
    """
    base_score = 0
    score_details = []
    
    # 1. Score base por CNN (0-40 puntos)
    if cnn_confidence >= 90:
        base_score += 40
        score_details.append("CNN muy confiable (+40)")
    elif cnn_confidence >= 80:
        base_score += 35
        score_details.append("CNN confiable (+35)")
    elif cnn_confidence >= 70:
        base_score += 25
        score_details.append("CNN moderada (+25)")
    elif cnn_confidence >= 60:
        base_score += 15
        score_details.append("CNN baja (+15)")
    else:
        base_score += 5
        score_details.append("CNN muy baja (+5)")
    
    # 2. Multiplicador por extensión multi-imagen (0-30 puntos)
    if consensus_data:
        agreement = consensus_data.get('agreement_level', 0)  # 0-1
        extension_score = int(agreement * 30)
        base_score += extension_score
        score_details.append(f"Extensión multi-imagen (+{extension_score})")
    
    # 3. Palabras de urgencia (+20 puntos)
    urgency_words = [
        'urgente', 'rápido', 'extiende', 'muchas plantas', 'todo el cultivo',
        'se mueren', 'perdiendo', 'dañando', 'empeora', 'crítico', 'grave'
    ]
    if user_query and any(word in user_query.lower() for word in urgency_words):
        base_score += 20
        score_details.append("Palabras de urgencia (+20)")
    
    # 4. Progresión temporal si hay historial (+10 puntos)
    if conversation_history and st.session_state.conversation_memory['user_profile']['last_diagnosis']:
        last_diag = st.session_state.conversation_memory['user_profile']['last_diagnosis']
        if last_diag.get('disease') in ['Tizon_tardio', 'Tizón tardío']:
            base_score += 10
            score_details.append("Caso en seguimiento (+10)")
    
    # Clasificación final
    if base_score >= 80:
        nivel = "CRÍTICO"
        color = "🔴"
    elif base_score >= 60:
        nivel = "ALTO"
        color = "🟠"
    elif base_score >= 40:
        nivel = "MODERADO"
        color = "🟡"
    else:
        nivel = "BAJO"
        color = "🟢"
    
    return nivel, base_score, score_details, color

def calculate_economic_impact_cop(severity_level, classification="Tizon_tardio", user_context="general", num_plants=100):
    """
    Calculadora económica en pesos colombianos
    
    Args:
        severity_level: CRÍTICO, ALTO, MODERADO, BAJO
        classification: Tipo de enfermedad detectada
        user_context: campesino, tecnico, urgencia, general
        num_plants: Número de plantas afectadas
    
    Returns:
        dict: Análisis económico completo
    """
    
    # ===== DATOS BASE COLOMBIA 2024 =====
    PRECIO_TOMATE_KG = 4200  # COP por kg (promedio mayorista Bogotá)
    RENDIMIENTO_PLANTA = 8   # kg por planta en condiciones normales
    COSTO_PLANTA = 1200     # Costo de producir una planta
    
    # Precios de fungicidas (COP)
    FUNGICIDAS_PRECIOS = {
        # Tratamientos económicos (campesino)
        "mancozeb_economico": {"precio": 25000, "dosis_total": 0.9, "nombre": "Mancozeb (contacto)"},
        "cobre_economico": {"precio": 18000, "dosis_total": 1.2, "nombre": "Sulfato de Cobre"},
        
        # Tratamientos intermedios (técnico)
        "metalaxil_intermedio": {"precio": 85000, "dosis_total": 0.8, "nombre": "Metalaxil (sistémico)"},
        "dimetomorf_intermedio": {"precio": 120000, "dosis_total": 0.6, "nombre": "Dimetomorf"},
        
        # Tratamientos premium (urgencia/empresarial)
        "mandipropamida_premium": {"precio": 180000, "dosis_total": 0.4, "nombre": "Mandipropamida"},
        "fluopicolide_premium": {"precio": 220000, "dosis_total": 0.3, "nombre": "Fluopicolide"}
    }
    
    # Factores de pérdida por severidad
    PERDIDAS_POR_SEVERIDAD = {
        "CRÍTICO": {"sin_tratamiento": 0.85, "con_tratamiento": 0.15},
        "ALTO": {"sin_tratamiento": 0.60, "con_tratamiento": 0.08},
        "MODERADO": {"sin_tratamiento": 0.35, "con_tratamiento": 0.05},
        "BAJO": {"sin_tratamiento": 0.15, "con_tratamiento": 0.02}
    }
    
    # ===== CÁLCULOS =====
    
    # Valor total del cultivo
    valor_total_cultivo = num_plants * RENDIMIENTO_PLANTA * PRECIO_TOMATE_KG
    
    # Pérdidas potenciales sin tratamiento
    factor_perdida = PERDIDAS_POR_SEVERIDAD[severity_level]["sin_tratamiento"]
    perdida_sin_tratamiento = valor_total_cultivo * factor_perdida
    
    # Pérdidas con tratamiento
    factor_perdida_tratado = PERDIDAS_POR_SEVERIDAD[severity_level]["con_tratamiento"]
    perdida_con_tratamiento = valor_total_cultivo * factor_perdida_tratado
    
    # Seleccionar tratamiento según contexto (manejar None)
    user_context = user_context or "general"  # Fallback si es None
    
    if user_context == "campesino" or (user_context and "barato" in user_context):
        tratamiento_key = "mancozeb_economico"
    elif user_context == "urgencia" or severity_level == "CRÍTICO":
        tratamiento_key = "mandipropamida_premium"
    else:  # técnico o general
        tratamiento_key = "metalaxil_intermedio"
    
    tratamiento_info = FUNGICIDAS_PRECIOS[tratamiento_key]
    costo_tratamiento = tratamiento_info["precio"] * tratamiento_info["dosis_total"]
    
    # Cálculos finales
    ahorro_bruto = perdida_sin_tratamiento - perdida_con_tratamiento
    ahorro_neto = ahorro_bruto - costo_tratamiento
    roi_percentage = (ahorro_neto / costo_tratamiento) * 100 if costo_tratamiento > 0 else 0
    
    # Generar recomendación
    ventana_accion = {
        "CRÍTICO": "12-24 horas",
        "ALTO": "24-48 horas", 
        "MODERADO": "2-5 días",
        "BAJO": "1-2 semanas"
    }
    
    return {
        'valor_cultivo_cop': valor_total_cultivo,
        'perdida_sin_tratamiento_cop': perdida_sin_tratamiento,
        'perdida_con_tratamiento_cop': perdida_con_tratamiento,
        'costo_tratamiento_cop': costo_tratamiento,
        'ahorro_bruto_cop': ahorro_bruto,
        'ahorro_neto_cop': ahorro_neto,
        'roi_percentage': roi_percentage,
        'tratamiento_recomendado': tratamiento_info["nombre"],
        'ventana_accion': ventana_accion[severity_level],
        'num_plantas': num_plants,
        'severity_level': severity_level,
        
        # Formateado para mostrar
        'valor_cultivo_formatted': f"${valor_total_cultivo:,.0f}",
        'perdida_potencial_formatted': f"${perdida_sin_tratamiento:,.0f}",
        'costo_tratamiento_formatted': f"${costo_tratamiento:,.0f}",
        'ahorro_neto_formatted': f"${ahorro_neto:,.0f}",
        'roi_formatted': f"{roi_percentage:,.0f}%"
    }

def save_conversation_to_memory(query: str, response: str, classification: str = None, execution_data: dict = None):
    """Guardar conversación en la memoria del sistema"""
    memory = st.session_state.conversation_memory
    
    # Incrementar contador de sesión
    memory['session_count'] += 1
    
    # Crear registro de conversación
    conversation_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'session_id': memory['session_count'],
        'query': query,
        'response': response,
        'classification': classification,
        'detected_context': execution_data.get('contexto_detectado') if execution_data else None,
        'n_docs_used': len(execution_data.get('documentos_encontrados', [])) if execution_data else 0
    }
    
    # Agregar a historial (mantener últimas 10 conversaciones)
    memory['history'].append(conversation_entry)
    if len(memory['history']) > 10:
        memory['history'].pop(0)
    
    # Actualizar perfil del usuario
    if execution_data:
        context = execution_data.get('contexto_detectado')
        if context:
            memory['user_profile']['detected_context'] = context
            memory['user_profile']['preference_style'] = context
        
        # Guardar último diagnóstico
        if classification:
            memory['user_profile']['last_diagnosis'] = {
                'disease': classification,
                'timestamp': conversation_entry['timestamp'],
                'treatment_recommended': True
            }
            
            # Agregar a casos previos
            case_entry = {
                'disease': classification,
                'timestamp': conversation_entry['timestamp'],
                'context': context,
                'query_type': 'diagnosis'
            }
            memory['user_profile']['previous_cases'].append(case_entry)
            
            # Mantener últimos 5 casos
            if len(memory['user_profile']['previous_cases']) > 5:
                memory['user_profile']['previous_cases'].pop(0)

def detect_follow_up_query(query: str) -> tuple:
    """Detectar si es una pregunta de seguimiento y obtener contexto relevante"""
    follow_up_indicators = [
        'como va', 'cómo va', 'como está', 'cómo está', 'funcionó', 'funciona',
        'resultado', 'mejora', 'empeoró', 'empeoro', 'sigue igual', 'tratamiento',
        'que paso', 'qué pasó', 'después', 'ahora que', 'ya aplique', 'ya apliqué',
        'segunda dosis', 'otra aplicación'
    ]
    
    query_lower = query.lower()
    is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
    
    if is_follow_up and st.session_state.conversation_memory['history']:
        # Buscar la conversación más reciente con diagnóstico
        recent_conversations = st.session_state.conversation_memory['history'][-3:]
        last_diagnosis = None
        last_treatment = None
        
        for conv in reversed(recent_conversations):
            if conv['classification']:
                last_diagnosis = conv['classification']
                last_treatment = conv['response'][:200] + "..."
                break
        
        return is_follow_up, last_diagnosis, last_treatment
    
    return False, None, None

def get_conversation_context() -> str:
    """Obtener contexto de conversaciones previas para incluir en el prompt"""
    memory = st.session_state.conversation_memory
    
    if not memory['history']:
        return ""
    
    # Obtener últimas 2-3 conversaciones relevantes
    recent_history = memory['history'][-3:]
    
    context_parts = []
    context_parts.append("📚 CONTEXTO DE CONVERSACIONES PREVIAS:")
    
    for i, conv in enumerate(recent_history, 1):
        context_parts.append(f"\n{i}. {conv['timestamp']} - Usuario: {conv['query'][:100]}...")
        if conv['classification']:
            context_parts.append(f"   Diagnóstico: {conv['classification']}")
        context_parts.append(f"   Contexto detectado: {conv['detected_context']}")
    
    # Información del perfil del usuario
    profile = memory['user_profile']
    if profile['preference_style']:
        context_parts.append(f"\n👤 PERFIL USUARIO: Estilo preferido: {profile['preference_style']}")
    
    if profile['last_diagnosis']:
        last_diag = profile['last_diagnosis']
        context_parts.append(f"🩺 ÚLTIMO DIAGNÓSTICO: {last_diag['disease']} el {last_diag['timestamp']}")
    
    return "\n".join(context_parts)

def get_langgraph_response(query, classification=None):
    """Obtener respuesta del sistema LangGraph con memoria conversacional"""
    try:
        if st.session_state.langgraph_system:
            # 🧠 NUEVA: Detectar si es pregunta de seguimiento
            is_follow_up, last_diagnosis, last_treatment = detect_follow_up_query(query)
            
            # 🧠 NUEVA: Obtener contexto de conversaciones previas
            conversation_context = get_conversation_context()
            
            # Preparar query enriquecida con contexto
            if is_follow_up and last_diagnosis:
                enhanced_query = f"""
PREGUNTA DE SEGUIMIENTO DETECTADA:
Usuario pregunta: {query}

CONTEXTO DE CASO ANTERIOR:
- Último diagnóstico: {last_diagnosis}
- Tratamiento previo recomendado: {last_treatment}

{conversation_context}

INSTRUCCIÓN ESPECIAL: Esta es una consulta de seguimiento. Haz referencia al caso anterior y proporciona continuidad en el tratamiento.
PREGUNTA ACTUAL: {query}
"""
            else:
                enhanced_query = query
                if conversation_context:
                    enhanced_query = f"{conversation_context}\n\nPREGUNTA ACTUAL: {query}"
            
            # Procesar con LangGraph (usando query enriquecida o original)
            resultado = st.session_state.langgraph_system.procesar_consulta(enhanced_query, classification)
            
            # Guardar el flujo de ejecución
            st.session_state.last_execution_flow = resultado
            
            # 🧠 NUEVA: Guardar conversación en memoria
            response_text = resultado.get('respuesta_final', 'Sin respuesta')
            save_conversation_to_memory(query, response_text, classification, resultado)
            
            # Agregar indicador de seguimiento si aplica
            if is_follow_up:
                response_text = f"📋 **SEGUIMIENTO DE CASO** - {last_diagnosis}\n\n{response_text}"
            
            return response_text, resultado
        return "Sistema LangGraph no disponible", None
    except Exception as e:
        return f"Error en LangGraph: {e}", None

def render_chat_message(role, content, image=None, classification=None, execution_data=None):
    """Renderizar mensaje de chat"""
    if role == "user":
        if image:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="📸 Imagen para análisis", use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class="user-message fade-in">
                    <strong>🧑‍🌾 Usuario:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="user-message fade-in">
                <strong>🧑‍🌾 Usuario:</strong><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
    
    elif role == "assistant":
        # Información de contexto si está disponible
        context_info = ""
        if execution_data:
            contexto = execution_data.get('contexto_detectado', 'general')
            contexto_class = f"context-{contexto}"
            context_info = f"""
            <div style="margin-bottom: 1rem;">
                <span class="context-indicator {contexto_class}">
                    🎯 Contexto: {contexto.title()}
                </span>
            </div>
            """
        
        # Información de clasificación si está disponible
        classification_info = ""
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
            
            classification_info = f"""
            <div style="margin-bottom: 1rem;">
                <span class="status-indicator {status_class}">
                    {icon} Diagnóstico: {classification.replace('_', ' ').title()}
                </span>
            </div>
            """
        
        # Renderizar componentes por separado para evitar problemas de HTML
        
        # Contexto
        if context_info:
            st.markdown(context_info, unsafe_allow_html=True)
        
        # Clasificación  
        if classification_info:
            st.markdown(classification_info, unsafe_allow_html=True)
            
            # Determinar tipo de respuesta
        is_general = execution_data and execution_data.get('is_general_query', False)
        context_type = "Consulta General" if is_general else "Análisis Contextualizado"
        context_icon = "🤖" if is_general else "🔬"
        
        # Mensaje principal del RAG con más prominencia
        st.markdown(f"""
        <div class="assistant-message fade-in">
            <strong>{context_icon} Asistente Agrónomo LangGraph RAG</strong>
            <div style="font-size: 0.85rem; color: #6b7280; margin-top: 0.5rem;">
                ✨ {context_type} - Basado en 472 documentos especializados
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Contenido RAG con estilo destacado y texto visible
        st.markdown(f"""
        <div style="background: #1f2937; color: #f9fafb; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0 8px 8px 0; margin: 1rem 0; line-height: 1.6;">
        {content}
        </div>
        """, unsafe_allow_html=True)

def sidebar_controls():
    """Controles en la sidebar"""
    with st.sidebar:
        st.markdown("### 🎛️ Controles LangGraph")
        

        
        # Upload de múltiples imágenes
        st.markdown("#### 📸 Análisis Multi-Imagen")
        uploaded_files = st.file_uploader(
            "Sube fotos de diferentes partes de tu planta",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Hojas superiores, inferiores, tallos, frutos - máx 5 imágenes"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} imagen(es) seleccionada(s)")
            if len(uploaded_files) > 5:
                st.warning("⚠️ Máximo 5 imágenes. Se procesarán las primeras 5.")
                uploaded_files = uploaded_files[:5]
        
        # Ejemplos de contexto
        st.markdown("#### 💬 Ejemplos por Contexto")
        
        ejemplos = {
            "🚜 Campesino": [
                "¿Qué le echo a mis tomates que sea barato?",
                "Se me están dañando las plantas, mano",
                "Cómo trato esta enfermedad sin químicos"
            ],
            "🔬 Técnico": [
                "Fungicida sistémico contra Phytophthora",
                "Mecanismo de resistencia del patógeno",
                "Ingredientes activos más efectivos"
            ],
            "🚨 Urgencia": [
                "¡URGENTE! Se extiende rápido la enfermedad",
                "Necesito solución inmediata para el tizón",
                "¡Se me está muriendo el cultivo!"
            ],
            "📚 General": [
                "¿Cuál es el mejor tratamiento?",
                "¿Cómo prevenir el tizón tardío?",
                "Manejo integrado de la enfermedad"
            ]
        }
        
        for contexto, preguntas in ejemplos.items():
            with st.expander(contexto):
                for pregunta in preguntas:
                    if st.button(f"📝 {pregunta[:30]}...", key=f"ejemplo_{hash(pregunta)}", use_container_width=True):
                        # Simular input del usuario
                        st.session_state.pending_question = pregunta
        
        # 🎯 DESTACAR VALOR DEL RAG
        st.markdown("#### 🎯 Potencia del RAG")
        if st.session_state.system_ready:
            st.success("✅ LangGraph + CNN Activos")
            st.markdown("""
            **🧠 Base de Conocimiento:**
            - 📚 472 documentos especializados
            - 🔬 Literatura científica 
            - 👨‍🌾 Guías prácticas de campo
            - 💊 Tratamientos verificados
            
            **🎨 Características Únicas:**
            - 🎯 Contexto adaptativo
            - 🧠 Memoria conversacional  
            - 📸 Análisis visual CNN
            - 💰 Calculadora económica opcional
            """)
        else:
            st.warning("⏳ Cargando sistemas...")
        
        # 🧠 Memoria Conversacional
        if st.session_state.conversation_memory['history']:
            st.markdown("#### 🧠 Memoria Conversacional")
            memory = st.session_state.conversation_memory
            
            # Mostrar perfil del usuario
            profile = memory['user_profile']
            if profile['preference_style']:
                st.markdown(f"**Perfil:** {profile['preference_style'].title()}")
            
            # Mostrar último diagnóstico
            if profile['last_diagnosis']:
                last_diag = profile['last_diagnosis']
                st.markdown(f"**Último caso:** {last_diag['disease']}")
                st.caption(f"📅 {last_diag['timestamp']}")
            
            # Mostrar historial reciente (últimas 3 conversaciones)
            with st.expander("💬 Historial Reciente", expanded=False):
                recent_history = memory['history'][-3:]
                for i, conv in enumerate(reversed(recent_history), 1):
                    st.markdown(f"**{i}.** {conv['query'][:50]}...")
                    if conv['classification']:
                        st.caption(f"🔍 {conv['classification']} | {conv['detected_context']}")
                    st.caption(f"🕒 {conv['timestamp']}")
                    if i < len(recent_history):
                        st.divider()
            
            # Botón para limpiar memoria
            if st.button("🧠 Limpiar Memoria", use_container_width=True):
                st.session_state.conversation_memory = {
                    'history': [],
                    'user_profile': {
                        'detected_context': None,
                        'preference_style': None,
                        'previous_cases': [],
                        'last_diagnosis': None,
                        'follow_up_pending': False
                    },
                    'session_count': 0
                }
                st.success("✅ Memoria limpiada")
                st.rerun()
        
        # Estadísticas
        st.markdown("#### 📊 Estadísticas")
        st.metric("Mensajes", len(st.session_state.chat_history))
        st.metric("Conversaciones", st.session_state.conversation_memory['session_count'])
        if st.session_state.last_execution_flow:
            contexto = st.session_state.last_execution_flow.get('contexto_detectado', 'N/A')
            st.metric("Último contexto", contexto.title() if contexto != 'N/A' else 'N/A')
        
        # Limpiar chat
        if st.button("🗑️ Limpiar Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_execution_flow = None
            if "temp_image_display" in st.session_state:
                del st.session_state["temp_image_display"]
            st.rerun()
        
        return uploaded_files

def handle_multiple_images_upload(uploaded_files):
    """Manejar subida de múltiples imágenes con análisis consensuado"""
    if uploaded_files and st.session_state.system_ready:
        # Preparar imágenes
        images_with_labels = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                image = Image.open(uploaded_file)
                label = f"Imagen {i+1}"
                images_with_labels.append((image, label))
            except Exception as e:
                st.error(f"❌ Error procesando {uploaded_file.name}: {e}")
                continue
        
        if not images_with_labels:
            return False
        
        with st.spinner(f"🔍 Analizando {len(images_with_labels)} imágenes..."):
            # Realizar análisis multi-imagen
            multi_analysis = analyze_multiple_images(images_with_labels)
            
            if multi_analysis:
                # Guardar resultado completo para visualización
                st.session_state.temp_multi_image_display = {
                    "images_with_labels": images_with_labels,
                    "analysis": multi_analysis,
                    "upload_timestamp": time.time()
                }
                return True
            else:
                st.error("❌ No se pudo procesar ninguna imagen correctamente.")
    return False

def handle_image_upload(uploaded_files):
    """Función de compatibilidad - ahora maneja múltiples imágenes"""
    if isinstance(uploaded_files, list):
        return handle_multiple_images_upload(uploaded_files)
    elif uploaded_files:
        # Convertir archivo único a lista para compatibilidad
        return handle_multiple_images_upload([uploaded_files])
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
    
    # Layout principal con columnas
    col1, col2 = st.columns([3, 1])
    
    # Header en la columna principal
    with col1:
        create_header()
    
    with col2:
        # Sidebar en columna derecha
        uploaded_files = sidebar_controls()
    
    with col1:
        # Manejar upload de múltiples imágenes
        if uploaded_files:
            handle_image_upload(uploaded_files)
        

        
        # Contenedor de imagen temporal
        if "temp_image_display" in st.session_state:
            with st.container():
                st.markdown("### 📸 Resultado del Análisis")
                col_img, col_result = st.columns([1, 2])
                with col_img:
                    st.image(st.session_state.temp_image_display["image"], 
                            caption="Imagen analizada", use_container_width=True)
                with col_result:
                    pred = st.session_state.temp_image_display['prediction']
                    conf = st.session_state.temp_image_display['confidence']
                    
                    if pred == "Tizon_tardio":
                        st.error(f"🍄 **TIZÓN TARDÍO** (Confianza: {conf:.1f}%)")
                    elif pred == "Sana":
                        st.success(f"🌱 **PLANTA SANA** (Confianza: {conf:.1f}%)")
                    else:
                        st.warning(f"⚠️ **OTRA ENFERMEDAD** (Confianza: {conf:.1f}%)")
                    
                    if st.button("❌ Ocultar", key="hide_temp"):
                        del st.session_state.temp_image_display
                        st.rerun()
            st.markdown("---")
        
        # 🆕 Contenedor para análisis multi-imagen
        if "temp_multi_image_display" in st.session_state:
            multi_data = st.session_state.temp_multi_image_display
            analysis = multi_data['analysis']
            images_with_labels = multi_data['images_with_labels']
            
            with st.container():
                st.markdown("### 🖼️ Análisis Multi-Imagen")
                
                # Mostrar resultado consensuado prominente
                consensus = analysis['consensus']
                st.markdown(f"#### 🎯 Diagnóstico Consensuado")
                
                # Color según el resultado
                if "TIZÓN TARDÍO" in consensus['interpretation']:
                    st.error(f"**{consensus['final_prediction']}** ({consensus['votes']} imágenes)")
                    st.error(consensus['interpretation'])
                elif "SANA" in consensus['interpretation']:
                    st.success(f"**{consensus['final_prediction']}** ({consensus['votes']} imágenes)")
                    st.success(consensus['interpretation'])
                else:
                    st.warning(f"**{consensus['final_prediction']}** ({consensus['votes']} imágenes)")
                    st.warning(consensus['interpretation'])
                
                # 🆕 MOSTRAR ANÁLISIS DE SEVERIDAD (SUTIL, NO DOMINANTE)
                if analysis.get('severity_analysis'):
                    severity = analysis['severity_analysis']
                    
                    # Barra de severidad más discreta
                    col_info, col_calc = st.columns([2, 1])
                    
                    with col_info:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ef4444, #dc2626); color: white; padding: 1rem 1.5rem; border-radius: 8px; margin: 0.5rem 0;">
                            <span style="font-size: 0.9rem; font-weight: 600;">🚨 Severidad: {severity['level']} (Score: {severity['score']}/100)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_calc:
                        # Botón para abrir calculadora económica (OPCIONAL)
                        if st.button("💰 Calculadora Económica", use_container_width=True, key="open_calc_multi"):
                            st.session_state.show_economic_calculator = True
                    
                    # Mostrar calculadora solo si se solicita explícitamente
                    if st.session_state.get('show_economic_calculator', False):
                        st.markdown("---")
                        st.markdown("#### 💰 Calculadora de Impacto Económico Personalizada")
                        st.info("📋 **Completa la información para calcular el impacto económico para tu caso específico:**")
                        
                        with st.form(key="economic_calculator_form"):
                            col_form1, col_form2 = st.columns(2)
                            
                            with col_form1:
                                num_plants = st.number_input(
                                    "🌱 Número total de plantas en tu cultivo:",
                                    min_value=1,
                                    max_value=10000,
                                    value=100,
                                    step=10,
                                    help="Ingresa el número total de plantas que tienes"
                                )
                                
                                area_cultivo = st.number_input(
                                    "� Área del cultivo (m²):",
                                    min_value=1.0,
                                    max_value=100000.0,
                                    value=1000.0,
                                    step=50.0,
                                    help="Área total del cultivo en metros cuadrados"
                                )
                            
                            with col_form2:
                                affected_percentage = st.slider(
                                    "🎯 ¿Qué porcentaje del cultivo está afectado?",
                                    min_value=5,
                                    max_value=100,
                                    value=30,
                                    step=5,
                                    help="Estima qué porcentaje de tus plantas muestran síntomas"
                                )
                                
                                user_budget = st.selectbox(
                                    "💵 ¿Cuál es tu presupuesto para tratamiento?",
                                    [
                                        "Económico (hasta $50.000 COP)",
                                        "Intermedio ($50.000 - $150.000 COP)", 
                                        "Sin restricción (lo que sea necesario)"
                                    ],
                                    help="Selecciona tu rango de presupuesto disponible"
                                )
                            
                            calcular_economics = st.form_submit_button(
                                "🧮 Calcular Impacto Económico", 
                                use_container_width=True
                            )
                        
                        # Mostrar cálculo si se presiona el botón
                        if calcular_economics:
                            # Determinar contexto según presupuesto
                            if "Económico" in user_budget:
                                budget_context = "campesino"
                            elif "Sin restricción" in user_budget:
                                budget_context = "urgencia"
                            else:
                                budget_context = "tecnico"
                            
                            # Calcular plantas afectadas
                            plants_affected = int(num_plants * (affected_percentage / 100))
                            
                            # Calcular impacto económico personalizado
                            economics = calculate_economic_impact_cop(
                                severity_level=severity['level'],
                                classification="Tizon_tardio",
                                user_context=budget_context,
                                num_plants=plants_affected
                            )
                            
                            # Guardar en session_state para persistir
                            st.session_state.calculated_economics = {
                                'economics': economics,
                                'user_inputs': {
                                    'num_plants': num_plants,
                                    'plants_affected': plants_affected,
                                    'area_cultivo': area_cultivo,
                                    'affected_percentage': affected_percentage,
                                    'budget_context': budget_context
                                }
                            }
                        
                        # Mostrar resultados si existen
                        if hasattr(st.session_state, 'calculated_economics'):
                            economics = st.session_state.calculated_economics['economics']
                            inputs = st.session_state.calculated_economics['user_inputs']
                            
                            st.markdown("---")
                            st.markdown("#### 📈 Resultados del Análisis Económico")
                            
                            # Layout en 3 columnas para métricas principales
                            col_econ1, col_econ2, col_econ3 = st.columns(3)
                            
                            with col_econ1:
                                perdida_formatted = economics['perdida_potencial_formatted']
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
                                    <h3 style="margin: 0; font-size: 1.1rem;">💰 PÉRDIDA POTENCIAL</h3>
                                    <p style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{perdida_formatted}</p>
                                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">COP sin tratamiento</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_econ2:
                                costo_formatted = economics['costo_tratamiento_formatted']
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
                                    <h3 style="margin: 0; font-size: 1.1rem;">🧪 COSTO TRATAMIENTO</h3>
                                    <p style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{costo_formatted}</p>
                                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">COP inversión</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_econ3:
                                roi_formatted = economics['roi_formatted']
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
                                    <h3 style="margin: 0; font-size: 1.1rem;">📈 ROI TRATAMIENTO</h3>
                                    <p style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{roi_formatted}</p>
                                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Retorno inversión</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Detalles económicos personalizados
                            st.markdown("##### 💼 Tu Análisis Personalizado")
                            col_det1, col_det2 = st.columns(2)
                            
                            with col_det1:
                                st.markdown(f"""
                                **🌱 Información del Cultivo:**
                                - Plantas totales: `{inputs['num_plants']:,}`
                                - Plantas afectadas: `{inputs['plants_affected']:,}` ({inputs['affected_percentage']}%)
                                - Área cultivada: `{inputs['area_cultivo']:,.0f} m²`
                                - Densidad: `{inputs['num_plants']/inputs['area_cultivo']:.1f} plantas/m²`
                                """)
                            
                            with col_det2:
                                st.markdown(f"""
                                **💵 Análisis Financiero:**
                                - Tratamiento recomendado: `{economics['tratamiento_recomendado']}`
                                - Costo por planta: `${economics['costo_tratamiento_cop']/inputs['plants_affected']:,.0f} COP`
                                - **Ahorro neto: `{economics['ahorro_neto_formatted']} COP`**
                                - Ventana de acción: `{economics['ventana_accion']}`
                                """)
                            
                            # Alerta según severidad y ROI
                            if severity['level'] == "CRÍTICO" and economics['roi_percentage'] > 1000:
                                st.error(f"""
                                🚨 **ACCIÓN INMEDIATA CRÍTICA** 
                                
                                Con {inputs['plants_affected']:,} plantas afectadas, cada hora cuenta. 
                                ROI de {roi_formatted} justifica tratamiento URGENTE.
                                Aplique dentro de {economics['ventana_accion']}.
                                """)
                            elif economics['roi_percentage'] > 500:
                                st.warning(f"""
                                ⚠️ **TRATAMIENTO ALTAMENTE RENTABLE**
                                
                                ROI de {roi_formatted} indica excelente oportunidad. 
                                Cada peso invertido genera ${economics['roi_percentage']/100:.0f} de retorno.
                                """)
                            else:
                                st.info(f"""
                                ℹ️ **TRATAMIENTO RECOMENDADO**
                                
                                Análisis personalizado para {inputs['plants_affected']:,} plantas afectadas.
                                ROI: {roi_formatted} - Inversión justificada.
                                """)
                            
                            # Botones de acción
                            col_recalc, col_close = st.columns(2)
                            with col_recalc:
                                if st.button("🔄 Recalcular", key="recalc_econ", use_container_width=True):
                                    if hasattr(st.session_state, 'calculated_economics'):
                                        del st.session_state.calculated_economics
                                    st.rerun()
                            with col_close:
                                if st.button("❌ Cerrar Calculadora", key="close_calc", use_container_width=True):
                                    st.session_state.show_economic_calculator = False
                                    if hasattr(st.session_state, 'calculated_economics'):
                                        del st.session_state.calculated_economics
                                    st.rerun()
                    
                    # Mostrar factores de severidad de forma compacta
                    with st.expander("🔍 Ver factores de severidad detectados"):
                        for detail in severity['details']:
                            st.write(f"• {detail}")
                    
                    st.markdown("---")
                
                # Grid de imágenes individuales
                st.markdown("#### 🔍 Análisis Individual")
                
                # Crear columnas dinámicas según número de imágenes
                num_images = len(images_with_labels)
                if num_images <= 3:
                    cols = st.columns(num_images)
                else:
                    # Para más de 3 imágenes, usar filas de 3
                    for row in range(0, num_images, 3):
                        end_idx = min(row + 3, num_images)
                        cols = st.columns(end_idx - row)
                        
                        for i, col in enumerate(cols):
                            img_idx = row + i
                            if img_idx < len(analysis['individual_results']):
                                result = analysis['individual_results'][img_idx]
                                image, label = images_with_labels[img_idx]
                                
                                with col:
                                    st.image(image, caption=label, use_container_width=True)
                                    
                                    pred = result['prediction']
                                    conf = result['confidence']
                                    
                                    if pred == "Tizon_tardio":
                                        st.error(f"🍄 {pred}")
                                        st.caption(f"Confianza: {conf:.1f}%")
                                    elif pred == "Sana":
                                        st.success(f"🌱 {pred}")
                                        st.caption(f"Confianza: {conf:.1f}%")
                                    else:
                                        st.warning(f"⚠️ {pred}")
                                        st.caption(f"Confianza: {conf:.1f}%")
                
                # Mostrar si es primera fila (<=3 imágenes)
                if num_images <= 3:
                    for i, col in enumerate(cols):
                        if i < len(analysis['individual_results']):
                            result = analysis['individual_results'][i]
                            image, label = images_with_labels[i]
                            
                            with col:
                                st.image(image, caption=label, use_container_width=True)
                                
                                pred = result['prediction']
                                conf = result['confidence']
                                
                                if pred == "Tizon_tardio":
                                    st.error(f"🍄 {pred}")
                                    st.caption(f"Confianza: {conf:.1f}%")
                                elif pred == "Sana":
                                    st.success(f"🌱 {pred}")
                                    st.caption(f"Confianza: {conf:.1f}%")
                                else:
                                    st.warning(f"⚠️ {pred}")
                                    st.caption(f"Confianza: {conf:.1f}%")
                
                # Botón para ocultar
                if st.button("❌ Ocultar Análisis", key="hide_multi_temp"):
                    del st.session_state.temp_multi_image_display
                    st.rerun()
            
            st.markdown("---")
        
        # Mostrar historial de chat
        if st.session_state.chat_history:
            st.markdown("### 💬 Conversación")
            for message in st.session_state.chat_history:
                render_chat_message(
                    message["role"],
                    message["content"],
                    message.get("image"),
                    message.get("classification"),
                    message.get("execution_data")
                )
        else:
            # Mensaje de bienvenida enfocado en RAG
            st.markdown("""
            <div class="assistant-message fade-in">
                <strong>🤖 Asistente Agrónomo Especializado - LangGraph RAG</strong><br><br>
                ¡Hola! Soy tu asistente especializado en el manejo del tizón tardío en tomate. 
                Mi conocimiento está basado en <strong>472 documentos científicos y técnicos</strong> 
                especializados en fitopatología.<br><br>
            </div>
            """, unsafe_allow_html=True)
            
            # Características enfocadas en RAG
            st.markdown("""
            ### 🎯 **¿Cómo puedo ayudarte?**
            
            **🔬 Consultas Técnicas Especializadas:**
            - Diagnóstico diferencial de enfermedades
            - Ingredientes activos más efectivos  
            - Resistencia de patógenos
            - Manejo integrado de enfermedades
            
          
            **🚀 Funciones Avanzadas:**
            - 📸 **Análisis visual**: Sube fotos para diagnóstico CNN
            - 🧠 **Memoria conversacional**: Recuerdo casos anteriores  
            - 💰 **Calculadora económica**: Análisis costo-beneficio (opcional)
            - 🎯 **Respuestas contextuales**: Adaptadas a tu perfil
            
            ---
            
            ### � **Ejemplos de preguntas que puedo responder:**
            """)
            
            # Ejemplos específicos de RAG
            col_ej1, col_ej2 = st.columns(2)
            
            with col_ej1:
                st.markdown("""
                **🔬 Técnicas:**
                - "¿Cuál es el mejor sistémico contra Phytophthora?"
                - "¿Cómo funciona el Metalaxil en la planta?"
                - "¿Qué resistencias tiene P. infestans?"
                """)
            
            with col_ej2:
                st.markdown("""
                **👨‍🌾 Prácticas:**
                - "¿Qué le echo a mis tomates que sea barato?"
                - "¿Cada cuánto aplico el fungicida?"
                - "¿Cómo prevenir el tizón en invierno?"
                """)
            
            st.markdown("### 📸 **¡Sube una imagen para empezar el análisis!**")
        
        # Input de texto
        with st.container():
            # Verificar si hay imagen cargada (individual o multi-imagen)
            has_single_image = "temp_image_display" in st.session_state
            has_multi_image = "temp_multi_image_display" in st.session_state
            image_required = not (has_single_image or has_multi_image)
            
            if image_required:
                col_warning, col_bypass = st.columns([3, 1])
                
                with col_warning:
                    st.markdown("""
                    <div style="background: #fef3c7; border: 2px solid #f59e0b; color: #7c2d12; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <strong>💡 Recomendado: Sube imagen para contexto</strong><br>
                        Para respuestas más precisas, sube imágenes de tu planta.
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_bypass:
                    if st.button("🤖 Consulta General", help="Hacer pregunta sin imagen", use_container_width=True):
                        st.session_state.allow_general_questions = True
                        st.rerun()
            
            # Permitir preguntas generales si se autoriza
            if st.session_state.get('allow_general_questions', False) and image_required:
                st.info("💬 **Modo consulta general activado** - Respuestas basadas solo en RAG sin contexto visual")
                image_required = False  # Desbloquear input
            
            with st.form(key="chat_form", clear_on_submit=True):
                col_input, col_button = st.columns([4, 1])
                
                with col_input:
                    user_input = st.text_input(
                        "💬 Escribe tu pregunta...",
                        placeholder="Primero sube una imagen..." if image_required else "Ej: ¿Qué fungicida barato me recomiendas? (contexto campesino)",
                        label_visibility="collapsed",
                        key="user_input",
                        value=st.session_state.get("pending_question", "") if not image_required else "",
                        disabled=image_required
                    )
                
                with col_button:
                    send_button = st.form_submit_button("🚀 Procesar", use_container_width=True, disabled=image_required)
        
        # Limpiar pregunta pendiente
        if "pending_question" in st.session_state:
            del st.session_state.pending_question
        
        # Procesar input (si hay imagen o se permite consulta general)
        can_process = (has_single_image or has_multi_image or 
                      st.session_state.get('allow_general_questions', False))
        
        if (send_button and user_input and user_input.strip() and 
            st.session_state.system_ready and can_process):
            
            # 🆕 Guardar query actual para análisis de severidad
            st.session_state.current_user_query = user_input.strip()
            
            # Agregar mensaje del usuario
            user_message = {
                "role": "user",
                "content": user_input.strip(),
                "timestamp": time.time()
            }
            st.session_state.chat_history.append(user_message)
            
            # Determinar clasificación de imagen si existe
            classification = None
            image_context = ""
            is_general_query = not (has_single_image or has_multi_image)
            
            if is_general_query:
                # Consulta general sin imagen
                image_context = "CONSULTA GENERAL - Sin análisis visual disponible. Respuesta basada únicamente en base de conocimientos RAG."
            elif has_multi_image:
                # Priorizar análisis multi-imagen si está disponible
                multi_data = st.session_state.temp_multi_image_display
                consensus = multi_data['analysis']['consensus']
                classification = consensus['final_prediction']
                
                # Crear contexto enriquecido para LangGraph
                image_context = f"""
ANÁLISIS MULTI-IMAGEN DISPONIBLE:
- Diagnóstico consensuado: {consensus['final_prediction']}
- Acuerdo: {consensus['votes']} de {multi_data['analysis']['total_images']} imágenes
- Confianza promedio: {consensus['consensus_confidence']:.1f}%
- Interpretación: {consensus['interpretation']}
- Distribución: {consensus['class_distribution']}
"""
            elif has_single_image:
                # Imagen individual
                classification = st.session_state.temp_image_display["prediction"]
                confidence = st.session_state.temp_image_display["confidence"]
                image_context = f"Imagen única analizada: {classification} (confianza: {confidence:.1f}%)"
            
            # Generar respuesta LangGraph con contexto de imagen
            enhanced_query = user_input.strip()
            if image_context:
                enhanced_query = f"{user_input.strip()}\n\n{image_context}"
            
            with st.spinner("🔗 Ejecutando flujo LangGraph..."):
                response, execution_data = get_langgraph_response(enhanced_query, classification)
            
            # Agregar respuesta del asistente con contexto
            assistant_message = {
                "role": "assistant",
                "content": response,
                "timestamp": time.time(),
                "classification": classification,
                "execution_data": execution_data,
                "is_general_query": is_general_query  # 🆕 Marcar si es consulta general
            }
            st.session_state.chat_history.append(assistant_message)
            
            # Rerun para mostrar
            st.rerun()

if __name__ == "__main__":
    main()