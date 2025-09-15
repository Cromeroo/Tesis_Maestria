#!/usr/bin/env python3
"""
Interfaz web CONVERSACIONAL que REALMENTE usa el LLM
1. Ejecuta el sistema LangGraph completo (CNN → RAG → LLM)
2. Mantiene contexto de conversación
3. Genera respuestas inteligentes personalizadas
"""

import streamlit as st
import tempfile
import os
from datetime import datetime
from PIL import Image
import sys
from pathlib import Path

# Configurar path
sys.path.append(str(Path(__file__).parent))

def initialize_session_state():
    """Inicializar estado de sesión"""
    if 'graph_system' not in st.session_state:
        st.session_state.graph_system = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_classification' not in st.session_state:
        st.session_state.current_classification = None
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None

def initialize_system():
    """Inicializar sistema LangGraph"""
    if st.session_state.graph_system is None:
        try:
            from langgraph_system_simple import TomatoDiseaseGraph
            st.session_state.graph_system = TomatoDiseaseGraph()
            return True
        except Exception as e:
            st.error(f"Error inicializando sistema: {e}")
            return False
    return True

def build_conversation_context():
    """Construir contexto conversacional"""
    if not st.session_state.conversation_history:
        return ""
    
    recent_history = st.session_state.conversation_history[-4:]  # Últimas 4 interacciones
    context_parts = ["HISTORIAL PREVIO:"]
    
    for msg in recent_history:
        if msg['role'] == 'user':
            context_parts.append(f"USUARIO: {msg['content']}")
        else:
            summary = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
            context_parts.append(f"ASISTENTE: {summary}")
    
    context_parts.append("NUEVA CONSULTA:")
    return "\n".join(context_parts)

def process_with_full_langgraph(user_question, image_path, classification):
    """Procesar usando el sistema LangGraph COMPLETO"""
    try:
        # Construir contexto conversacional
        conversation_context = build_conversation_context()
        
        # Crear prompt que incluye historial + pregunta actual
        full_prompt = f"""
{conversation_context}

PREGUNTA ACTUAL: {user_question}

INSTRUCCIONES:
- Considera el historial de conversación previo
- Responde específicamente a la pregunta actual
- Si es sobre causas, explica factores específicos
- Si es sobre tratamiento, da recomendaciones concretas
- Si es sobre prevención, enfócate en medidas futuras
- Usa la información técnica de los documentos RAG
- Responde de forma conversacional y coherente
"""
        
        # Estado para LangGraph
        state = {
            "image_path": image_path,
            "user_text": full_prompt,
            "disease_prediction": classification,
            "rag_context": None,
            "final_response": None,
            "error": None
        }
        
        print(f"🚀 Ejecutando LangGraph completo...")
        print(f"📝 Pregunta: {user_question}")
        print(f"🧠 Contexto conversacional: {len(conversation_context)} chars")
        
        # Ejecutar el grafo COMPLETO
        result = st.session_state.graph_system.graph.invoke(state)
        
        print(f"📊 Resultado keys: {list(result.keys())}")
        print(f"🤖 Final response length: {len(result.get('final_response', ''))}")
        print(f"📚 RAG context length: {len(result.get('rag_context', ''))}")
        
        # Verificar si el LLM generó respuesta válida
        llm_response = result.get('final_response', '').strip()
        rag_context = result.get('rag_context', '')
        
        if llm_response and len(llm_response) > 100:
            print("✅ LLM generó respuesta exitosamente")
            return {
                'response': llm_response,
                'rag_context': rag_context,
                'response_type': '🤖 RESPUESTA INTELIGENTE (LLM + RAG)',
                'llm_used': True,
                'success': True
            }
        else:
            print("⚠️ LLM no generó respuesta válida, creando respuesta analítica")
            return create_analytical_response(user_question, classification, rag_context)
            
    except Exception as e:
        print(f"❌ Error en LangGraph: {e}")
        return {
            'response': f"Error procesando consulta: {str(e)}",
            'rag_context': '',
            'response_type': '❌ ERROR DEL SISTEMA',
            'llm_used': False,
            'success': False
        }

def create_analytical_response(user_question, classification, rag_context):
    """Crear respuesta analítica cuando LLM falla"""
    
    disease_name = classification.get('class_name', 'Enfermedad detectada')
    confidence = classification.get('confidence', 0)
    question_lower = user_question.lower()
    
    # Analizar tipo de pregunta
    if any(word in question_lower for word in ['causa', 'por que', 'porque', 'razon', 'origen']):
        focus = "📋 ANÁLISIS DE CAUSAS"
        answer = f"""
**¿Por qué apareció {disease_name} en tu cultivo?**

**FACTORES CAUSALES PRINCIPALES:**
• **Condiciones ambientales**: Humedad >80%, temperatura 15-25°C
• **Presencia del patógeno**: Phytophthora infestans en el ambiente
• **Susceptibilidad**: Variedad sin resistencia + manejo inadecuado
• **Vías de infección**: Esporas por viento, agua, herramientas

**CONDICIONES ESPECÍFICAS QUE LO FAVORECIERON:**
• Riego por aspersión o humedad foliar prolongada
• Densidad alta con poca ventilación
• Rastrojos infectados de cultivos anteriores
• Herramientas sin desinfectar entre plantas
"""
        
    elif any(word in question_lower for word in ['tratamiento', 'control', 'combatir', 'curar']):
        focus = "💊 PLAN DE TRATAMIENTO"
        answer = f"""
**Plan de tratamiento para {disease_name}:**

**CONTROL QUÍMICO INMEDIATO:**
• **Sistémicos**: Ridomil Gold MZ (metalaxil + mancozeb) - 2.5 kg/ha
• **Contacto**: Mancozeb 80% WP - 2-2.5 kg/ha cada 7 días
• **Alternativa**: Curzate M8 (cymoxanil + mancozeb) - 2 kg/ha

**CONTROL CULTURAL SIMULTÁNEO:**
• Eliminar plantas infectadas inmediatamente
• Mejorar ventilación y espaciamiento
• Cambiar a riego por goteo exclusivamente
• Poda sanitaria de hojas basales

**MONITOREO CONTINUO:**
• Inspección diaria en condiciones húmedas
• Aplicaciones preventivas antes de lluvia
"""
        
    elif any(word in question_lower for word in ['prevenir', 'evitar', 'futuro', 'proximo']):
        focus = "🛡️ PREVENCIÓN FUTURA"
        answer = f"""
**Cómo prevenir {disease_name} en futuros cultivos:**

**VARIEDADES RESISTENTES:**
• Mountain Fresh Plus, Phoenix, Celebrity
• Iron Lady, Defiant PhR, Mountain Pride

**MANEJO PREVENTIVO:**
• Rotación con no-solanáceas por 2-3 años
• Espaciamiento adecuado (50-60 cm entre plantas)
• Riego localizado desde el inicio
• Monitoreo climático: aplicar preventivos cuando HR >80%

**PROGRAMA DE APLICACIONES:**
• Semanas 1-4: Cobre preventivo cada 10 días
• Semanas 5-8: Alternancia cobre + mancozeb
• Vigilancia constante en condiciones húmedas
"""
        
    elif any(word in question_lower for word in ['urgente', 'rapido', 'inmediato', 'emergencia']):
        focus = "🚨 PROTOCOLO DE EMERGENCIA"
        answer = f"""
**Acciones URGENTES para {disease_name}:**

**PRÓXIMAS 24 HORAS:**
• Aplicación inmediata de Ridomil Gold MZ (2.5 kg/ha)
• Eliminar plantas muy infectadas y destruir
• Suspender todo riego foliar/aspersión

**48-72 HORAS:**
• Segunda aplicación: Curzate M8 (2 kg/ha)
• Poda sanitaria intensiva
• Maximizar ventilación en invernaderos

**SEGUIMIENTO SEMANAL:**
• Monitoreo diario al amanecer
• Aplicaciones cada 5-7 días según humedad
• Vigilar nuevas lesiones constantemente

**CRÍTICO**: Actuar SIN demora - la enfermedad puede destruir el cultivo en 10-14 días.
"""
        
    else:
        focus = "📖 INFORMACIÓN GENERAL"
        answer = f"""
**Información sobre {disease_name}:**

**DIAGNÓSTICO CONFIRMADO:**
• Confianza: {confidence:.1%}
• Patógeno: Phytophthora infestans

**MANEJO INTEGRAL RECOMENDADO:**
• Control químico: Alternar sistémicos y contacto
• Control cultural: Ventilación, riego localizado
• Monitoreo: Inspección diaria en húmedo
• Prevención: Variedades resistentes, rotación

**TU CONSULTA:** "{user_question}"
Basándome en tu pregunta y el diagnóstico confirmado, estas son las recomendaciones más apropiadas.
"""
    
    # Agregar información del RAG si está disponible
    if rag_context and len(rag_context.strip()) > 100:
        rag_excerpt = extract_relevant_rag_info(rag_context, question_lower)
        if rag_excerpt:
            answer += f"\n\n**📚 INFORMACIÓN DE DOCUMENTOS TÉCNICOS:**\n{rag_excerpt}"
    
    return {
        'response': answer,
        'rag_context': rag_context,
        'response_type': f'{focus} (Análisis + RAG)',
        'llm_used': False,
        'success': True
    }

def extract_relevant_rag_info(rag_context, question_lower):
    """Extraer información relevante del RAG según la pregunta"""
    try:
        lines = rag_context.split('\n')
        relevant_lines = []
        
        # Keywords según tipo de pregunta
        if any(word in question_lower for word in ['causa', 'por que', 'origen']):
            keywords = ['condiciones', 'favorece', 'causa', 'propicia', 'factor']
        elif any(word in question_lower for word in ['tratamiento', 'control']):
            keywords = ['tratamiento', 'control', 'fungicida', 'aplicar', 'producto']
        elif any(word in question_lower for word in ['prevenir', 'evitar']):
            keywords = ['prevenir', 'evitar', 'resistente', 'rotación', 'preventivo']
        else:
            keywords = ['tizón', 'phytophthora', 'manejo', 'cultivo']
        
        # Filtrar líneas relevantes
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords) and len(line.strip()) > 20:
                relevant_lines.append(line.strip())
                if len(relevant_lines) >= 3:  # Máximo 3 líneas
                    break
        
        return '\n'.join(relevant_lines) if relevant_lines else ""
        
    except Exception:
        return ""

def main():
    st.set_page_config(
        page_title="Sistema RAG Conversacional - Tizón Tardío",
        page_icon="🍅",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🍅 Sistema RAG Conversacional - Especialista en Tizón Tardío")
    st.markdown("**Sistema inteligente que mantiene contexto y usa LLM para respuestas personalizadas**")
    
    # Layout principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Análisis de Imagen")
        
        uploaded_file = st.file_uploader(
            "Sube una imagen de la planta afectada",
            type=['png', 'jpg', 'jpeg'],
            help="Imagen clara de hojas con síntomas"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen para análisis", use_container_width=True)
            
            # Guardar temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image.save(tmp_file.name)
                st.session_state.current_image_path = tmp_file.name
            
            if st.button("🔬 Analizar Imagen", type="primary"):
                if not initialize_system():
                    return
                
                with st.spinner("🔄 Procesando imagen..."):
                    try:
                        from components.image_classifier import TomatoImageClassifier
                        classifier = TomatoImageClassifier()
                        result = classifier.classify_image(st.session_state.current_image_path)
                        
                        st.session_state.current_classification = result
                        
                        st.success("✅ Análisis completado")
                        st.info(f"**Diagnóstico:** {result['class_name']}")
                        st.info(f"**Confianza:** {result['confidence']:.1%}")
                        
                        if result['class_name'].lower() == 'tizon_tardio':
                            st.warning("⚠️ Tizón tardío detectado - Requiere acción inmediata")
                        
                    except Exception as e:
                        st.error(f"Error en análisis: {e}")
    
    with col2:
        st.header("💬 Consulta Especializada")
        
        # Mostrar historial
        if st.session_state.conversation_history:
            st.subheader("📋 Conversación")
            
            # Contenedor con scroll para historial
            history_container = st.container()
            with history_container:
                for i, msg in enumerate(st.session_state.conversation_history[-8:]):  # Últimas 8 interacciones
                    if msg['role'] == 'user':
                        st.markdown(f"**👤 Tu:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 Especialista:** {msg.get('response_type', 'Respuesta')}")
                        with st.expander("Ver respuesta", expanded=(i == len(st.session_state.conversation_history[-8:]) - 1)):
                            st.markdown(msg['content'])
                    st.markdown("---")
        
        # Nueva consulta
        st.subheader("❓ Nueva Consulta")
        
        user_question = st.text_area(
            "Escribe tu pregunta:",
            placeholder="Ejemplo: ¿Por qué apareció el tizón tardío en mi cultivo?",
            key="user_input",
            height=100,
            help="El sistema recordará el contexto de preguntas anteriores"
        )
        
        # Ejemplos de preguntas
        examples = [
            "¿Por qué apareció esta enfermedad?",
            "¿Qué tratamiento específico me recomiendas?",
            "¿Cómo prevenir esto en futuros cultivos?",
            "¿Qué tan urgente es actuar?",
            "¿Qué productos debo aplicar y cada cuánto?",
            "¿Hay alternativas orgánicas?",
            "¿Cuánto tiempo tomará controlar la enfermedad?"
        ]
        
        col_send, col_example = st.columns([1, 2])
        
        with col_send:
            send_button = st.button("📤 Consultar", type="primary")
        
        with col_example:
            example_selected = st.selectbox("O elige un ejemplo:", [""] + examples)
        
        # Procesar consulta
        question_to_process = None
        if send_button and user_question.strip():
            question_to_process = user_question.strip()
        elif example_selected:
            question_to_process = example_selected
        
        if question_to_process:
            if not st.session_state.current_classification:
                st.warning("⚠️ Primero analiza una imagen para obtener el diagnóstico")
            else:
                if not initialize_system():
                    return
                
                with st.spinner("🤔 Generando respuesta especializada..."):
                    # Procesar con LangGraph completo
                    result = process_with_full_langgraph(
                        question_to_process,
                        st.session_state.current_image_path,
                        st.session_state.current_classification
                    )
                    
                    # Agregar al historial
                    timestamp = datetime.now().strftime("%H:%M")
                    
                    st.session_state.conversation_history.append({
                        'role': 'user',
                        'content': question_to_process,
                        'timestamp': timestamp
                    })
                    
                    st.session_state.conversation_history.append({
                        'role': 'assistant',
                        'content': result['response'],
                        'response_type': result['response_type'],
                        'llm_used': result['llm_used'],
                        'timestamp': timestamp
                    })
                    
                    # Mostrar resultado inmediato
                    if result['success']:
                        st.success("✅ Respuesta generada")
                        with st.expander("Ver respuesta completa", expanded=True):
                            st.markdown(result['response'])
                    else:
                        st.error("❌ Error generando respuesta")
                    
                    # Recargar para mostrar en historial
                    st.rerun()
    
    # Panel lateral con información
    with st.sidebar:
        st.header("📊 Estado del Sistema")
        
        if st.session_state.current_classification:
            st.success("✅ Imagen analizada")
            disease = st.session_state.current_classification['class_name']
            confidence = st.session_state.current_classification['confidence']
            st.metric("Diagnóstico", disease)
            st.metric("Confianza", f"{confidence:.1%}")
        
        num_questions = len([msg for msg in st.session_state.conversation_history if msg['role'] == 'user'])
        if num_questions > 0:
            st.success(f"💬 {num_questions} consultas realizadas")
            
            # Mostrar tipos de respuestas
            response_types = set()
            for msg in st.session_state.conversation_history:
                if msg['role'] == 'assistant':
                    resp_type = msg.get('response_type', 'Sin tipo').split('(')[0].strip()
                    response_types.add(resp_type)
            
            if response_types:
                st.info("📋 Tipos de consultas:")
                for rtype in sorted(response_types):
                    st.text(f"• {rtype}")
        
        st.markdown("---")
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.conversation_history = []
            st.rerun()
        
        if st.button("🔄 Nuevo Caso"):
            st.session_state.conversation_history = []
            st.session_state.current_classification = None
            st.session_state.current_image_path = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("🔬 **Sistema RAG Conversacional** - Diagnóstico inteligente con contexto de conversación")

if __name__ == "__main__":
    main()
