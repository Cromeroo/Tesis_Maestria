#!/usr/bin/env python3
"""
INTERFAZ DEFINITIVA - Sistema RAG Tizón Tardío
✅ LLM Gemini 2.5 Flash (modelo corregido que funciona)
✅ Contexto conversacional
✅ Respuestas inteligentes personalizadas
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

def init_session():
    """Inicializar sesión de Streamlit"""
    if 'graph_system' not in st.session_state:
        st.session_state.graph_system = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'current_diagnosis' not in st.session_state:
        st.session_state.current_diagnosis = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

def load_system():
    """Cargar sistema LangGraph"""
    if st.session_state.graph_system is None:
        try:
            with st.spinner("🔄 Inicializando sistema RAG..."):
                from langgraph_system_simple import TomatoDiseaseGraph
                st.session_state.graph_system = TomatoDiseaseGraph()
            st.success("✅ Sistema cargado - Gemini 2.5 Flash activo")
            return True
        except Exception as e:
            st.error(f"❌ Error cargando sistema: {e}")
            return False
    return True

def process_image(image_file):
    """Procesar imagen subida"""
    try:
        # Guardar imagen temporalmente
        image = Image.open(image_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        st.session_state.current_image = temp_path
        
        # Clasificar imagen
        with st.spinner("🔬 Analizando imagen con CNN..."):
            from components.image_classifier import TomatoImageClassifier
            classifier = TomatoImageClassifier()
            result = classifier.classify_image(temp_path)
            
        st.session_state.current_diagnosis = result
        return result
        
    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None

def ask_question(question):
    """Hacer pregunta al sistema completo (RAG + LLM)"""
    if not st.session_state.current_diagnosis:
        st.warning("⚠️ Primero analiza una imagen")
        return None
    
    try:
        with st.spinner("🤔 Consultando documentos técnicos y generando respuesta..."):
            # Preparar estado para LangGraph
            state = {
                "image_path": st.session_state.current_image,
                "user_text": question,
                "disease_prediction": None,
                "rag_context": None,
                "final_response": None,
                "error": None
            }
            
            # Ejecutar sistema completo LangGraph
            result = st.session_state.graph_system.graph.invoke(state)
            
            # Verificar respuesta del LLM
            llm_response = result.get('final_response')
            rag_context = result.get('rag_context')
            error = result.get('error')
            
            # Si el LLM funcionó, usar su respuesta
            if llm_response and len(llm_response.strip()) > 50:
                return {
                    'response': llm_response,
                    'rag_used': bool(rag_context),
                    'llm_used': True,
                    'error': error
                }
            
            # Si el LLM falló, generar respuesta inteligente local
            else:
                return {
                    'response': generate_smart_fallback(question, st.session_state.current_diagnosis, rag_context),
                    'rag_used': bool(rag_context),
                    'llm_used': False,
                    'error': error or "LLM no disponible, usando respuesta local inteligente"
                }
                
    except Exception as e:
        st.error(f"Error en consulta: {e}")
        return None

def generate_smart_fallback(question, diagnosis, rag_context):
    """Generar respuesta inteligente sin LLM"""
    q = question.lower()
    disease = diagnosis['class_name']
    confidence = diagnosis['confidence']
    
    # Usar contexto RAG si está disponible
    rag_info = ""
    if rag_context:
        rag_info = f"\n\n**📚 Información de documentos técnicos:**\n{rag_context[:500]}...\n"
    
    response = f"""**Respuesta Especializada - {disease}**
*Diagnóstico CNN: {confidence:.1%} de confianza*

"""
    
    if any(word in q for word in ['causa', 'por que', 'porque', 'razón']):
        response += """**🔍 ANÁLISIS DE CAUSAS**

**Factores principales que causaron el tizón tardío:**

**🌧️ Condiciones ambientales:**
• Humedad relativa >80% por períodos prolongados
• Temperatura entre 15-25°C (óptima para Phytophthora)
• Presencia de agua libre en hojas (rocío, lluvia, riego)

**🦠 Presencia del patógeno:**
• Phytophthora infestans activo en el ambiente
• Esporas transportadas por viento desde cultivos vecinos
• Supervivencia en rastrojos de cosechas anteriores

**🌱 Susceptibilidad del cultivo:**
• Variedad sin resistencia genética
• Densidad de plantación alta (poca ventilación)
• Estrés nutricional o hídrico de las plantas

**⚠️ La combinación de estos factores creó las condiciones perfectas para la infección.**"""

    elif any(word in q for word in ['cultural', 'organico', 'sin quimico', 'natural']):
        response += """**🌱 MANEJO CULTURAL INTEGRAL**

**Acciones inmediatas (próximas 24h):**
• **Eliminación sanitaria:** Cortar y destruir plantas muy infectadas
• **Mejorar circulación:** Podar hojas basales y brotes axilares
• **Riego modificado:** Solo por goteo, suspender aspersión
• **Ventilación:** Abrir todas las aberturas en invernaderos

**Medidas a mediano plazo:**
• **Espaciamiento:** Aumentar distancia entre plantas
• **Mulching:** Cobertura plástica para evitar salpicadura
• **Manejo nutricional:** Reducir nitrógeno, aumentar potasio
• **Sanidad:** Desinfectar herramientas con alcohol 70%

**Prevención futura:**
• **Rotación:** 2-3 años con cultivos no-solanáceas
• **Variedades resistentes:** Mountain Fresh Plus, Phoenix
• **Monitoreo climático:** Aplicar preventivos cuando HR >75%"""

    elif any(word in q for word in ['urgente', 'rapido', 'inmediato', 'emergencia']):
        response += """**🚨 PROTOCOLO DE EMERGENCIA**

**PRÓXIMAS 4-6 HORAS:**
1. **Suspender riego:** Evitar mojado foliar inmediatamente
2. **Eliminar focos:** Cortar plantas muy infectadas y sacar del lote
3. **Preparar aplicación:** Alistar equipo para fungicida sistémico

**PRÓXIMAS 24 HORAS:**
1. **Aplicación de choque:** Metalaxil + Mancozeb 2.5 kg/ha
2. **Mejorar ventilación:** Abrir todas las estructuras posibles
3. **Monitoreo intensivo:** Inspeccionar cultivo cada 6 horas

**48-72 HORAS:**
1. **Segunda aplicación:** Dimetomorf + Mancozeb 2 kg/ha
2. **Evaluación:** Verificar avance o control de la enfermedad
3. **Ajuste de estrategia:** Según respuesta observada

**⚠️ CRÍTICO:** Cada hora cuenta. El tizón tardío puede expandirse exponencialmente."""

    elif any(word in q for word in ['economico', 'barato', 'costo', 'precio', 'dinero']):
        response += """**💰 ESTRATEGIA ECONÓMICA EFECTIVA**

**Productos costo-beneficio:**
• **Mancozeb 80%:** $15-20/kg → 2 kg/ha = $30-40/aplicación
• **Hidróxido de cobre:** $8-12/kg → 2.5 kg/ha = $20-30/aplicación
• **Mezcla tanque:** Cobre + Mancozeb = $45/aplicación (vs $80 sistémicos)

**Estrategia de ahorro:**
1. **Preventivo económico:** Cobre cada 10 días en clima seco
2. **Sistémico selectivo:** Solo en condiciones críticas de humedad
3. **Timing inteligente:** Aplicar antes de lluvia (mayor persistencia)

**Manejo cultural (costo $0):**
• Poda sanitaria regular → Reduce presión de enfermedad
• Mejora de drenaje → Menos condiciones favorables
• Riego matutino → Secado rápido de follaje

**Cálculo económico:**
• **Estrategia económica:** $150-200/ha por ciclo
• **Manejo premium:** $400-500/ha por ciclo
• **Ahorro potencial:** 50-60% manteniendo eficacia"""

    else:
        response += """**📋 PLAN DE MANEJO INTEGRAL**

**Control químico escalonado:**
• **Aplicación inmediata:** Ridomil Gold MZ 2.5 kg/ha
• **Seguimiento (7 días):** Mancozeb 80% WP 2 kg/ha
• **Preventivo:** Hidróxido de cobre 2.5 kg/ha en clima seco

**Manejo cultural complementario:**
• **Eliminación:** Plantas muy infectadas destruidas fuera del lote
• **Ventilación:** Mejorar circulación de aire entre plantas
• **Riego:** Exclusivamente por goteo, horario 6-8 AM
• **Monitoreo:** Inspección diaria en condiciones húmedas

**Seguimiento y evaluación:**
• **Eficacia:** Evaluar control a los 5-7 días
• **Reaplicación:** Según evolución y condiciones climáticas
• **Prevención:** Implementar programa para próximo ciclo

**Prioridad:** ALTA - Iniciar tratamiento en máximo 24 horas"""
    
    # Agregar información RAG si está disponible
    response += rag_info
    
    return response

def main():
    """Función principal"""
    st.set_page_config(
        page_title="Sistema RAG Definitivo",
        page_icon="🍅",
        layout="wide"
    )
    
    init_session()
    
    st.title("🍅 Sistema RAG Definitivo - Tizón Tardío")
    st.markdown("**CNN + RAG + Gemini 2.5 Flash → Respuestas Inteligentes**")
    
    # Sidebar para diagnóstico
    with st.sidebar:
        st.header("📸 Análisis de Imagen")
        
        uploaded_file = st.file_uploader(
            "Sube imagen de la planta",
            type=['png', 'jpg', 'jpeg'],
            help="Imagen clara de hojas con síntomas"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
            
            if st.button("🔬 Analizar con CNN", type="primary"):
                result = process_image(uploaded_file)
                if result:
                    st.success("✅ Análisis completado")
                    st.info(f"🔬 **{result['class_name']}**")
                    st.info(f"📊 **{result['confidence']:.1%}** confianza")
        
        # Mostrar diagnóstico activo
        if st.session_state.current_diagnosis:
            st.markdown("---")
            st.markdown("**📋 Diagnóstico Activo:**")
            st.success(f"🔬 {st.session_state.current_diagnosis['class_name']}")
            st.info(f"📊 {st.session_state.current_diagnosis['confidence']:.1%}")
            
            # Botón para limpiar diagnóstico
            if st.button("🗑️ Nuevo Análisis"):
                st.session_state.current_diagnosis = None
                st.session_state.current_image = None
                st.session_state.conversation = []
                st.rerun()
    
    # Cargar sistema
    if not load_system():
        st.stop()
    
    # Área principal - Conversación
    st.header("💬 Consulta Especializada")
    
    # Mostrar historial si existe
    if st.session_state.conversation:
        st.subheader("📋 Historial de Consultas")
        
        # Contenedor con scroll para historial
        with st.container():
            for i, msg in enumerate(st.session_state.conversation):
                if msg['role'] == 'user':
                    st.markdown(f"**👤 Tu pregunta:** {msg['content']}")
                else:
                    st.markdown("**🤖 Agrónomo Especialista:**")
                    st.markdown(msg['content'])
                    
                    # Indicadores de estado
                    col1, col2 = st.columns(2)
                    with col1:
                        if msg.get('llm_used'):
                            st.success("✅ Gemini 2.5 Flash")
                        else:
                            st.warning("⚠️ Respuesta local")
                    with col2:
                        if msg.get('rag_used'):
                            st.info("📚 Con documentos RAG")
                        else:
                            st.warning("📖 Sin documentos")
                    
                    if msg.get('error'):
                        st.caption(f"ℹ️ {msg['error']}")
                
                st.markdown("---")
    
    # Nueva consulta
    st.subheader("❓ Nueva Consulta")
    
    # Input de pregunta
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "Escribe tu pregunta específica:",
            placeholder="Ej: ¿Por qué apareció esta enfermedad?",
            key="question_input"
        )
    
    with col2:
        ask_button = st.button("📤 Consultar", type="primary")
    
    # Preguntas frecuentes
    st.markdown("**💡 Preguntas Frecuentes:**")
    
    frequent_questions = [
        "¿Por qué apareció esta enfermedad en mi cultivo?",
        "¿Qué tratamiento cultural puedo usar sin químicos?", 
        "¿Qué tan urgente es actuar contra esta enfermedad?",
        "¿Cuáles son las opciones de tratamiento más económicas?",
        "¿Cómo puedo prevenir esto en futuros cultivos?"
    ]
    
    # Mostrar preguntas en grid
    cols = st.columns(3)
    for i, question in enumerate(frequent_questions):
        col_idx = i % 3
        if cols[col_idx].button(f"❓ {question[:30]}...", key=f"freq_{i}"):
            user_question = question
            ask_button = True
    
    # Procesar consulta
    if ask_button and user_question:
        if not st.session_state.current_diagnosis:
            st.warning("⚠️ Primero sube y analiza una imagen de la planta")
        else:
            # Ejecutar consulta
            response_data = ask_question(user_question)
            
            if response_data:
                # Agregar a historial
                timestamp = datetime.now().strftime("%H:%M")
                
                st.session_state.conversation.append({
                    'role': 'user',
                    'content': user_question,
                    'timestamp': timestamp
                })
                
                st.session_state.conversation.append({
                    'role': 'assistant',
                    'content': response_data['response'],
                    'llm_used': response_data['llm_used'],
                    'rag_used': response_data['rag_used'],
                    'error': response_data.get('error'),
                    'timestamp': timestamp
                })
                
                # Mostrar respuesta inmediata
                st.markdown("**🤖 Respuesta del Especialista:**")
                st.markdown(response_data['response'])
                
                # Estado de la respuesta
                col1, col2 = st.columns(2)
                with col1:
                    if response_data['llm_used']:
                        st.success("✅ Respuesta generada por Gemini 2.5 Flash")
                    else:
                        st.warning("⚠️ Respuesta generada localmente (LLM no disponible)")
                
                with col2:
                    if response_data['rag_used']:
                        st.info("📚 Basada en documentos técnicos especializados")
                    else:
                        st.warning("📖 Basada en conocimiento general")
                
                if response_data.get('error'):
                    st.error(f"ℹ️ Nota técnica: {response_data['error']}")
                
                # Limpiar input y recargar
                st.rerun()
    
    # Footer informativo
    st.markdown("---")
    st.markdown("""
    **🔬 Tecnología del Sistema:**
    - **CNN:** Clasificación de imágenes con 99%+ precisión
    - **RAG:** Base de 472 documentos técnicos especializados
    - **LLM:** Gemini 2.5 Flash para respuestas inteligentes
    - **Contexto:** Mantiene historial de conversación completo
    """)

if __name__ == "__main__":
    main()
