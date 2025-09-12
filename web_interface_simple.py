#!/usr/bin/env python3
"""
Interfaz web simple para el sistema RAG de diagnóstico de enfermedades en tomate
Sin caracteres especiales que causen problemas de sintaxis
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
    """Inicializar el estado de la sesión"""
    if 'graph_system' not in st.session_state:
        st.session_state.graph_system = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_classification' not in st.session_state:
        st.session_state.current_classification = None
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None

def initialize_system():
    """Inicializa el sistema LangGraph una sola vez"""
    if st.session_state.graph_system is None:
        try:
            from langgraph_system_simple import TomatoDiseaseGraph
            st.session_state.graph_system = TomatoDiseaseGraph()
            return True
        except Exception as e:
            st.error(f"Error inicializando sistema: {e}")
            return False
    return True

def generate_rag_response(user_question, classification, rag_context=""):
    """Genera respuesta usando el contexto RAG"""
    
    question_lower = user_question.lower()
    disease_name = classification.get('class_name', 'Enfermedad detectada')
    confidence = classification.get('confidence', 0)
    
    # Mostrar si estamos usando RAG
    using_rag = rag_context and len(rag_context.strip()) > 50
    
    response_header = f"""
**DIAGNOSTICO: {disease_name}**
*Confianza: {confidence:.1%}*
"""
    
    if using_rag:
        response_header += "\n**USANDO BASE DE DOCUMENTOS TECNICOS (RAG)**\n"
        
        # Análisis de la pregunta con RAG
        if any(word in question_lower for word in ['causa', 'por que', 'porque', 'razon']):
            if any(word in question_lower for word in ['cultural', 'no quimico', 'sin fungicida']):
                response_type = "MANEJO CULTURAL CON RAG"
                specific_response = f"""
**INFORMACION DE DOCUMENTOS CIENTIFICOS:**
{rag_context[:800]}...

**MANEJO CULTURAL RECOMENDADO:**
- Eliminacion sanitaria de plantas infectadas
- Mejora de ventilacion y espaciamiento
- Riego localizado por goteo
- Rotacion de cultivos
- Preparacion adecuada del suelo
"""
            else:
                response_type = "ANALISIS DE CAUSAS CON RAG"
                specific_response = f"""
**BASE TECNICA DE DOCUMENTOS:**
{rag_context[:1000]}...

**FACTORES CAUSALES IDENTIFICADOS:**
- Condiciones ambientales favorables
- Presencia del patogeno en el area
- Susceptibilidad de la variedad
- Practicas de manejo inadecuadas
"""
        else:
            response_type = "RESPUESTA COMPLETA CON RAG"
            specific_response = f"""
**INFORMACION ESPECIALIZADA:**
{rag_context[:1200]}...

**RECOMENDACIONES BASADAS EN LITERATURA CIENTIFICA:**
Segun la pregunta: "{user_question}", esta es la informacion mas relevante de los documentos tecnicos.
"""
    else:
        response_header += "\n**SIN ACCESO A BASE DOCUMENTAL - RESPUESTA BASICA**\n"
        response_type = "RESPUESTA BASICA"
        specific_response = f"""
**INFORMACION GENERAL:**
- Diagnostico confirmado mediante CNN
- Para informacion detallada se requiere acceso a documentos RAG
- Se recomienda consultar especialista para tratamiento especifico

**NOTA:** El sistema RAG no esta disponible en este momento.
"""
    
    return f"{response_header}\n**TIPO DE RESPUESTA:** {response_type}\n{specific_response}", response_type

def main():
    st.set_page_config(
        page_title="Sistema RAG - Diagnostico de Enfermedades en Tomate",
        page_icon="🍅",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🍅 Sistema RAG para Diagnostico de Enfermedades en Tomate")
    st.markdown("**Sube una imagen y haz preguntas especificas sobre el diagnostico**")
    
    # Layout en columnas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Subir Imagen")
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de la planta",
            type=['png', 'jpg', 'jpeg'],
            help="Sube una imagen clara de las hojas afectadas"
        )
        
        if uploaded_file:
            # Mostrar imagen
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_container_width=True)
            
            # Guardar imagen temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image.save(tmp_file.name)
                temp_path = tmp_file.name
                st.session_state.current_image_path = temp_path
            
            # Clasificar imagen
            if st.button("🔬 Analizar Imagen", type="primary"):
                if not initialize_system():
                    st.error("Error inicializando sistema")
                    return
                
                with st.spinner("🔄 Analizando imagen..."):
                    try:
                        # Solo clasificar imagen (sin pregunta aún)
                        state = {
                            "image_path": temp_path,
                            "user_question": "",
                            "classification": None,
                            "conversation_history": []
                        }
                        
                        # Ejecutar solo la clasificación
                        from components.image_classifier import TomatoImageClassifier
                        classifier = TomatoImageClassifier()
                        result = classifier.classify_image(temp_path)
                        
                        st.session_state.current_classification = result
                        
                        # Mostrar resultado
                        st.success("✅ Imagen analizada correctamente")
                        st.info(f"**Diagnostico:** {result['class_name']}")
                        st.info(f"**Confianza:** {result['confidence']:.1%}")
                        
                    except Exception as e:
                        st.error(f"Error en análisis: {e}")
    
    with col2:
        st.header("💬 Conversacion")
        
        # Mostrar historial de conversación
        if st.session_state.conversation_history:
            st.subheader("📋 Historial de Conversacion")
            for i, msg in enumerate(st.session_state.conversation_history):
                if msg['role'] == 'user':
                    st.markdown(f"**👤 Tu:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 Agronomo:** {msg.get('response_type', 'RESPUESTA')}")
                    st.markdown(msg['content'])
                st.markdown("---")
        
        # Input para nueva pregunta
        st.subheader("❓ Hacer Pregunta")
        
        # Ejemplos de preguntas
        example_questions = [
            "¿Por que aparecio esta enfermedad en mi cultivo?",
            "Dame solo tratamiento cultural, no quimicos",
            "¿Que opciones economicas tengo para el tratamiento?",
            "¿Como puedo prevenir esto en el futuro?",
            "¿Que tan urgente es tratar esta enfermedad?"
        ]
        
        with st.container():
            user_question = st.text_input(
                "Tu pregunta:",
                placeholder="Escribe tu pregunta especifica aqui...",
                key="user_input"
            )
            
            send_button = st.button("📤 Enviar Pregunta", type="primary")
            
            st.markdown("**Ejemplos de preguntas:**")
            example_clicked = st.selectbox(
                "Selecciona un ejemplo:",
                [""] + example_questions,
                key="examples"
            )
        
        # Procesar pregunta
        question_to_process = None
        if send_button and user_question:
            question_to_process = user_question
        elif example_clicked:
            question_to_process = example_clicked
        
        if question_to_process:
            if not st.session_state.current_classification:
                st.warning("⚠️ Primero sube y analiza una imagen")
            else:
                if not initialize_system():
                    st.error("Error inicializando sistema")
                    return
                
                with st.spinner("🤔 Generando respuesta personalizada..."):
                    try:
                        # Ejecutar sistema LangGraph completo para obtener contexto RAG
                        state = {
                            "user_question": question_to_process,
                            "image_path": st.session_state.current_image_path,
                            "classification": st.session_state.current_classification,
                            "conversation_history": [msg['content'] for msg in st.session_state.conversation_history[-6:]]
                        }
                        
                        result = st.session_state.graph_system.graph.invoke(state)
                        rag_context = result.get('rag_context', '')
                        
                        # Generar respuesta con contexto RAG
                        response, response_type = generate_rag_response(
                            question_to_process,
                            st.session_state.current_classification,
                            rag_context
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
                            'content': response,
                            'response_type': response_type,
                            'timestamp': timestamp
                        })
                        
                        # Limpiar input
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generando respuesta: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔬 **Sistema RAG con LangGraph** - Diagnostico inteligente de enfermedades en tomate"
    )

if __name__ == "__main__":
    main()
