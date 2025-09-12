import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import tempfile
import os
import uuid
from datetime import datetime

# Configurar página
st.set_page_config(
    page_title="🍅 Diagnóstico Conversacional de Tomate",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar path del sistema
sys.path.append(str(Path(__file__).parent))

# CSS personalizado mejorado
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
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        margin-right: auto;
        border-left: 4px solid #4caf50;
    }
    .classification-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .response-type-indicator {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_classification' not in st.session_state:
    st.session_state.current_classification = None
if 'graph_system' not in st.session_state:
    st.session_state.graph_system = None

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

def generate_specific_answer(user_question, disease_name, confidence):
    """Genera respuesta específica analizando las palabras clave de la pregunta"""
    
    question_lower = user_question.lower()
    
    # Detectar palabras clave específicas en la pregunta
    if 'causa' in question_lower or 'por qué' in question_lower or 'porque' in question_lower:
        return f"""
Las principales causas que llevaron a que apareciera {disease_name} en tu cultivo fueron:

**CONDICIONES AMBIENTALES:**
• Humedad relativa alta (>80%) por períodos prolongados
• Temperatura entre 15-25°C (ideal para Phytophthora)
• Presencia de agua libre en las hojas (rocío, lluvia, riego)

**FACTORES DEL CULTIVO:**
• Densidad de plantación alta (poca ventilación)
• Variedades susceptibles sin resistencia genética
• Presencia de rastrojos infectados de cosechas anteriores

**FUENTES DE INÓCULO:**
• Esporas transportadas por el viento desde cultivos vecinos
• Material vegetal infectado no eliminado completamente
• Herramientas o equipos contaminados
"""
    
    elif 'urgente' in question_lower or 'inmediato' in question_lower or 'rápido' in question_lower:
        return f"""
**ACCIONES INMEDIATAS (próximas 24-48 horas):**

1. **Eliminar focos**: Cortar y destruir plantas/hojas infectadas
2. **Aplicación de emergencia**: Producto sistémico + contacto
3. **Suspender riego**: Evitar mojado foliar hasta control inicial
4. **Mejorar ventilación**: Abrir todas las vías de circulación de aire

**CRÍTICO**: {disease_name} puede expandirse muy rápidamente en condiciones favorables.
"""
    
    elif any(word in question_lower for word in ['cómo', 'como', 'qué hacer', 'que hacer']):
        return f"""
Para manejar {disease_name} (detectado con {confidence:.1%} confianza):

**LO MÁS IMPORTANTE:**
• Actuar rápidamente - la enfermedad se expande muy rápido
• Combinar control químico + cultural
• Monitorear diariamente en condiciones húmedas
• Aplicar productos específicos según condiciones climáticas

**PASOS A SEGUIR:**
1. Control inmediato con fungicidas
2. Mejorar condiciones ambientales
3. Implementar monitoreo continuo
4. Planificar prevención para próximos ciclos
"""
    
    else:
        return f"""
Basándome en tu pregunta específica sobre {disease_name}, la respuesta más relevante incluye:

• **Diagnóstico confirmado** con {confidence:.1%} de precisión
• **Factores específicos** de tu consulta analizados
• **Recomendaciones personalizadas** para tu situación
• **Plan de acción** adaptado a tus necesidades particulares

Si necesitas información más específica, puedes preguntar sobre aspectos concretos como causas, prevención, tratamientos económicos, o manejo técnico.
"""

def generate_contextual_response(user_question, classification, rag_context=""):
    """Genera respuesta personalizada analizando específicamente la pregunta del usuario"""
    
    question_lower = user_question.lower()
    disease_name = classification.get('class_name', 'Enfermedad detectada')
    confidence = classification.get('confidence', 0)
    
    # Mostrar si estamos usando RAG
    using_rag = rag_context and len(rag_context.strip()) > 50
    
    # Análisis específico de la pregunta CON contexto RAG si está disponible
    if any(word in question_lower for word in ['por qué', 'porque', 'causa', 'razón', 'origen']):
        if any(word in question_lower for word in ['no', 'sin', 'evitar']) and any(word in question_lower for word in ['tratamiento', 'químico', 'fungicida']):
            # Pregunta sobre causas SIN tratamiento
            response_type = "🔍 ANÁLISIS DE CAUSAS" + (" (con RAG)" if using_rag else " (sin RAG)")
            
            if using_rag:
                response = f"""
**ANALISIS BASADO EN DOCUMENTOS TECNICOS: Por que aparecio Tizon Tardio?**
*Confianza del diagnóstico: {confidence:.1%}*

**📚 INFORMACIÓN DE LA BASE DE CONOCIMIENTO:**
{rag_context[:800]}...

**🔬 FACTORES CAUSALES SEGÚN LITERATURA CIENTÍFICA:**
• **Condiciones ambientales**: Humedad relativa >80%, temperatura 15-25°C
• **Presencia del patógeno**: Phytophthora infestans activo en el ambiente
• **Susceptibilidad del hospedante**: Variedad sin resistencia + condiciones de estrés
• **Vías de infección**: Esporas transportadas por viento, agua, herramientas contaminadas
"""
            else:
                response = f"""
**⚠️ ANÁLISIS BÁSICO (SIN ACCESO A BASE DOCUMENTAL)**
*Confianza del diagnóstico: {confidence:.1%}*

**CAUSAS GENERALES DEL TIZÓN TARDÍO:**
• Condiciones de humedad alta y temperatura moderada
• Presencia del patógeno Phytophthora infestans
• Variedades susceptibles
• Prácticas culturales inadecuadas

**Nota**: Para un análisis más detallado con información científica específica, necesitamos acceso a la base de documentos RAG.
"""
                
        else:
            response_type = "🔍 CAUSAS COMPLETAS" + (" (con RAG)" if using_rag else "")
            if using_rag:
                response = f"""
**ANÁLISIS COMPLETO DE CAUSAS - Con Base Técnica**
*Diagnóstico: {confidence:.1%}*

**📚 INFORMACIÓN DE DOCUMENTOS ESPECIALIZADOS:**
{rag_context[:1000]}

**CONCLUSIÓN BASADA EN EVIDENCIA CIENTÍFICA:**
El análisis de tu caso específico indica múltiples factores contributivos al desarrollo de la enfermedad.
"""
            else:
                response = generate_specific_answer(user_question, disease_name, confidence)
                
    elif any(word in question_lower for word in ['cultural', 'no químico', 'sin fungicida', 'orgánico']):
        response_type = "🌱 MANEJO CULTURAL" + (" (con RAG)" if using_rag else "")
        
        if using_rag:
            response = f"""
**MANEJO CULTURAL BASADO EN LITERATURA CIENTÍFICA**
*Para {disease_name} - {confidence:.1%} confianza*

**� RECOMENDACIONES DE DOCUMENTOS TÉCNICOS:**
{rag_context[:800]}

**🌿 PRÁCTICAS CULTURALES VALIDADAS:**
• Eliminación sanitaria según protocolos técnicos
• Manejo de espaciamiento y ventilación
• Sistemas de riego localizados
• Rotación y preparación del suelo
"""
        else:
            response = f"""
**MANEJO CULTURAL - {disease_name}**
*Confianza: {confidence:.1%}*

**⚠️ Información básica (sin acceso a documentos técnicos):**
• Eliminación de plantas infectadas
• Mejora de ventilación
• Riego por goteo
• Rotación de cultivos

**Para recomendaciones más específicas y técnicas, se requiere acceso a la base documental.**
"""
            
    else:
        # Respuesta general
        response_type = "📚 RESPUESTA COMPLETA" + (" (con RAG)" if using_rag else " (básica)")
        
        if using_rag:
            response = f"""
**INFORMACIÓN ESPECIALIZADA - {disease_name}**
*Confianza: {confidence:.1%}*

**📚 CONTEXTO DE DOCUMENTOS CIENTÍFICOS:**
{rag_context[:1200]}

**🎯 ANÁLISIS DE TU CONSULTA:**
Basándome en la literatura científica disponible y tu pregunta específica: "{user_question}", esta información es la más relevante para tu situación particular.
"""
        else:
            response = f"""
**DIAGNÓSTICO BÁSICO: {disease_name}**
*Confianza: {confidence:.1%}*

**⚠️ SISTEMA RAG NO DISPONIBLE**
Esta es una respuesta básica. Para obtener información detallada basada en documentos técnicos, el sistema RAG debe estar funcionando.

**Información general disponible:**
• Diagnóstico confirmado mediante CNN
• Recomendaciones básicas de manejo
• Sugerencia de consultar especialistas para tratamiento específico
"""
"""
        else:
            # Pregunta general sobre causas
            response_type = "🔍 ANÁLISIS DE CAUSAS"
            response = f"""
**Por que aparecio {disease_name}?**

**FACTORES QUE LO CAUSARON:**
• Condiciones ambientales favorables (humedad, temperatura)
• Presencia del patógeno en el área
• Susceptibilidad de la variedad
• Prácticas de manejo inadecuadas

**PARA PREVENIR:**
• Mejorar ventilación y drenaje
• Usar variedades resistentes
• Aplicar medidas preventivas según clima
• Mantener sanidad en herramientas y suelo
"""
    
    elif any(word in question_lower for word in ['económico', 'barato', 'costo', 'presupuesto', 'precio', 'dinero']):
        response_type = "💰 ECONÓMICO"
        response = f"""
**TRATAMIENTO ECONÓMICO para {disease_name}**
*Detectado con {confidence:.1%} de confianza*

**PRODUCTOS MÁS ECONÓMICOS:**
• **Mancozeb genérico**: $15-20/kg (vs $35-45 marcas premium)
• **Hidróxido de cobre**: $12-15/kg (opción orgánica económica)
• **Oxicloruro de cobre**: $10-14/kg (preventivo barato)

**ESTRATEGIA DE AHORRO:**
• Mezclar productos: Cobre + Mancozeb reduce costo 30%
• Aplicar preventivo en clima seco (menos frecuencia)
• Usar sistémicos solo en emergencias
• Manejo cultural (gratis): poda, ventilación, riego adecuado

**COSTO APROXIMADO:** $150-200/ha vs $400-500/ha con productos premium
"""
    
    elif any(word in question_lower for word in ['prevenir', 'prevención', 'futuro', 'evitar', 'próximas']):
        response_type = "🌱 PREVENTIVO"
        response = f"""
**CÓMO PREVENIR {disease_name} EN FUTURAS SIEMBRAS**

**ANTES DE SEMBRAR:**
• **Variedades resistentes**: Mountain Fresh, Phoenix, Celebrity
• **Rotación**: 2-3 años con gramíneas/leguminosas
• **Preparación del suelo**: Eliminar rastrojos, mejorar drenaje

**DURANTE EL CULTIVO:**
• **Espaciamiento adecuado**: 40-50 cm entre plantas
• **Riego localizado**: Goteo, nunca aspersión
• **Monitoreo climático**: Preventivos cuando HR >80%
• **Aplicaciones calendario**: Cobre preventivo cada 15 días

**MANEJO PREVENTIVO:**
• Inspección semanal en condiciones húmedas
• Poda sanitaria regular
• Desinfección de herramientas
• Control de malezas hospederas
"""
    
    elif any(word in question_lower for word in ['técnico', 'fitopatológico', 'completo', 'detallado', 'análisis']):
        response_type = "🔬 TÉCNICO ESPECIALIZADO"
        response = f"""
**ANÁLISIS FITOPATOLÓGICO DETALLADO**
*{disease_name} - Confianza: {confidence:.1%}*

**ETIOLOGÍA:**
• **Agente causal**: *Phytophthora infestans* (Mont.) de Bary
• **Reino**: Stramenopila, Clase: Oomycetes
• **Ciclo**: Asexual (esporangios) y sexual (oosporas de resistencia)

**EPIDEMIOLOGÍA:**
• **Condiciones óptimas**: 18-22°C, HR >90%, mojado >6h
• **Dispersión**: Viento, lluvia, herramientas, insectos
• **Supervivencia**: Oosporas en suelo, micelia en rastrojos

**SINTOMATOLOGÍA PATOGNOMÓNICA:**
• Lesiones irregulares con halo clorótico
• Esporulación blanca en envés foliar (condiciones húmedas)
• Progresión necrótica sistémica descendente
• Lesiones en frutos: firmes, café-rojizas, superficie granulosa

**MANEJO INTEGRADO (MIP):**
• Resistencia genética + control cultural + químico rotacional
• Monitoreo con modelos predictivos (TOMCAST)
• Umbrales de acción basados en condiciones ambientales
"""
    
    elif any(word in question_lower for word in ['invernadero', 'humedad', 'temperatura', 'ambiente']):
        response_type = "🌾 MANEJO AMBIENTAL"
        response = f"""
**MANEJO ESPECÍFICO PARA CONDICIONES AMBIENTALES**
*{disease_name} detectado - Confianza: {confidence:.1%}*

**ANÁLISIS DE TU AMBIENTE:**
• **Alta humedad mencionada**: Factor crítico para el desarrollo
• **Ambiente protegido**: Requiere manejo especializado

**CONTROL AMBIENTAL ESPECÍFICO:**
• **Ventilación**: Ventanas cenitales + laterales 24h
• **Deshumidificación**: Equipos si HR >85% persistente
• **Calefacción nocturna**: Evitar condensación en hojas
• **Monitoreo**: Sensores de HR/temperatura con alarmas

**AJUSTES EN APLICACIONES:**
• Frecuencia: Cada 5 días (vs 7 días campo abierto)
• Volumen: 600-800 L/ha (mejor cobertura)
• Productos: Translaminar para mejor penetración

**MEDIDAS CULTURALES ESPECÍFICAS:**
• Poda más intensiva para circulación de aire
• Mulching plástico para evitar salpicadura
• Riego por goteo exclusivamente
• Desinfección herramientas con amonio cuaternario
"""
    
    else:
        # Respuesta personalizada basada en la pregunta exacta
        response_type = "💬 RESPUESTA PERSONALIZADA"
        response = f"""
**Respondiendo específicamente a tu pregunta:**
"{user_question}"

**DIAGNÓSTICO:** {disease_name} (Confianza: {confidence:.1%})

**RESPUESTA ESPECÍFICA A TU CONSULTA:**
Basándome en tu pregunta específica y el análisis de la imagen, te proporciono la información más relevante para tu situación particular.

{generate_specific_answer(user_question, disease_name, confidence)}
"""
    
    return response, response_type
    
    
    return response, response_type

def display_conversation():
    """Muestra el historial de conversación"""
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 Usuario ({message['timestamp']}):</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            response_type = message.get('response_type', 'GENERAL')
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="response-type-indicator">
                    🤖 Agrónomo Especialista - Respuesta {response_type}
                </div>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🍅 Sistema Conversacional de Diagnóstico</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💬 Chat Inteligente con Especialista Virtual")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Panel de Control")
        
        if st.button("🗑️ Limpiar Conversación"):
            st.session_state.conversation_history = []
            st.session_state.current_image = None
            st.session_state.current_classification = None
            st.rerun()
        
        st.markdown("---")
        st.header("📊 Estado Actual")
        
        if st.session_state.current_image:
            st.success("✅ Imagen cargada")
        else:
            st.info("📸 Sin imagen")
            
        if st.session_state.current_classification:
            cls = st.session_state.current_classification
            st.metric("Enfermedad", cls.get('class_name', 'N/A'))
            st.metric("Confianza", f"{cls.get('confidence', 0):.1%}")
        
        st.markdown("---")
        st.header("💡 Tipos de Pregunta")
        st.markdown("""
        - 💰 **Económica**: "tratamiento más barato"
        - 🌱 **Preventiva**: "cómo prevenir"
        - 🔬 **Técnica**: "análisis fitopatológico"
        - 🌾 **Contextual**: "invernadero, humedad"
        - 🚨 **Urgente**: "ayuda inmediata"
        """)

    # Layout principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📸 Imagen del Cultivo")
        
        # Subir nueva imagen
        uploaded_file = st.file_uploader(
            "Nueva imagen:",
            type=['jpg', 'jpeg', 'png'],
            help="Sube una imagen para análisis inicial"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Nueva imagen", use_column_width=True)
            
            if st.button("🔬 Analizar Esta Imagen"):
                if not initialize_system():
                    st.error("Error inicializando sistema")
                    return
                
                with st.spinner("Analizando..."):
                    try:
                        # Guardar temporalmente
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            image.save(tmp.name)
                            temp_path = tmp.name
                        
                        # Clasificar solo la imagen
                        result = st.session_state.graph_system.process_input(
                            image_path=temp_path,
                            user_text=""
                        )
                        
                        os.unlink(temp_path)
                        
                        if result["success"] and result["disease_prediction"]:
                            st.session_state.current_image = image
                            st.session_state.current_classification = result["disease_prediction"]
                            
                            # Agregar al historial
                            timestamp = datetime.now().strftime("%H:%M")
                            st.session_state.conversation_history.append({
                                'role': 'user',
                                'content': f"📸 [Nueva imagen subida: {uploaded_file.name}]",
                                'timestamp': timestamp
                            })
                            
                            cls = result["disease_prediction"]
                            response = f"""
                            <div class="classification-box">
                                <h3>🔬 ANÁLISIS INICIAL</h3>
                                <p><strong>Enfermedad Detectada:</strong> {cls.get('class_name', 'N/A')}</p>
                                <p><strong>Nivel de Confianza:</strong> {cls.get('confidence', 0):.1%}</p>
                            </div>
                            
                            **📋 Imagen analizada exitosamente.**
                            
                            💬 **Ahora puedes hacer preguntas específicas sobre esta imagen:**
                            - "¿Cuál es el tratamiento más económico?"
                            - "¿Cómo prevenir esto en futuras siembras?"
                            - "Necesito un plan de manejo técnico completo"
                            - "Mi cultivo está en invernadero con alta humedad"
                            """
                            
                            st.session_state.conversation_history.append({
                                'role': 'assistant',
                                'content': response,
                                'response_type': '🔬 ANÁLISIS INICIAL',
                                'timestamp': timestamp
                            })
                            
                            st.success("✅ Imagen analizada! Ahora puedes hacer preguntas específicas.")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # Mostrar imagen actual
        if st.session_state.current_image:
            st.markdown("### 📷 Imagen Actual")
            st.image(st.session_state.current_image, caption="En análisis", use_column_width=True)

    with col2:
        st.header("💬 Conversación")
        
        # Mostrar historial
        if st.session_state.conversation_history:
            with st.container():
                display_conversation()
        else:
            st.info("👋 ¡Hola! Sube una imagen para comenzar el diagnóstico.")
        
        # Input de pregunta
        st.markdown("### ✍️ Hacer Pregunta")
        
        user_question = st.text_input(
            "Escribe tu pregunta:",
            placeholder="Ej: ¿Cuál es el tratamiento más económico para esta enfermedad?",
            key="user_input"
        )
        
        col_send, col_examples = st.columns([1, 2])
        
        with col_send:
            send_button = st.button("📤 Enviar", type="primary")
        
        with col_examples:
            example_questions = [
                "¿Tratamiento más económico?",
                "¿Cómo prevenir en futuras siembras?", 
                "Plan técnico completo",
                "Manejo en invernadero"
            ]
            
            example_clicked = st.selectbox(
                "Ejemplos rápidos:",
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
                        # Ejecutar sistema LangGraph para obtener contexto RAG
                        state = {
                            "user_question": question_to_process,
                            "image_path": st.session_state.current_image_path,
                            "classification": st.session_state.current_classification,
                            "conversation_history": [msg['content'] for msg in st.session_state.conversation_history[-6:]]
                        }
                        
                        result = st.session_state.graph_system.graph.invoke(state)
                        rag_context = result.get('rag_context', '')
                        
                        # Generar respuesta contextual con RAG
                        response, response_type = generate_contextual_response(
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
    st.markdown("🍅 **Sistema Conversacional** | LangGraph + CNN + RAG + Gemini 2.0")

if __name__ == "__main__":
    main()
