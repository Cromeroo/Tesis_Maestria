#!/usr/bin/env python3
"""
Sistema RAG con LangGraph para diagnóstico de enfermedades en tomate
Flujo: Imagen + Texto -> Clasificación -> RAG -> LLM Response
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

# Configurar path
sys.path.append(str(Path(__file__).parent))

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Google Cloud imports
from langchain_google_vertexai import ChatVertexAI
import google.auth
from google.oauth2 import service_account

# Componentes locales
from components.image_classifier import TomatoImageClassifier
from components.vector_store import VectorStore
from config.settings import Settings

class State(TypedDict):
    """Estado del grafo"""
    image_path: str
    user_text: str
    disease_prediction: Optional[Dict[str, Any]]
    rag_context: Optional[str]
    final_response: Optional[str]
    error: Optional[str]

class TomatoDiseaseGraph:
    """Grafo LangGraph para diagnóstico integrado de enfermedades"""
    
    def __init__(self):
        self.settings = Settings()
        self._setup_credentials()
        self._initialize_components()
        self._build_graph()
    
    def _setup_credentials(self):
        """Configurar credenciales de Google Cloud"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.settings.GOOGLE_CREDENTIALS_PATH,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.settings.GOOGLE_CREDENTIALS_PATH
            print("✅ Credenciales Google Cloud configuradas")
        except Exception as e:
            print(f"⚠️ Error configurando credenciales: {e}")
    
    def _initialize_components(self):
        """Inicializar componentes del sistema"""
        try:
            # Clasificador de imágenes
            self.image_classifier = TomatoImageClassifier()
            print("✅ Clasificador de imágenes inicializado")
            
            # Vector store para RAG
            self.vector_store = VectorStore()
            print("✅ Vector store inicializado")
            
            # LLM con Gemini 2.0 Flash
            self.llm = ChatVertexAI(
                model_name="gemini-2.0-flash-exp",
                temperature=self.settings.LLM_TEMPERATURE,
                max_output_tokens=self.settings.LLM_MAX_OUTPUT_TOKENS,
                project=json.loads(Path(self.settings.GOOGLE_CREDENTIALS_PATH).read_text())["project_id"],
                location=self.settings.GOOGLE_LOCATION
            )
            print("✅ Gemini 2.0 Flash inicializado")
            
        except Exception as e:
            print(f"❌ Error inicializando componentes: {e}")
            raise
    
    def _build_graph(self):
        """Construir el grafo LangGraph"""
        workflow = StateGraph(State)
        
        # Añadir nodos
        workflow.add_node("classify_image", self._classify_image)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        
        # Definir flujo
        workflow.set_entry_point("classify_image")
        workflow.add_conditional_edges(
            "classify_image",
            self._check_late_blight,
            {
                "proceed": "retrieve_context",
                "end": END
            }
        )
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)
        
        self.graph = workflow.compile()
        print("✅ Grafo LangGraph compilado")
    
    def _classify_image(self, state: State) -> State:
        """Clasificar imagen con CNN"""
        try:
            print("🔬 Clasificando imagen...")
            
            if not state["image_path"]:
                state["error"] = "No se proporcionó imagen"
                return state
            
            # Clasificar imagen
            result = self.image_classifier.classify_image(state["image_path"])
            state["disease_prediction"] = result
            
            print(f"   Resultado: {result['predicted_class']}")
            print(f"   Confianza: {result['confidence']:.2%}")
            
            return state
            
        except Exception as e:
            state["error"] = f"Error en clasificación: {e}"
            return state
    
    def _check_late_blight(self, state: State) -> str:
        """Verificar si es tizón tardío"""
        if state.get("error"):
            return "end"
        
        if not state.get("disease_prediction"):
            return "end"
        
        # Usar class_name en lugar de predicted_class
        class_name = state["disease_prediction"].get('class_name', '').lower()
        
        # Solo proceder si es tizón tardío
        if 'tizon_tardio' in class_name or 'late_blight' in class_name:
            print("✅ Tizón tardío detectado - Procediendo con RAG")
            return "proceed"
        else:
            print(f"ℹ️ Otra enfermedad detectada ({class_name}) - Finalizando")
            return "end"
    
    def _retrieve_context(self, state: State) -> State:
        """Buscar contexto con RAG"""
        try:
            print("📚 Buscando contexto en documentos...")
            
            # Preparar query
            query = f"tizón tardío tomate manejo control {state.get('user_text', '')}"
            
            # Buscar documentos
            collection = self.vector_store.get_or_create_collection(
                self.settings.COLLECTION_NAME, 
                self.settings.COLLECTION_DESCRIPTION
            )
            results = self.vector_store.search_documents(collection, query, n_results=3)
            
            # Construir contexto
            context_parts = []
            if results:
                for i, result in enumerate(results, 1):
                    if isinstance(result, dict):
                        content = result.get('text', '') or result.get('content', '') or str(result)
                    else:
                        content = str(result)
                    context_parts.append(f"Documento {i}:\n{content}\n")
            
            state["rag_context"] = "\n".join(context_parts) if context_parts else "Información técnica sobre tizón tardío"
            print(f"   Encontrados {len(results)} documentos relevantes")
            
            return state
            
        except Exception as e:
            state["error"] = f"Error en RAG: {e}"
            return state
    
    def _generate_response(self, state: State) -> State:
        """Generar respuesta con LLM"""
        try:
            print("🤖 Generando respuesta con Gemini 2.0 Flash...")
            
            # Prompt especializado
            system_prompt = """
Eres un agrónomo especialista en fitopatología. 

INSTRUCCIONES IMPORTANTES:
- Responde de forma ESTRUCTURADA pero CONCISA (300-400 palabras)
- Incluye: Diagnóstico + Control químico + Manejo cultural + Productos específicos + Urgencia
- Usa terminología técnica pero clara
- Enfócate en acciones INMEDIATAS y plan a corto plazo
- Incluye dosificaciones específicas y frecuencias

Tu respuesta debe ser completa pero práctica para uso en campo.
"""
            
            # Construir prompt
            user_prompt = f"""
DIAGNÓSTICO: Tizón Tardío (Phytophthora infestans) confirmado

CONTEXTO TÉCNICO:
{state.get('rag_context', '')[:1500]}

CONSULTA: {state.get('user_text', 'Plan de manejo urgente para tizón tardío')}

Proporciona un plan de manejo completo pero conciso (300-400 palabras) con productos específicos y dosificaciones.
"""
            
            # Generar respuesta
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            state["final_response"] = response.content
            
            print("✅ Respuesta generada con éxito")
            return state
            
        except Exception as e:
            # Respuesta de fallback
            fallback_response = """
**DIAGNÓSTICO CONFIRMADO: Tizón Tardío (Phytophthora infestans)**

**CONTROL QUÍMICO INMEDIATO:**
• **Sistémicos:** Ridomil Gold MZ (metalaxil + mancozeb) - 2.5 kg/ha
• **Contacto:** Mancozeb 80% WP - 2-2.5 kg/ha
• **Alternativa:** Curzate M8 (cymoxanil + mancozeb) - 2 kg/ha
• **Frecuencia:** Alternar cada 5-7 días según humedad

**MANEJO CULTURAL:**
• Eliminar inmediatamente plantas/hojas infectadas y destruir fuera del lote
• Mejorar ventilación: podar hojas basales, ajustar densidad de siembra
• Riego localizado por goteo, evitar aspersión foliar
• Aplicar en horas tempranas (6-8 AM) para mejor absorción

**MONITOREO:**
• Inspección diaria especialmente en condiciones húmedas (>80% HR)
• Vigilar envés de hojas para detectar esporulación temprana
• Aplicaciones preventivas antes de lluvia o neblina

**URGENCIA:** CRÍTICA - Implementar medidas en las próximas 24 horas
La enfermedad puede devastar el cultivo en 7-10 días bajo condiciones favorables.
"""
            
            state["final_response"] = fallback_response
            state["error"] = f"Error LLM (usando fallback): {e}"
            print(f"⚠️ Error LLM, usando respuesta de fallback: {e}")
            return state
    
    def process_input(self, image_path: str, user_text: str = "") -> Dict[str, Any]:
        """Procesar entrada completa"""
        print("🚀 Iniciando procesamiento con LangGraph")
        print(f"📸 Imagen: {image_path}")
        print(f"💬 Texto: {user_text}")
        print()
        
        # Estado inicial
        initial_state = {
            "image_path": image_path,
            "user_text": user_text,
            "disease_prediction": None,
            "rag_context": None,
            "final_response": None,
            "error": None
        }
        
        try:
            # Ejecutar el grafo
            final_state = self.graph.invoke(initial_state)
            
            return {
                "success": True,
                "disease_prediction": final_state.get("disease_prediction"),
                "rag_context": final_state.get("rag_context"),
                "final_response": final_state.get("final_response"),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error ejecutando grafo: {e}",
                "disease_prediction": None,
                "final_response": None
            }

if __name__ == "__main__":
    print("🧪 Inicializando sistema LangGraph...")
    
    try:
        graph_system = TomatoDiseaseGraph()
        print("✅ Sistema LangGraph inicializado correctamente")
        
        # Probar con imagen de ejemplo
        test_image = "test_images/1-36.png"
        if Path(test_image).exists():
            result = graph_system.process_input(
                image_path=test_image,
                user_text="Necesito ayuda urgente para manejar esta enfermedad en mi cultivo de tomate"
            )
            
            print("\n" + "="*70)
            print("🎯 RESULTADO DEL PROCESAMIENTO CON LANGGRAPH:")
            print("="*70)
            
            if result["success"]:
                print("✅ Procesamiento exitoso con LangGraph")
                
                if result["disease_prediction"]:
                    pred = result["disease_prediction"]
                    print(f"🔬 Clasificación: {pred.get('class_name', 'N/A')}")
                    print(f"📊 Confianza: {pred['confidence']:.2%}")
                
                if result["final_response"]:
                    print("\n🤖 RESPUESTA DEL AGENTE CON GEMINI 2.0 FLASH:")
                    print("-" * 50)
                    print(result["final_response"])
                
                if result["error"]:
                    print(f"\n⚠️ Advertencias: {result['error']}")
                    
                print("\n" + "="*70)
                print("💡 TECNOLOGÍAS UTILIZADAS:")
                print("   🧠 LLM: Gemini 2.0 Flash Experimental")
                print("   🔀 Framework: LangGraph para flujo condicional")
                print("   📚 RAG: ChromaDB + sentence-transformers")
                print("   🔬 CNN: PyTorch (HuggingFace)")
                print("   ☁️ Cloud: Google Cloud Vertex AI")
                
            else:
                print(f"❌ Error: {result['error']}")
        else:
            print("⚠️ No se encontró imagen de prueba")
            
    except Exception as e:
        print(f"❌ Error inicializando sistema: {e}")
