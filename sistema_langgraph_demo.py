#!/usr/bin/env python3
"""
SISTEMA RAG CON LANGGRAPH - DEMO
Versión de sistema_final.py convertida a LangGraph para mostrar el flujo
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

sys.path.append(str(Path(__file__).parent))

# Imports de LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class RAGState(TypedDict):
    """Estado del grafo RAG"""
    # Inputs
    pregunta: str
    clasificacion_imagen: Optional[str]
    
    # 🧠 NUEVA: Memoria conversacional
    contexto_conversacional: Optional[str]  # Historial de conversaciones previas
    es_seguimiento: bool  # Si es pregunta de seguimiento
    
    # Estados intermedios
    contexto_detectado: str
    query_rag: str
    n_docs: int
    advertencia_especialidad: str
    documentos_encontrados: List[str]
    contexto_rag: str
    
    # Output
    respuesta_final: str
    error: Optional[str]

class SistemaRAGLangGraph:
    """Sistema RAG implementado con LangGraph"""
    
    def __init__(self):
        self.collection = None
        self.llm = None
        self.graph = None
        self._setup_system()
        self._build_graph()
    
    def _setup_system(self):
        """Configurar ChromaDB y VertexAI (mismo que sistema_final.py)"""
        try:
            # Configurar credenciales
            credentials_path = r"C:\Users\danil\OneDrive\Escritorio\Tesis\credentials.json"
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Archivo de credenciales no encontrado: {credentials_path}")
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
            # Configurar ChromaDB
            import chromadb
            chroma_path = "C:/Users/danil/OneDrive/Escritorio/Tesis/chroma_db"
            if not os.path.exists(chroma_path):
                raise FileNotFoundError(f"Base de datos ChromaDB no encontrada: {chroma_path}")
            
            client = chromadb.PersistentClient(path=chroma_path)
            collections = client.list_collections()
            
            if not collections:
                raise Exception("No hay colecciones en ChromaDB. Ejecuta primero el procesamiento de documentos.")
            
            self.collection = collections[0]
            print(f"[OK] ChromaDB cargado: {self.collection.name} con {self.collection.count()} documentos")
            
            # Configurar VertexAI
            import vertexai
            from langchain_google_vertexai import VertexAI
            
            vertexai.init(project="stately-moon-451804-a9", location="us-central1")
            self.llm = VertexAI(
                model_name="gemini-2.5-flash",
                temperature=0.3,
                max_output_tokens=2500,
                top_p=0.9,
                top_k=40
            )
            
            print("[OK] VertexAI configurado")
            
        except Exception as e:
            print(f"[ERROR] Error en setup_system: {e}")
            # Asegurar que las variables estén en None si hay error
            self.collection = None
            self.llm = None
            import traceback
            traceback.print_exc()
            # Re-lanzar el error para que se maneje en la interfaz
            raise e
    
    def _detectar_contexto_node(self, state: RAGState) -> RAGState:
        """Nodo 1: Detectar contexto del usuario"""
        print("🔍 NODO 1: Detectando contexto...")
        
        pregunta_lower = state["pregunta"].lower()
        
        # Indicadores específicos para cada contexto (mismo que sistema_final.py)
        campesino_indicators = [
            'bicho', 'se está dañando', 'se me está muriendo', 'qué le echo',
            'como elimino', 'barato', 'sin químicos', 'natural', 'casero'
        ]
        
        tecnico_indicators = [
            'phytophthora', 'fungicida', 'resistencia', 'mecanismo', 'sistémico',
            'ingrediente activo', 'patógeno', 'oomiceto', 'científic'
        ]
        
        urgencia_indicators = [
            'urgente', 'rápido', 'inmediato', 'se está extendiendo', 'emergencia',
            'ya', 'ahora', 'pronto'
        ]
        
        # Priorizar detección (urgencia > campesino > técnico > general)
        if any(word in pregunta_lower for word in urgencia_indicators):
            contexto = "urgencia"
        elif any(word in pregunta_lower for word in campesino_indicators):
            contexto = "campesino"
        elif any(word in pregunta_lower for word in tecnico_indicators):
            contexto = "tecnico"
        else:
            contexto = "general"
        
        state["contexto_detectado"] = contexto
        print(f"   🎯 Contexto detectado: {contexto.upper()}")
        
        return state
    
    def _configurar_estrategia_rag_node(self, state: RAGState) -> RAGState:
        """Nodo 2: Configurar estrategia RAG según clasificación de imagen"""
        print("⚙️ NODO 2: Configurando estrategia RAG...")
        
        clasificacion_imagen = state.get("clasificacion_imagen")
        pregunta = state["pregunta"]
        
        if clasificacion_imagen:
            if clasificacion_imagen == "Tizon_tardio":
                # RAG completo para tizón tardío (especialidad)
                state["query_rag"] = f"tizón tardío Phytophthora infestans tomate {pregunta}"
                state["n_docs"] = 4
                state["advertencia_especialidad"] = ""
                print("   🎯 RAG ESPECIALIZADO: Tizón tardío")
                
            elif clasificacion_imagen == "Sana":
                # RAG preventivo para plantas sanas
                state["query_rag"] = f"prevención enfermedades tomate manejo preventivo {pregunta}"
                state["n_docs"] = 2
                state["advertencia_especialidad"] = "\n**NOTA**: Tu planta está sana. Las siguientes son recomendaciones preventivas generales."
                print("   🌱 RAG PREVENTIVO: Planta sana")
                
            else:
                # RAG limitado para otras enfermedades
                state["query_rag"] = f"enfermedades tomate diagnóstico manejo {pregunta}"
                state["n_docs"] = 2
                state["advertencia_especialidad"] = f"\n**ESPECIALIDAD LIMITADA**: Este sistema está especializado en tizón tardío. Para '{clasificacion_imagen}' la información puede ser limitada. Se recomienda consultar especialista."
                print("   [WARNING] RAG LIMITADO: Otra enfermedad")
        else:
            # Sin imagen, RAG general
            state["query_rag"] = pregunta
            state["n_docs"] = 3
            state["advertencia_especialidad"] = ""
            print("   [INFO] RAG GENERAL: Sin imagen")
        
        return state
    
    def _buscar_documentos_node(self, state: RAGState) -> RAGState:
        """Nodo 3: Buscar documentos relevantes en ChromaDB"""
        print("[SEARCH] NODO 3: Buscando documentos en ChromaDB...")
        
        try:
            # Verificar que ChromaDB esté disponible
            if self.collection is None:
                state["error"] = "Error en búsqueda RAG: ChromaDB no inicializado correctamente"
                print("   [ERROR] ChromaDB no inicializado")
                return state
                
            query_rag = state["query_rag"]
            n_docs = state["n_docs"]
            
            # Búsqueda RAG adaptada
            results = self.collection.query(query_texts=[query_rag], n_results=n_docs)
            documents = results['documents'][0] if results['documents'] else []
            context = "\n\n".join(documents[:n_docs])
            
            state["documentos_encontrados"] = documents
            state["contexto_rag"] = context
            
            print(f"   [INFO] Documentos encontrados: {len(documents)}")
            
        except Exception as e:
            state["error"] = f"Error en búsqueda RAG: {e}"
            print(f"   [ERROR] {e}")
        
        return state
    
    def _generar_respuesta_node(self, state: RAGState) -> RAGState:
        """Nodo 4: Generar respuesta final con LLM"""
        print("[LLM] NODO 4: Generando respuesta con LLM...")
        
        try:
            if state.get("error"):
                state["respuesta_final"] = f"Error: {state['error']}"
                return state
            
            pregunta = state["pregunta"]
            context = state["contexto_rag"]
            advertencia = state["advertencia_especialidad"]
            contexto_usuario = state.get("contexto_detectado", "general")  # AHORA SI USAMOS EL CONTEXTO
            
            # PROMPT PERSONALIZADO SEGUN CONTEXTO DETECTADO CON MEMORIA
            prompt = self._construir_prompt_contextualizado(contexto_usuario, pregunta, context, state)
            
            # Generar respuesta con LLM
            respuesta = self.llm.invoke(prompt)
            
            # Agregar advertencia de especialidad si aplica
            if advertencia:
                respuesta = advertencia + "\n\n" + respuesta
            
            state["respuesta_final"] = respuesta
            print(f"   [OK] Respuesta generada con contexto {contexto_usuario}: {len(respuesta)} caracteres")

        except Exception as e:
            state["error"] = f"Error generando respuesta: {e}"
            state["respuesta_final"] = f"Error: {e}"
            print(f"   [ERROR] {e}")

        return state

    def _construir_prompt_contextualizado(self, contexto_usuario: str, pregunta: str, context: str, state: RAGState) -> str:
        """Construir prompt personalizado según el contexto del usuario con memoria conversacional"""
        print(f"   🎯 Personalizando prompt para contexto: {contexto_usuario.upper()}")
        
        # 🧠 NUEVA: Incluir contexto conversacional si está disponible
        contexto_memoria = ""
        if state.get("contexto_conversacional"):
            contexto_memoria = f"""
🧠 MEMORIA CONVERSACIONAL ACTIVA:
{state["contexto_conversacional"]}

🔄 INSTRUCCIÓN ESPECIAL: Usa esta información previa para dar continuidad y coherencia en tu respuesta.
"""
            print(f"   🧠 Incluyendo memoria conversacional en prompt")
        
        # Indicador de seguimiento
        seguimiento_info = ""
        if state.get("es_seguimiento"):
            seguimiento_info = "🔄 NOTA: Esta es una consulta de SEGUIMIENTO de un caso anterior. Proporciona continuidad."
        
        # Configuración base común
        base_info = f"""
{contexto_memoria}
{seguimiento_info}

CONTEXTO TÉCNICO DISPONIBLE:
{context[:1000]}

PREGUNTA DEL USUARIO: {pregunta}
"""
        
        if contexto_usuario == "campesino":
            return f"""Eres un agrónomo práctico que habla DIRECTO con campesinos. Tu trabajo es ser MUY CLARO y PRÁCTICO.

🎯 PERFIL USUARIO: CAMPESINO/PRODUCTOR RURAL
- Necesita soluciones INMEDIATAS y ECONÓMICAS
- Prefiere lenguaje SENCILLO, sin términos técnicos complicados
- Le importan más las ACCIONES CONCRETAS que la teoría

INSTRUCCIONES ESPECIALES PARA CAMPESINOS:
1. USA palabras simples (evita jerga científica)
2. Da PASOS CLAROS y ESPECÍFICOS (qué hacer, cómo, cuándo)
3. Menciona COSTOS APROXIMADOS cuando sea posible
4. Sugiere alternativas CASERAS o ECONÓMICAS si las hay
5. Sé DIRECTO: "Haz esto...", "Aplica...", "Compra..."

TONO: Como un veterano agrónomo rural hablándole a un productor - cercano, práctico, confiable.

ESTRUCTURA RESPUESTA:
1. "Tienes [problema]..." (confirmación directa)
2. "Para solucionarlo debes..." (acción inmediata)
3. "También te recomiendo..." (2-3 consejos adicionales prácticos)
4. "Cuidado con..." (advertencia si es necesaria)

{base_info}

RESPUESTA DIRECTA Y PRÁCTICA:"""

        elif contexto_usuario == "tecnico":
            return f"""Eres un agrónomo especialista brindando consultoría técnica profesional.

🎯 PERFIL USUARIO: TÉCNICO/ESPECIALISTA AGRÍCOLA
- Maneja conocimientos avanzados en fitosanidad
- Busca información precisa y fundamentada científicamente
- Necesita detalles técnicos, mecanismos de acción, resistencias

INSTRUCCIONES ESPECIALES PARA TÉCNICOS:
1. USA terminología científica apropiada (nombres latinos, ingredientes activos)
2. EXPLICA mecanismos de acción y fundamentos
3. Menciona ESTUDIOS o EVIDENCIA cuando sea relevante
4. Incluye consideraciones de RESISTENCIA y MANEJO INTEGRADO
5. Proporciona DOSIS exactas y especificaciones técnicas

TONO: Profesional-a-profesional, técnico pero accesible, respaldado en ciencia.

ESTRUCTURA RESPUESTA:
1. "El problema corresponde a..." (diagnóstico técnico)
2. "El mecanismo involucra..." (explicación científica)
3. "Las opciones de manejo incluyen..." (alternativas técnicas)
4. "Consideraciones importantes..." (manejo integrado/resistencias)

{base_info}

RESPUESTA TÉCNICA ESPECIALIZADA:"""

        elif contexto_usuario == "urgencia":
            return f"""Eres un agrónomo de EMERGENCIA. El productor tiene una situación CRÍTICA que requiere acción INMEDIATA.

🚨 PERFIL USUARIO: SITUACIÓN DE URGENCIA
- La enfermedad se está extendiendo AHORA
- Necesita solución INMEDIATA o perderá cultivo
- No tiene tiempo para explicaciones largas

INSTRUCCIONES PARA URGENCIAS:
1. PRIORIZA la acción más RÁPIDA y EFECTIVA
2. Da INSTRUCCIONES PASO A PASO para acción inmediata
3. Sé CONCISO pero COMPLETO en la solución
4. Menciona qué hacer HOY, MAÑANA, y ESTA SEMANA
5. Si es grave, recomienda consultar especialista local

TONO: Urgente pero calmado, directo, orientado a la acción inmediata.

ESTRUCTURA RESPUESTA:
1. "ACCIÓN INMEDIATA:" (qué hacer HOY)
2. "SIGUIENTES 48 HORAS:" (seguimiento crítico)
3. "ESTA SEMANA:" (consolidar tratamiento)
4. "PREVENIR REINCIDENCIA:" (medidas futuras)

{base_info}

RESPUESTA DE EMERGENCIA:"""

        else:  # general
            return f"""Eres un agrónomo práctico especializado en tomate. Responde de manera directa y útil.

🎯 PERFIL USUARIO: CONSULTA GENERAL
- Nivel técnico intermedio
- Busca información balanceada entre práctica y fundamentos
- Necesita solución completa pero accesible

INSTRUCCIONES GENERALES:
1. ESCUCHA lo que realmente pregunta
2. RESPONDE directamente a su consulta específica 
3. COMPLEMENTA con 2-3 recomendaciones adicionales clave
4. USA lenguaje intermedio (explica términos técnicos importantes)
5. SÉ CONCISO pero completo en información esencial

ESTRUCTURA DE RESPUESTA:
1. Confirmación del problema (1-2 líneas)
2. Solución directa a lo que pregunta (productos, dosis, cómo aplicar)
3. 2-3 medidas complementarias importantes
4. Una advertencia de seguridad si aplica
5. CIERRE: Termina con una frase de cierre completa

TONO: Como un agrónomo experimentado hablando con un productor - profesional pero cercano, directo pero servicial.

{base_info}

RESPUESTA COMO AGRÓNOMO ESPECIALISTA:"""
    
    def _build_graph(self):
        """Construir el grafo LangGraph"""
        print("[BUILD] Construyendo grafo LangGraph...")
        
        # Crear el grafo
        workflow = StateGraph(RAGState)
        
        # Agregar nodos
        workflow.add_node("detectar_contexto", self._detectar_contexto_node)
        workflow.add_node("configurar_estrategia", self._configurar_estrategia_rag_node)
        workflow.add_node("buscar_documentos", self._buscar_documentos_node)
        workflow.add_node("generar_respuesta", self._generar_respuesta_node)
        
        # Definir flujo secuencial
        workflow.set_entry_point("detectar_contexto")
        workflow.add_edge("detectar_contexto", "configurar_estrategia")
        workflow.add_edge("configurar_estrategia", "buscar_documentos")
        workflow.add_edge("buscar_documentos", "generar_respuesta")
        workflow.add_edge("generar_respuesta", END)
        
        # Compilar grafo
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        
        print("[OK] Grafo LangGraph compilado")
    
    def procesar_consulta(self, pregunta: str, clasificacion_imagen: Optional[str] = None) -> Dict[str, Any]:
        """Procesar consulta usando el grafo LangGraph con memoria conversacional"""
        print(f"\n🚀 INICIANDO FLUJO LANGGRAPH")
        print(f"📝 Pregunta: {pregunta}")
        if clasificacion_imagen:
            print(f"🖼️ Clasificación imagen: {clasificacion_imagen}")
        
        # 🧠 NUEVA: Detectar contexto conversacional en la pregunta
        contexto_conversacional = None
        es_seguimiento = False
        
        if "PREGUNTA DE SEGUIMIENTO DETECTADA:" in pregunta:
            es_seguimiento = True
            contexto_conversacional = pregunta
            # Extraer la pregunta real
            lines = pregunta.split('\n')
            for line in lines:
                if line.startswith('PREGUNTA ACTUAL:'):
                    pregunta = line.replace('PREGUNTA ACTUAL:', '').strip()
                    break
            print(f"🧠 SEGUIMIENTO DETECTADO: {pregunta}")
        elif "CONTEXTO DE CONVERSACIONES PREVIAS:" in pregunta:
            contexto_conversacional = pregunta
            lines = pregunta.split('\n')
            for line in lines:
                if line.startswith('PREGUNTA ACTUAL:'):
                    pregunta = line.replace('PREGUNTA ACTUAL:', '').strip()
                    break
            print(f"🧠 CONTEXTO CONVERSACIONAL INCLUIDO: {pregunta}")
        
        print("=" * 60)
        
        # Estado inicial
        initial_state = {
            "pregunta": pregunta,
            "clasificacion_imagen": clasificacion_imagen,
            "contexto_conversacional": contexto_conversacional,
            "es_seguimiento": es_seguimiento,
            "contexto_detectado": "",
            "query_rag": "",
            "n_docs": 0,
            "advertencia_especialidad": "",
            "documentos_encontrados": [],
            "contexto_rag": "",
            "respuesta_final": "",
            "error": None
        }
        
        # Ejecutar grafo
        try:
            config = {"configurable": {"thread_id": "demo_thread"}}
            final_state = self.graph.invoke(initial_state, config)
            
            print("=" * 60)
            print("[DONE] FLUJO COMPLETADO")
            print(f"[OK] Contexto detectado: {final_state['contexto_detectado']}")
            print(f"[INFO] Documentos encontrados: {len(final_state['documentos_encontrados'])}")
            print(f"[INFO] Respuesta: {len(final_state['respuesta_final'])} caracteres")
            
            return final_state
            
        except Exception as e:
            print(f"[ERROR] Error en flujo: {e}")
            return {"error": str(e), "respuesta_final": f"Error: {e}"}
    
    def mostrar_estructura_grafo(self):
        """Mostrar la estructura del grafo"""
        print("\n🌐 ESTRUCTURA DEL GRAFO LANGGRAPH")
        print("=" * 50)
        print("""
        ┌─────────────────┐
        │  INPUT:         │
        │  - pregunta     │
        │  - clasificación│
        └─────────┬───────┘
                  │
        ┌─────────▼───────┐
        │ NODO 1:         │
        │ Detectar        │
        │ Contexto        │
        └─────────┬───────┘
                  │
        ┌─────────▼───────┐
        │ NODO 2:         │
        │ Configurar      │
        │ Estrategia RAG  │
        └─────────┬───────┘
                  │
        ┌─────────▼───────┐
        │ NODO 3:         │
        │ Buscar          │
        │ Documentos      │
        └─────────┬───────┘
                  │
        ┌─────────▼───────┐
        │ NODO 4:         │
        │ Generar         │
        │ Respuesta       │
        └─────────┬───────┘
                  │
        ┌─────────▼───────┐
        │ OUTPUT:         │
        │ Respuesta Final │
        └─────────────────┘
        """)
        
        print("[FLOW] FLUJO DE DATOS:")
        print("1. [DETECT] DETECTAR CONTEXTO -> contexto_detectado")
        print("2. [CONFIG] CONFIGURAR ESTRATEGIA -> query_rag, n_docs, advertencia")
        print("3. [SEARCH] BUSCAR DOCUMENTOS -> documentos_encontrados, contexto_rag")
        print("4. [LLM] GENERAR RESPUESTA -> respuesta_final")
        
        print("\n[INFO] VENTAJAS DE LANGGRAPH:")
        print("[OK] Flujo visual y claro")
        print("[OK] Estado persistente entre nodos")
        print("[OK] Facil debugging paso a paso")
        print("[OK] Posibilidad de agregar rutas condicionales")
        print("[OK] Checkpoints para recuperacion")

def demo_interactivo():
    """Demo interactivo del sistema LangGraph"""
    print("[DEMO] LANGGRAPH - SISTEMA RAG")
    print("=" * 40)
    
    # Inicializar sistema
    sistema = SistemaRAGLangGraph()
    
    if not sistema.collection or not sistema.llm:
        print("[ERROR] Sistema no pudo inicializarse")
        return
    
    # Mostrar estructura
    sistema.mostrar_estructura_grafo()
    
    while True:
        print(f"\n{'='*60}")
        print("👉 Opciones:")
        print("1. 🧪 Probar con pregunta personalizada")
        print("2. 🎯 Demo automático con casos de prueba")
        print("3. 📊 Ver estructura del grafo")
        print("4. 🚪 Salir")
        
        opcion = input("\n📝 Elige (1-4): ").strip()
        
        if opcion == "1":
            pregunta = input("❓ Tu pregunta: ").strip()
            if not pregunta:
                print("⚠️ Por favor escribe una pregunta")
                continue
            
            # Preguntar por clasificación de imagen (opcional)
            print("\n🖼️ ¿Hay clasificación de imagen? (opcional)")
            print("1. Tizon_tardio")
            print("2. Sana") 
            print("3. Otra_enfermedad")
            print("4. Sin imagen")
            
            img_choice = input("Elige (1-4, Enter=sin imagen): ").strip()
            
            clasificacion = None
            if img_choice == "1":
                clasificacion = "Tizon_tardio"
            elif img_choice == "2":
                clasificacion = "Sana"
            elif img_choice == "3":
                clasificacion = "Otra_enfermedad"
            
            # Procesar consulta
            resultado = sistema.procesar_consulta(pregunta, clasificacion)
            
            print(f"\n💡 RESPUESTA FINAL:")
            print("=" * 70)
            print(resultado["respuesta_final"])
            print("=" * 70)
            
        elif opcion == "2":
            print("\n🧪 DEMO AUTOMÁTICO - CASOS DE PRUEBA")
            
            casos = [
                {
                    "pregunta": "Se me está dañando el tomate con manchas",
                    "imagen": "Tizon_tardio",
                    "descripcion": "Caso campesino + tizón tardío"
                },
                {
                    "pregunta": "¿Cuál es el mecanismo de resistencia de Phytophthora?",
                    "imagen": None,
                    "descripcion": "Caso técnico sin imagen"
                },
                {
                    "pregunta": "¿Cómo mantengo mi planta sana?",
                    "imagen": "Sana",
                    "descripcion": "Caso preventivo"
                }
            ]
            
            for i, caso in enumerate(casos, 1):
                print(f"\n--- CASO {i}: {caso['descripcion']} ---")
                resultado = sistema.procesar_consulta(caso["pregunta"], caso["imagen"])
                print(f"Preview: {resultado['respuesta_final'][:150]}...")
                
                if i < len(casos):
                    input("Presiona ENTER para continuar...")
            
        elif opcion == "3":
            sistema.mostrar_estructura_grafo()
            
        elif opcion == "4":
            print("\n👋 ¡Gracias por probar el demo LangGraph!")
            break
            
        else:
            print("⚠️ Opción inválida")

if __name__ == "__main__":
    demo_interactivo()