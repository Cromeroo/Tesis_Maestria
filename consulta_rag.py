import chromadb
from chromadb.config import Settings
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class ConsultaRAG:
    def __init__(self, collection_name: str = "tizon_tardio"):
        """
        Inicializa el sistema de consulta RAG
        
        Args:
            collection_name: Nombre de la colección en ChromaDB
        """
        # Configurar ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Obtener colección
        self.collection = self.chroma_client.get_collection(name=collection_name)
        
        # Configurar Gemini
        self.llm = self._configurar_gemini()
        
        print("🔍 Sistema de consulta RAG inicializado")
    
    def _configurar_gemini(self):
        """Configura Gemini para generar respuestas"""
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\danil\OneDrive\Escritorio\Tesis\credentials.json"
            
            llm = ChatVertexAI(
                model_name="gemini-2.5-flash",
                temperature=0.7,
                max_output_tokens=2048,
                location="us-central1"
            )
            
            return llm
            
        except Exception as e:
            print(f"❌ Error al configurar Gemini: {e}")
            return None
    
    def buscar_documentos(self, consulta: str, n_results: int = 5):
        """
        Busca documentos relevantes en la base de conocimiento
        
        Args:
            consulta: Consulta del usuario
            n_results: Número de resultados a retornar
            
        Returns:
            Resultados de la búsqueda
        """
        try:
            results = self.collection.query(
                query_texts=[consulta],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error en la búsqueda: {e}")
            return None
    
    def generar_respuesta(self, consulta: str, documentos_relevantes: list):
        """
        Genera una respuesta basada en los documentos encontrados
        
        Args:
            consulta: Consulta original del usuario
            documentos_relevantes: Lista de documentos relevantes
            
        Returns:
            Respuesta generada
        """
        if not self.llm:
            return "❌ No se pudo configurar el modelo de lenguaje"
        
        # Preparar contexto
        contexto = "\n\n".join(documentos_relevantes)
        
        prompt = f"""
        Eres un experto en fitopatología especializado en tizón tardío del tomate.
        
        Basándote en la siguiente información de la base de conocimiento, responde la consulta del usuario.
        
        INFORMACIÓN DE LA BASE DE CONOCIMIENTO:
        {contexto}
        
        CONSULTA DEL USUARIO:
        {consulta}
        
        INSTRUCCIONES:
        1. Responde de manera clara y profesional en español
        2. Usa solo la información proporcionada en la base de conocimiento
        3. Si la información no es suficiente, indícalo claramente
        4. Proporciona recomendaciones prácticas cuando sea apropiado
        5. Mantén un tono educativo y útil
        
        RESPUESTA:
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"❌ Error al generar respuesta: {e}"
    
    def consultar(self, pregunta: str, n_results: int = 5):
        """
        Realiza una consulta completa al sistema RAG
        
        Args:
            pregunta: Pregunta del usuario
            n_results: Número de documentos a recuperar
            
        Returns:
            Respuesta completa
        """
        print(f"🔍 Buscando información sobre: {pregunta}")
        
        # Buscar documentos relevantes
        resultados = self.buscar_documentos(pregunta, n_results)
        
        if not resultados or not resultados['documents']:
            return "❌ No se encontraron documentos relevantes en la base de conocimiento."
        
        documentos = resultados['documents'][0]
        metadatos = resultados['metadatas'][0]
        distancias = resultados['distances'][0]
        
        print(f"📚 Encontrados {len(documentos)} documentos relevantes")
        
        # Mostrar información de los documentos encontrados
        print("\n📄 Documentos consultados:")
        for i, (doc, meta, dist) in enumerate(zip(documentos, metadatos, distancias)):
            print(f"   {i+1}. {meta.get('archivo', 'Desconocido')} (similitud: {1-dist:.3f})")
        
        # Generar respuesta
        print("\n🤖 Generando respuesta...")
        respuesta = self.generar_respuesta(pregunta, documentos)
        
        return respuesta

def main():
    """Función principal para probar el sistema de consulta"""
    consulta_rag = ConsultaRAG()
    
    # Ejemplos de consultas
    consultas_ejemplo = [
        "¿Cuáles son los síntomas del tizón tardío en tomate?",
        "¿Cómo se puede prevenir el tizón tardío?",
        "¿Qué tratamientos químicos son efectivos contra el tizón tardío?",
        "¿Cuáles son las condiciones ambientales favorables para el desarrollo del tizón tardío?"
    ]
    
    print("🧪 Probando sistema de consulta RAG...\n")
    
    for consulta in consultas_ejemplo:
        print(f"❓ Consulta: {consulta}")
        print("-" * 80)
        
        respuesta = consulta_rag.consultar(consulta)
        print(f"💡 Respuesta: {respuesta}")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 