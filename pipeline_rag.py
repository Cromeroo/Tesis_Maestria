import os
import fitz  # PyMuPDF
import re
import unidecode
import json
from typing import List, Dict, Optional
from langdetect import detect, LangDetectException
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import time

# Cargar variables de entorno
load_dotenv()

class PipelineRAG:
    def __init__(self, carpeta_pdfs: str, output_dir: str = None):
        """
        Inicializa el pipeline RAG para procesar documentos PDF
        
        Args:
            carpeta_pdfs: Ruta a la carpeta con documentos PDF
            output_dir: Carpeta de salida (opcional)
        """
        self.carpeta_pdfs = carpeta_pdfs
        self.output_dir = output_dir or os.path.join(carpeta_pdfs, "textos_procesados")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurar Gemini
        self.llm = self._configurar_gemini()
        
        # Configurar ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Configurar text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print("🚀 Pipeline RAG inicializado correctamente")
    
    def _configurar_gemini(self) -> Optional[ChatVertexAI]:
        """Configura Gemini usando las credenciales de Google Cloud"""
        try:
            # Configurar credenciales
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\danil\OneDrive\Escritorio\Tesis\credentials.json"
            
            # Inicializar el modelo
            llm = ChatVertexAI(
                model_name="gemini-2.5-flash",
                temperature=0.3,  # Temperatura baja para traducciones más consistentes
                max_output_tokens=2048,
                location="us-central1"
            )
            
            # Prueba de conexión
            response = llm.invoke("Hola, prueba de conexión.")
            print("✅ Gemini configurado exitosamente")
            return llm
            
        except Exception as e:
            print(f"❌ Error al configurar Gemini: {e}")
            return None
    
    def detectar_idioma(self, texto: str) -> str:
        """
        Detecta el idioma del texto
        
        Args:
            texto: Texto a analizar
            
        Returns:
            Código de idioma (ej: 'en', 'es')
        """
        try:
            # Tomar una muestra del texto para detectar idioma
            muestra = texto[:1000] if len(texto) > 1000 else texto
            idioma = detect(muestra)
            return idioma
        except LangDetectException:
            # Si no puede detectar, asumir inglés
            return 'en'
    
    def traducir_texto(self, texto: str, idioma_origen: str) -> str:
        """
        Traduce texto del idioma origen al español usando Gemini
        
        Args:
            texto: Texto a traducir
            idioma_origen: Código del idioma origen
            
        Returns:
            Texto traducido al español
        """
        if not self.llm:
            print("⚠️ Gemini no está configurado, devolviendo texto original")
            return texto
        
        if idioma_origen == 'es':
            return texto  # Ya está en español
        
        try:
            prompt = f"""
            Traduce el siguiente texto del {idioma_origen} al español.
            
            Instrucciones importantes:
            1. Mantén la terminología técnica y científica correcta
            2. Preserva el formato y estructura del texto
            3. Usa un español claro y profesional
            4. Si hay términos técnicos específicos, mantenlos en español cuando sea apropiado
            
            Texto a traducir:
            {texto}
            
            Traducción al español:
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"❌ Error en la traducción: {e}")
            return texto  # Devolver original si falla
    
    def limpiar_texto(self, texto: str) -> str:
        """
        Limpia y normaliza el texto
        
        Args:
            texto: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        # Remover acentos
        texto = unidecode.unidecode(texto)
        
        # Normalizar espacios y saltos de línea
        texto = texto.replace('\n', ' ')
        texto = re.sub(r'\s+', ' ', texto)
        
        # Remover referencias de página
        texto = re.sub(r'Page \d+', '', texto, flags=re.IGNORECASE)
        texto = re.sub(r'Página \d+', '', texto, flags=re.IGNORECASE)
        
        # Remover caracteres especiales innecesarios
        texto = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]]', '', texto)
        
        return texto.strip()
    
    def extraer_texto_pdf(self, ruta_pdf: str) -> str:
        """
        Extrae texto de un archivo PDF
        
        Args:
            ruta_pdf: Ruta al archivo PDF
            
        Returns:
            Texto extraído
        """
        try:
            doc = fitz.open(ruta_pdf)
            texto_total = ""
            
            for page in doc:
                texto_total += page.get_text()
            
            doc.close()
            return texto_total
            
        except Exception as e:
            print(f"❌ Error al extraer texto de {ruta_pdf}: {e}")
            return ""
    
    def procesar_documento(self, ruta_pdf: str) -> Dict:
        """
        Procesa un documento PDF completo
        
        Args:
            ruta_pdf: Ruta al archivo PDF
            
        Returns:
            Diccionario con información del documento procesado
        """
        nombre_archivo = os.path.basename(ruta_pdf)
        print(f"📄 Procesando: {nombre_archivo}")
        
        # Extraer texto
        texto_original = self.extraer_texto_pdf(ruta_pdf)
        if not texto_original:
            print(f"⚠️ No se pudo extraer texto de {nombre_archivo}")
            return None
        
        # Detectar idioma
        idioma = self.detectar_idioma(texto_original)
        print(f"🌍 Idioma detectado: {idioma}")
        
        # Traducir si es necesario
        if idioma != 'es':
            print(f"🔄 Traduciendo de {idioma} a español...")
            texto_traducido = self.traducir_texto(texto_original, idioma)
            time.sleep(1)  # Pausa para evitar rate limits
        else:
            texto_traducido = texto_original
        
        # Limpiar texto
        texto_limpio = self.limpiar_texto(texto_traducido)
        
        # Guardar archivo procesado
        nombre_txt = os.path.splitext(nombre_archivo)[0] + "__procesado.txt"
        ruta_txt = os.path.join(self.output_dir, nombre_txt)
        
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(texto_limpio)
        
        print(f"✅ Guardado: {ruta_txt}")
        
        return {
            "archivo_original": nombre_archivo,
            "archivo_procesado": nombre_txt,
            "idioma_origen": idioma,
            "texto_limpio": texto_limpio,
            "ruta_procesado": ruta_txt
        }
    
    def crear_chunks(self, texto: str, metadata: Dict) -> List[Dict]:
        """
        Divide el texto en chunks para ChromaDB
        
        Args:
            texto: Texto a dividir
            metadata: Metadatos del documento
            
        Returns:
            Lista de chunks con metadatos
        """
        chunks = self.text_splitter.split_text(texto)
        
        documentos_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
            
            documentos_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return documentos_chunks
    
    def cargar_en_chromadb(self, documentos: List[Dict], collection_name: str = "tizon_tardio"):
        """
        Carga los documentos procesados en ChromaDB
        
        Args:
            documentos: Lista de documentos procesados
            collection_name: Nombre de la colección en ChromaDB
        """
        try:
            # Crear o obtener colección
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Base de conocimiento sobre tizón tardío en tomate"}
            )
            
            print(f"📚 Cargando documentos en ChromaDB...")
            
            for doc in documentos:
                if not doc or not doc.get("texto_limpio"):
                    continue
                
                # Crear chunks
                metadata = {
                    "archivo": doc["archivo_original"],
                    "idioma_origen": doc["idioma_origen"],
                    "fecha_procesamiento": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                chunks = self.crear_chunks(doc["texto_limpio"], metadata)
                
                # Insertar chunks en ChromaDB
                for chunk in chunks:
                    collection.add(
                        documents=[chunk["text"]],
                        metadatas=[chunk["metadata"]],
                        ids=[f"{doc['archivo_original']}_chunk_{chunk['metadata']['chunk_id']}"]
                    )
                
                print(f"✅ {doc['archivo_original']}: {len(chunks)} chunks agregados")
            
            print(f"🎉 Base de conocimiento creada exitosamente en ChromaDB")
            print(f"📊 Total de documentos en la colección: {collection.count()}")
            
        except Exception as e:
            print(f"❌ Error al cargar en ChromaDB: {e}")
    
    def ejecutar_pipeline(self):
        """
        Ejecuta el pipeline completo de procesamiento
        """
        print("🚀 Iniciando pipeline de procesamiento RAG...")
        
        documentos_procesados = []
        
        # Procesar todos los archivos PDF
        for archivo in os.listdir(self.carpeta_pdfs):
            if archivo.lower().endswith(".pdf"):
                ruta_pdf = os.path.join(self.carpeta_pdfs, archivo)
                documento = self.procesar_documento(ruta_pdf)
                
                if documento:
                    documentos_procesados.append(documento)
        
        print(f"\n📊 Resumen del procesamiento:")
        print(f"   - Documentos procesados: {len(documentos_procesados)}")
        
        # Cargar en ChromaDB
        if documentos_procesados:
            self.cargar_en_chromadb(documentos_procesados)
        
        # Guardar metadatos del procesamiento
        metadata_file = os.path.join(self.output_dir, "metadata_procesamiento.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "fecha_procesamiento": time.strftime("%Y-%m-%d %H:%M:%S"),
                "documentos_procesados": len(documentos_procesados),
                "documentos": documentos_procesados
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Pipeline completado. Metadatos guardados en: {metadata_file}")

def main():
    """Función principal para ejecutar el pipeline"""
    carpeta_pdfs = r"C:\Users\danil\OneDrive\Escritorio\Tesis\Documentos"
    
    # Crear y ejecutar pipeline
    pipeline = PipelineRAG(carpeta_pdfs)
    pipeline.ejecutar_pipeline()

if __name__ == "__main__":
    main() 