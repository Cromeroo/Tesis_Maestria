import os
import time
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config.settings import Settings as AppSettings

class VectorStore:
    """Componente para manejo de ChromaDB"""
    
    def __init__(self, db_path: str = None):
        """
        Inicializa el vector store
        
        Args:
            db_path: Ruta a la base de datos ChromaDB
        """
        self.db_path = db_path or str(AppSettings.CHROMA_DB_DIR)
        self.client = self._initialize_client()
        self.embedding_function = self._initialize_embedding_function()
    
    def _initialize_client(self) -> chromadb.PersistentClient:
        """Inicializa el cliente de ChromaDB"""
        try:
            client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            print(f"✅ Cliente ChromaDB inicializado en: {self.db_path}")
            return client
        except Exception as e:
            print(f"❌ Error al inicializar ChromaDB: {e}")
            raise
    
    def _initialize_embedding_function(self):
        """Inicializa la función de embeddings con manejo robusto de errores"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Limpiar cache si hay problemas
                if attempt > 0:
                    import torch
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=AppSettings.EMBEDDING_MODEL,
                    device="cpu"  # Forzar CPU para evitar problemas de GPU
                )
                print(f"🔍 Función de embeddings inicializada: {AppSettings.EMBEDDING_MODEL}")
                return embedding_function
                
            except Exception as e:
                print(f"⚠️ Intento {attempt + 1} falló: {e}")
                if attempt == max_retries - 1:
                    # Último intento con modelo alternativo
                    try:
                        print("🔄 Intentando con modelo de embeddings alternativo...")
                        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                            model_name="all-MiniLM-L6-v2",  # Modelo más ligero
                            device="cpu"
                        )
                        print("✅ Modelo alternativo inicializado correctamente")
                        return embedding_function
                    except Exception as e2:
                        print(f"❌ Error crítico con embeddings: {e2}")
                        raise e2
                time.sleep(1)  # Esperar antes del siguiente intento
    
    def create_collection(self, collection_name: str, description: str = None) -> chromadb.Collection:
        """
        Crea una nueva colección
        
        Args:
            collection_name: Nombre de la colección
            description: Descripción de la colección
            
        Returns:
            Colección creada
        """
        try:
            # Eliminar colección existente si existe
            try:
                self.client.delete_collection(collection_name)
                print(f"🗑️ Colección '{collection_name}' eliminada para recrear")
            except:
                pass
            
            # Crear nueva colección
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": description or AppSettings.COLLECTION_DESCRIPTION},
                embedding_function=self.embedding_function
            )
            
            print(f"✅ Colección '{collection_name}' creada exitosamente")
            return collection
            
        except Exception as e:
            print(f"❌ Error al crear colección '{collection_name}': {e}")
            raise
    
    def get_or_create_collection(self, collection_name: str, description: str = None) -> chromadb.Collection:
        """
        Obtiene una colección existente o la crea si no existe
        
        Args:
            collection_name: Nombre de la colección
            description: Descripción de la colección
            
        Returns:
            Colección obtenida o creada
        """
        try:
            # Intentar obtener colección existente
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"✅ Colección '{collection_name}' obtenida (existente)")
            return collection
            
        except Exception:
            # Si no existe, crearla
            print(f"🔄 Colección '{collection_name}' no existe, creándola...")
            return self.create_collection(collection_name, description)
    
    def get_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Obtiene una colección existente
        
        Args:
            collection_name: Nombre de la colección
            
        Returns:
            Colección existente
        """
        try:
            collection = self.client.get_collection(collection_name)
            print(f"✅ Colección '{collection_name}' obtenida")
            return collection
        except Exception as e:
            print(f"❌ Error al obtener colección '{collection_name}': {e}")
            raise
    
    def add_documents(self, collection: chromadb.Collection, documents: List[Dict]) -> int:
        """
        Agrega documentos a la colección
        
        Args:
            collection: Colección de ChromaDB
            documents: Lista de documentos con texto y metadatos
            
        Returns:
            Número total de chunks agregados
        """
        if not documents:
            print("⚠️ No hay documentos para agregar")
            return 0
        
        total_chunks = 0
        
        try:
            print(f"📚 Agregando documentos a la colección...")
            
            for doc in documents:
                if not doc or not doc.get("texto_limpio"):
                    continue
                
                # Crear metadatos para el documento
                metadata = {
                    "archivo": doc["archivo_original"],
                    "idioma_origen": doc["idioma_origen"],
                    "fecha_procesamiento": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "caracteres_originales": doc.get("caracteres_originales", 0),
                    "caracteres_finales": doc.get("caracteres_finales", 0)
                }
                
                # Dividir texto en chunks
                from components.text_chunker import TextChunker
                chunker = TextChunker()
                chunks = chunker.create_chunks(doc["texto_limpio"], metadata)
                
                # Insertar chunks en ChromaDB
                for chunk in chunks:
                    try:
                        collection.add(
                            documents=[chunk["text"]],
                            metadatas=[chunk["metadata"]],
                            ids=[f"{doc['archivo_original']}_chunk_{chunk['metadata']['chunk_id']}"]
                        )
                    except Exception as chunk_error:
                        print(f"⚠️ Error al agregar chunk {chunk['metadata']['chunk_id']}: {chunk_error}")
                        continue
                
                total_chunks += len(chunks)
                print(f"✅ {doc['archivo_original']}: {len(chunks)} chunks agregados")
            
            print(f"🎉 Documentos agregados exitosamente")
            print(f"📊 Total de chunks: {total_chunks}")
            print(f"📊 Total en colección: {collection.count()}")
            
            return total_chunks
            
        except Exception as e:
            print(f"❌ Error al agregar documentos: {e}")
            raise
    
    def search_documents(self, collection: chromadb.Collection, query: str, 
                        n_results: int = 5) -> List[Dict]:
        """
        Busca documentos similares en la colección
        
        Args:
            collection: Colección de ChromaDB
            query: Consulta de búsqueda
            n_results: Número de resultados a retornar
            
        Returns:
            Lista de resultados de búsqueda
        """
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Formatear resultados
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                        "rank": i + 1
                    })
            
            print(f"🔍 Búsqueda completada: {len(formatted_results)} resultados")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error en la búsqueda: {e}")
            return []
    
    def get_collection_info(self, collection: chromadb.Collection) -> Dict:
        """
        Obtiene información de la colección
        
        Args:
            collection: Colección de ChromaDB
            
        Returns:
            Diccionario con información de la colección
        """
        try:
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata,
                "embedding_function": str(self.embedding_function)
            }
        except Exception as e:
            print(f"❌ Error al obtener información de la colección: {e}")
            return {}
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Elimina una colección
        
        Args:
            collection_name: Nombre de la colección a eliminar
            
        Returns:
            True si se eliminó exitosamente, False en caso contrario
        """
        try:
            self.client.delete_collection(collection_name)
            print(f"✅ Colección '{collection_name}' eliminada")
            return True
        except Exception as e:
            print(f"❌ Error al eliminar colección '{collection_name}': {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        Lista todas las colecciones disponibles
        
        Returns:
            Lista de nombres de colecciones
        """
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            print(f"📚 Colecciones disponibles: {collection_names}")
            return collection_names
        except Exception as e:
            print(f"❌ Error al listar colecciones: {e}")
            return [] 