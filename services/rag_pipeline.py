from typing import List
from pathlib import Path

from services.document_processor import DocumentProcessor
from components.vector_store import VectorStore
from models.document import ProcessedDocument
from config.settings import Settings

class RAGPipeline:
    """Pipeline principal del sistema RAG"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        # Intentar cargar colección existente
        try:
            self.collection = self.vector_store.get_collection(Settings.COLLECTION_NAME)
        except:
            self.collection = None
    
    def execute_full_pipeline(self, pdf_directory: str = None) -> bool:
        """
        Ejecuta el pipeline completo de procesamiento RAG
        
        Args:
            pdf_directory: Directorio con archivos PDF (opcional)
            
        Returns:
            True si el pipeline se ejecutó exitosamente, False en caso contrario
        """
        try:
            # Usar directorio por defecto si no se especifica
            if not pdf_directory:
                pdf_directory = str(Settings.DOCUMENTS_DIR)
            
            print("🚀 Iniciando pipeline de procesamiento RAG...")
            print("=" * 80)
            
            # Paso 1: Procesar documentos
            print("📚 PASO 1: Procesamiento de documentos")
            processed_documents = self._process_documents(pdf_directory)
            
            if not processed_documents:
                print("❌ No se procesaron documentos, abortando pipeline")
                return False
            
            # Paso 2: Crear base de conocimiento vectorial
            print("\n🔍 PASO 2: Creación de base de conocimiento vectorial")
            success = self._create_vector_knowledge_base(processed_documents)
            
            if not success:
                print("❌ Error al crear base de conocimiento vectorial")
                return False
            
            # Paso 3: Generar reportes
            print("\n📊 PASO 3: Generación de reportes")
            self._generate_reports(processed_documents)
            
            print("\n🎉 Pipeline RAG completado exitosamente!")
            return True
            
        except Exception as e:
            print(f"❌ Error en el pipeline RAG: {e}")
            return False
    
    def _process_documents(self, pdf_directory: str) -> List[ProcessedDocument]:
        """
        Procesa los documentos PDF
        
        Args:
            pdf_directory: Directorio con archivos PDF
            
        Returns:
            Lista de documentos procesados
        """
        print(f"📁 Procesando documentos desde: {pdf_directory}")
        
        # Verificar que el directorio existe
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            print(f"❌ Directorio no encontrado: {pdf_directory}")
            return []
        
        # Contar archivos PDF
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"📊 Total de archivos PDF encontrados: {len(pdf_files)}")
        
        if not pdf_files:
            print("⚠️ No se encontraron archivos PDF para procesar")
            return []
        
        # Procesar documentos
        processed_documents = self.document_processor.process_multiple_documents(pdf_directory)
        
        print(f"✅ Documentos procesados exitosamente: {len(processed_documents)}")
        return processed_documents
    
    def _create_vector_knowledge_base(self, documents: List[ProcessedDocument]) -> bool:
        """
        Crea la base de conocimiento vectorial
        
        Args:
            documents: Lista de documentos procesados
            
        Returns:
            True si se creó exitosamente, False en caso contrario
        """
        try:
            print("🔍 Creando base de conocimiento vectorial...")
            
            # Crear colección en ChromaDB
            self.collection = self.vector_store.create_collection(
                collection_name=Settings.COLLECTION_NAME,
                description=Settings.COLLECTION_DESCRIPTION
            )
            
            # Convertir documentos al formato esperado por ChromaDB
            chroma_documents = []
            for doc in documents:
                chroma_doc = {
                    "archivo_original": doc.original_filename,
                    "idioma_origen": doc.source_language,
                    "texto_limpio": doc.clean_text,
                    "caracteres_originales": doc.original_char_count,
                    "caracteres_finales": doc.final_char_count
                }
                chroma_documents.append(chroma_doc)
            
            # Agregar documentos a la colección
            total_chunks = self.vector_store.add_documents(self.collection, chroma_documents)
            
            if total_chunks > 0:
                print(f"✅ Base de conocimiento creada con {total_chunks} chunks")
                return True
            else:
                print("❌ No se pudieron agregar chunks a la base de conocimiento")
                return False
                
        except Exception as e:
            print(f"❌ Error al crear base de conocimiento vectorial: {e}")
            return False
    
    def _generate_reports(self, documents: List[ProcessedDocument]):
        """Genera reportes del procesamiento"""
        try:
            print("📊 Generando reportes...")
            
            # Guardar metadatos del procesamiento
            total_pdfs = len(list(Path(Settings.DOCUMENTS_DIR).glob("*.pdf")))
            metadata_file = self.document_processor.save_processing_metadata(documents, total_pdfs)
            
            # Obtener estadísticas
            stats = self.document_processor.get_processing_statistics(documents)
            
            # Imprimir resumen final
            self._print_final_summary(documents, stats, total_pdfs)
            
            print(f"✅ Reportes generados exitosamente")
            
        except Exception as e:
            print(f"⚠️ Error al generar reportes: {e}")
    
    def _print_final_summary(self, documents: List[ProcessedDocument], stats: dict, total_pdfs: int):
        """Imprime resumen final del procesamiento"""
        print(f"\n📊 RESUMEN FINAL DEL PROCESAMIENTO:")
        print(f"   - Archivos PDF encontrados: {total_pdfs}")
        print(f"   - Documentos procesados exitosamente: {len(documents)}")
        
        if stats:
            print(f"   - Total de caracteres originales: {stats.get('total_characters_original', 0):,}")
            print(f"   - Total de caracteres finales: {stats.get('total_characters_final', 0):,}")
            print(f"   - Ratio de compresión: {stats.get('compression_ratio', 0):.1f}%")
            
            # Distribución por idioma
            lang_dist = stats.get('language_distribution', {})
            if lang_dist:
                print(f"   - Distribución por idioma:")
                for lang, lang_stats in lang_dist.items():
                    print(f"     * {lang}: {lang_stats['count']} documentos "
                          f"({lang_stats['avg_chars']:.0f} chars promedio)")
        
        # Información de la base de conocimiento
        if self.collection:
            collection_info = self.vector_store.get_collection_info(self.collection)
            print(f"   - Base de conocimiento: {collection_info.get('name', 'N/A')}")
            print(f"   - Total de chunks: {collection_info.get('count', 0)}")
    
    def search_knowledge_base(self, query: str, n_results: int = 5) -> List[dict]:
        """
        Busca en la base de conocimiento
        
        Args:
            query: Consulta de búsqueda
            n_results: Número de resultados a retornar
            
        Returns:
            Lista de resultados de búsqueda
        """
        if not self.collection:
            print("❌ Base de conocimiento no disponible")
            return []
        
        try:
            print(f"🔍 Buscando: '{query}'")
            results = self.vector_store.search_documents(self.collection, query, n_results)
            
            if results:
                print(f"✅ Búsqueda completada: {len(results)} resultados encontrados")
                for i, result in enumerate(results):
                    print(f"   {i+1}. {result['metadata'].get('archivo', 'N/A')} "
                          f"(distancia: {result['distance']:.3f})")
            else:
                print("⚠️ No se encontraron resultados")
            
            return results
            
        except Exception as e:
            print(f"❌ Error en la búsqueda: {e}")
            return []
    
    def get_knowledge_base_info(self) -> dict:
        """
        Obtiene información de la base de conocimiento
        
        Returns:
            Diccionario con información de la base de conocimiento
        """
        if not self.collection:
            return {"error": "Base de conocimiento no disponible"}
        
        try:
            return self.vector_store.get_collection_info(self.collection)
        except Exception as e:
            return {"error": f"Error al obtener información: {e}"}
    
    def reset_knowledge_base(self) -> bool:
        """
        Reinicia la base de conocimiento
        
        Returns:
            True si se reinició exitosamente, False en caso contrario
        """
        try:
            if self.collection:
                collection_name = self.collection.name
                success = self.vector_store.delete_collection(collection_name)
                if success:
                    self.collection = None
                    print(f"✅ Base de conocimiento '{collection_name}' reiniciada")
                    return True
                else:
                    print(f"❌ Error al reiniciar base de conocimiento")
                    return False
            else:
                print("⚠️ No hay base de conocimiento activa para reiniciar")
                return True
                
        except Exception as e:
            print(f"❌ Error al reiniciar base de conocimiento: {e}")
            return False 