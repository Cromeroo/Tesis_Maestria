#!/usr/bin/env python3
"""
Script para cargar documentos en ChromaDB
Procesa PDFs y los carga en la base vectorial
"""

import os
import sys
from pathlib import Path

# Configurar path
sys.path.append(str(Path(__file__).parent))

def load_documents_to_chromadb():
    """Cargar documentos procesados en ChromaDB"""
    try:
        from components.vector_store import VectorStore
        from components.text_chunker import TextChunker
        from config.settings import Settings
        
        print("🚀 INICIANDO CARGA DE DOCUMENTOS")
        print("=" * 50)
        
        # Inicializar componentes
        vector_store = VectorStore()
        text_chunker = TextChunker()
        settings = Settings()
        
        # Crear/obtener colección
        collection = vector_store.get_or_create_collection(
            settings.COLLECTION_NAME,
            settings.COLLECTION_DESCRIPTION
        )
        
        # Verificar si ya hay documentos
        current_count = collection.count()
        if current_count > 0:
            print(f"⚠️ La colección ya tiene {current_count} documentos")
            response = input("¿Desea recargar todos los documentos? (s/n): ")
            if response.lower() != 's':
                print("✅ Conservando documentos existentes")
                return True
            else:
                # Eliminar colección y recrear
                vector_store.client.delete_collection(settings.COLLECTION_NAME)
                collection = vector_store.create_collection(
                    settings.COLLECTION_NAME,
                    settings.COLLECTION_DESCRIPTION
                )
        
        # Buscar documentos procesados
        docs_path = Path("Documentos/textos_procesados")
        if not docs_path.exists():
            print("❌ No existe directorio de textos procesados")
            print("🔄 Necesitas procesar los PDFs primero")
            return False
        
        # Cargar documentos procesados
        processed_files = list(docs_path.glob("*.txt"))
        if not processed_files:
            print("❌ No hay archivos de texto procesados")
            return False
            
        print(f"📚 Encontrados {len(processed_files)} documentos procesados")
        
        # Cargar cada documento
        documents = []
        metadatas = []
        ids = []
        
        for i, file_path in enumerate(processed_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Dividir en chunks más simple - usar directamente el splitter
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = splitter.split_text(content)
                
                for j, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 50:  # Solo chunks con contenido
                        documents.append(chunk)
                        metadatas.append({
                            "source": file_path.name,
                            "chunk": j,
                            "file_type": "pdf_processed"
                        })
                        ids.append(f"{file_path.stem}_chunk_{j}")
                
                print(f"✅ {file_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"❌ Error procesando {file_path.name}: {e}")
        
        if not documents:
            print("❌ No se encontraron documentos válidos para cargar")
            return False
        
        print(f"\n🔄 Cargando {len(documents)} chunks en ChromaDB...")
        
        # Cargar en lotes para evitar problemas de memoria
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            print(f"   Lote {i//batch_size + 1}: {len(batch_docs)} documentos")
        
        final_count = collection.count()
        print(f"\n✅ CARGA COMPLETADA")
        print(f"📊 Total de documentos en BD: {final_count}")
        
        # Prueba de búsqueda
        print("\n🔍 Probando búsqueda...")
        results = vector_store.search_documents(collection, "tizón tardío tomate", n_results=2)
        if results:
            print(f"✅ Búsqueda exitosa: {len(results)} resultados encontrados")
            print(f"   Primer resultado: {str(results[0])[:100]}...")
        else:
            print("⚠️ La búsqueda no devolvió resultados")
            
        return True
        
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        return False

def main():
    """Función principal"""
    success = load_documents_to_chromadb()
    
    if success:
        print("\n🎉 SISTEMA RAG LISTO PARA USAR")
        print("Ahora puedes ejecutar la interfaz web con documentos cargados.")
    else:
        print("\n💥 FALLO EN LA CARGA")
        print("Revisa los errores anteriores y corrige los problemas.")

if __name__ == "__main__":
    main()
