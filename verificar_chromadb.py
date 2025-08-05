import chromadb
from chromadb.config import Settings

def verificar_chromadb():
    """Verifica que ChromaDB esté funcionando correctamente"""
    
    try:
        # Conectar a la base de datos
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Obtener la colección
        collection = client.get_collection("tizon_tardio")
        
        print("�� Verificando ChromaDB...")
        print("=" * 50)
        
        # Información básica
        print(f"�� Total de elementos: {collection.count()}")
        
        # Obtener algunos ejemplos
        print("\n📄 Ejemplos de documentos:")
        results = collection.peek(limit=5)  # Aumentar a 5 para ver más ejemplos
        
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"\n--- Documento {i+1} ---")
            print(f"Archivo: {metadata['archivo']}")
            print(f"Idioma: {metadata['idioma_origen']}")
            print(f"Chunk ID: {metadata['chunk_id']}")
            print(f"Tamaño del chunk: {len(doc)} caracteres")
            print(f"Texto completo: {doc}")  # Mostrar texto completo, no solo primeros 100 chars
            print("-" * 40)
        
        # Probar búsqueda
        print("\n🔍 Probando búsqueda...")
        query_results = collection.query(
            query_texts=["tizón tardío"],
            n_results=3
        )
        
        print(f"Búsqueda exitosa: {len(query_results['documents'][0])} resultados encontrados")
        
        # Mostrar resultados de búsqueda
        if query_results['documents'][0]:
            print("\n📋 Resultados de búsqueda:")
            for i, (doc, metadata) in enumerate(zip(query_results['documents'][0], query_results['metadatas'][0])):
                print(f"\n--- Resultado {i+1} ---")
                print(f"Archivo: {metadata['archivo']}")
                print(f"Texto: {doc[:200]}...")
        
        print("\n✅ ChromaDB está funcionando correctamente!")
        
    except Exception as e:
        print(f"❌ Error al verificar ChromaDB: {e}")

if __name__ == "__main__":
    verificar_chromadb() 