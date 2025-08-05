import shutil
import os

def borrar_chromadb():
    """Elimina completamente la base de datos ChromaDB"""
    
    try:
        # Verificar si existe la carpeta
        if os.path.exists("./chroma_db"):
            # Eliminar toda la carpeta
            shutil.rmtree("./chroma_db")
            print("✅ Carpeta chroma_db eliminada exitosamente")
        else:
            print("⚠️ La carpeta chroma_db no existe")
        
        print("✅ ChromaDB eliminado. Puedes ejecutar limpieza.py nuevamente")
        
    except Exception as e:
        print(f"❌ Error al eliminar: {e}")

if __name__ == "__main__":
    borrar_chromadb() 