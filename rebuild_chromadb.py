#!/usr/bin/env python3
"""
RECONSTRUIR CHROMADB
Script para recargar los documentos en ChromaDB
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.append(str(Path(__file__).parent))

# Importar el módulo de carga de documentos
import cargar_documentos

if __name__ == "__main__":
    print("🔄 Reconstruyendo base de datos ChromaDB...")
    print("="*60)
    
    try:
        # Esto debería cargar todos los documentos
        cargar_documentos.main()
        print("\n✅ Base de datos ChromaDB reconstruida correctamente")
    except Exception as e:
        print(f"\n❌ Error al reconstruir ChromaDB: {e}")
        sys.exit(1)
