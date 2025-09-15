#!/usr/bin/env python3
"""
SCRIPT DE LIMPIEZA DEL PROYECTO
Elimina archivos de demo y prueba, mantiene solo lo esencial
"""

import os
import shutil
from pathlib import Path

def limpiar_proyecto():
    """Limpiar archivos innecesarios del proyecto"""
    
    base_path = Path("C:/Users/danil/OneDrive/Escritorio/Tesis")
    
    # ARCHIVOS PARA ELIMINAR (demos, pruebas, duplicados)
    archivos_eliminar = [
        # Demos y pruebas
        "demo_rag.py",
        "demo_universal.py", 
        "debug_rag.py",
        "diagnostico_rag.py",
        "evaluar_preguntas.py",
        "prueba_campesino.py",
        "prueba_final.py", 
        "prueba_longitud_optimizada.py",
        "test_longitud.py",
        "test_pipeline.py",
        "test_template_agronomo.py",
        
        # Pipelines obsoletos (mantener solo sistema_final.py)
        "asistente_campesino.py",
        "pipeline_conversacional.py",
        "pipeline_directo.py", 
        "pipeline_simple.py",
        "pipeline_ultra_simple.py",
        "pipeline_vertex.py",
        "rag_definitivo.py",
        "rag_final.py",
        "rag_universal.py",
        "langgraph_system_simple.py",
        
        # Interfaces web obsoletas (mantener solo la definitiva)
        "web_interface.py",
        "web_interface_clean.py",
        "web_interface_conversacional.py", 
        "web_interface_fixed.py",
        "web_interface_integrada.py",
        "web_interface_rapida.py",
        "web_interface_simple.py",
        
        # READMEs duplicados
        "README_INTEGRADO.md",
        "README_MODEL.md", 
        "README_WEB.md",
        
        # Otros archivos temporales
        "iniciar_web.bat",
    ]
    
    print("🗑️ LIMPIEZA DEL PROYECTO")
    print("=" * 30)
    print(f"📁 Directorio base: {base_path}")
    
    archivos_eliminados = []
    archivos_no_encontrados = []
    
    for archivo in archivos_eliminar:
        archivo_path = base_path / archivo
        if archivo_path.exists():
            try:
                if archivo_path.is_file():
                    archivo_path.unlink()  # Eliminar archivo
                elif archivo_path.is_dir():
                    shutil.rmtree(archivo_path)  # Eliminar directorio
                archivos_eliminados.append(archivo)
                print(f"✅ Eliminado: {archivo}")
            except Exception as e:
                print(f"❌ Error eliminando {archivo}: {e}")
        else:
            archivos_no_encontrados.append(archivo)
    
    # Limpiar __pycache__
    pycache_dirs = list(base_path.rglob("__pycache__"))
    for pycache in pycache_dirs:
        try:
            shutil.rmtree(pycache)
            print(f"✅ Eliminado: {pycache.relative_to(base_path)}")
        except Exception as e:
            print(f"❌ Error eliminando cache: {e}")
    
    print(f"\n📊 RESUMEN:")
    print(f"   • Archivos eliminados: {len(archivos_eliminados)}")
    print(f"   • No encontrados: {len(archivos_no_encontrados)}")
    
    print(f"\n📁 ARCHIVOS ESENCIALES MANTENIDOS:")
    archivos_esenciales = [
        "sistema_final.py",
        "cargar_documentos.py", 
        "web_interface_definitivo.py",
        "web_interface_final.py",
        "credentials.json",
        "requirements.txt",
        "chroma_db/",
        "components/",
        "config/",
        "Documentos/"
    ]
    
    for archivo in archivos_esenciales:
        archivo_path = base_path / archivo
        if archivo_path.exists():
            print(f"   ✅ {archivo}")
        else:
            print(f"   ⚠️ {archivo} (no encontrado)")
    
    return archivos_eliminados

def gitignore_archivos_temp():
    """Agregar archivos temporales al .gitignore"""
    
    gitignore_path = Path("C:/Users/danil/OneDrive/Escritorio/Tesis/.gitignore")
    
    entradas_nuevas = [
        "",
        "# Archivos temporales y de prueba",
        "test_*.py",
        "prueba_*.py", 
        "demo_*.py",
        "debug_*.py",
        "*_test.py",
        "*_prueba.py",
        "",
        "# Cache Python",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "",
        "# Archivos de sistema",
        ".DS_Store",
        "Thumbs.db",
    ]
    
    try:
        # Leer contenido actual
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                contenido_actual = f.read()
        else:
            contenido_actual = ""
        
        # Agregar nuevas entradas si no existen
        contenido_nuevo = contenido_actual
        for entrada in entradas_nuevas:
            if entrada.strip() and entrada not in contenido_actual:
                contenido_nuevo += f"\n{entrada}"
        
        # Escribir .gitignore actualizado
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(contenido_nuevo)
        
        print(f"✅ .gitignore actualizado")
        
    except Exception as e:
        print(f"❌ Error actualizando .gitignore: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO LIMPIEZA DEL PROYECTO")
    print("ADVERTENCIA: Esto eliminará archivos permanentemente")
    
    respuesta = input("¿Continuar? (sí/no): ").strip().lower()
    
    if respuesta in ['sí', 'si', 's', 'yes', 'y']:
        archivos_eliminados = limpiar_proyecto()
        gitignore_archivos_temp()
        
        print(f"\n🎉 LIMPIEZA COMPLETADA")
        print(f"✅ {len(archivos_eliminados)} archivos eliminados")
        print(f"📁 Proyecto más limpio y organizado")
        
        print(f"\n💡 SIGUIENTES PASOS:")
        print(f"1. Ejecuta: git add .")
        print(f"2. Ejecuta: git commit -m 'Limpieza del proyecto'")
        print(f"3. Ejecuta: git push")
        print(f"4. Esto evitará que los archivos regresen")
        
    else:
        print("❌ Limpieza cancelada")