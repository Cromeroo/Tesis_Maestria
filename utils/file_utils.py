import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

class FileUtils:
    """Utilidades para manejo de archivos"""
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """
        Asegura que un directorio existe, creándolo si es necesario
        
        Args:
            path: Ruta del directorio
            
        Returns:
            Path del directorio
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict:
        """
        Obtiene información de un archivo
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Diccionario con información del archivo
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": "Archivo no encontrado"}
        
        stat = path.stat()
        return {
            "name": path.name,
            "size": stat.st_size,
            "size_human": FileUtils._format_file_size(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": path.suffix.lower(),
            "is_file": path.is_file(),
            "is_directory": path.is_dir()
        }
    
    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Formatea el tamaño de archivo en formato legible"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    @staticmethod
    def find_files_by_extension(directory: str, extensions: List[str]) -> List[Path]:
        """
        Encuentra archivos por extensión en un directorio
        
        Args:
            directory: Directorio a buscar
            extensions: Lista de extensiones (ej: ['.pdf', '.txt'])
            
        Returns:
            Lista de archivos encontrados
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            return []
        
        files = []
        for ext in extensions:
            files.extend(directory_path.glob(f"*{ext}"))
            files.extend(directory_path.glob(f"*{ext.upper()}"))
        
        return sorted(files)
    
    @staticmethod
    def copy_file_with_timestamp(source: str, destination_dir: str, 
                               prefix: str = "", suffix: str = "") -> str:
        """
        Copia un archivo con timestamp en el nombre
        
        Args:
            source: Ruta del archivo fuente
            destination_dir: Directorio de destino
            prefix: Prefijo para el nombre del archivo
            suffix: Sufijo para el nombre del archivo
            
        Returns:
            Ruta del archivo copiado
        """
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Archivo fuente no encontrado: {source}")
        
        # Crear directorio de destino si no existe
        dest_dir = Path(destination_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = [prefix, source_path.stem, timestamp, suffix]
        name_parts = [part for part in name_parts if part]  # Remover partes vacías
        
        new_filename = "_".join(name_parts) + source_path.suffix
        dest_path = dest_dir / new_filename
        
        # Copiar archivo
        shutil.copy2(source_path, dest_path)
        
        return str(dest_path)
    
    @staticmethod
    def save_json(data: Dict, file_path: str, indent: int = 2, 
                  ensure_ascii: bool = False) -> bool:
        """
        Guarda datos en formato JSON
        
        Args:
            data: Datos a guardar
            file_path: Ruta del archivo
            indent: Indentación del JSON
            ensure_ascii: Si debe asegurar ASCII
            
        Returns:
            True si se guardó exitosamente, False en caso contrario
        """
        try:
            # Crear directorio si no existe
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, 
                         default=str)  # Convertir datetime a string
            
            return True
        except Exception as e:
            print(f"❌ Error al guardar JSON en {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Dict]:
        """
        Carga datos desde un archivo JSON
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Datos cargados o None si hay error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error al cargar JSON desde {file_path}: {e}")
            return None
    
    @staticmethod
    def backup_file(file_path: str, backup_dir: str = None) -> Optional[str]:
        """
        Crea una copia de seguridad de un archivo
        
        Args:
            file_path: Ruta del archivo a respaldar
            backup_dir: Directorio de respaldo (opcional)
            
        Returns:
            Ruta del archivo de respaldo o None si hay error
        """
        try:
            if not backup_dir:
                backup_dir = str(Path(file_path).parent / "backup")
            
            return FileUtils.copy_file_with_timestamp(
                source=file_path,
                destination_dir=backup_dir,
                prefix="backup",
                suffix=""
            )
        except Exception as e:
            print(f"❌ Error al crear respaldo de {file_path}: {e}")
            return None
    
    @staticmethod
    def cleanup_old_files(directory: str, days_old: int = 30, 
                          extensions: List[str] = None) -> int:
        """
        Limpia archivos antiguos de un directorio
        
        Args:
            directory: Directorio a limpiar
            days_old: Días de antigüedad para considerar archivo como "antiguo"
            extensions: Extensiones de archivos a considerar (None = todas)
            
        Returns:
            Número de archivos eliminados
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return 0
            
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
            deleted_count = 0
            
            for file_path in directory_path.iterdir():
                if file_path.is_file():
                    # Verificar extensión si se especifica
                    if extensions and file_path.suffix.lower() not in extensions:
                        continue
                    
                    # Verificar antigüedad
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            print(f"🗑️ Eliminado archivo antiguo: {file_path.name}")
                        except Exception as e:
                            print(f"⚠️ Error al eliminar {file_path.name}: {e}")
            
            print(f"✅ Limpieza completada: {deleted_count} archivos eliminados")
            return deleted_count
            
        except Exception as e:
            print(f"❌ Error en la limpieza: {e}")
            return 0
    
    @staticmethod
    def get_directory_size(directory: str) -> Dict:
        """
        Calcula el tamaño total de un directorio
        
        Args:
            directory: Ruta del directorio
            
        Returns:
            Diccionario con información del tamaño
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return {"error": "Directorio no encontrado"}
            
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for path in directory_path.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
                    file_count += 1
                elif path.is_dir():
                    dir_count += 1
            
            return {
                "total_size": total_size,
                "total_size_human": FileUtils._format_file_size(total_size),
                "file_count": file_count,
                "directory_count": dir_count,
                "path": str(directory_path)
            }
            
        except Exception as e:
            return {"error": f"Error al calcular tamaño: {e}"} 