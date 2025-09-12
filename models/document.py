from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

@dataclass
class DocumentChunk:
    """Modelo para chunks de documentos"""
    text: str
    chunk_id: int
    total_chunks: int
    chunk_size: int
    metadata: Dict

@dataclass
class ProcessedDocument:
    """Modelo para documentos procesados"""
    # Información del archivo
    original_filename: str
    processed_filename: str
    translation_filename: Optional[str]
    
    # Rutas
    original_path: str
    processed_path: str
    translation_path: Optional[str]
    
    # Contenido del texto
    original_text: str
    filtered_text: str
    translated_text: str
    clean_text: str
    
    # Metadatos
    source_language: str
    
    # Estadísticas
    original_char_count: int
    filtered_char_count: int
    translated_char_count: int
    final_char_count: int
    
    # Timestamps
    processing_date: datetime
    
    # Valores por defecto (deben ir al final)
    target_language: str = "es"
    translation_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario para serialización"""
        return {
            "original_filename": self.original_filename,
            "processed_filename": self.processed_filename,
            "translation_filename": self.translation_filename,
            "original_path": self.original_path,
            "processed_path": self.processed_path,
            "translation_path": self.translation_path,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "original_char_count": self.original_char_count,
            "filtered_char_count": self.filtered_char_count,
            "translated_char_count": self.translated_char_count,
            "final_char_count": self.final_char_count,
            "processing_date": self.processing_date.isoformat(),
            "translation_date": self.translation_date.isoformat() if self.translation_date else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessedDocument':
        """Crear instancia desde diccionario"""
        return cls(
            original_filename=data["original_filename"],
            processed_filename=data["processed_filename"],
            translation_filename=data.get("translation_filename"),
            original_path=data["original_path"],
            processed_path=data["processed_path"],
            translation_path=data.get("translation_path"),
            original_text=data.get("original_text", ""),
            filtered_text=data.get("filtered_text", ""),
            translated_text=data.get("translated_text", ""),
            clean_text=data.get("clean_text", ""),
            source_language=data["source_language"],
            original_char_count=data["original_char_count"],
            filtered_char_count=data["filtered_char_count"],
            translated_char_count=data["translated_char_count"],
            final_char_count=data["final_char_count"],
            processing_date=datetime.fromisoformat(data["processing_date"]),
            target_language=data.get("target_language", "es"),
            translation_date=datetime.fromisoformat(data["translation_date"]) if data.get("translation_date") else None
        )

@dataclass
class ProcessingMetadata:
    """Metadatos del procesamiento completo"""
    processing_date: datetime
    total_pdfs_found: int
    documents_processed: int
    documents: List[ProcessedDocument]
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario para serialización"""
        return {
            "processing_date": self.processing_date.isoformat(),
            "total_pdfs_found": self.total_pdfs_found,
            "documents_processed": self.documents_processed,
            "documents": [doc.to_dict() for doc in self.documents]
        } 