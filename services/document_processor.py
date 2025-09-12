import os
import json
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from components.pdf_extractor import PDFExtractor
from components.language_detector import LanguageDetector
from components.text_translator import TextTranslator
from components.text_cleaner import TextCleaner
from models.document import ProcessedDocument, ProcessingMetadata
from config.settings import Settings

class DocumentProcessor:
    """Servicio para procesamiento completo de documentos"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.language_detector = LanguageDetector()
        self.text_translator = TextTranslator()
        self.text_cleaner = TextCleaner()
        
        # Crear directorios necesarios
        Settings.create_directories()
    
    def process_single_document(self, pdf_path: str) -> Optional[ProcessedDocument]:
        """
        Procesa un documento PDF individual
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Documento procesado o None si hay error
        """
        filename = os.path.basename(pdf_path)
        print(f"\n📄 Procesando: {filename}")
        print("=" * 60)
        
        # Verificar si ya existe archivo procesado
        processed_file_path = self._get_processed_file_path(filename)
        if processed_file_path.exists():
            print(f"✅ Archivo ya procesado encontrado: {processed_file_path.name}")
            print("   Cargando contenido existente...")
            
            try:
                with open(processed_file_path, "r", encoding="utf-8") as f:
                    clean_text = f.read()
                
                # Crear documento procesado con información mínima
                processed_doc = ProcessedDocument(
                    original_filename=filename,
                    processed_filename=processed_file_path.name,
                    translation_filename=None,  # Se determinará después
                    original_path=pdf_path,
                    processed_path=str(processed_file_path),
                    translation_path=None,
                    original_text="",  # No disponible
                    filtered_text="",  # No disponible
                    translated_text="",  # No disponible
                    clean_text=clean_text,
                    source_language="es",  # Asumimos español por defecto
                    original_char_count=0,  # No disponible
                    filtered_char_count=0,  # No disponible
                    translated_char_count=0,  # No disponible
                    final_char_count=len(clean_text),
                    processing_date=datetime.now(),
                    translation_date=None
                )
                
                print(f"   Texto cargado: {len(clean_text)} caracteres")
                self._print_processing_summary(processed_doc)
                return processed_doc
                
            except Exception as e:
                print(f"⚠️ Error cargando archivo procesado: {e}")
                print("   Continuando con procesamiento normal...")
        
        try:
            # Extraer texto del PDF
            original_text = self.pdf_extractor.extract_text(pdf_path)
            if not original_text:
                print(f"⚠️ No se pudo extraer texto de {filename}")
                return None
            
            print(f"📏 Texto original: {len(original_text)} caracteres")
            
            # Detectar idioma
            source_language = self.language_detector.detect_language(original_text)
            print(f"🌍 Idioma detectado: {source_language}")
            
            # Filtrar secciones científicas ANTES de la traducción
            filtered_text = self.text_cleaner.filter_scientific_sections(original_text, source_language)
            print(f"🔍 Texto después del filtrado: {len(filtered_text)} caracteres")
            
            # Traducir si es necesario
            translated_text = self._handle_translation(filtered_text, source_language, filename)
            
            # Limpiar texto final
            clean_text = self.text_cleaner.clean_text(translated_text)
            
            # Guardar archivo procesado
            processed_filename = self._save_processed_file(filename, clean_text)
            
            # Crear objeto de documento procesado
            processed_doc = ProcessedDocument(
                original_filename=filename,
                processed_filename=processed_filename,
                translation_filename=os.path.basename(self._get_translation_path(filename)) if source_language != 'es' else None,
                original_path=pdf_path,
                processed_path=str(Settings.OUTPUT_DIR / processed_filename),
                translation_path=str(self._get_translation_path(filename)) if source_language != 'es' else None,
                original_text=original_text,
                filtered_text=filtered_text,
                translated_text=translated_text,
                clean_text=clean_text,
                source_language=source_language,
                original_char_count=len(original_text),
                filtered_char_count=len(filtered_text),
                translated_char_count=len(translated_text),
                final_char_count=len(clean_text),
                processing_date=datetime.now(),
                translation_date=datetime.now() if source_language != 'es' else None
            )
            
            self._print_processing_summary(processed_doc)
            return processed_doc
            
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
            return None
    
    def _handle_translation(self, text: str, source_language: str, filename: str) -> str:
        """
        Maneja la traducción del texto
        
        Args:
            text: Texto a traducir
            source_language: Idioma origen
            filename: Nombre del archivo
            
        Returns:
            Texto traducido
        """
        if source_language == 'es':
            print("✅ Texto ya está en español")
            return text
        
        # Verificar si ya existe traducción
        translation_path = self._get_translation_path(filename)
        if translation_path.exists():
            print(f"✅ Usando traducción existente: {translation_path.name}")
            with open(translation_path, "r", encoding="utf-8") as f:
                translated_text = f.read()
            print(f"   Texto traducido cargado: {len(translated_text)} caracteres")
            return translated_text
        
        # Realizar traducción
        print(f"🔄 Traduciendo de {source_language} a español...")
        translated_text = self.text_translator.translate_text(text, source_language, "es")
        print(f"   Texto traducido: {len(translated_text)} caracteres")
        
        # Guardar traducción
        with open(translation_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        
        print(f"✅ Texto traducido guardado: {translation_path}")
        return translated_text
    
    def _get_translation_path(self, filename: str) -> Path:
        """Obtiene la ruta para el archivo de traducción"""
        base_name = os.path.splitext(filename)[0]
        return Settings.TRANSLATIONS_DIR / f"{base_name}__traducido.txt"
    
    def _get_processed_file_path(self, filename: str) -> Path:
        """Obtiene la ruta para el archivo procesado"""
        base_name = os.path.splitext(filename)[0]
        return Settings.OUTPUT_DIR / f"{base_name}__procesado.txt"
    
    def _save_processed_file(self, filename: str, clean_text: str) -> str:
        """Guarda el archivo procesado"""
        base_name = os.path.splitext(filename)[0]
        processed_filename = f"{base_name}__procesado.txt"
        processed_path = Settings.OUTPUT_DIR / processed_filename
        
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        
        print(f"✅ Guardado: {processed_path}")
        return processed_filename
    
    def _print_processing_summary(self, doc: ProcessedDocument):
        """Imprime resumen del procesamiento"""
        print(f"📊 Resumen del documento:")
        print(f"   - Idioma original: {doc.source_language}")
        print(f"   - Caracteres originales: {doc.original_char_count}")
        print(f"   - Caracteres después del filtrado: {doc.filtered_char_count}")
        print(f"   - Caracteres traducidos: {doc.translated_char_count}")
        print(f"   - Caracteres finales: {doc.final_char_count}")
        print("=" * 60)
    
    def process_multiple_documents(self, pdf_directory: str) -> List[ProcessedDocument]:
        """
        Procesa múltiples documentos PDF
        
        Args:
            pdf_directory: Directorio con archivos PDF
            
        Returns:
            Lista de documentos procesados
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            print(f"❌ Directorio no encontrado: {pdf_directory}")
            return []
        
        # Encontrar archivos PDF
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"📁 Encontrados {len(pdf_files)} archivos PDF para procesar")
        
        if not pdf_files:
            print("⚠️ No se encontraron archivos PDF")
            return []
        
        processed_documents = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n🔄 Procesando archivo {i}/{len(pdf_files)}")
            processed_doc = self.process_single_document(str(pdf_file))
            
            if processed_doc:
                processed_documents.append(processed_doc)
        
        return processed_documents
    
    def save_processing_metadata(self, documents: List[ProcessedDocument], 
                                total_pdfs_found: int) -> Path:
        """
        Guarda metadatos del procesamiento
        
        Args:
            documents: Lista de documentos procesados
            total_pdfs_found: Total de PDFs encontrados
            
        Returns:
            Ruta al archivo de metadatos
        """
        metadata = ProcessingMetadata(
            processing_date=datetime.now(),
            total_pdfs_found=total_pdfs_found,
            documents_processed=len(documents),
            documents=documents
        )
        
        metadata_file = Settings.OUTPUT_DIR / "metadata_procesamiento.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metadatos guardados en: {metadata_file}")
        return metadata_file
    
    def get_processing_statistics(self, documents: List[ProcessedDocument]) -> dict:
        """
        Obtiene estadísticas del procesamiento
        
        Args:
            documents: Lista de documentos procesados
            
        Returns:
            Diccionario con estadísticas
        """
        if not documents:
            return {}
        
        # Estadísticas por idioma
        language_stats = {}
        for doc in documents:
            lang = doc.source_language
            if lang not in language_stats:
                language_stats[lang] = {
                    "count": 0,
                    "total_chars": 0,
                    "avg_chars": 0
                }
            language_stats[lang]["count"] += 1
            language_stats[lang]["total_chars"] += doc.original_char_count
        
        # Calcular promedios
        for lang in language_stats:
            language_stats[lang]["avg_chars"] = language_stats[lang]["total_chars"] / language_stats[lang]["count"]
        
        # Estadísticas generales
        total_chars_original = sum(doc.original_char_count for doc in documents)
        total_chars_final = sum(doc.final_char_count for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_characters_original": total_chars_original,
            "total_characters_final": total_chars_final,
            "compression_ratio": (total_chars_original - total_chars_final) / total_chars_original * 100,
            "language_distribution": language_stats
        } 