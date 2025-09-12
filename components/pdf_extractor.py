import fitz  # PyMuPDF
import re
from typing import Optional
from pathlib import Path

class PDFExtractor:
    """Componente para extraer texto de archivos PDF"""
    
    def __init__(self):
        self.text_cleaning_patterns = [
            # Separar palabras pegadas
            (r'(?<=[a-z])(?=[A-Z])', ' '),  # aA -> a A
            (r'(?<=[a-z])(?=\d)', ' '),     # a1 -> a 1
            (r'(?<=\d)(?=[A-Za-z])', ' '),  # 1a -> 1 a
            (r'(?<=[A-Z])(?=[A-Z][a-z])', ' '),  # AAa -> A Aa
        ]
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """
        Extrae texto de un archivo PDF
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Texto extraído o None si hay error
        """
        try:
            doc = fitz.open(pdf_path)
            text_total = ""
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                cleaned_text = self._clean_page_text(page_text)
                text_total += cleaned_text + "\n"
                
                print(f"📄 Página {page_num + 1}: {len(cleaned_text)} caracteres")
            
            doc.close()
            
            if text_total.strip():
                print(f"✅ Total extraído: {len(text_total)} caracteres")
                return text_total
            else:
                print("⚠️ No se pudo extraer texto del PDF")
                return None
                
        except Exception as e:
            print(f"❌ Error al extraer texto de {pdf_path}: {e}")
            return None
    
    def _clean_page_text(self, page_text: str) -> str:
        """
        Limpia el texto de una página específica
        
        Args:
            page_text: Texto de la página
            
        Returns:
            Texto limpio
        """
        cleaned_text = page_text
        
        # Aplicar patrones de limpieza
        for pattern, replacement in self.text_cleaning_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Normalizar espacios
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Obtiene información básica del PDF
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Diccionario con información del PDF
        """
        try:
            doc = fitz.open(pdf_path)
            info = {
                "page_count": len(doc),
                "file_size": Path(pdf_path).stat().st_size,
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", "")
            }
            doc.close()
            return info
        except Exception as e:
            print(f"❌ Error al obtener información del PDF {pdf_path}: {e}")
            return {} 