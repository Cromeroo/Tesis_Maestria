import re
from typing import List, Dict

class TextCleaner:
    """Componente para limpieza y filtrado de texto"""
    
    def __init__(self):
        self.scientific_sections_patterns = self._initialize_scientific_patterns()
    
    def _initialize_scientific_patterns(self) -> Dict[str, List[str]]:
        """Inicializa patrones para filtrar secciones científicas"""
        return {
            'en': [
                # Referencias
                r'\b(?:References|Bibliography|Literature Cited|Works Cited)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Materiales y métodos
                r'\b(?:Materials and Methods|Methods|Methodology|Experimental Methods)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Abstract
                r'\b(?:Abstract)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Agradecimientos
                r'\b(?:Acknowledgments|Acknowledgements)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Apéndices
                r'\b(?:Appendix|Appendices)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Contribuciones
                r'\b(?:Author Contributions|Contributions)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Conflictos de interés
                r'\b(?:Conflict of Interest|Conflicts of Interest)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Financiamiento
                r'\b(?:Funding|Financial Support)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Disponibilidad de datos
                r'\b(?:Data Availability|Availability of Data)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Material suplementario
                r'\b(?:Supplementary Material|Supplementary Information)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Figuras y tablas específicas
                r'\b(?:Figure \d+|Table \d+)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # DOI y fechas
                r'\b(?:DOI|doi:)\s*[^\n]*\n',
                r'\b(?:Received|Accepted|Published)\s*[^\n]*\n',
                # Palabras clave
                r'\b(?:Keywords|Key words)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Correspondencia
                r'\b(?:Corresponding Author|Correspondence)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Información de página
                r'Page \d+',
                r'Página \d+',
            ],
            'es': [
                # Referencias
                r'\b(?:Referencias|Bibliografía|Literatura Citada|Obras Citadas)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Materiales y métodos
                r'\b(?:Materiales y Métodos|Métodos|Metodología|Métodos Experimentales)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Resumen
                r'\b(?:Resumen|Abstract)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Agradecimientos
                r'\b(?:Agradecimientos)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Apéndices
                r'\b(?:Apéndice|Apéndices)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Contribuciones
                r'\b(?:Contribuciones de Autores|Contribuciones)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Conflictos de interés
                r'\b(?:Conflicto de Intereses|Conflictos de Interés)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Financiamiento
                r'\b(?:Financiamiento|Apoyo Financiero)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Disponibilidad de datos
                r'\b(?:Disponibilidad de Datos|Accesibilidad de Datos)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Material suplementario
                r'\b(?:Material Suplementario|Información Suplementaria)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Figuras y tablas
                r'\b(?:Figura \d+|Tabla \d+)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # DOI y fechas
                r'\b(?:DOI|doi:)\s*[^\n]*\n',
                r'\b(?:Recibido|Aceptado|Publicado)\s*[^\n]*\n',
                # Palabras clave
                r'\b(?:Palabras clave|Palabras Clave)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Correspondencia
                r'\b(?:Autor Correspondiente|Correspondencia)\s*\n.*?(?=\n\s*[A-Z][a-z]+\s*\n|\Z)',
                # Información de página
                r'Page \d+',
                r'Página \d+',
            ]
        }
    
    def filter_scientific_sections(self, text: str, language: str) -> str:
        """
        Filtra y descarta secciones específicas de artículos científicos
        
        Args:
            text: Texto a filtrar
            language: Código del idioma del texto
            
        Returns:
            Texto filtrado sin secciones no deseadas
        """
        print("🔍 Filtrando secciones de artículo científico...")
        
        # Obtener patrones según el idioma
        patterns = self.scientific_sections_patterns.get(language, self.scientific_sections_patterns['en'])
        
        # Agregar patrones adicionales para metadatos
        additional_patterns = [
            # Títulos de documentos (más agresivo)
            r'^.*?(?=\n\s*[A-Z][a-z]{3,}\s*\n|\n\s*Resumen|\n\s*Introduccion|\n\s*Abstract|\n\s*Introduction)',
            # Palabras clave
            r'Palabras clave.*?\n',
            r'Keywords.*?\n',
            # Catalogación
            r'Catalogación.*?\n',
            r'Cataloging.*?\n',
            # Páginas con puntos
            r'Pág\.+\d+',
            r'Page \d+',
            # Información de autoría
            r'\[.*?\]',
            r'\(.*?\)',
            # Información de publicación
            r'AGROSAVIA.*?\n',
            r'Universidad.*?\n',
            r'Instituto.*?\n',
            r'Centro.*?\n',
            r'Facultad.*?\n',
            # Años solos
            r'^\s*\d{4}\s*$',
            # Número de páginas
            r'^\s*\d+\s*paginas?\s*$',
            r'^\s*\d+\s*pages?\s*$',
        ]
        
        # Agregar patrones adicionales
        patterns.extend(additional_patterns)
        
        filtered_text = text
        removed_sections = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, filtered_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            for match in matches:
                section_found = match.group(0)
                # Extraer el título de la sección para el reporte
                title_match = re.search(r'\b[A-Z][a-zA-Z\s]+\b', section_found)
                title = title_match.group(0) if title_match else "Sección desconocida"
                removed_sections.append(title)
                
                # Reemplazar la sección con espacio
                filtered_text = filtered_text.replace(section_found, ' ')
        
        # Normalizar espacios múltiples
        filtered_text = re.sub(r'\s+', ' ', filtered_text)
        filtered_text = filtered_text.strip()
        
        # Limpiar líneas vacías y líneas con solo caracteres especiales
        lines = filtered_text.split('\n')
        clean_lines = []
        for line in lines:
            clean_line = line.strip()
            # Mantener solo líneas con contenido significativo
            if (len(clean_line) > 20 and  # Aumentar longitud mínima
                not re.match(r'^\s*[\d\.\-\s]+\s*$', clean_line) and
                not re.match(r'^\s*[A-Z][a-z]+\s*$', clean_line) and
                not re.match(r'^\s*[A-Z][a-z]+\s*[A-Z][a-z]+\s*$', clean_line)):  # Títulos muy cortos
                clean_lines.append(clean_line)
        
        filtered_text = '\n'.join(clean_lines)
        
        print(f"✅ Filtrado completado:")
        print(f"   - Caracteres originales: {len(text)}")
        print(f"   - Caracteres después del filtrado: {len(filtered_text)}")
        if removed_sections:
            print(f"   - Secciones eliminadas: {', '.join(set(removed_sections))}")
        else:
            print(f"   - No se encontraron secciones para eliminar")
        
        return filtered_text
    
    def clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        print("🧹 Limpiando texto...")
        
        # Normalizar espacios y saltos de línea
        cleaned_text = text.replace('\n', ' ')
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Remover referencias de página
        cleaned_text = re.sub(r'Page \d+', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'Página \d+', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remover caracteres realmente problemáticos
        cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned_text)
        
        final_text = cleaned_text.strip()
        print(f"✅ Texto limpio. Longitud: {len(final_text)} caracteres")
        
        return final_text
    
    def remove_accent_marks(self, text: str) -> str:
        """
        Remueve acentos del texto (opcional)
        
        Args:
            text: Texto con acentos
            
        Returns:
            Texto sin acentos
        """
        try:
            import unidecode
            return unidecode.unidecode(text)
        except ImportError:
            print("⚠️ unidecode no está disponible, manteniendo acentos")
            return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normaliza espacios en blanco
        
        Args:
            text: Texto con espacios irregulares
            
        Returns:
            Texto con espacios normalizados
        """
        # Normalizar espacios múltiples
        normalized = re.sub(r'\s+', ' ', text)
        # Normalizar espacios al inicio y final
        return normalized.strip() 