from typing import List, Dict, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import Settings
import re

class ThematicTextChunker:
    """Componente para dividir texto en chunks temáticos para RAG agrícola"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Inicializa el thematic text chunker
        
        Args:
            chunk_size: Tamaño de cada chunk
            chunk_overlap: Solapamiento entre chunks
        """
        self.chunk_size = chunk_size or Settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Settings.CHUNK_OVERLAP
        
        # Configurar text splitter estándar como respaldo
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Definir patrones temáticos para tizón tardío
        self.thematic_patterns = {
            "sintomas_identificacion": [
                r"identificación y síntomas",
                r"síntomas",
                r"reconocer el problema",
                r"manchas.*agua",
                r"crecimiento.*algodonoso",
                r"hojas.*marchitan"
            ],
            "condiciones_favorables": [
                r"condiciones favorables",
                r"antecedentes",
                r"temperatura.*húmedo",
                r"15-20.*°c",
                r"viento.*agua.*propagar"
            ],
            "control_preventivo": [
                r"control preventivo",
                r"prevención",
                r"medidas preventivas",
                r"semillas.*sanas",
                r"rotación.*cultivos"
            ],
            "control_cultural": [
                r"control cultural",
                r"manejo.*cultivo",
                r"prácticas.*culturales",
                r"riego.*aspersión",
                r"densidades.*plantación",
                r"fertilización.*equilibrada"
            ],
            "control_quimico": [
                r"control químico",
                r"fungicidas",
                r"aspersiones",
                r"mancozeb",
                r"clorotalonil",
                r"metalaxil",
                r"dosis.*aplicación"
            ],
            "control_biologico": [
                r"control biológico",
                r"trichoderma",
                r"bacillus",
                r"pseudomonas",
                r"microorganismos",
                r"biocontrol"
            ],
            "variedades_resistentes": [
                r"variedades resistentes",
                r"control genético",
                r"materiales.*resistentes",
                r"variedades inmunes"
            ],
            "manejo_integrado": [
                r"manejo integrado",
                r"rotación.*productos",
                r"alternar.*combinar",
                r"sistémico.*preventivo"
            ]
        }
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Divide el texto en chunks (mantiene compatibilidad con código existente)
        
        Args:
            text: Texto a dividir
            metadata: Metadatos del documento
            
        Returns:
            Lista de chunks con metadatos
        """
        # Detectar si es documento maestro por nombre de archivo
        is_master = False
        if metadata and metadata.get('archivo'):
            filename = metadata['archivo'].lower()
            master_indicators = ['compendio', 'maestro', 'principal', 'integrado', 'completo']
            is_master = any(indicator in filename for indicator in master_indicators)
        
        return self.create_thematic_chunks(text, metadata, is_master)
    
    def create_thematic_chunks(self, text: str, metadata: Dict = None, 
                             is_master_document: bool = True) -> List[Dict]:
        """
        Crea chunks temáticos para documentos agrícolas
        
        Args:
            text: Texto a dividir
            metadata: Metadatos del documento
            is_master_document: Si es el documento maestro principal
            
        Returns:
            Lista de chunks temáticos con metadatos enriquecidos
        """
        if not text or not text.strip():
            print("⚠️ Texto vacío, no se pueden crear chunks")
            return []
        
        print(f"✂️ Creando chunks temáticos...")
        print(f"   - Documento maestro: {is_master_document}")
        
        # Dividir texto por secciones principales
        sections = self._split_by_headers(text)
        
        thematic_chunks = []
        
        for section_title, section_content in sections:
            if not section_content.strip():
                continue
                
            # Identificar tema de la sección
            theme = self._identify_theme(section_title, section_content)
            
            print(f"   📋 Procesando sección: {section_title} -> Tema: {theme}")
            
            if is_master_document and len(section_content) > self.chunk_size:
                # Para documento maestro: mantener secciones completas cuando sea posible
                sub_chunks = self._create_smart_sub_chunks(section_content, theme)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    chunk_metadata = self._create_chunk_metadata(
                        metadata, theme, section_title, i, len(sub_chunks), True
                    )
                    # Actualizar tamaño del chunk
                    chunk_metadata["chunk_size"] = len(sub_chunk)
                    
                    thematic_chunks.append({
                        "text": sub_chunk,
                        "metadata": chunk_metadata
                    })
            else:
                # Sección completa como un chunk
                chunk_metadata = self._create_chunk_metadata(
                    metadata, theme, section_title, 0, 1, is_master_document
                )
                full_text = f"{section_title}\n\n{section_content}"
                # Actualizar tamaño del chunk
                chunk_metadata["chunk_size"] = len(full_text)
                
                thematic_chunks.append({
                    "text": full_text,
                    "metadata": chunk_metadata
                })
        
        # Si no se encontraron secciones claras, usar chunking estándar
        if not thematic_chunks:
            print("   ⚡ No se encontraron secciones temáticas, usando chunking estándar")
            return self._create_standard_chunks_with_themes(text, metadata)
        
        print(f"✅ Se crearon {len(thematic_chunks)} chunks temáticos")
        return thematic_chunks
    
    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """Divide el texto por encabezados"""
        # Patrones para detectar encabezados
        header_patterns = [
            r'^#{1,3}\s+(.+)$',  # Markdown headers
            r'^(.+)\n[=-]{3,}$',  # Underlined headers
            r'^\*\*(.+)\*\*$',   # Bold headers
            r'^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑa-záéíóúñ\s]{5,50})\s*$'  # Title case headers
        ]
        
        sections = []
        lines = text.split('\n')
        current_section_title = "Introducción"
        current_section_content = ""
        
        for line in lines:
            is_header = False
            
            for pattern in header_patterns:
                match = re.search(pattern, line, re.MULTILINE)
                if match:
                    # Guardar sección anterior
                    if current_section_content.strip():
                        sections.append((current_section_title, current_section_content.strip()))
                    
                    # Iniciar nueva sección
                    current_section_title = match.group(1).strip()
                    current_section_content = ""
                    is_header = True
                    break
            
            if not is_header:
                current_section_content += line + "\n"
        
        # Agregar última sección
        if current_section_content.strip():
            sections.append((current_section_title, current_section_content.strip()))
        
        return sections
    
    def _identify_theme(self, section_title: str, section_content: str) -> str:
        """Identifica el tema de una sección"""
        combined_text = f"{section_title} {section_content}".lower()
        
        theme_scores = {}
        
        for theme, patterns in self.thematic_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
                score += matches
            
            if score > 0:
                theme_scores[theme] = score
        
        if theme_scores:
            best_theme = max(theme_scores, key=theme_scores.get)
            return best_theme
        
        return "general"
    
    def _create_smart_sub_chunks(self, content: str, theme: str) -> List[str]:
        """Crea sub-chunks inteligentes manteniendo coherencia temática"""
        # Para secciones largas, dividir por párrafos completos
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Si agregar este párrafo no excede el límite, agregarlo
            if len(current_chunk + paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Guardar chunk actual si no está vacío
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Si el párrafo es muy largo, dividirlo
                if len(paragraph) > self.chunk_size:
                    sub_chunks = self.text_splitter.split_text(paragraph)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] + "\n\n" if sub_chunks else ""
                else:
                    current_chunk = paragraph + "\n\n"
        
        # Agregar último chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_chunk_metadata(self, base_metadata: Dict, theme: str, 
                              section_title: str, chunk_index: int, 
                              total_sub_chunks: int, is_master: bool) -> Dict:
        """Crea metadatos enriquecidos para chunks temáticos"""
        chunk_metadata = base_metadata.copy() if base_metadata else {}
        
        chunk_metadata.update({
            # Mantener compatibilidad con código existente
            "chunk_id": chunk_index,
            "total_chunks": total_sub_chunks,
            "chunk_size": 0,  # Se actualizará después
            "chunk_position": f"{chunk_index+1}/{total_sub_chunks}",
            
            # Nuevos metadatos temáticos
            "theme": theme,
            "section_title": section_title,
            "chunk_index": chunk_index,
            "total_sub_chunks": total_sub_chunks,
            "is_master_document": is_master,
            "chunk_type": "thematic",
            "relevance_keywords": ",".join(self._get_theme_keywords(theme)),
            "control_types": ",".join(self._get_control_types(theme))
        })
        
        return chunk_metadata
    
    def _get_theme_keywords(self, theme: str) -> List[str]:
        """Obtiene palabras clave relevantes para un tema"""
        keyword_map = {
            "sintomas_identificacion": ["manchas", "hojas", "frutos", "algodonoso", "marchitan"],
            "condiciones_favorables": ["humedad", "temperatura", "viento", "hospedadores"],
            "control_preventivo": ["semillas", "rotación", "prevención", "limpieza"],
            "control_cultural": ["riego", "fertilización", "densidad", "ventilación"],
            "control_quimico": ["fungicidas", "mancozeb", "aplicación", "dosis"],
            "control_biologico": ["trichoderma", "bacillus", "microorganismos"],
            "variedades_resistentes": ["resistencia", "variedades", "genético"],
            "manejo_integrado": ["integrado", "combinación", "rotación", "alternado"]
        }
        
        return keyword_map.get(theme, [])
    
    def _get_control_types(self, theme: str) -> List[str]:
        """Obtiene tipos de control asociados al tema"""
        control_map = {
            "sintomas_identificacion": ["diagnostico"],
            "condiciones_favorables": ["preventivo"],
            "control_preventivo": ["preventivo", "cultural"],
            "control_cultural": ["cultural", "preventivo"],
            "control_quimico": ["quimico", "curativo"],
            "control_biologico": ["biologico", "preventivo"],
            "variedades_resistentes": ["genetico", "preventivo"],
            "manejo_integrado": ["integrado", "preventivo", "curativo"]
        }
        
        return control_map.get(theme, ["general"])
    
    def _create_standard_chunks_with_themes(self, text: str, metadata: Dict) -> List[Dict]:
        """Crea chunks estándar pero con identificación temática"""
        standard_chunks = self.text_splitter.split_text(text)
        
        thematic_chunks = []
        for i, chunk in enumerate(standard_chunks):
            theme = self._identify_theme("", chunk)
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(standard_chunks),
                "chunk_size": len(chunk),
                "chunk_position": f"{i+1}/{len(standard_chunks)}",
                "theme": theme,
                "chunk_index": i,
                "chunk_type": "standard_with_theme",
                "relevance_keywords": ",".join(self._get_theme_keywords(theme)),
                "control_types": ",".join(self._get_control_types(theme))
            })
            
            thematic_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return thematic_chunks
    
    def create_overlapping_chunks(self, text: str, metadata: Dict = None, 
                                custom_chunk_size: int = None, 
                                custom_overlap: int = None) -> List[Dict]:
        """
        Crea chunks con parámetros personalizados (mantiene compatibilidad)
        
        Args:
            text: Texto a dividir
            metadata: Metadatos del documento
            custom_chunk_size: Tamaño personalizado de chunk
            custom_overlap: Solapamiento personalizado
            
        Returns:
            Lista de chunks con metadatos
        """
        if custom_chunk_size or custom_overlap:
            # Crear splitter temporal con parámetros personalizados
            temp_splitter = RecursiveCharacterTextSplitter(
                chunk_size=custom_chunk_size or self.chunk_size,
                chunk_overlap=custom_overlap or self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = temp_splitter.split_text(text)
            
            # Crear chunks con metadatos básicos
            documents_chunks = []
            for i, chunk in enumerate(chunks):
                theme = self._identify_theme("", chunk)
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "chunk_position": f"{i+1}/{len(chunks)}",
                    "chunk_type": "custom_overlap",
                    "theme": theme,
                    "relevance_keywords": ",".join(self._get_theme_keywords(theme)),
                    "control_types": ",".join(self._get_control_types(theme))
                })
                
                documents_chunks.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })
            
            return documents_chunks
        else:
            return self.create_chunks(text, metadata)
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Obtiene estadísticas de los chunks creados
        
        Args:
            chunks: Lista de chunks
            
        Returns:
            Diccionario con estadísticas
        """
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk["text"]) for chunk in chunks]
        
        # Estadísticas temáticas
        themes = {}
        for chunk in chunks:
            theme = chunk["metadata"].get("theme", "unknown")
            themes[theme] = themes.get(theme, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunk_size_variance": self._calculate_variance(chunk_sizes),
            "themes_distribution": themes,
            "master_document_chunks": len([c for c in chunks if c["metadata"].get("is_master_document", False)])
        }
    
    def get_thematic_statistics(self, chunks: List[Dict]) -> Dict:
        """Obtiene estadísticas de chunks temáticos"""
        if not chunks:
            return {}
        
        themes = {}
        control_types = {}
        
        for chunk in chunks:
            theme = chunk["metadata"].get("theme", "unknown")
            themes[theme] = themes.get(theme, 0) + 1
            
            chunk_control_types = chunk["metadata"].get("control_types", [])
            for control_type in chunk_control_types:
                control_types[control_type] = control_types.get(control_type, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "themes_distribution": themes,
            "control_types_distribution": control_types,
            "master_document_chunks": len([c for c in chunks if c["metadata"].get("is_master_document", False)])
        }
    
    def _calculate_variance(self, values: List[int]) -> float:
        """Calcula la varianza de una lista de valores"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return squared_diff_sum / (len(values) - 1)
    
    def validate_chunks(self, chunks: List[Dict]) -> bool:
        """
        Valida que los chunks sean correctos
        
        Args:
            chunks: Lista de chunks a validar
            
        Returns:
            True si los chunks son válidos, False en caso contrario
        """
        if not chunks:
            print("❌ No hay chunks para validar")
            return False
        
        for i, chunk in enumerate(chunks):
            if "text" not in chunk or "metadata" not in chunk:
                print(f"❌ Chunk {i} no tiene estructura válida")
                return False
            
            if not chunk["text"] or not chunk["text"].strip():
                print(f"❌ Chunk {i} tiene texto vacío")
                return False
            
            # Validar metadatos temáticos
            metadata = chunk["metadata"]
            if "theme" not in metadata:
                print(f"❌ Chunk {i} no tiene tema asignado")
                return False
        
        print("✅ Todos los chunks son válidos")
        return True


# Alias para mantener compatibilidad
TextChunker = ThematicTextChunker 