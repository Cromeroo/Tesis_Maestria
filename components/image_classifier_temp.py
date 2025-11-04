import re
from typing import Optional
from langdetect import detect, LangDetectException

class LanguageDetector:
    """Componente para detectar el idioma del texto"""
    
    def __init__(self, min_text_length: int = 50, sample_size: int = 2000):
        self.min_text_length = min_text_length
        self.sample_size = sample_size
    
    def detect_language(self, text: str) -> str:
        """
        Detecta el idioma del texto con mejor manejo de errores
        
        Args:
            text: Texto a analizar
            
        Returns:
            Código de idioma (ej: 'en', 'es')
        """
        try:
            # Tomar una muestra del texto para detectar idioma
            sample = text[:self.sample_size] if len(text) > self.sample_size else text
            
            # Limpiar muestra para mejor detección
            clean_sample = self._clean_sample_for_detection(sample)
            
            if len(clean_sample) < self.min_text_length:
                print(f"⚠️ Texto muy corto para detectar idioma ({len(clean_sample)} caracteres), asumiendo inglés")
                return 'en'
            
            language = detect(clean_sample)
            print(f"🌍 Muestra analizada: {len(clean_sample)} caracteres")
            print(f"🌍 Idioma detectado: {language}")
            return language
            
        except LangDetectException as e:
            print(f"⚠️ Error en detección de idioma: {e}")
            return 'en'
        except Exception as e:
            print(f"❌ Error inesperado en detección de idioma: {e}")
            return 'en'
    
    def _clean_sample_for_detection(self, sample: str) -> str:
        """
        Limpia la muestra de texto para mejor detección de idioma
        
        Args:
            sample: Muestra de texto
            
        Returns:
            Muestra limpia
        """
        # Remover caracteres especiales que pueden interferir
        clean_sample = re.sub(r'[^\w\s]', '', sample)
        
        # Normalizar espacios
        clean_sample = re.sub(r'\s+', ' ', clean_sample)
        
        # Remover números que pueden interferir
        clean_sample = re.sub(r'\d+', '', clean_sample)
        
        return clean_sample.strip()
    
    def is_spanish(self, text: str) -> bool:
        """
        Verifica si el texto está en español
        
        Args:
            text: Texto a verificar
            
        Returns:
            True si es español, False en caso contrario
        """
        detected_lang = self.detect_language(text)
        return detected_lang == 'es'
    
    def get_language_name(self, language_code: str) -> str:
        """
        Obtiene el nombre del idioma a partir del código
        
        Args:
            language_code: Código del idioma
            
        Returns:
            Nombre del idioma
        """
        language_names = {
            'es': 'Español',
            'en': 'Inglés',
            'fr': 'Francés',
            'de': 'Alemán',
            'pt': 'Portugués',
            'it': 'Italiano',
            'nl': 'Holandés',
            'ru': 'Ruso',
            'zh': 'Chino',
            'ja': 'Japonés',
            'ko': 'Coreano'
        }
        
        return language_names.get(language_code, f'Idioma {language_code}') 