import time
from typing import Optional
from langchain_google_vertexai import ChatVertexAI
from config.settings import Settings

class TextTranslator:
    """Componente para traducción de texto usando Gemini"""
    
    def __init__(self):
        self.llm = self._configure_gemini()
        self.max_chunk_size = Settings.MAX_CHUNK_SIZE_FOR_TRANSLATION
        self.translation_delay = Settings.TRANSLATION_DELAY
    
    def _configure_gemini(self) -> Optional[ChatVertexAI]:
        """Configura Gemini usando las credenciales de Google Cloud"""
        try:
            if not Settings.get_google_credentials():
                print("❌ No se encontraron credenciales de Google Cloud")
                return None
            
            # Inicializar el modelo con configuración robusta
            llm = ChatVertexAI(
                model_name=Settings.LLM_MODEL_NAME,
                temperature=Settings.LLM_TEMPERATURE,
                max_output_tokens=Settings.LLM_MAX_OUTPUT_TOKENS,
                location=Settings.GOOGLE_LOCATION,
                max_retries=Settings.LLM_MAX_RETRIES
            )
            
            # Prueba de conexión
            print("🔄 Probando conexión con Gemini...")
            try:
                response = llm.invoke("Traduce 'Hello world' al español.")
                if response and hasattr(response, 'content') and response.content:
                    print("✅ Gemini configurado exitosamente")
                    print(f"   Prueba: {response.content}")
                    return llm
                else:
                    print("❌ Gemini no respondió correctamente")
                    return None
            except Exception as test_error:
                print(f"❌ Error en prueba de conexión: {test_error}")
                return None
            
        except Exception as e:
            print(f"❌ Error al configurar Gemini: {e}")
            return None
    
    def translate_text(self, text: str, source_language: str, target_language: str = "es") -> str:
        """
        Traduce texto del idioma origen al idioma destino
        
        Args:
            text: Texto a traducir
            source_language: Código del idioma origen
            target_language: Código del idioma destino
            
        Returns:
            Texto traducido
        """
        if not self.llm:
            print("⚠️ Gemini no está configurado, devolviendo texto original")
            return text
        
        if source_language == target_language:
            print("✅ Texto ya está en el idioma destino, no necesita traducción")
            return text
        
        try:
            print(f"🔄 Iniciando traducción de {source_language} a {target_language}...")
            print(f"   Longitud del texto: {len(text)} caracteres")
            
            # Dividir texto en chunks más pequeños
            chunks = self._split_text_into_chunks(text)
            
            translated_text = ""
            
            for i, chunk in enumerate(chunks):
                print(f"🔄 Traduciendo chunk {i+1}/{len(chunks)}...")
                
                # Prompt optimizado para traducción
                prompt = self._create_translation_prompt(chunk, source_language, target_language)
                
                try:
                    response = self.llm.invoke(prompt)
                    
                    if response and hasattr(response, 'content') and response.content:
                        translated_text += response.content + " "
                        print(f"✅ Chunk {i+1} traducido exitosamente")
                    else:
                        print(f"⚠️ Respuesta vacía para chunk {i+1}, usando texto original")
                        translated_text += chunk + " "
                        
                except Exception as chunk_error:
                    print(f"⚠️ Error en chunk {i+1}: {chunk_error}")
                    translated_text += chunk + " "
                
                # Pausa entre chunks para evitar rate limits
                if i < len(chunks) - 1:
                    time.sleep(self.translation_delay)
            
            print(f"✅ Traducción completada. Longitud final: {len(translated_text)} caracteres")
            return translated_text.strip()
            
        except Exception as e:
            print(f"❌ Error en la traducción: {e}")
            print("⚠️ Devolviendo texto original")
            return text
    
    def _split_text_into_chunks(self, text: str) -> list:
        """
        Divide el texto en chunks para traducción
        
        Args:
            text: Texto a dividir
            
        Returns:
            Lista de chunks
        """
        return [text[i:i+self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]
    
    def _create_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Crea el prompt para traducción
        
        Args:
            text: Texto a traducir
            source_lang: Idioma origen
            target_lang: Idioma destino
            
        Returns:
            Prompt formateado
        """
        language_names = {
            'en': 'inglés',
            'es': 'español',
            'fr': 'francés',
            'de': 'alemán',
            'pt': 'portugués'
        }
        
        source_name = language_names.get(source_lang, source_lang)
        target_name = language_names.get(target_lang, target_lang)
        
        return f"""Traduce el siguiente texto del {source_name} al {target_name}.

Instrucciones importantes:
1. Mantén la terminología técnica y científica correcta
2. Preserva el formato y estructura del texto
3. Usa un {target_name} claro y profesional
4. Si hay términos técnicos específicos, mantenlos en {target_name} cuando sea apropiado
5. No agregues información adicional ni interpretes el contenido

Texto a traducir:
{text}

Traducción al {target_name}:"""
    
    def is_available(self) -> bool:
        """
        Verifica si el servicio de traducción está disponible
        
        Returns:
            True si está disponible, False en caso contrario
        """
        return self.llm is not None 