# pipeline_rag_corregido.py
import os
import fitz  # PyMuPDF
import re
import unidecode
import json
from typing import List, Dict, Optional
from langdetect import detect, LangDetectException
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import time

# Cargar variables de entorno
load_dotenv()

class PipelineRAGCorregido:
    def __init__(self, carpeta_pdfs: str, output_dir: str = None):
        """
        Inicializa el pipeline RAG corregido para procesar documentos PDF
        
        Args:
            carpeta_pdfs: Ruta a la carpeta con documentos PDF
            output_dir: Carpeta de salida (opcional)
        """
        self.carpeta_pdfs = carpeta_pdfs
        self.output_dir = output_dir or os.path.join(carpeta_pdfs, "textos_procesados")
        self.traducciones_dir = os.path.join(carpeta_pdfs, "textos_traducidos")
        
        # Crear directorios
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.traducciones_dir, exist_ok=True)
        
        # Configurar Gemini
        self.llm = self._configurar_gemini()
        
        # Configurar ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Configurar text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print("🚀 Pipeline RAG corregido inicializado correctamente")
    
    def _configurar_gemini(self) -> Optional[ChatVertexAI]:
        """Configura Gemini usando las credenciales de Google Cloud"""
        try:
            # Configurar credenciales
            creds_path = r"C:\Users\danil\OneDrive\Escritorio\Tesis\credentials.json"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            
            if not os.path.exists(creds_path):
                print(f"❌ Archivo de credenciales no encontrado: {creds_path}")
                return None
            
            # Inicializar el modelo con configuración más robusta
            llm = ChatVertexAI(
                model_name="gemini-2.5-flash",  # Cambiar a modelo más estable
                temperature=0.1,  # Temperatura muy baja para traducciones
                max_output_tokens=4096,  # Aumentar tokens de salida
                location="us-central1",
                max_retries=3  # Reintentos automáticos
            )
            
            # Prueba de conexión más robusta
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
            print("💡 Verifica:")
            print("   - Que el archivo credentials.json existe y es válido")
            print("   - Que tienes permisos en Google Cloud")
            print("   - Que la región es correcta")
            print("   - Que tienes cuota disponible para Gemini")
            return None
    

    
    def detectar_idioma(self, texto: str) -> str:
        """
        Detecta el idioma del texto con mejor manejo de errores
        
        Args:
            texto: Texto a analizar
            
        Returns:
            Código de idioma (ej: 'en', 'es')
        """
        try:
            # Tomar una muestra del texto para detectar idioma
            muestra = texto[:2000] if len(texto) > 2000 else texto
            
            # Limpiar muestra para mejor detección
            muestra_limpia = re.sub(r'[^\w\s]', '', muestra)
            muestra_limpia = re.sub(r'\s+', ' ', muestra_limpia).strip()
            
            if len(muestra_limpia) < 50:
                print("⚠️ Texto muy corto para detectar idioma, asumiendo inglés")
                return 'en'
            
            idioma = detect(muestra_limpia)
            print(f"�� Muestra analizada: {len(muestra_limpia)} caracteres")
            return idioma
            
        except LangDetectException as e:
            print(f"⚠️ Error en detección de idioma: {e}")
            return 'en'
        except Exception as e:
            print(f"❌ Error inesperado en detección de idioma: {e}")
            return 'en'
    
    def traducir_texto(self, texto: str, idioma_origen: str) -> str:
        """
        Traduce texto del idioma origen al español usando Gemini
        
        Args:
            texto: Texto a traducir
            idioma_origen: Código del idioma origen
            
        Returns:
            Texto traducido al español
        """
        if not self.llm:
            print("⚠️ Gemini no está configurado, devolviendo texto original")
            return texto
        
        if idioma_origen == 'es':
            print("✅ Texto ya está en español, no necesita traducción")
            return texto
        
        try:
            print(f"🔄 Iniciando traducción de {idioma_origen} a español...")
            print(f" Longitud del texto: {len(texto)} caracteres")
            
            # Dividir texto en chunks más pequeños
            max_chunk_size = 1500  # Chunks más pequeños
            chunks = [texto[i:i+max_chunk_size] for i in range(0, len(texto), max_chunk_size)]
            
            texto_traducido = ""
            
            for i, chunk in enumerate(chunks):
                print(f"🔄 Traduciendo chunk {i+1}/{len(chunks)}...")
                
                # Prompt más simple y directo
                prompt = f"Traduce al español: {chunk}"
                
                try:
                    response = self.llm.invoke(prompt)
                    
                    if response and hasattr(response, 'content') and response.content:
                        texto_traducido += response.content + " "
                        print(f"✅ Chunk {i+1} traducido exitosamente")
                    else:
                        print(f"⚠️ Respuesta vacía para chunk {i+1}, usando texto original")
                        texto_traducido += chunk + " "
                        
                except Exception as chunk_error:
                    print(f"⚠️ Error en chunk {i+1}: {chunk_error}")
                    texto_traducido += chunk + " "
                
                # Pausa más larga entre chunks
                if i < len(chunks) - 1:
                    time.sleep(5)  # Pausa más larga
            
            print(f"✅ Traducción completada. Longitud final: {len(texto_traducido)} caracteres")
            return texto_traducido.strip()
            
        except Exception as e:
            print(f"❌ Error en la traducción: {e}")
            print("⚠️ Devolviendo texto original")
            return texto
    
    def filtrar_secciones_cientificas(self, texto: str, idioma: str) -> str:
        """
        Filtra y descarta secciones específicas de artículos científicos
        
        Args:
            texto: Texto a filtrar
            idioma: Código del idioma del texto
            
        Returns:
            Texto filtrado sin secciones no deseadas
        """
        print("🔍 Filtrando secciones de artículo científico...")
        
        # Definir patrones de secciones a descartar (inglés y español)
        secciones_a_descartar = {
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
        
        # Obtener patrones según el idioma
        patrones = secciones_a_descartar.get(idioma, secciones_a_descartar['en'])
        
        # Agregar patrones más específicos para metadatos
        patrones_adicionales = [
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
        patrones.extend(patrones_adicionales)
        
        texto_filtrado = texto
        secciones_eliminadas = []
        
        for patron in patrones:
            matches = re.finditer(patron, texto_filtrado, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            for match in matches:
                seccion_encontrada = match.group(0)
                # Extraer el título de la sección para el reporte
                titulo_match = re.search(r'\b[A-Z][a-zA-Z\s]+\b', seccion_encontrada)
                titulo = titulo_match.group(0) if titulo_match else "Sección desconocida"
                secciones_eliminadas.append(titulo)
                
                # Reemplazar la sección con espacio
                texto_filtrado = texto_filtrado.replace(seccion_encontrada, ' ')
        
        # SOLO normalizar espacios múltiples, NO tocar espacios entre palabras
        texto_filtrado = re.sub(r'\s+', ' ', texto_filtrado)
        texto_filtrado = texto_filtrado.strip()
        
        # Limpiar líneas vacías y líneas con solo caracteres especiales
        lineas = texto_filtrado.split('\n')
        lineas_limpias = []
        for linea in lineas:
            linea_limpia = linea.strip()
            # Mantener solo líneas con contenido significativo
            if (len(linea_limpia) > 20 and  # Aumentar longitud mínima
                not re.match(r'^\s*[\d\.\-\s]+\s*$', linea_limpia) and
                not re.match(r'^\s*[A-Z][a-z]+\s*$', linea_limpia) and
                not re.match(r'^\s*[A-Z][a-z]+\s*[A-Z][a-z]+\s*$', linea_limpia)):  # Títulos muy cortos
                lineas_limpias.append(linea_limpia)
        
        texto_filtrado = '\n'.join(lineas_limpias)
        
        print(f"✅ Filtrado completado:")
        print(f"   - Caracteres originales: {len(texto)}")
        print(f"   - Caracteres después del filtrado: {len(texto_filtrado)}")
        if secciones_eliminadas:
            print(f"   - Secciones eliminadas: {', '.join(set(secciones_eliminadas))}")
        else:
            print(f"   - No se encontraron secciones para eliminar")
        
        return texto_filtrado

    def limpiar_texto(self, texto: str) -> str:
        """
        Limpia y normaliza el texto
        
        Args:
            texto: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        print(" Limpiando texto...")
        
        # Normalizar espacios y saltos de línea
        texto = texto.replace('\n', ' ')
        texto = re.sub(r'\s+', ' ', texto)
        
        # Remover referencias de página
        texto = re.sub(r'Page \d+', '', texto, flags=re.IGNORECASE)
        texto = re.sub(r'Página \d+', '', texto, flags=re.IGNORECASE)
        
        # SOLO remover caracteres realmente problemáticos
        texto = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', texto)
        
        texto_limpio = texto.strip()
        print(f"✅ Texto limpio. Longitud: {len(texto_limpio)} caracteres")
        
        return texto_limpio
    
    def extraer_texto_pdf(self, ruta_pdf: str) -> str:
        """
        Extrae texto de un archivo PDF
        
        Args:
            ruta_pdf: Ruta al archivo PDF
            
        Returns:
            Texto extraído
        """
        try:
            doc = fitz.open(ruta_pdf)
            texto_total = ""
            
            for page_num, page in enumerate(doc):
                texto_pagina = page.get_text()
                
                # ARREGLAR PALABRAS PEGADAS DESDE LA EXTRACCIÓN
                # Agregar espacios entre palabras que están pegadas
                texto_pagina = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', texto_pagina)  # aA -> a A
                texto_pagina = re.sub(r'(?<=[a-z])(?=\d)', ' ', texto_pagina)     # a1 -> a 1
                texto_pagina = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', texto_pagina)  # 1a -> 1 a
                texto_pagina = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', texto_pagina)  # AAa -> A Aa
                
                # Normalizar espacios
                texto_pagina = re.sub(r'\s+', ' ', texto_pagina)
                
                texto_total += texto_pagina + "\n"
                print(f" Página {page_num + 1}: {len(texto_pagina)} caracteres")
            
            doc.close()
            print(f"📊 Total extraído: {len(texto_total)} caracteres")
            return texto_total
            
        except Exception as e:
            print(f"❌ Error al extraer texto de {ruta_pdf}: {e}")
            return ""
    
    def procesar_documento(self, ruta_pdf: str) -> Dict:
        """
        Procesa un documento PDF completo con información detallada
        
        Args:
            ruta_pdf: Ruta al archivo PDF
            
        Returns:
            Diccionario con información del documento procesado
        """
        nombre_archivo = os.path.basename(ruta_pdf)
        print(f"\n📄 Procesando: {nombre_archivo}")
        print("=" * 60)
        
        # Extraer texto
        texto_original = self.extraer_texto_pdf(ruta_pdf)
        if not texto_original:
            print(f"⚠️ No se pudo extraer texto de {nombre_archivo}")
            return None
        
        print(f"📏 Texto original: {len(texto_original)} caracteres")
        
        # Detectar idioma
        idioma = self.detectar_idioma(texto_original)
        print(f"🌍 Idioma detectado: {idioma}")
        
        # Filtrar secciones científicas ANTES de la traducción
        texto_filtrado = self.filtrar_secciones_cientificas(texto_original, idioma)
        print(f"🔍 Texto después del filtrado: {len(texto_filtrado)} caracteres")
        
        # Traducir si es necesario (solo el texto filtrado)
        if idioma != 'es':
            # Verificar si ya existe traducción
            nombre_traducido = os.path.splitext(nombre_archivo)[0] + "__traducido.txt"
            ruta_traducido = os.path.join(self.traducciones_dir, nombre_traducido)
            
            if os.path.exists(ruta_traducido):
                print(f"✅ Usando traducción existente: {nombre_traducido}")
                with open(ruta_traducido, "r", encoding="utf-8") as f:
                    texto_traducido = f.read()
                print(f" Texto traducido cargado: {len(texto_traducido)} caracteres")
            else:
                print(f"🔄 Traduciendo de {idioma} a español...")
                texto_traducido = self.traducir_texto(texto_filtrado, idioma)
                print(f" Texto traducido: {len(texto_traducido)} caracteres")
                
                # Guardar texto traducido ANTES de la limpieza
                with open(ruta_traducido, "w", encoding="utf-8") as f:
                    f.write(texto_traducido)
                
                print(f"✅ Texto traducido guardado: {ruta_traducido}")
            
        else:
            texto_traducido = texto_filtrado
            print("✅ Texto ya está en español")
        
        # Limpiar texto
        texto_limpio = self.limpiar_texto(texto_traducido)
        
        # Guardar archivo procesado (después de limpieza)
        nombre_txt = os.path.splitext(nombre_archivo)[0] + "__procesado.txt"
        ruta_txt = os.path.join(self.output_dir, nombre_txt)
        
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(texto_limpio)
        
        print(f"✅ Guardado: {ruta_txt}")
        print(f"📊 Resumen del documento:")
        print(f"   - Idioma original: {idioma}")
        print(f"   - Caracteres originales: {len(texto_original)}")
        print(f"   - Caracteres después del filtrado: {len(texto_filtrado)}")
        print(f"   - Caracteres traducidos: {len(texto_traducido)}")
        print(f"   - Caracteres finales: {len(texto_limpio)}")
        print("=" * 60)
        
        return {
            "archivo_original": nombre_archivo,
            "archivo_procesado": nombre_txt,
            "archivo_traducido": nombre_traducido if idioma != 'es' else None,
            "idioma_origen": idioma,
            "texto_limpio": texto_limpio,
            "texto_traducido": texto_traducido,
            "ruta_procesado": ruta_txt,
            "ruta_traducido": ruta_traducido if idioma != 'es' else None,
            "caracteres_originales": len(texto_original),
            "caracteres_filtrados": len(texto_filtrado),
            "caracteres_traducidos": len(texto_traducido),
            "caracteres_finales": len(texto_limpio)
        }
    
    def crear_chunks(self, texto: str, metadata: Dict) -> List[Dict]:
        """
        Divide el texto en chunks para ChromaDB
        
        Args:
            texto: Texto a dividir
            metadata: Metadatos del documento
            
        Returns:
            Lista de chunks con metadatos
        """
        chunks = self.text_splitter.split_text(texto)
        
        documentos_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
            
            documentos_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return documentos_chunks
    
    def cargar_en_chromadb(self, documentos: List[Dict], collection_name: str = "tizon_tardio"):
        """
        Carga los documentos procesados en ChromaDB
        
        Args:
            documentos: Lista de documentos procesados
            collection_name: Nombre de la colección en ChromaDB
        """
        try:
            # Eliminar colección existente si existe
            try:
                self.chroma_client.delete_collection(collection_name)
                print(f"🗑️ Colección '{collection_name}' eliminada para recrear")
            except:
                pass
            
            from chromadb.utils import embedding_functions
            
            # ELIGE UNA OPCIÓN:
            
            # OPCIÓN 1: Modelo inglés (más rápido)
            #embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            #    model_name="sentence-transformers/all-MiniLM-L6-v2"
            #)
            #print("🚀 Usando modelo: all-MiniLM-L6-v2 (inglés)")
            
            # OPCIÓN 2: Modelo multilingüe (mejor para español)
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                 model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
             )
            print("🔍 Usando modelo: paraphrase-multilingual-MiniLM-L12-v2 (multilingüe)")
            
            # Crear nueva colección con embedding function
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Base de conocimiento sobre tizón tardío en tomate"},
                embedding_function=embedding_function
            )
            
            print(f"📚 Cargando documentos en ChromaDB...")
            
            total_chunks = 0
            for doc in documentos:
                if not doc or not doc.get("texto_limpio"):
                    continue
                
                # Crear chunks
                metadata = {
                    "archivo": doc["archivo_original"],
                    "idioma_origen": doc["idioma_origen"],
                    "fecha_procesamiento": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                chunks = self.crear_chunks(doc["texto_limpio"], metadata)
                
                # Insertar chunks en ChromaDB con manejo de errores
                for chunk in chunks:
                    try:
                        collection.add(
                            documents=[chunk["text"]],
                            metadatas=[chunk["metadata"]],
                            ids=[f"{doc['archivo_original']}_chunk_{chunk['metadata']['chunk_id']}"]
                        )
                    except Exception as chunk_error:
                        print(f"⚠️ Error al agregar chunk {chunk['metadata']['chunk_id']}: {chunk_error}")
                        continue
                
                total_chunks += len(chunks)
                print(f"✅ {doc['archivo_original']}: {len(chunks)} chunks agregados")
            
            print(f"🎉 Base de conocimiento creada exitosamente en ChromaDB")
            print(f"📊 Total de documentos: {len(documentos)}")
            print(f" Total de chunks: {total_chunks}")
            print(f" Total en colección: {collection.count()}")
            
        except Exception as e:
            print(f"❌ Error al cargar en ChromaDB: {e}")
            print("💡 Sugerencia: Intenta eliminar la carpeta chroma_db y ejecutar nuevamente")
    
    def ejecutar_pipeline(self):
        """
        Ejecuta el pipeline completo de procesamiento
        """
        print("�� Iniciando pipeline de procesamiento RAG corregido...")
        print("=" * 80)
        
        documentos_procesados = []
        archivos_pdf = [f for f in os.listdir(self.carpeta_pdfs) if f.lower().endswith(".pdf")]
        
        print(f"📁 Encontrados {len(archivos_pdf)} archivos PDF para procesar")
        
        for i, archivo in enumerate(archivos_pdf, 1):
            print(f"\n�� Procesando archivo {i}/{len(archivos_pdf)}")
            ruta_pdf = os.path.join(self.carpeta_pdfs, archivo)
            documento = self.procesar_documento(ruta_pdf)
            
            if documento:
                documentos_procesados.append(documento)
        
        print(f"\n📊 Resumen final del procesamiento:")
        print(f"   - Archivos PDF encontrados: {len(archivos_pdf)}")
        print(f"   - Documentos procesados exitosamente: {len(documentos_procesados)}")
        
        if documentos_procesados:
            # Estadísticas por idioma
            idiomas = {}
            for doc in documentos_procesados:
                idioma = doc["idioma_origen"]
                idiomas[idioma] = idiomas.get(idioma, 0) + 1
            
            print(f"   - Distribución por idioma:")
            for idioma, count in idiomas.items():
                print(f"     * {idioma}: {count} documentos")
            
            # Cargar en ChromaDB
            self.cargar_en_chromadb(documentos_procesados)
        
        # Guardar metadatos del procesamiento
        metadata_file = os.path.join(self.output_dir, "metadata_procesamiento.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "fecha_procesamiento": time.strftime("%Y-%m-%d %H:%M:%S"),
                "archivos_pdf_encontrados": len(archivos_pdf),
                "documentos_procesados": len(documentos_procesados),
                "documentos": documentos_procesados
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Pipeline completado. Metadatos guardados en: {metadata_file}")

def main():
    """Función principal para ejecutar el pipeline"""
    carpeta_pdfs = r"C:\Users\danil\OneDrive\Escritorio\Tesis\Documentos"
    
    # Crear y ejecutar pipeline
    pipeline = PipelineRAGCorregido(carpeta_pdfs)
    pipeline.ejecutar_pipeline()

if __name__ == "__main__":
    main()