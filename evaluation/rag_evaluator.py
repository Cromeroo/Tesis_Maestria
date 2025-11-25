#!/usr/bin/env python3
"""
SISTEMA DE EVALUACIÓN RAG AVANZADO
Módulo para evaluación de sistemas RAG
"""

import time
import json
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import torch
from datetime import datetime
import re

# Intentar importar GPUtil, si no está disponible continuar sin él
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

@dataclass
class RetrievalMetrics:
    """Métricas de calidad del retriever"""
    context_precision: float
    context_recall: float
    mrr_score: float  # Mean Reciprocal Rank
    ndcg_score: float  # Normalized Discounted Cumulative Gain
    query: str
    num_docs_retrieved: int
    retrieval_time_ms: float
    similarity_scores: List[float]

@dataclass
class GenerationMetrics:
    """Métricas de calidad de la respuesta generada"""
    faithfulness_score: float
    answer_relevancy_score: float
    context_utilization_score: float
    response_length: int
    generation_time_ms: float
    hallucination_score: float
    sources_cited: int

@dataclass
class ComputationalMetrics:
    """Métricas de rendimiento computacional"""
    total_time_ms: float
    cpu_percent: float
    ram_usage_mb: float
    gpu_usage_percent: Optional[float]
    gpu_memory_mb: Optional[float]
    inference_time_ms: float
    retrieval_time_ms: float

@dataclass
class EvaluationResult:
    """Resultado completo de evaluación"""
    timestamp: str
    query: str
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    computational_metrics: ComputationalMetrics
    context_quality_score: float

class RAGEvaluator:
    """Evaluador principal del sistema RAG"""
    
    def __init__(self, log_dir: str = "evaluation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Cargar frases clave para detección de alucinaciones
        self.factual_indicators = [
            "según el documento", "como se menciona en", "de acuerdo con",
            "la fuente indica", "la literatura señala"
        ]
        
        # Patrones específicos por tipo de consulta
        self.query_patterns = {
            'tratamiento': [
                r'(?i)(como|qu[ée]|cu[áa]l).*(trat|control|aplic)',
                r'(?i)(que|como).*(?:hago|echo|pongo)',
                r'(?i)(ayuda|auxilio|consejo).*(?:cultivo|planta)'
            ],
            'ingredientes': [
                r'(?i)(ingrediente|activo|principio).*(?:efectivo|mejor)',
                r'(?i)(fungicida|producto).*(?:recomendado|usado)'
            ],
            'urgencia': [
                r'(?i)(auxilio|urgente|rapido|ayuda).*(?:muere|daña)',
                r'(?i)(emergencia|ya|ahora|pronto).*(?:control|trat)'
            ]
        }
        
        # Inicializar métricas acumuladas
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reiniciar acumuladores de métricas"""
        self.total_queries = 0
        self.total_retrieval_time = 0
        self.total_generation_time = 0
        self.faithfulness_scores = []
        self.relevancy_scores = []
        self.context_scores = []
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[str], 
                         retrieval_time_ms: float) -> RetrievalMetrics:
        """
        Evaluar calidad de los documentos recuperados usando múltiples criterios:
        1. Context Precision: relevancia de fragmentos recuperados
        2. Similitud semántica con la consulta
        3. Métricas de ranking (MRR, NDCG)
        """
        # Keywords y conceptos específicos del dominio (tizón tardío)
        domain_keywords = {
            # Enfermedad y agente causal
            'tizón', 'phytophthora', 'infestans', 'enfermedad', 'patógeno',
            'hongo', 'lesiones', 'manchas', 'síntomas', 'signos',
            
            # Cultivos afectados
            'tomate', 'papa', 'solanáceas', 'cultivo', 'planta',
            'follaje', 'hoja', 'tallo', 'fruto', 'tubérculo',
            
            # Tratamientos y control
            'fungicida', 'tratamiento', 'control', 'prevención', 'manejo',
            'aplicación', 'dosis', 'producto', 'ingrediente', 'activo',
            'protectante', 'sistémico', 'curativo', 'preventivo',
            
            # Condiciones y factores
            'humedad', 'temperatura', 'clima', 'lluvia', 'riego',
            'ventilación', 'susceptible', 'resistente', 'severidad'
        }
        
        # Patrones de texto relevantes
        relevant_patterns = [
            r'tiz[oó]n\s+tard[ií]o',
            r'phytophthora\s+infestans',
            r'tratamiento|control|manejo',
            r'fungicida|producto|aplicaci[oó]n',
            r's[ií]ntomas|signos|lesiones'
        ]
        
        def calculate_relevance_score(text: str) -> float:
            """Calcula un score compuesto de relevancia usando múltiples criterios"""
            text = text.lower()
            
            # 1. Relevancia por keywords del dominio (40%)
            text_words = set(text.split())
            keyword_overlap = domain_keywords.intersection(text_words)
            keyword_score = len(keyword_overlap) / len(domain_keywords)
            
            # 2. Relevancia por patrones (30%)
            pattern_matches = sum(1 for pattern in relevant_patterns if re.search(pattern, text))
            pattern_score = pattern_matches / len(relevant_patterns)
            
            # 3. Coherencia del fragmento (30%)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                coherence_score = 0
            else:
                # Mide la consistencia temática entre oraciones
                coherent_sentences = sum(1 for s in sentences 
                                      if any(k in s for k in domain_keywords))
                coherence_score = coherent_sentences / len(sentences)
            
            # Normalización y boost de scores
            keyword_score = min(1.0, keyword_score * 2)  # Duplicar score de keywords
            pattern_score = min(1.0, pattern_score * 1.5)  # Boost de 50% a patrones
            
            # Score final ponderado con más peso en keywords
            final_score = (0.5 * keyword_score + 
                         0.3 * pattern_score + 
                         0.2 * coherence_score)
            
            # Aplicar boost final si hay matches críticos
            if 'tizón tardío' in text or 'phytophthora infestans' in text:
                final_score = min(1.0, final_score * 1.3)
            
            return final_score
        
        # Stop words en español
        spanish_stop_words = {
            'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra',
            'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella',
            'ellas', 'ellos', 'en', 'entre', 'era', 'erais', 'eran', 'eras', 'eres', 'es',
            'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estaba', 'estado', 'estais', 'estamos',
            'estan', 'estar', 'estas', 'este', 'esto', 'estos', 'estoy', 'etc', 'fue', 'fueron',
            'ha', 'hab\u00E9is', 'haber', 'habia', 'habla', 'hace', 'haceis', 'hacemos', 'hacen',
            'hacer', 'haces', 'hago', 'han', 'has', 'hasta', 'hay', 'he', 'hemos', 'hicieron',
            'hizo', 'la', 'las', 'le', 'les', 'lo', 'los', 'mas', 'me', 'mi', 'mientras',
            'mio', 'mis', 'mucho', 'muchos', 'muy', 'nada', 'ni', 'no', 'nos', 'nosotros',
            'nuestra', 'nuestras', 'nuestro', 'nuestros', 'o', 'os', 'otra', 'otras', 'otro',
            'otros', 'para', 'pero', 'poco', 'por', 'porque', 'que', 'quien', 'quienes',
            'qu\u00E9', 'se', 'sea', 'seais', 'semos', 'ser', 'sera', 'sera', 'seras',
            'si', 'sido', 'siendo', 'sin', 'sobre', 'sois', 'somos', 'son', 'soy', 'su',
            'sus', 'suya', 'suyas', 'suyo', 'suyos', 'también', 'tanto', 'te', 'teneis',
            'tenemos', 'tener', 'tengo', 'ti', 'tiene', 'tienen', 'todo', 'todos', 'tu',
            'tus', 'tuya', 'tuyo', 'tú', 'un', 'una', 'uno', 'unos', 'vosotras', 'vosotros',
            'vuestra', 'vuestras', 'vuestro', 'vuestros', 'y', 'ya', 'yo'
        }
        
        if not retrieved_docs:
            return RetrievalMetrics(
                context_precision=0.0,
                context_recall=0.0,
                mrr_score=0.0,
                ndcg_score=0.0,
                query=query,
                num_docs_retrieved=0,
                retrieval_time_ms=retrieval_time_ms,
                similarity_scores=[]
            )
        
        # Calcular scores de relevancia para cada documento
        relevance_scores = []
        semantic_scores = []
        combined_scores = []
        
        try:
            # 1. SBERT para similitud semántica
            from sentence_transformers import SentenceTransformer
            
            # Usar el modelo multilingüe de SBERT
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            
            # Obtener embeddings
            query_embedding = model.encode(query, convert_to_tensor=True)
            doc_embeddings = model.encode(retrieved_docs, convert_to_tensor=True)
            
            # Calcular similitudes coseno con SBERT
            for i, doc_embedding in enumerate(doc_embeddings):
                # Score semántico (SBERT)
                semantic_score = float(torch.nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=0))
                semantic_scores.append(semantic_score)
                
                # Score de relevancia de dominio
                relevance_score = calculate_relevance_score(retrieved_docs[i-1])
                relevance_scores.append(relevance_score)
                
                # Normalizar scores
                semantic_score = min(1.0, semantic_score * 2)  # Boost similitud semántica
                
                # Detectar tipo de consulta
                query_type = 'general'
                for qtype, patterns in self.query_patterns.items():
                    if any(re.search(pattern, query) for pattern in patterns):
                        query_type = qtype
                        break

                # Aplicar pesos según tipo de consulta - Optimizado para SBERT
                if query_type == 'tratamiento':
                    combined_score = (0.4 * relevance_score) + (0.6 * semantic_score)  # Mayor peso a similitud semántica
                elif query_type == 'ingredientes':
                    combined_score = (0.3 * relevance_score) + (0.7 * semantic_score)  # SBERT es mejor para detalles técnicos
                elif query_type == 'urgencia':
                    combined_score = (0.35 * relevance_score) + (0.65 * semantic_score)  # Balance con énfasis en semántica
                else:
                    combined_score = (0.3 * relevance_score) + (0.7 * semantic_score)  # Por defecto confiar más en SBERT
                
                # Boost por coincidencia de términos clave
                text = retrieved_docs[i-1].lower()
                key_terms = {
                    'tratamiento': ['tizón tardío', 'phytophthora', 'tratamiento', 'control'],
                    'ingredientes': ['ingrediente activo', 'fungicida', 'principio activo'],
                    'urgencia': ['control inmediato', 'emergencia', 'acción rápida']
                }
                
                if query_type in key_terms and any(term in text for term in key_terms[query_type]):
                    combined_score *= 1.2  # 20% boost for matching key terms
                
                combined_scores.append(min(1.0, combined_score))  # Cap at 1.0
                
                print(f"\nAnálisis de documento {i}:")
                print(f"• Relevancia de dominio: {relevance_score:.3f}")
                print(f"• Similitud semántica: {semantic_score:.3f}")
                print(f"• Score combinado: {combined_score:.3f}")
                print(f"• Tipo de consulta: {query_type}")
                
                if any(term in text for term in key_terms.get(query_type, [])):
                    print("✓ Contiene términos clave relevantes para la consulta")
        
        except Exception as e:
            print(f"Error en análisis semántico: {e}")
            combined_scores = [calculate_relevance_score(doc) for doc in retrieved_docs]
        
        # Calcular Context Precision con umbral adaptado a SBERT
        threshold = 0.45  # SBERT produce scores más altos y consistentes
        relevant_docs = sum(1 for score in combined_scores if score > threshold)
        context_precision = relevant_docs / len(retrieved_docs) if retrieved_docs else 0
        
        print(f"\nAnálisis de precisión:")
        print(f"• Documentos sobre umbral ({threshold}): {relevant_docs}/{len(retrieved_docs)}")
        print(f"• Context Precision: {context_precision:.3f}")
        
        # Calcular MRR y NDCG con los scores combinados
        mrr = self._calculate_mrr(combined_scores)
        ndcg = self._calculate_ndcg(combined_scores)
        
        print(f"• MRR Score: {mrr:.3f}")
        print(f"• NDCG Score: {ndcg:.3f}")
        
        metrics = RetrievalMetrics(
            context_precision=context_precision,
            context_recall=0.0,  # Requiere ground truth
            mrr_score=mrr,
            ndcg_score=ndcg,
            query=query,
            num_docs_retrieved=len(retrieved_docs),
            retrieval_time_ms=retrieval_time_ms,
            similarity_scores=combined_scores
        )
        
        return metrics
    
    def evaluate_generation(self, response: str, context: str, query: str,
                          generation_time_ms: float) -> GenerationMetrics:
        """Evaluar calidad de la respuesta generada"""
        
        # 1. Faithfulness (veracidad)
        faithfulness = self._calculate_faithfulness(response, context)
        
        # 2. Answer Relevancy
        relevancy = self._calculate_relevancy(response, query)
        
        # 3. Context Utilization
        context_score = self._calculate_context_utilization(response, context)
        
        # 4. Hallucination Detection
        hallucination_score = self._detect_hallucinations(response, context)
        
        # 5. Source Citation Analysis
        sources_cited = self._count_source_citations(response)
        
        metrics = GenerationMetrics(
            faithfulness_score=faithfulness,
            answer_relevancy_score=relevancy,
            context_utilization_score=context_score,
            response_length=len(response),
            generation_time_ms=generation_time_ms,
            hallucination_score=hallucination_score,
            sources_cited=sources_cited
        )
        
        return metrics
    
    def evaluate_computational(self, start_time: float) -> ComputationalMetrics:
        """Evaluar métricas de rendimiento computacional"""
        
        # Tiempo total
        total_time = (time.time() - start_time) * 1000
        
        # CPU y RAM más precisos
        process = psutil.Process()
        
        # Tomar múltiples muestras de CPU para mayor precisión
        cpu_samples = []
        for _ in range(5):
            cpu_samples.append(process.cpu_percent(interval=0.1))
        cpu_percent = sum(cpu_samples) / len(cpu_samples)
        
        # RAM en MB (solo del proceso actual)
        ram_info = process.memory_info()
        ram = ram_info.rss / (1024 * 1024)  # RSS: Resident Set Size
        
        # GPU si está disponible
        gpu_usage = None
        gpu_memory = None
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primera GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory = gpu.memoryUsed
            except:
                pass
        
        metrics = ComputationalMetrics(
            total_time_ms=total_time,
            cpu_percent=cpu_percent,
            ram_usage_mb=ram,
            gpu_usage_percent=gpu_usage,
            gpu_memory_mb=gpu_memory,
            inference_time_ms=0.0,  # Se actualiza externamente
            retrieval_time_ms=0.0  # Se actualiza externamente
        )
        
        return metrics
    
    def _is_doc_relevant(self, doc: str, query: str) -> bool:
        """Determinar si un documento es relevante para la consulta"""
        # Implementación simple - buscar palabras clave en común
        query_words = set(query.lower().split())
        doc_words = set(doc.lower().split())
        overlap = len(query_words.intersection(doc_words))
        return overlap >= 2  # Al menos 2 palabras en común
    
    def _calculate_mrr(self, scores: List[float]) -> float:
        """Calcular Mean Reciprocal Rank con umbral adaptativo"""
        if not scores:
            return 0.0
        
        # Umbral adaptativo basado en los scores disponibles
        threshold = max(0.3, np.mean(scores) + np.std(scores))
        
        # Encontrar el primer documento relevante
        for i, score in enumerate(scores, 1):
            if score > threshold:
                return 1.0 / i
        return 0.0
    
    def _calculate_ndcg(self, scores: List[float]) -> float:
        """Calcular Normalized Discounted Cumulative Gain con pesos mejorados"""
        if not scores:
            return 0.0
        
        # Función de ganancia exponencial
        def gain(score):
            return 2**score - 1
        
        # DCG con ganancia exponencial
        dcg = gain(scores[0])
        for i, score in enumerate(scores[1:], 2):
            dcg += gain(score) / np.log2(i + 1)
        
        # IDCG con scores ordenados
        ideal_scores = sorted(scores, reverse=True)
        idcg = gain(ideal_scores[0])
        for i, score in enumerate(ideal_scores[1:], 2):
            idcg += gain(score) / np.log2(i + 1)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Ajuste final para penalizar conjuntos pequeños de documentos
        if len(scores) < 3:
            ndcg *= 0.8  # Penalización del 20% por conjunto pequeño
            
        return ndcg
    
    def _calculate_faithfulness(self, response: str, context: str) -> float:
        """
        Calcular score de fidelidad/veracidad (NLTK + SBERT)
        
        Implementación según tesis 4.2.4:
        1. Tokenizar respuesta en oraciones usando NLTK
        2. Comparar cada oración contra contexto con embeddings SBERT
        3. Usar similitud coseno para determinar si está soportada
        """
        try:
            import nltk
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Descargar tokenizador de NLTK si es necesario
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            # 1. Tokenizar respuesta en oraciones usando NLTK
            sentences = nltk.sent_tokenize(response, language='spanish')
            
            if not sentences:
                return 1.0
            
            # 2. Cargar modelo SBERT multilingüe
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            
            # 3. Obtener embedding del contexto completo
            context_embedding = model.encode(context, convert_to_tensor=True)
            
            # 4. Verificar cada oración contra el contexto
            supported_sentences = 0
            threshold = 0.5  # Umbral de similitud coseno
            
            for sentence in sentences:
                if len(sentence.strip()) < 5:  # Ignorar oraciones muy cortas
                    continue
                
                # Obtener embedding de la oración
                sentence_embedding = model.encode(sentence, convert_to_tensor=True)
                
                # Calcular similitud coseno
                similarity = float(cosine_similarity(
                    sentence_embedding.reshape(1, -1),
                    context_embedding.reshape(1, -1)
                )[0][0])
                
                # Si similitud > threshold, está soportada
                if similarity > threshold:
                    supported_sentences += 1
            
            # Calcular fidelidad como proporción de oraciones soportadas
            faithfulness = supported_sentences / len([s for s in sentences if len(s.strip()) >= 5])
            
            return min(1.0, max(0.0, faithfulness))
            
        except ImportError as e:
            print(f"⚠️ Error importando dependencias para faithfulness: {e}")
            print("   Usando método fallback basado en keywords...")
            # Fallback a método simple si faltan dependencias
            return self._calculate_faithfulness_fallback(response, context)
    
    def _calculate_faithfulness_fallback(self, response: str, context: str) -> float:
        """
        Método fallback para fidelidad si faltan NLTK/SBERT
        Usa solo word overlap + similitud básica
        """
        sentences = response.split('.')
        supported = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 5:
                continue
            
            # Verificar si hay suficiente overlap de palabras con contexto
            sentence_words = set(sentence.lower().split())
            context_words = set(context.lower().split())
            overlap = len(sentence_words.intersection(context_words))
            
            if len(sentence_words) > 0 and overlap / len(sentence_words) > 0.3:
                supported += 1
        
        valid_sentences = len([s for s in sentences if len(s.strip()) >= 5])
        return supported / valid_sentences if valid_sentences > 0 else 1.0
    
    def _calculate_relevancy(self, response: str, query: str) -> float:
        """
        Calcular relevancia de la respuesta a la pregunta (Spacy + Yake)
        
        Implementación según tesis 4.2.5:
        1. Usar Spacy para análisis lingüístico avanzado
        2. Usar Yake para extracción de palabras clave
        3. Análisis de entidades nombradas (NER)
        4. Análisis de dependencias sintácticas
        5. Similitud semántica con SBERT (complementario)
        """
        try:
            import spacy
            import yake
            from sentence_transformers import SentenceTransformer
            
            # 1. Cargar modelo de Spacy para español
            try:
                nlp = spacy.load('es_core_news_sm')
            except OSError:
                print("⚠️  Descargando modelo de Spacy español...")
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', 'es_core_news_sm'], 
                             capture_output=True)
                nlp = spacy.load('es_core_news_sm')
            
            # Procesar query y response
            query_doc = nlp(query)
            response_doc = nlp(response)
            
            # 2. Extracción de palabras clave con Yake
            kw_extractor = yake.KeywordExtractor(
                lan="es",
                n=3,  # Palabras clave de hasta 3 palabras
                top=15,  # Top 15 keywords
                features=None
            )
            
            query_keywords = [kw[0] for kw in kw_extractor.extract_keywords(query)]
            response_keywords = [kw[0] for kw in kw_extractor.extract_keywords(response)]
            
            # 3. Análisis de entidades nombradas (NER)
            query_entities = set([ent.text for ent in query_doc.ents])
            response_entities = set([ent.text for ent in response_doc.ents])
            entity_overlap = len(query_entities.intersection(response_entities))
            
            # 4. Análisis de dependencias sintácticas
            # Extraer palabras con dependencia de raíz (verbos principales)
            query_roots = set([token.text for token in query_doc if token.dep_ == "ROOT"])
            response_roots = set([token.text for token in response_doc if token.dep_ == "ROOT"])
            root_overlap = len(query_roots.intersection(response_roots))
            
            # 5. Calcular coincidencia de palabras clave (normalizado)
            # Buscar palabras clave que compartan tokens comunes
            keyword_matches = 0
            for qkw in query_keywords:
                qkw_tokens = set(qkw.lower().split())
                for rkw in response_keywords:
                    rkw_tokens = set(rkw.lower().split())
                    # Si comparten 40%+ de tokens, es una coincidencia
                    overlap = len(qkw_tokens.intersection(rkw_tokens))
                    total = max(len(qkw_tokens), len(rkw_tokens))
                    if total > 0 and overlap / total > 0.4:
                        keyword_matches += 1
                        break
            
            keyword_score = keyword_matches / len(query_keywords) if query_keywords else 0.0
            
            # 6. Calcular similitud de tokens (lemmas)
            query_lemmas = set([token.lemma_ for token in query_doc if not token.is_stop])
            response_lemmas = set([token.lemma_ for token in response_doc if not token.is_stop])
            lemma_overlap = len(query_lemmas.intersection(response_lemmas))
            lemma_score = lemma_overlap / len(query_lemmas) if query_lemmas else 0.0
            
            # 7. Similitud semántica con SBERT
            try:
                model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
                query_embedding = model.encode(query, convert_to_tensor=True)
                response_embedding = model.encode(response, convert_to_tensor=True)
                semantic_score = float(torch.nn.functional.cosine_similarity(query_embedding, response_embedding, dim=0))
                semantic_score = max(0.0, min(1.0, semantic_score))
            except:
                semantic_score = 0.0
            
            # 8. Combinar scores con pesos
            # Palabras clave (35%), Lemmas (25%), Similitud Semántica (25%), Entidades (10%), Raíces (5%)
            combined_score = (
                0.35 * keyword_score +
                0.25 * lemma_score +
                0.25 * semantic_score +
                0.10 * (entity_overlap / max(len(query_entities), 1)) +
                0.05 * (root_overlap / max(len(query_roots), 1))
            )
            
            return min(1.0, max(0.0, combined_score))
            
        except ImportError as e:
            print(f"⚠️  Error importando Spacy/Yake: {e}")
            print("   Usando método fallback...")
            return self._calculate_relevancy_fallback(response, query)
    
    def _calculate_relevancy_fallback(self, response: str, query: str) -> float:
        """
        Fallback para relevancia si faltan Spacy/Yake
        Usa overlap de lemmas básicos
        """
        # Simple overlap normalizado
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        overlap = len(query_words.intersection(response_words))
        return overlap / len(query_words) if query_words else 0.0
    
    def _calculate_context_utilization(self, response: str, context: str) -> float:
        """Calcular qué tan bien se utilizó el contexto"""
        # 1. Extraer frases clave del contexto
        context_phrases = set(self._extract_key_phrases(context))
        
        # 2. Contar cuántas se usaron en la respuesta
        used_phrases = sum(1 for phrase in context_phrases 
                         if phrase.lower() in response.lower())
        
        return used_phrases / len(context_phrases) if context_phrases else 0.0
    
    def _detect_hallucinations(self, response: str, context: str) -> float:
        """Detectar y cuantificar alucinaciones"""
        # 1. Extraer afirmaciones específicas
        statements = self._extract_statements(response)
        
        # 2. Verificar cada afirmación contra el contexto
        unsupported = 0
        for statement in statements:
            if not self._is_statement_supported(statement, context):
                unsupported += 1
        
        return unsupported / len(statements) if statements else 0.0
    
    def _count_source_citations(self, response: str) -> int:
        """Contar referencias a fuentes en la respuesta"""
        citation_indicators = [
            "según", "como indica", "de acuerdo a",
            "la fuente", "el documento", "la literatura"
        ]
        
        count = 0
        for indicator in citation_indicators:
            count += response.lower().count(indicator)
        
        return count
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extraer frases clave de un texto"""
        # Implementación simple - dividir por puntos y filtrar por longitud
        sentences = text.split('.')
        return [s.strip() for s in sentences if len(s.split()) > 3]
    
    def _extract_statements(self, text: str) -> List[str]:
        """Extraer afirmaciones factuales de un texto"""
        # Similar a key phrases pero filtrando por indicadores factuales
        statements = []
        sentences = text.split('.')
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in self.factual_indicators):
                statements.append(sentence.strip())
        
        return statements
    
    def _is_statement_supported(self, statement: str, context: str) -> bool:
        """Verificar si una afirmación está soportada por el contexto"""
        # Implementación básica - buscar overlap significativo
        statement_words = set(statement.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(statement_words.intersection(context_words))
        return overlap / len(statement_words) > 0.5
    
    def save_evaluation(self, result: EvaluationResult):
        """Guardar resultado de evaluación en archivo JSON"""
        filename = f"eval_{result.timestamp.replace(':', '-')}.json"
        filepath = self.log_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            print(f"📊 Evaluación guardada: {filename}")
        except Exception as e:
            print(f"❌ Error guardando evaluación: {e}")
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generar reporte agregado de evaluaciones"""
        if not results:
            return {"error": "No hay resultados para analizar"}
        
        # Promedios de métricas principales
        retrieval_metrics = {
            'avg_precision': np.mean([r.retrieval_metrics.context_precision for r in results]),
            'avg_mrr': np.mean([r.retrieval_metrics.mrr_score for r in results]),
            'avg_ndcg': np.mean([r.retrieval_metrics.ndcg_score for r in results]),
            'avg_retrieval_time': np.mean([r.retrieval_metrics.retrieval_time_ms for r in results])
        }
        
        generation_metrics = {
            'avg_faithfulness': np.mean([r.generation_metrics.faithfulness_score for r in results]),
            'avg_relevancy': np.mean([r.generation_metrics.answer_relevancy_score for r in results]),
            'avg_context_use': np.mean([r.generation_metrics.context_utilization_score for r in results]),
            'avg_hallucination': np.mean([r.generation_metrics.hallucination_score for r in results])
        }
        
        computational_metrics = {
            'avg_total_time': np.mean([r.computational_metrics.total_time_ms for r in results]),
            'avg_cpu': np.mean([r.computational_metrics.cpu_percent for r in results]),
            'avg_ram': np.mean([r.computational_metrics.ram_usage_mb for r in results])
        }
        
        # Análisis de calidad general
        quality_analysis = {
            'overall_score': np.mean([r.context_quality_score for r in results]),
            'high_quality_responses': len([r for r in results if r.context_quality_score > 0.8]),
            'low_quality_responses': len([r for r in results if r.context_quality_score < 0.5])
        }
        
        return {
            'num_evaluations': len(results),
            'retrieval_metrics': retrieval_metrics,
            'generation_metrics': generation_metrics,
            'computational_metrics': computational_metrics,
            'quality_analysis': quality_analysis,
            'timestamp': datetime.now().isoformat()
        }