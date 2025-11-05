#!/usr/bin/env python3
"""
RAG EVALUATOR FIXED
Evaluador de métricas para sistemas RAG con cálculo correcto de MRR y NDCG
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class RetrievalMetrics:
    """Métricas de recuperación de documentos"""
    mrr_score: float  # Mean Reciprocal Rank
    ndcg_score: float  # Normalized Discounted Cumulative Gain
    context_precision: float  # Precisión del contexto
    similarity_scores: List[float]  # Scores de similitud de cada doc
    relevance_scores: List[float]  # Scores de relevancia de cada doc
    doc_rankings: List[int]  # Ranking de posición de cada documento

@dataclass
class GenerationMetrics:
    """Métricas de generación de respuestas"""
    faithfulness_score: float
    answer_relevancy_score: float
    rouge_score: float

class RAGEvaluator:
    """
    Evaluador de sistemas RAG
    
    Calcula:
    1. MRR: Posición del primer documento relevante
    2. NDCG: Ranking normalizado considerando múltiples niveles de relevancia
    3. Precision@k: Documentos relevantes en top-k
    4. Faithfulness: Fidelidad de la respuesta al contexto
    5. Answer Relevancy: Relevancia de la respuesta a la pregunta
    """
    
    def __init__(self, relevance_threshold: float = 0.7):
        """
        Inicializar evaluador
        
        Args:
            relevance_threshold: Umbral de similitud para considerar un doc como relevante
        """
        self.relevance_threshold = relevance_threshold
        self.relevance_keywords = {
            "tizon": ["tizon", "tizón", "phytophthora", "late blight"],
            "tomate": ["tomate", "tomato", "cultivo", "crop"],
            "tratamiento": ["fungicida", "control", "tratamiento", "fungicide", "spray"],
            "sintomas": ["síntoma", "symptoms", "mancha", "spots", "pudrición"]
        }
    
    def evaluate_retrieval(self, 
                          query: str,
                          retrieved_docs: List[Dict],
                          retrieval_time_ms: float = 0.0,
                          k: int = 10) -> RetrievalMetrics:
        """
        Evaluar métricas de recuperación de documentos
        
        ⭐ AQUÍ ES DONDE SE CALCULA LA RELEVANCIA DE CADA DOCUMENTO
        
        Args:
            query: Pregunta del usuario
            retrieved_docs: Lista de documentos recuperados con scores
            retrieval_time_ms: Tiempo de recuperación en ms
            k: Top-k documentos a considerar
            
        Returns:
            RetrievalMetrics con MRR, NDCG y otras métricas
        """
        
        # Paso 1: CALCULAR RELEVANCIA DE CADA DOCUMENTO
        relevance_scores = []
        similarity_scores = []
        
        for doc in retrieved_docs[:k]:
            # Extraer puntuación de similitud
            similarity = doc.get("distance", 0.0)
            if isinstance(similarity, (int, float)):
                # Si es distancia, convertir a similitud (0 a 1)
                # Fórmula: similitud = 1 / (1 + distancia)
                similarity = 1.0 / (1.0 + similarity)
            
            similarity_scores.append(similarity)
            
            # Paso 2: CALCULAR RELEVANCIA BASADA EN:
            # 1. Similitud semántica (distance/embedding)
            # 2. Contenido del documento (palabras clave)
            # 3. Metadatos
            
            relevance = self._calculate_document_relevance(
                query=query,
                doc=doc,
                similarity_score=similarity
            )
            relevance_scores.append(relevance)
        
        # Paso 3: CALCULAR MRR (Mean Reciprocal Rank)
        # MRR = 1 / posición del primer documento relevante
        mrr_score = self._calculate_mrr(relevance_scores)
        
        # Paso 4: CALCULAR NDCG (Normalized Discounted Cumulative Gain)
        ndcg_score = self._calculate_ndcg(relevance_scores, k=k)
        
        # Paso 5: CALCULAR PRECISION DE CONTEXTO
        # Qué porcentaje de documentos recuperados son relevantes
        context_precision = sum(1 for score in relevance_scores if score >= 0.5) / len(relevance_scores) if relevance_scores else 0.0
        
        # Paso 6: RANKING DE DOCUMENTOS
        doc_rankings = self._get_document_rankings(relevance_scores)
        
        return RetrievalMetrics(
            mrr_score=mrr_score,
            ndcg_score=ndcg_score,
            context_precision=context_precision,
            similarity_scores=similarity_scores,
            relevance_scores=relevance_scores,
            doc_rankings=doc_rankings
        )
    
    def _calculate_document_relevance(self, 
                                     query: str, 
                                     doc: Dict, 
                                     similarity_score: float) -> float:
        """
        🔴 AQUÍ SE CALCULA LA RELEVANCIA DE UN DOCUMENTO
        
        Combina 3 factores:
        1. Similitud semántica (60% peso)
        2. Coincidencia de palabras clave (25% peso)
        3. Puntuación de metadatos (15% peso)
        
        Args:
            query: Pregunta del usuario
            doc: Documento recuperado
            similarity_score: Score de similitud (0-1)
            
        Returns:
            Score de relevancia (0-1)
        """
        
        # FACTOR 1: SIMILITUD SEMÁNTICA (60%)
        # Es el score del embedding/distancia
        semantic_relevance = min(1.0, max(0.0, similarity_score))
        
        # FACTOR 2: COINCIDENCIA DE PALABRAS CLAVE (25%)
        # Cuenta cuántas palabras clave de la query aparecen en el documento
        keyword_relevance = self._calculate_keyword_relevance(query, doc)
        
        # FACTOR 3: PUNTUACIÓN DE METADATOS (15%)
        # Bonus por metadatos confiables
        metadata_relevance = self._calculate_metadata_relevance(doc)
        
        # COMBINACIÓN PONDERADA
        combined_relevance = (
            semantic_relevance * 0.60 +      # 60% es similitud semántica
            keyword_relevance * 0.25 +       # 25% es coincidencia de keywords
            metadata_relevance * 0.15        # 15% es metadatos
        )
        
        # Normalizar a rango 0-1
        relevance_score = min(1.0, max(0.0, combined_relevance))
        
        return relevance_score
    
    def _calculate_keyword_relevance(self, query: str, doc: Dict) -> float:
        """
        Calcular relevancia basada en coincidencia de palabras clave
        
        Busca palabras de la query en el contenido del documento
        """
        query_lower = query.lower()
        doc_content = doc.get("page_content", "").lower()
        doc_metadata = str(doc.get("metadata", {})).lower()
        
        # Palabras importantes de la query (removiendo stopwords)
        query_words = [w for w in query_lower.split() if len(w) > 3]
        
        if not query_words:
            return 0.5
        
        # Contar coincidencias
        matches = 0
        for word in query_words:
            if word in doc_content or word in doc_metadata:
                matches += 1
        
        # Score: qué porcentaje de palabras de la query aparecen en el documento
        keyword_score = matches / len(query_words) if query_words else 0.0
        
        return min(1.0, keyword_score)
    
    def _calculate_metadata_relevance(self, doc: Dict) -> float:
        """
        Calcular relevancia basada en metadatos
        
        Bonus si el documento viene de fuentes confiables
        """
        metadata = doc.get("metadata", {})
        
        # Bonus por documento compilado (es manual, muy confiable)
        if "5984b65c-112e-4362-afcc-777f9ea963ae.pdf" in str(metadata.get("archivo", "")):
            return 0.9
        
        # Bonus por documento maestro de manejo integrado
        if "62-manejo integrado" in str(metadata.get("archivo", "")).lower():
            return 0.8
        
        # Neutral para otros documentos
        return 0.6
    
    def _calculate_mrr(self, relevance_scores: List[float]) -> float:
        """
        Calcular MRR (Mean Reciprocal Rank)
        
        MRR = 1 / posición del primer documento con relevancia >= threshold
        
        Ejemplo:
        - Si el doc más relevante está en posición 1: MRR = 1/1 = 1.0 (perfecto)
        - Si el doc más relevante está en posición 3: MRR = 1/3 = 0.333
        - Si ninguno supera threshold: MRR = 0.0
        """
        
        for position, relevance in enumerate(relevance_scores, start=1):
            if relevance >= self.relevance_threshold:
                return 1.0 / position
        
        # Si ninguno supera el umbral
        return 0.0
    
    def _calculate_ndcg(self, relevance_scores: List[float], k: int = 10) -> float:
        """
        Calcular NDCG (Normalized Discounted Cumulative Gain)
        
        NDCG considera:
        1. Múltiples niveles de relevancia (0-1, no binario)
        2. Penaliza documentos mal posicionados
        3. Se normaliza contra el ranking ideal
        
        Fórmula:
        DCG@k = sum(relevance_i / log2(i+1)) para i=1 a k
        IDCG@k = DCG del ranking ideal (scores ordenados descendente)
        NDCG = DCG / IDCG
        """
        
        if not relevance_scores:
            return 0.0
        
        # DCG: Discounted Cumulative Gain actual
        dcg = 0.0
        for position, relevance in enumerate(relevance_scores[:k], start=1):
            # Descuento logarítmico: documentos al inicio valen más
            discount = np.log2(position + 1)
            dcg += relevance / discount
        
        # IDCG: Ranking ideal (todos ordenados de mayor a menor relevancia)
        ideal_ranking = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for position, relevance in enumerate(ideal_ranking[:k], start=1):
            discount = np.log2(position + 1)
            idcg += relevance / discount
        
        # NDCG normalizado
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return min(1.0, max(0.0, ndcg))
    
    def _get_document_rankings(self, relevance_scores: List[float]) -> List[int]:
        """
        Obtener ranking de documentos ordenados por relevancia
        
        Returns:
            Lista de posiciones ordenadas por relevancia descendente
        """
        rankings = sorted(
            range(len(relevance_scores)),
            key=lambda i: relevance_scores[i],
            reverse=True
        )
        return rankings
    
    def evaluate_generation(self,
                           response: str,
                           context: str,
                           query: str,
                           generation_time_ms: float = 0.0) -> GenerationMetrics:
        """
        Evaluar métricas de generación de respuesta
        
        Args:
            response: Respuesta generada
            context: Contexto usado (documentos recuperados)
            query: Pregunta original
            generation_time_ms: Tiempo de generación
            
        Returns:
            GenerationMetrics con faithfulness, relevance, etc.
        """
        
        # Calcular fidelidad (respuesta usa contenido del contexto)
        faithfulness = self._calculate_faithfulness(response, context)
        
        # Calcular relevancia de respuesta a la pregunta
        answer_relevancy = self._calculate_answer_relevancy(response, query)
        
        # Calcular ROUGE (similitud entre respuesta y query)
        rouge = self._calculate_rouge_similarity(response, query)
        
        return GenerationMetrics(
            faithfulness_score=faithfulness,
            answer_relevancy_score=answer_relevancy,
            rouge_score=rouge
        )
    
    def _calculate_faithfulness(self, response: str, context: str) -> float:
        """
        Calcular fidelidad: ¿La respuesta es fiel al contexto?
        
        Mide si la respuesta contiene información presente en el contexto
        """
        if not context or not response:
            return 0.0
        
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Palabras importantes de la respuesta
        response_words = [w for w in response_lower.split() if len(w) > 4]
        
        if not response_words:
            return 0.5
        
        # Contar cuántas aparecen en el contexto
        matches = sum(1 for word in response_words if word in context_lower)
        
        faithfulness = matches / len(response_words) if response_words else 0.0
        
        return min(1.0, max(0.0, faithfulness))
    
    def _calculate_answer_relevancy(self, response: str, query: str) -> float:
        """
        Calcular relevancia de respuesta: ¿La respuesta contesta la pregunta?
        
        Compara palabras clave entre query y respuesta
        """
        query_lower = query.lower()
        response_lower = response.lower()
        
        query_words = [w for w in query_lower.split() if len(w) > 3]
        
        if not query_words:
            return 0.5
        
        # Contar coincidencias
        matches = sum(1 for word in query_words if word in response_lower)
        
        relevancy = matches / len(query_words) if query_words else 0.0
        
        return min(1.0, max(0.0, relevancy))
    
    def _calculate_rouge_similarity(self, text1: str, text2: str) -> float:
        """
        Calcular similitud ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
        
        Mide overlap de palabras entre dos textos
        """
        text1_words = set(text1.lower().split())
        text2_words = set(text2.lower().split())
        
        if not text1_words or not text2_words:
            return 0.0
        
        intersection = text1_words & text2_words
        union = text1_words | text2_words
        
        rouge = len(intersection) / len(union) if union else 0.0
        
        return min(1.0, max(0.0, rouge))

# Ejemplo de uso
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DEMO: Cómo se calcula la RELEVANCIA de cada documento")
    print("="*60)
    
    evaluator = RAGEvaluator(relevance_threshold=0.7)
    
    # Documentos de ejemplo
    docs = [
        {
            "page_content": "El tizón tardío causado por Phytophthora infestans es una enfermedad fúngica que afecta principalmente a la papa y el tomate. Se controla con fungicidas sistémicos.",
            "metadata": {"archivo": "documento1.pdf", "tema": "tizon"},
            "distance": 0.15
        },
        {
            "page_content": "Los tomates rojos son ricos en licopeno y vitamina C. Se cultivan en diferentes climas.",
            "metadata": {"archivo": "documento2.pdf", "tema": "nutricion"},
            "distance": 0.45
        },
        {
            "page_content": "Fungicidas como clorotalonil y mancozeb son efectivos contra Phytophthora. Se aplican cada 10 días.",
            "metadata": {"archivo": "62-manejo integrado del tizon tardio y estrategias de control quimico.pdf"},
            "distance": 0.20
        }
    ]
    
    query = "¿Cómo tratar el tizón tardío en tomate con fungicidas?"
    
    print(f"\n📝 Pregunta: {query}\n")
    print("="*60)
    print("ANÁLISIS DE RELEVANCIA DE CADA DOCUMENTO")
    print("="*60)
    
    # Calcular métricas
    metrics = evaluator.evaluate_retrieval(
        query=query,
        retrieved_docs=docs,
        k=3
    )
    
    # Mostrar análisis por documento
    for i, (doc, relevance, similarity) in enumerate(zip(docs, metrics.relevance_scores, metrics.similarity_scores), 1):
        print(f"\n📄 Documento {i}:")
        print(f"   Archivo: {doc['metadata'].get('archivo', 'desconocido')}")
        print(f"   Contenido: {doc['page_content'][:60]}...")
        print(f"   ├─ Similitud semántica: {similarity:.3f}")
        print(f"   ├─ Relevancia TOTAL: {relevance:.3f}")
        print(f"   └─ ¿Relevante?: {'✅ SÍ' if relevance >= 0.7 else '❌ NO'}")
    
    print(f"\n{'='*60}")
    print("MÉTRICAS FINALES")
    print(f"{'='*60}")
    print(f"MRR Score: {metrics.mrr_score:.3f}")
    print(f"  → Posición del primer doc relevante: {1/metrics.mrr_score:.0f} " if metrics.mrr_score > 0 else "  → No hay documento relevante")
    print(f"NDCG Score: {metrics.ndcg_score:.3f}")
    print(f"Context Precision: {metrics.context_precision:.3f}")
    print(f"\n{'✅ Sistema RAG funcionando correctamente!' if metrics.mrr_score > 0 else '❌ Necesita mejorar ranking de documentos'}")
