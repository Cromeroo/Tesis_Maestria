#!/usr/bin/env python3
"""
Backup completo del archivo con correcciones de sintaxis
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
import google.generativeai as genai
from config.settings import Settings

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from components.image_classifier import TomatoImageClassifier
from services.rag_pipeline import RAGPipeline


class IntegratedRAGClassifier:
    """Servicio integrado que combina clasificación de imágenes con RAG"""
    
    def __init__(self, model_repo="DaniloR2011/Tomato_accuracy"):
        """
        Inicializar el servicio integrado
        
        Args:
            model_repo: Repositorio de Hugging Face con el modelo
        """
        self.image_classifier = TomatoImageClassifier(model_repo=model_repo)
        self.rag_pipeline = RAGPipeline()
        
        # Configurar Gemini para respuestas con LLM
        genai.configure(api_key=Settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt especializado para manejo integrado
        self.integrated_management_prompt = """
Eres un agrónomo práctico especializado en tomate. Responde de manera directa y útil.

PRINCIPIOS:
1. ESCUCHA lo que el campesino realmente pregunta
2. RESPONDE directamente a su consulta específica 
3. COMPLEMENTA con 2-3 recomendaciones adicionales clave
4. USA lenguaje sencillo, sin jerga técnica excesiva
5. SÉ CONCISO pero completo en información esencial

ESTRUCTURA DE RESPUESTA:
1. Confirmación del problema (1-2 líneas)
2. Solución directa a lo que pregunta (productos, dosis, cómo aplicar)
3. 2-3 medidas complementarias importantes
4. Una advertencia de seguridad si aplica

LONGITUD: Máximo 200 palabras para respuestas básicas, 300 para casos complejos.

TONO: Como un agrónomo experimentado hablando con un productor - profesional pero cercano, directo pero servicial.

EVITAR: 
- Listas excesivamente largas
- Emojis (máximo 2-3 en toda la respuesta)
- Términos muy técnicos sin explicación
- Opciones múltiples cuando el usuario ya eligió una ruta

---

CONSULTA DEL CAMPESINO: {user_query}
DIAGNÓSTICO DE IMAGEN: {classification} (Confianza: {confidence:.0%})
INFORMACIÓN TÉCNICA: {rag_context}

Responde como agrónomo práctico siguiendo los principios arriba.
"""
        
        print("🚀 Inicializando servicio integrado RAG + Clasificador...")
        
    def diagnose_tomato_with_context(self, image_path_or_pil, user_query: str = "", 
                                   n_rag_results: int = 5) -> Dict:
        """
        Diagnóstico completo con contexto de usuario usando prompt de manejo integrado
        """
        print(f"🔍 Iniciando diagnóstico con contexto de usuario...")
        print(f"👤 Consulta usuario: {user_query}")
        
        # Paso 1: Clasificar imagen (si se proporciona)
        if image_path_or_pil is not None:
            print("📊 Clasificando imagen...")
            classification_result = self.image_classifier.classify_image(image_path_or_pil)
            
            if "error" in classification_result:
                return {
                    "success": False,
                    "error": classification_result["error"],
                    "classification": None,
                    "rag_context": [],
                    "integrated_response": ""
                }
        else:
            # Sin imagen, asumir consulta general
            classification_result = {
                "class_name": "Consulta_general",
                "confidence": 1.0,
                "probabilities": {}
            }
        
        # Paso 2: Obtener consultas RAG
        if image_path_or_pil is not None:
            rag_queries = self.image_classifier.get_rag_queries_for_classification(classification_result)
        else:
            rag_queries = [user_query]  # Solo la consulta del usuario
        
        # Agregar consultas de diversificación
        diversification_queries = self._get_diversification_queries(user_query, classification_result["class_name"])
        rag_queries.extend(diversification_queries)
        
        print(f"🔍 Ejecutando {len(rag_queries)} consultas RAG especializadas...")
        
        # Paso 3: Ejecutar consultas RAG
        all_rag_context = []
        document_counts = {}
        
        for query in rag_queries:
            print(f"   Consultando: {query[:50]}...")
            rag_results = self.rag_pipeline.search_knowledge_base(query, n_rag_results)
            
            if rag_results:
                diversified_results = self._diversify_document_sources(rag_results, document_counts, max_per_doc=2)
                all_rag_context.extend(diversified_results)
                
                for result in diversified_results:
                    doc_name = result["metadata"]["archivo"]
                    document_counts[doc_name] = document_counts.get(doc_name, 0) + 1
        
        # Paso 4: Formatear contexto para el prompt
        rag_context_text = self._format_rag_context_for_prompt(all_rag_context)
        
        # Paso 5: Generar respuesta con Gemini
        integrated_response = self._generate_integrated_management_response(
            user_query=user_query,
            classification=classification_result["class_name"],
            confidence=classification_result["confidence"],
            rag_context=rag_context_text
        )
        
        # Paso 6: Resultado final
        final_result = {
            "success": True,
            "user_query": user_query,
            "classification": classification_result,
            "rag_context": all_rag_context[:10],
            "integrated_response": integrated_response,
            "system_confidence": classification_result["confidence"],
            "knowledge_sources": len(all_rag_context),
            "suggested_actions": self._get_suggested_actions(classification_result["class_name"])
        }
        
        print("✅ Diagnóstico con contexto completado")
        return final_result
    
    def _format_rag_context_for_prompt(self, rag_results: List[Dict]) -> str:
        """Formatear resultados RAG para incluir en el prompt"""
        if not rag_results:
            return "No se encontró información específica en la base de conocimiento."
        
        formatted_context = []
        
        for i, result in enumerate(rag_results[:8]):
            content = result.get("text", "")  # Cambié "content" por "text"
            metadata = result.get("metadata", {})
            
            doc_info = ""
            if metadata.get("archivo"):
                doc_info = f"Fuente: {metadata['archivo']}"
                if metadata.get("theme"):
                    doc_info += f" (Tema: {metadata['theme']})"
            
            formatted_context.append(f"""
**FUENTE {i+1}:** {doc_info}
**CONTENIDO:** {content[:300]}...
**RELEVANCIA:** {result.get('distance', 'N/A')}
""")
        
        return "\\n".join(formatted_context)
    
    def _generate_integrated_management_response(self, user_query: str, 
                                               classification: str, 
                                               confidence: float,
                                               rag_context: str) -> str:
        """
        Generar respuesta práctica usando Gemini con el prompt especializado
        """
        try:
            full_prompt = self.integrated_management_prompt.format(
                user_query=user_query,
                classification=classification,
                confidence=confidence,
                rag_context=rag_context
            )
            
            print("🤖 Generando respuesta con Gemini...")
            
            response = self.model.generate_content(full_prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return self._get_fallback_response(classification, user_query)
                
        except Exception as e:
            print(f"⚠️  Error con Gemini: {e}")
            return self._get_fallback_response(classification, user_query)
    
    def _get_fallback_response(self, classification: str, user_query: str) -> str:
        """Respuesta de respaldo si falla Gemini"""
        query_lower = user_query.lower()
        
        if classification == "Tizon_tardio":
            if any(word in query_lower for word in ["quimico", "fungicida"]):
                return """Confirmo tizón tardío en tu tomate. Tratamiento químico directo:

**APLICACIÓN INMEDIATA:**
- Mancozeb 1.5 kg/ha + Metalaxil-M 200g/ha
- 400 litros agua por hectárea
- Aplicar temprano (6-8 AM)

**MEDIDAS COMPLEMENTARIAS:**
- Eliminar hojas infectadas y quemar
- Cambiar a riego por goteo
- Repetir en 7-10 días si persiste humedad

⚠️ Usar equipo de protección. No cosechar por 15 días."""
            else:
                return """Tienes tizón tardío. Requiere acción inmediata:

**TRATAMIENTO:**
- Fungicida sistémico (mancozeb + metalaxil)
- Eliminar hojas infectadas
- Mejorar ventilación

**PREVENCIÓN:**
- Riego por goteo exclusivamente
- Espaciamiento adecuado entre plantas
- Monitoreo semanal

La enfermedad avanza rápido con humedad. Actuar pronto es clave."""
        
        elif classification == "Sana":
            return """Tu planta se ve sana. Mantén la prevención:

**PRÁCTICAS PREVENTIVAS:**
- Riego temprano, nunca mojar hojas
- Espaciamiento mínimo 80cm
- Poda de ventilación semanal

**MONITOREO:**
- Revisar semanalmente por manchas
- Actuar rápido ante primeros síntomas

La prevención es más económica que el tratamiento."""
        
        else:
            return """Se detecta posible problema en la planta:

**RECOMENDACIÓN:**
- Monitoreo cercano de síntomas
- Mejorar condiciones de cultivo
- Consultar con técnico local si empeora

**PREVENCIÓN GENERAL:**
- Riego adecuado
- Nutrición balanceada
- Manejo integrado de plagas"""
    
    def _get_diversification_queries(self, user_query: str, classification: str) -> List[str]:
        """Generar consultas adicionales para diversificar fuentes RAG"""
        diversification_queries = []
        
        compilation_queries = [
            "rotación cultivos papa viceversa hospedantes",
            "malezas género Solanum trasplante",
            "surcos paralelos viento ventilación densidades",
            "variedades resistentes componente plan manejo",
            "control genético variedades inmunes susceptibles", 
            "aspersiones Manzate Dithane control preventivo"
        ]
        
        base_queries = [
            "tomate cultivo prácticas",
            "control enfermedades manejo", 
            "fungicidas aplicación",
            "manejo integrado cultivo"
        ]
        
        query_lower = user_query.lower()
        
        if "cultural" in query_lower or "biologico" in query_lower:
            diversification_queries.extend([
                "prácticas culturales drenaje terreno",
                "control orgánico natural rotación", 
                "riego ventilación espaciamiento",
                "eliminación restos plantas infectadas"
            ])
            diversification_queries.extend(compilation_queries[:2])
            
        elif "quimico" in query_lower or "fungicida" in query_lower:
            diversification_queries.extend([
                "fungicidas aplicación dosis",
                "mancozeb metalaxil control",
                "aspersión productos químicos",
                "tratamiento preventivo curativo"
            ])
            diversification_queries.extend(compilation_queries[4:6])
            
        elif "integrado" in query_lower:
            diversification_queries.extend([
                "estrategias control integrado manejo",
                "combinación métodos control prevención",
                "alternativas tratamiento variedades",
                "plan manejo componente"
            ])
            diversification_queries.extend(compilation_queries[2:5])
        
        if not any("rotación" in q or "variedades" in q for q in diversification_queries):
            diversification_queries.extend(compilation_queries[:2])
        
        diversification_queries.extend(base_queries[:2])
        
        return diversification_queries[:8]
    
    def _diversify_document_sources(self, rag_results: List[Dict], 
                                   document_counts: Dict[str, int], 
                                   max_per_doc: int = 2) -> List[Dict]:
        """Diversificar fuentes de documentos, priorizando el documento compilado"""
        compilation_doc_uuid = "5984b65c-112e-4362-afcc-777f9ea963ae.pdf"
        
        diversified_results = []
        temp_counts = document_counts.copy()
        
        # PRIORIDAD 1: Incluir resultados del documento compilado
        compilation_results = [r for r in rag_results if compilation_doc_uuid in r["metadata"]["archivo"]]
        if compilation_results:
            compilation_results.sort(key=lambda x: x["distance"], reverse=False)
            for result in compilation_results[:2]:
                diversified_results.append(result)
                doc_name = result["metadata"]["archivo"] 
                temp_counts[doc_name] = temp_counts.get(doc_name, 0) + 1
            print(f"🎯 Priorizando {len(compilation_results[:2])} resultados del documento compilado")
        
        # PRIORIDAD 2: Diversificar otros documentos
        other_results = [r for r in rag_results if compilation_doc_uuid not in r["metadata"]["archivo"]]
        
        master_doc_name = "62-manejo integrado del tizon tardio y estrategias de control quimico.pdf"
        master_results = []
        diverse_results = []
        
        for result in other_results:
            if master_doc_name in result["metadata"]["archivo"]:
                master_results.append(result)
            else:
                diverse_results.append(result)
        
        # Primero documentos diversos
        for result in diverse_results:
            doc_name = result["metadata"]["archivo"]
            current_count = temp_counts.get(doc_name, 0)
            
            if current_count < max_per_doc and len(diversified_results) < 6:
                diversified_results.append(result)
                temp_counts[doc_name] = current_count + 1
        
        # Luego máximo 1 del documento maestro
        if len(diversified_results) < 5 and master_results:
            master_results.sort(key=lambda x: x["distance"], reverse=False)
            doc_name = master_results[0]["metadata"]["archivo"]
            current_count = temp_counts.get(doc_name, 0)
            if current_count == 0:
                diversified_results.append(master_results[0])
                temp_counts[doc_name] = 1
        
        # Log para monitoreo
        source_summary = {}
        for result in diversified_results:
            source = result["metadata"]["archivo"]
            if compilation_doc_uuid in source:
                key = "📄 Documento Compilado Usuario"
            elif master_doc_name in source:
                key = "📋 Doc. Maestro Dominante" 
            else:
                key = f"📖 {result['metadata']['archivo'][:30]}..."
            source_summary[key] = source_summary.get(key, 0) + 1
            
        print(f"📚 Fuentes finales: {source_summary}")
        
        return diversified_results
    
    def _get_suggested_actions(self, classification: str) -> List[str]:
        """Obtener acciones sugeridas según la clasificación"""
        if classification == "Tizon_tardio":
            return [
                "Aplicar fungicida inmediatamente",
                "Eliminar hojas infectadas",
                "Mejorar ventilación del cultivo",
                "Cambiar método de riego"
            ]
        elif classification == "Sana":
            return [
                "Mantener programa preventivo",
                "Monitoreo semanal",
                "Riego temprano en la mañana",
                "Espaciamiento adecuado"
            ]
        else:
            return [
                "Monitorear síntomas de cerca",
                "Mejorar condiciones generales",
                "Consultar técnico local",
                "Aplicar medidas preventivas"
            ]

if __name__ == "__main__":
    # Test básico
    classifier = IntegratedRAGClassifier()
    print("✅ Sistema integrado inicializado correctamente")
