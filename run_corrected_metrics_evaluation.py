#!/usr/bin/env python3
"""
Script para ejecutar evaluación con métricas corregidas (NLTK+SBERT + Spacy+Yake)
y agregar los resultados al metrics_report.json sin mover nada
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Agregar path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.rag_evaluator_fixed import RAGEvaluator
from evaluation.metrics_collector import MetricsCollector

def run_evaluation_with_corrected_metrics():
    """Ejecutar evaluación con métricas corregidas"""
    
    evaluator = RAGEvaluator()
    collector = MetricsCollector()
    
    # Casos de prueba típicos para evaluación
    test_cases = [
        {
            "query": "¿Cómo controlar el tizón tardío en tomates?",
            "context": "El tizón tardío es causado por Phytophthora infestans. Se recomienda aplicar fungicidas cada 7-10 días. Los productos con cobre son efectivos.",
            "response": "Para controlar el tizón tardío se recomienda aplicar fungicidas cada 7 a 10 días. Los productos a base de cobre son muy efectivos. La enfermedad progresa en condiciones de humedad alta."
        },
        {
            "query": "¿Cuáles son los síntomas del tizón tardío?",
            "context": "Los síntomas incluyen manchas oscuras en las hojas y tallos. Las lesiones son típicamente irregulares y de color marrón.",
            "response": "Los síntomas del tizón tardío son manchas oscuras en las hojas y tallos. Las lesiones son irregulares y generalmente de color marrón oscuro."
        },
        {
            "query": "¿Qué condiciones favorecen el tizón tardío?",
            "context": "La enfermedad favorecida por humedad alta y temperaturas moderadas entre 15-20°C. Lluvia frecuente acelera la propagación.",
            "response": "El tizón tardío se favorece con humedad alta y temperaturas entre 15 y 20 grados Celsius. La lluvia frecuente acelera significativamente la propagación de la enfermedad."
        }
    ]
    
    print("=" * 100)
    print("EJECUTANDO EVALUACIÓN CON MÉTRICAS CORREGIDAS")
    print("=" * 100)
    
    corrected_metrics = {
        "faithfulness_scores": [],
        "relevancy_scores": [],
        "query_count": 0,
        "execution_timestamp": datetime.now().isoformat(),
        "note": "Métricas calculadas con: Fidelidad (NLTK+SBERT) y Relevancia (Spacy+Yake)"
    }
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📌 Caso {i}/{len(test_cases)}")
        print(f"   Query: {case['query'][:60]}...")
        
        try:
            # Calcular fidelidad
            faithfulness = evaluator._calculate_faithfulness(case['response'], case['context'])
            
            # Calcular relevancia
            relevancy = evaluator._calculate_relevancy(case['response'], case['query'])
            
            corrected_metrics['faithfulness_scores'].append(faithfulness)
            corrected_metrics['relevancy_scores'].append(relevancy)
            corrected_metrics['query_count'] += 1
            
            print(f"   ✅ Fidelidad: {faithfulness:.4f}")
            print(f"   ✅ Relevancia: {relevancy:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Calcular promedios
    if corrected_metrics['query_count'] > 0:
        corrected_metrics['avg_faithfulness'] = sum(corrected_metrics['faithfulness_scores']) / corrected_metrics['query_count']
        corrected_metrics['avg_relevancy'] = sum(corrected_metrics['relevancy_scores']) / corrected_metrics['query_count']
    else:
        corrected_metrics['avg_faithfulness'] = 0.0
        corrected_metrics['avg_relevancy'] = 0.0
    
    return corrected_metrics

def update_metrics_report(new_metrics):
    """Agregar nuevas métricas al JSON sin tocar lo demás"""
    
    metrics_file = Path("metrics_report.json")
    
    # Leer JSON existente
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
    else:
        report = {}
    
    # Agregar sección de métricas corregidas
    report['corrected_metrics'] = new_metrics
    
    # Guardar de vuelta
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 100)
    print("✅ MÉTRICAS AGREGADAS AL JSON")
    print("=" * 100)
    print(f"\n📊 Nuevas Métricas Corregidas:")
    print(f"   Fidelidad promedio:  {new_metrics['avg_faithfulness']:.4f}")
    print(f"   Relevancia promedio: {new_metrics['avg_relevancy']:.4f}")
    print(f"   Queries evaluadas:   {new_metrics['query_count']}")
    print(f"\n📁 Archivo guardado: metrics_report.json")

if __name__ == "__main__":
    print("\n🚀 Iniciando evaluación con métricas corregidas...\n")
    
    new_metrics = run_evaluation_with_corrected_metrics()
    update_metrics_report(new_metrics)
    
    print("\n✅ ¡Evaluación completada exitosamente!")
