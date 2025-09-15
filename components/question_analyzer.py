#!/usr/bin/env python3
"""
ANALIZADOR DE PREGUNTAS CON DEBUGGING
Componente separado para analizar tipos de preguntas
"""

import re
from typing import Tuple, Dict, List

class QuestionAnalyzer:
    """Analizador inteligente de preguntas con logs detallados"""
    
    def __init__(self):
        # Patrones de palabras clave por categoría
        self.patterns = {
            'causas': [
                'causa', 'causas', 'por que', 'porque', 'por qué', 'porqué',
                'razón', 'razon', 'motivo', 'origen', 'debido', 'culpa',
                'apareció', 'aparecio', 'surgió', 'surgio', 'desarrollo'
            ],
            'cultural': [
                'cultural', 'culturales', 'no quimico', 'no químico', 
                'sin fungicida', 'sin químicos', 'sin quimicos', 'organico', 
                'orgánico', 'natural', 'ecologico', 'ecológico', 'biologico', 'biológico'
            ],
            'economico': [
                'economico', 'económico', 'economica', 'económica', 'economicas', 'económicas',
                'barato', 'baratos', 'barata', 'baratas', 'costo', 'costos',
                'precio', 'precios', 'dinero', 'gastar', 'presupuesto', 'ahorro',
                'ahorrar', 'efectivo', 'costo-efectivo'
            ],
            'urgencia': [
                'urgente', 'rapido', 'rápido', 'inmediato', 'emergencia',
                'que tan', 'qué tan', 'cuanto tiempo', 'cuánto tiempo', 'prioridad'
            ],
            'prevencion': [
                'prevenir', 'evitar', 'futuro', 'próxima', 'proxima',
                'siguiente', 'prevención', 'prevencion', 'proteger'
            ]
        }
    
    def analyze_question(self, question: str) -> Tuple[str, str, Dict]:
        """
        Analiza una pregunta y determina su tipo
        
        Returns:
            (response_type, category, debug_info)
        """
        print(f"\n🔍 ANALIZANDO PREGUNTA:")
        print(f"   Texto original: '{question}'")
        
        question_lower = question.lower()
        print(f"   Texto normalizado: '{question_lower}'")
        
        # Debug info
        debug_info = {
            'question_original': question,
            'question_normalized': question_lower,
            'matches_found': {},
            'decision_logic': []
        }
        
        # Buscar coincidencias en cada categoría
        category_matches = {}
        
        for category, keywords in self.patterns.items():
            matches = []
            for keyword in keywords:
                if keyword in question_lower:
                    matches.append(keyword)
            
            if matches:
                category_matches[category] = matches
                print(f"   ✅ {category.upper()}: {matches}")
            else:
                print(f"   ❌ {category.upper()}: sin coincidencias")
        
        debug_info['matches_found'] = category_matches
        
        # Lógica de decisión con prioridades
        if 'causas' in category_matches:
            if 'cultural' in category_matches:
                response_type = "🌱 CAUSAS + MANEJO CULTURAL"
                category = "causas_cultural"
                debug_info['decision_logic'].append("Causas + Cultural detectado")
            else:
                response_type = "🔍 ANÁLISIS DE CAUSAS"
                category = "causas"
                debug_info['decision_logic'].append("Solo causas detectado")
        
        elif 'urgencia' in category_matches:
            response_type = "🚨 EVALUACIÓN DE URGENCIA"
            category = "urgencia"
            debug_info['decision_logic'].append("Urgencia detectada")
        
        elif 'economico' in category_matches:
            response_type = "💰 SOLUCIÓN ECONÓMICA"
            category = "economico"
            debug_info['decision_logic'].append("Económico detectado")
        
        elif 'cultural' in category_matches:
            response_type = "🌱 MANEJO CULTURAL"
            category = "cultural"
            debug_info['decision_logic'].append("Cultural detectado")
            
        elif 'prevencion' in category_matches:
            response_type = "🛡️ PREVENCIÓN FUTURA"
            category = "prevencion"
            debug_info['decision_logic'].append("Prevención detectada")
        
        else:
            response_type = "📚 INFORMACIÓN GENERAL"
            category = "general"
            debug_info['decision_logic'].append("Ninguna categoría específica - usando general")
        
        print(f"   🎯 RESULTADO: {response_type} (categoría: {category})")
        print(f"   🧠 LÓGICA: {debug_info['decision_logic']}")
        
        return response_type, category, debug_info

def test_analyzer():
    """Función de prueba del analizador"""
    analyzer = QuestionAnalyzer()
    
    test_questions = [
        "¿Por qué apareció esta enfermedad en mi cultivo?",
        "Dame solo tratamiento cultural, no químicos",
        "¿Qué opciones económicas tengo?",
        "¿Qué tan urgente es actuar?",
        "¿Cómo prevengo esto en el futuro?",
        "¿Cuál es el mejor fungicida?"
    ]
    
    print("🧪 PRUEBAS DEL ANALIZADOR DE PREGUNTAS")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. PREGUNTA DE PRUEBA:")
        response_type, category, debug = analyzer.analyze_question(question)
        print(f"   ➡️ Tipo: {response_type}")
        print(f"   ➡️ Categoría: {category}")

if __name__ == "__main__":
    test_analyzer()