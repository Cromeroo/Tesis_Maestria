#!/usr/bin/env python3
"""
SISTEMA DE MÉTRICAS Y MONITOREO AVANZADO
Módulo complementario para recopilar métricas detalladas del sistema RAG LangGraph
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class NodeMetrics:
    """Métricas de ejecución de un nodo individual"""
    node_name: str
    start_time: float
    end_time: float
    execution_time_ms: float
    success: bool
    input_size: int
    output_size: int
    memory_usage_mb: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class RAGMetrics:
    """Métricas específicas de recuperación RAG"""
    query: str
    query_hash: str
    documents_retrieved: int
    avg_similarity_score: float
    max_similarity_score: float
    min_similarity_score: float
    collection_size: int
    search_time_ms: float
    unique_sources: int

@dataclass
class CNNMetrics:
    """Métricas de clasificación CNN"""
    image_path: str
    prediction: str
    confidence: float
    preprocessing_time_ms: float
    inference_time_ms: float
    model_load_time_ms: Optional[float] = None
    image_size_kb: Optional[float] = None

@dataclass
class ConversationMetrics:
    """Métricas de sesión conversacional"""
    session_id: str
    total_interactions: int
    avg_response_time_ms: float
    context_switches: int
    follow_up_detected: int
    user_satisfaction_indicators: Dict[str, int]

@dataclass
class SystemMetrics:
    """Métricas globales del sistema"""
    timestamp: str
    session_id: str
    total_execution_time_ms: float
    node_metrics: List[NodeMetrics]
    rag_metrics: Optional[RAGMetrics]
    cnn_metrics: Optional[CNNMetrics]
    conversation_metrics: Optional[ConversationMetrics]
    final_response_length: int
    context_detected: str
    success: bool
    error_count: int

class MetricsCollector:
    """Recolector central de métricas del sistema"""
    
    def __init__(self, log_directory: str = "metrics_logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        self.current_session_metrics = []
        self.active_node_metrics = {}
        self.session_start_time = time.time()
        
        # Contadores globales
        self.total_queries = 0
        self.total_errors = 0
        self.context_distribution = {}
        
    def start_session(self, session_id: str = None) -> str:
        """Iniciar nueva sesión de métricas"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.session_start_time = time.time()
        self.current_session_metrics = []
        
        print(f"📊 MÉTRICAS: Sesión iniciada - {session_id}")
        return session_id
    
    def start_node_execution(self, node_name: str, input_data: Any) -> str:
        """Marcar inicio de ejecución de nodo"""
        execution_id = f"{node_name}_{int(time.time() * 1000)}"
        
        input_size = len(str(input_data)) if input_data else 0
        
        self.active_node_metrics[execution_id] = {
            'node_name': node_name,
            'start_time': time.time(),
            'input_size': input_size
        }
        
        print(f"📊 NODO START: {node_name} (ID: {execution_id[-8:]})")
        return execution_id
    
    def end_node_execution(self, execution_id: str, output_data: Any, success: bool = True, error: str = None):
        """Marcar fin de ejecución de nodo"""
        if execution_id not in self.active_node_metrics:
            print(f"⚠️ MÉTRICAS: Execution ID no encontrado: {execution_id}")
            return
        
        metrics_data = self.active_node_metrics[execution_id]
        end_time = time.time()
        execution_time = (end_time - metrics_data['start_time']) * 1000  # ms
        
        output_size = len(str(output_data)) if output_data else 0
        
        node_metrics = NodeMetrics(
            node_name=metrics_data['node_name'],
            start_time=metrics_data['start_time'],
            end_time=end_time,
            execution_time_ms=round(execution_time, 2),
            success=success,
            input_size=metrics_data['input_size'],
            output_size=output_size,
            error_message=error
        )
        
        self.current_session_metrics.append(node_metrics)
        del self.active_node_metrics[execution_id]
        
        status = "✅" if success else "❌"
        print(f"📊 NODO END: {metrics_data['node_name']} {status} ({execution_time:.1f}ms)")
    
    def record_rag_metrics(self, query: str, documents: List[Dict], search_time_ms: float, collection_size: int):
        """Registrar métricas de recuperación RAG"""
        if not documents:
            similarity_scores = [0.0]
        else:
            similarity_scores = [doc.get('score', 0.0) for doc in documents]
        
        # Generar hash único de la query para análisis de patrones
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        # Contar fuentes únicas
        sources = set()
        for doc in documents:
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'unknown')
            sources.add(source)
        
        rag_metrics = RAGMetrics(
            query=query[:100] + "..." if len(query) > 100 else query,
            query_hash=query_hash,
            documents_retrieved=len(documents),
            avg_similarity_score=round(sum(similarity_scores) / len(similarity_scores), 3),
            max_similarity_score=round(max(similarity_scores), 3),
            min_similarity_score=round(min(similarity_scores), 3),
            collection_size=collection_size,
            search_time_ms=round(search_time_ms, 2),
            unique_sources=len(sources)
        )
        
        print(f"📊 RAG: {len(documents)} docs, avg_sim: {rag_metrics.avg_similarity_score}")
        return rag_metrics
    
    def record_cnn_metrics(self, image_path: str, prediction: str, confidence: float, 
                          preprocessing_time: float, inference_time: float):
        """Registrar métricas de clasificación CNN"""
        
        # Obtener tamaño de imagen si existe
        image_size_kb = None
        try:
            if Path(image_path).exists():
                image_size_kb = Path(image_path).stat().st_size / 1024
        except:
            pass
        
        cnn_metrics = CNNMetrics(
            image_path=str(Path(image_path).name),  # Solo nombre del archivo
            prediction=prediction,
            confidence=round(confidence, 3),
            preprocessing_time_ms=round(preprocessing_time * 1000, 2),
            inference_time_ms=round(inference_time * 1000, 2),
            image_size_kb=round(image_size_kb, 2) if image_size_kb else None
        )
        
        print(f"📊 CNN: {prediction} ({confidence:.1%}) - {inference_time*1000:.1f}ms")
        return cnn_metrics
    
    def finalize_session_metrics(self, final_response: str, context_detected: str, 
                                success: bool, error_count: int = 0) -> SystemMetrics:
        """Finalizar y guardar métricas de la sesión"""
        end_time = time.time()
        total_time = (end_time - self.session_start_time) * 1000
        
        # Actualizar contadores globales
        self.total_queries += 1
        self.total_errors += error_count
        
        if context_detected in self.context_distribution:
            self.context_distribution[context_detected] += 1
        else:
            self.context_distribution[context_detected] = 1
        
        system_metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            session_id=getattr(self, 'session_id', 'unknown'),
            total_execution_time_ms=round(total_time, 2),
            node_metrics=self.current_session_metrics.copy(),
            rag_metrics=None,  # Se asignará externamente si aplica
            cnn_metrics=None,  # Se asignará externamente si aplica
            conversation_metrics=None,  # Se asignará externamente si aplica
            final_response_length=len(final_response),
            context_detected=context_detected,
            success=success,
            error_count=error_count
        )
        
        print(f"📊 SESIÓN COMPLETA: {total_time:.1f}ms total, {len(self.current_session_metrics)} nodos")
        
        # Guardar métricas
        self._save_metrics(system_metrics)
        
        return system_metrics
    
    def _save_metrics(self, metrics: SystemMetrics):
        """Guardar métricas en archivo JSON"""
        filename = f"metrics_{metrics.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_directory / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
            print(f"📊 Métricas guardadas: {filename}")
        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generar reporte de rendimiento agregado"""
        if not self.current_session_metrics:
            return {"error": "No hay métricas disponibles"}
        
        # Análisis por nodo
        node_performance = {}
        for metric in self.current_session_metrics:
            node_name = metric.node_name
            if node_name not in node_performance:
                node_performance[node_name] = {
                    'executions': 0,
                    'total_time_ms': 0,
                    'avg_time_ms': 0,
                    'success_rate': 0,
                    'errors': 0
                }
            
            node_performance[node_name]['executions'] += 1
            node_performance[node_name]['total_time_ms'] += metric.execution_time_ms
            if not metric.success:
                node_performance[node_name]['errors'] += 1
        
        # Calcular promedios
        for node_name, data in node_performance.items():
            data['avg_time_ms'] = round(data['total_time_ms'] / data['executions'], 2)
            data['success_rate'] = round((data['executions'] - data['errors']) / data['executions'] * 100, 1)
        
        total_time = sum(m.execution_time_ms for m in self.current_session_metrics)
        
        report = {
            'session_summary': {
                'total_nodes_executed': len(self.current_session_metrics),
                'total_execution_time_ms': round(total_time, 2),
                'avg_node_time_ms': round(total_time / len(self.current_session_metrics), 2),
                'success_rate': round(sum(1 for m in self.current_session_metrics if m.success) / len(self.current_session_metrics) * 100, 1)
            },
            'node_performance': node_performance,
            'global_stats': {
                'total_queries_processed': self.total_queries,
                'total_errors': self.total_errors,
                'context_distribution': self.context_distribution
            }
        }
        
        return report
    
    def print_performance_summary(self):
        """Imprimir resumen de rendimiento en consola"""
        report = self.generate_performance_report()
        
        print("\n" + "="*60)
        print("📊 REPORTE DE RENDIMIENTO")
        print("="*60)
        
        summary = report['session_summary']
        print(f"🕐 Tiempo total: {summary['total_execution_time_ms']:.1f}ms")
        print(f"📈 Nodos ejecutados: {summary['total_nodes_executed']}")
        print(f"⚡ Tiempo promedio por nodo: {summary['avg_node_time_ms']:.1f}ms")
        print(f"✅ Tasa de éxito: {summary['success_rate']:.1f}%")
        
        print("\n📋 RENDIMIENTO POR NODO:")
        for node_name, perf in report['node_performance'].items():
            print(f"  {node_name}:")
            print(f"    • Tiempo promedio: {perf['avg_time_ms']:.1f}ms")
            print(f"    • Éxito: {perf['success_rate']:.1f}%")
        
        print("\n🌍 ESTADÍSTICAS GLOBALES:")
        print(f"  • Total consultas: {report['global_stats']['total_queries_processed']}")
        print(f"  • Total errores: {report['global_stats']['total_errors']}")
        
        if report['global_stats']['context_distribution']:
            print("  • Distribución de contextos:")
            for context, count in report['global_stats']['context_distribution'].items():
                print(f"    - {context}: {count}")
        
        print("="*60)

# Instancia global del collector
metrics_collector = MetricsCollector()

def track_node_execution(node_name: str):
    """Decorador para trackear automáticamente la ejecución de nodos"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            execution_id = metrics_collector.start_node_execution(node_name, args[1] if len(args) > 1 else None)
            
            try:
                result = func(*args, **kwargs)
                metrics_collector.end_node_execution(execution_id, result, success=True)
                return result
            except Exception as e:
                metrics_collector.end_node_execution(execution_id, None, success=False, error=str(e))
                raise
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # Demo de uso del sistema de métricas
    print("🚀 DEMO DEL SISTEMA DE MÉTRICAS")
    
    # Inicializar sesión
    session_id = metrics_collector.start_session("demo_metrics")
    
    # Simular ejecución de nodos
    exec_id_1 = metrics_collector.start_node_execution("detectar_contexto", {"pregunta": "¿Cómo tratar el tizón?"})
    time.sleep(0.1)  # Simular procesamiento
    metrics_collector.end_node_execution(exec_id_1, {"contexto": "campesino"}, success=True)
    
    exec_id_2 = metrics_collector.start_node_execution("buscar_documentos", {"query": "tizon tardio"})
    time.sleep(0.2)  # Simular búsqueda
    
    # Simular métricas RAG
    docs = [{"score": 0.85}, {"score": 0.72}, {"score": 0.69}]
    rag_metrics = metrics_collector.record_rag_metrics("tizon tardio", docs, 150.5, 472)
    
    metrics_collector.end_node_execution(exec_id_2, docs, success=True)
    
    # Finalizar sesión
    final_metrics = metrics_collector.finalize_session_metrics(
        final_response="Respuesta de ejemplo sobre el tizón tardío...",
        context_detected="campesino",
        success=True
    )
    
    # Mostrar reporte
    metrics_collector.print_performance_summary()