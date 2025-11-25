#!/usr/bin/env python3
"""
METRICS COLLECTOR
Recolector de métricas de rendimiento y calidad para sistemas RAG.
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class CollectorConfig:
    """Configuración del recolector de métricas"""
    save_dir: str = "metrics_logs"
    auto_save: bool = True
    track_metrics: List[str] = None
    
    def __post_init__(self):
        if self.track_metrics is None:
            self.track_metrics = [
                "retrieval_time",
                "generation_time",
                "total_time",
                "context_quality",
                "response_quality",
                "resource_usage"
            ]

class MetricsCollector:
    """
    Recolector de métricas para evaluación de sistemas RAG
    
    Características:
    - Recolección automática de tiempos
    - Tracking de calidad de contexto y respuestas
    - Métricas de uso de recursos
    - Exportación de reportes en JSON
    """
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.metrics_dir = Path(self.config.save_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "metrics": []
        }
        
        # Inicializar contadores
        self.reset_counters()
    
    def reset_counters(self):
        """Reiniciar contadores de métricas"""
        self.query_count = 0
        self.total_retrieval_time = 0
        self.total_generation_time = 0
        self.context_quality_scores = []
        self.response_quality_scores = []
    
    def start_query(self) -> Dict[str, float]:
        """Iniciar tracking de una nueva consulta"""
        self.query_count += 1
        timestamp = time.time()
        
        return {
            "query_start": timestamp,
            "retrieval_start": timestamp
        }
    
    def record_retrieval(self, timing_info: Dict[str, float], 
                        docs_retrieved: int, 
                        similarity_scores: List[float]) -> None:
        """Registrar métricas de recuperación"""
        retrieval_time = time.time() - timing_info["retrieval_start"]
        self.total_retrieval_time += retrieval_time
        
        metric = {
            "type": "retrieval",
            "timestamp": datetime.now().isoformat(),
            "retrieval_time_ms": retrieval_time * 1000,
            "docs_retrieved": docs_retrieved,
            "avg_similarity": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
            "max_similarity": max(similarity_scores) if similarity_scores else 0
        }
        
        self.current_session["metrics"].append(metric)
    
    def record_generation(self, timing_info: Dict[str, float], 
                         response_length: int,
                         context_tokens: int,
                         response_tokens: int) -> None:
        """Registrar métricas de generación"""
        generation_time = time.time() - timing_info.get("generation_start", timing_info["query_start"])
        self.total_generation_time += generation_time
        
        metric = {
            "type": "generation",
            "timestamp": datetime.now().isoformat(),
            "generation_time_ms": generation_time * 1000,
            "response_length": response_length,
            "context_tokens": context_tokens,
            "response_tokens": response_tokens,
            "tokens_per_second": response_tokens / generation_time if generation_time > 0 else 0
        }
        
        self.current_session["metrics"].append(metric)
    
    def record_quality_metrics(self, 
                             context_quality: float,
                             response_quality: float,
                             faithfulness: float,
                             relevance: float) -> None:
        """Registrar métricas de calidad"""
        self.context_quality_scores.append(context_quality)
        self.response_quality_scores.append(response_quality)
        
        metric = {
            "type": "quality",
            "timestamp": datetime.now().isoformat(),
            "context_quality": context_quality,
            "response_quality": response_quality,
            "faithfulness": faithfulness,
            "relevance": relevance
        }
        
        self.current_session["metrics"].append(metric)
    
    def record_resource_usage(self, cpu_percent: float, ram_mb: float,
                            gpu_usage: Optional[float] = None,
                            gpu_memory: Optional[float] = None) -> None:
        """Registrar métricas de uso de recursos"""
        metric = {
            "type": "resources",
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "ram_usage_mb": ram_mb
        }
        
        if gpu_usage is not None:
            metric["gpu_usage_percent"] = gpu_usage
            metric["gpu_memory_mb"] = gpu_memory
            
        self.current_session["metrics"].append(metric)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Obtener resumen de la sesión actual"""
        if not self.current_session["metrics"]:
            return {"error": "No hay métricas registradas en esta sesión"}
        
        retrieval_metrics = [m for m in self.current_session["metrics"] 
                           if m["type"] == "retrieval"]
        generation_metrics = [m for m in self.current_session["metrics"]
                            if m["type"] == "generation"]
        quality_metrics = [m for m in self.current_session["metrics"]
                         if m["type"] == "quality"]
        
        summary = {
            "session_info": {
                "start_time": self.current_session["start_time"],
                "end_time": datetime.now().isoformat(),
                "total_queries": self.query_count
            },
            "retrieval_stats": {
                "avg_time_ms": sum(m["retrieval_time_ms"] for m in retrieval_metrics) / len(retrieval_metrics)
                if retrieval_metrics else 0,
                "avg_docs": sum(m["docs_retrieved"] for m in retrieval_metrics) / len(retrieval_metrics)
                if retrieval_metrics else 0,
                "avg_similarity": sum(m["avg_similarity"] for m in retrieval_metrics) / len(retrieval_metrics)
                if retrieval_metrics else 0
            },
            "generation_stats": {
                "avg_time_ms": sum(m["generation_time_ms"] for m in generation_metrics) / len(generation_metrics)
                if generation_metrics else 0,
                "avg_tokens": sum(m["response_tokens"] for m in generation_metrics) / len(generation_metrics)
                if generation_metrics else 0,
                "avg_tokens_per_sec": sum(m["tokens_per_second"] for m in generation_metrics) / len(generation_metrics)
                if generation_metrics else 0
            },
            "quality_stats": {
                "avg_context_quality": sum(m["context_quality"] for m in quality_metrics) / len(quality_metrics)
                if quality_metrics else 0,
                "avg_response_quality": sum(m["response_quality"] for m in quality_metrics) / len(quality_metrics)
                if quality_metrics else 0,
                "avg_faithfulness": sum(m["faithfulness"] for m in quality_metrics) / len(quality_metrics)
                if quality_metrics else 0,
                "avg_relevance": sum(m["relevance"] for m in quality_metrics) / len(quality_metrics)
                if quality_metrics else 0
            }
        }
        
        return summary
    
    def save_metrics(self, filename: Optional[str] = None) -> None:
        """Guardar métricas actuales en archivo JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.metrics_dir / filename
        
        try:
            summary = self.get_session_summary()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": summary,
                    "detailed_metrics": self.current_session
                }, f, indent=2, ensure_ascii=False)
            print(f"📊 Métricas guardadas: {filename}")
        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")
    
    def get_full_report(self) -> Dict[str, Any]:
        """Generar reporte completo de métricas"""
        summary = self.get_session_summary()
        
        # Agregar análisis adicional
        resource_metrics = [m for m in self.current_session["metrics"]
                          if m["type"] == "resources"]
        
        if resource_metrics:
            summary["resource_stats"] = {
                "avg_cpu_percent": sum(m["cpu_percent"] for m in resource_metrics) / len(resource_metrics),
                "avg_ram_mb": sum(m["ram_usage_mb"] for m in resource_metrics) / len(resource_metrics),
            }
            
            # Agregar métricas GPU si existen
            if "gpu_usage_percent" in resource_metrics[0]:
                summary["resource_stats"].update({
                    "avg_gpu_percent": sum(m["gpu_usage_percent"] for m in resource_metrics) / len(resource_metrics),
                    "avg_gpu_memory_mb": sum(m["gpu_memory_mb"] for m in resource_metrics) / len(resource_metrics)
                })
        
        # Análisis de tendencias
        if len(self.context_quality_scores) > 1:
            summary["trends"] = {
                "context_quality_trend": "improving" if self.context_quality_scores[-1] > self.context_quality_scores[0]
                                      else "declining",
                "response_quality_trend": "improving" if self.response_quality_scores[-1] > self.response_quality_scores[0]
                                       else "declining"
            }
        
        return summary

# Ejemplo de uso
if __name__ == "__main__":
    # Crear colector con configuración personalizada
    config = CollectorConfig(
        save_dir="custom_metrics",
        auto_save=True,
        track_metrics=["retrieval_time", "generation_time", "quality"]
    )
    
    collector = MetricsCollector(config)
    
    # Simular tracking de consulta
    timing = collector.start_query()
    
    # Simular retrieval
    time.sleep(0.2)  # Simular trabajo
    collector.record_retrieval(
        timing_info=timing,
        docs_retrieved=3,
        similarity_scores=[0.92, 0.85, 0.76]
    )
    
    # Simular generación
    timing["generation_start"] = time.time()
    time.sleep(0.3)  # Simular trabajo
    collector.record_generation(
        timing_info=timing,
        response_length=150,
        context_tokens=512,
        response_tokens=64
    )
    
    # Registrar métricas de calidad
    collector.record_quality_metrics(
        context_quality=0.89,
        response_quality=0.92,
        faithfulness=0.95,
        relevance=0.88
    )
    
    # Registrar uso de recursos
    collector.record_resource_usage(
        cpu_percent=45.2,
        ram_mb=1250.5,
        gpu_usage=65.8,
        gpu_memory=2048.0
    )
    
    # Obtener y mostrar reporte
    report = collector.get_full_report()
    print("\n📊 REPORTE DE MÉTRICAS")
    print("="*50)
    
    for section, metrics in report.items():
        print(f"\n{section}:")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  • {k}: {v:.2f}")
                else:
                    print(f"  • {k}: {v}")
        else:
            print(f"  {metrics}")
    
    # Guardar métricas
    collector.save_metrics()