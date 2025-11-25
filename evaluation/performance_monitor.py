#!/usr/bin/env python3
"""
PERFORMANCE MONITOR
Monitor de rendimiento para sistemas RAG
"""

import time
import psutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch
import GPUtil
from threading import Thread
from queue import Queue
import signal
import sys

class PerformanceMonitor:
    """
    Monitor de rendimiento para sistemas RAG
    
    Características:
    - Monitoreo continuo de CPU/RAM/GPU
    - Tracking de latencias
    - Detección de cuellos de botella
    - Exportación de métricas
    """
    
    def __init__(self, log_dir: str = "performance_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.monitoring = False
        self.metrics_queue = Queue()
        self.samples = []
        
        # Configurar manejador de señales
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # Detectar disponibilidad de GPU
        self.gpu_available = torch.cuda.is_available()
        
        # Inicializar contadores
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reiniciar contadores y métricas"""
        self.start_time = time.time()
        self.peak_memory = 0
        self.peak_cpu = 0
        self.peak_gpu = 0 if self.gpu_available else None
        self.latencies = []
    
    def start_monitoring(self, interval: float = 1.0):
        """Iniciar monitoreo continuo"""
        if self.monitoring:
            print("⚠️ El monitoreo ya está activo")
            return
        
        self.monitoring = True
        self.monitor_thread = Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("✅ Monitoreo iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("🛑 Monitoreo detenido")
    
    def _monitor_loop(self, interval: float):
        """Loop principal de monitoreo"""
        while self.monitoring:
            metrics = self._collect_metrics()
            self.metrics_queue.put(metrics)
            self._update_peaks(metrics)
            self.samples.append(metrics)
            time.sleep(interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Recolectar métricas actuales"""
        timestamp = time.time()
        process = psutil.Process()
        
        metrics = {
            "timestamp": timestamp,
            "cpu_percent": psutil.cpu_percent(),
            "ram_total": psutil.virtual_memory().total / (1024 * 1024),  # MB
            "ram_used": process.memory_info().rss / (1024 * 1024),  # MB
            "ram_percent": psutil.virtual_memory().percent,
            "swap_used": psutil.swap_memory().used / (1024 * 1024),  # MB
            "swap_percent": psutil.swap_memory().percent,
            "disk_io": {
                "read_bytes": process.io_counters().read_bytes / (1024 * 1024),  # MB
                "write_bytes": process.io_counters().write_bytes / (1024 * 1024)  # MB
            }
        }
        
        # Agregar métricas GPU si está disponible
        if self.gpu_available:
            try:
                gpu = GPUtil.getGPUs()[0]  # Primera GPU
                metrics.update({
                    "gpu_load": gpu.load * 100,  # Porcentaje
                    "gpu_memory_total": gpu.memoryTotal,  # MB
                    "gpu_memory_used": gpu.memoryUsed,  # MB
                    "gpu_temperature": gpu.temperature  # °C
                })
            except Exception as e:
                print(f"⚠️ Error al obtener métricas GPU: {e}")
        
        return metrics
    
    def _update_peaks(self, metrics: Dict[str, Any]):
        """Actualizar valores pico"""
        self.peak_memory = max(self.peak_memory, metrics["ram_used"])
        self.peak_cpu = max(self.peak_cpu, metrics["cpu_percent"])
        
        if self.gpu_available and "gpu_load" in metrics:
            self.peak_gpu = max(self.peak_gpu or 0, metrics["gpu_load"])
    
    def _handle_interrupt(self, signum, frame):
        """Manejar interrupción (Ctrl+C)"""
        print("\n⚠️ Interrupción detectada, guardando métricas...")
        self.stop_monitoring()
        self.save_metrics()
        sys.exit(0)
    
    def record_latency(self, operation: str, duration_ms: float):
        """Registrar latencia de operación"""
        self.latencies.append({
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": time.time()
        })
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Obtener uso actual de recursos"""
        return self._collect_metrics()
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calcular métricas promedio"""
        if not self.samples:
            return {}
        
        avg_metrics = {
            "avg_cpu_percent": sum(s["cpu_percent"] for s in self.samples) / len(self.samples),
            "avg_ram_used": sum(s["ram_used"] for s in self.samples) / len(self.samples),
            "avg_ram_percent": sum(s["ram_percent"] for s in self.samples) / len(self.samples)
        }
        
        if self.gpu_available and "gpu_load" in self.samples[0]:
            avg_metrics.update({
                "avg_gpu_load": sum(s["gpu_load"] for s in self.samples) / len(self.samples),
                "avg_gpu_memory": sum(s["gpu_memory_used"] for s in self.samples) / len(self.samples)
            })
        
        return avg_metrics
    
    def get_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Calcular estadísticas de latencia por operación"""
        if not self.latencies:
            return {}
        
        stats = {}
        for op in set(l["operation"] for l in self.latencies):
            op_latencies = [l["duration_ms"] for l in self.latencies 
                          if l["operation"] == op]
            
            stats[op] = {
                "min": min(op_latencies),
                "max": max(op_latencies),
                "avg": sum(op_latencies) / len(op_latencies),
                "p95": sorted(op_latencies)[int(len(op_latencies) * 0.95)]
            }
        
        return stats
    
    def check_performance_issues(self) -> List[Dict[str, str]]:
        """Detectar posibles problemas de rendimiento"""
        issues = []
        metrics = self.get_current_usage()
        
        # CPU alto
        if metrics["cpu_percent"] > 80:
            issues.append({
                "type": "high_cpu",
                "severity": "warning",
                "message": f"Uso de CPU elevado: {metrics['cpu_percent']:.1f}%"
            })
        
        # RAM alta
        if metrics["ram_percent"] > 85:
            issues.append({
                "type": "high_memory",
                "severity": "critical",
                "message": f"Uso de RAM elevado: {metrics['ram_percent']:.1f}%"
            })
        
        # Swap alto
        if metrics["swap_percent"] > 60:
            issues.append({
                "type": "high_swap",
                "severity": "warning",
                "message": f"Uso de SWAP elevado: {metrics['swap_percent']:.1f}%"
            })
        
        # GPU si está disponible
        if self.gpu_available and "gpu_load" in metrics:
            if metrics["gpu_load"] > 90:
                issues.append({
                    "type": "high_gpu",
                    "severity": "warning",
                    "message": f"Uso de GPU elevado: {metrics['gpu_load']:.1f}%"
                })
            
            if metrics["gpu_temperature"] > 80:
                issues.append({
                    "type": "high_temp",
                    "severity": "critical",
                    "message": f"Temperatura GPU alta: {metrics['gpu_temperature']:.1f}°C"
                })
        
        return issues
    
    def save_metrics(self, filename: Optional[str] = None) -> None:
        """Guardar métricas en archivo JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        # Preparar reporte
        report = {
            "session_info": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": time.time() - self.start_time
            },
            "peaks": {
                "cpu_percent": self.peak_cpu,
                "ram_mb": self.peak_memory,
                "gpu_percent": self.peak_gpu if self.gpu_available else None
            },
            "averages": self.get_average_metrics(),
            "latency_stats": self.get_latency_stats(),
            "performance_issues": self.check_performance_issues(),
            "detailed_samples": self.samples
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📊 Métricas de rendimiento guardadas: {filename}")
        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generar reporte completo de rendimiento"""
        current = self.get_current_usage()
        averages = self.get_average_metrics()
        latencies = self.get_latency_stats()
        issues = self.check_performance_issues()
        
        report = {
            "summary": {
                "duration_seconds": time.time() - self.start_time,
                "samples_collected": len(self.samples),
                "current_status": "healthy" if not issues else "warning"
            },
            "current_metrics": current,
            "peak_metrics": {
                "cpu_percent": self.peak_cpu,
                "ram_mb": self.peak_memory,
                "gpu_percent": self.peak_gpu if self.gpu_available else None
            },
            "average_metrics": averages,
            "latency_statistics": latencies,
            "performance_issues": issues
        }
        
        # Agregar análisis de tendencias
        if len(self.samples) > 1:
            report["trends"] = self._analyze_trends()
        
        return report
    
    def _analyze_trends(self) -> Dict[str, str]:
        """Analizar tendencias en las métricas"""
        if len(self.samples) < 2:
            return {}
        
        # Tomar primeras y últimas muestras para comparación
        window_size = min(10, len(self.samples))
        early_samples = self.samples[:window_size]
        recent_samples = self.samples[-window_size:]
        
        trends = {}
        
        # CPU
        early_cpu = sum(s["cpu_percent"] for s in early_samples) / window_size
        recent_cpu = sum(s["cpu_percent"] for s in recent_samples) / window_size
        trends["cpu_trend"] = "increasing" if recent_cpu > early_cpu * 1.1 else \
                            "decreasing" if recent_cpu < early_cpu * 0.9 else \
                            "stable"
        
        # RAM
        early_ram = sum(s["ram_used"] for s in early_samples) / window_size
        recent_ram = sum(s["ram_used"] for s in recent_samples) / window_size
        trends["ram_trend"] = "increasing" if recent_ram > early_ram * 1.1 else \
                            "decreasing" if recent_ram < early_ram * 0.9 else \
                            "stable"
        
        # GPU si está disponible
        if self.gpu_available and "gpu_load" in self.samples[0]:
            early_gpu = sum(s["gpu_load"] for s in early_samples) / window_size
            recent_gpu = sum(s["gpu_load"] for s in recent_samples) / window_size
            trends["gpu_trend"] = "increasing" if recent_gpu > early_gpu * 1.1 else \
                                "decreasing" if recent_gpu < early_gpu * 0.9 else \
                                "stable"
        
        return trends

# Ejemplo de uso
if __name__ == "__main__":
    # Crear monitor
    monitor = PerformanceMonitor()
    
    # Iniciar monitoreo
    monitor.start_monitoring(interval=0.5)
    
    try:
        # Simular carga de trabajo
        print("🔄 Simulando carga de trabajo...")
        for i in range(5):
            # Simular operación de retrieval
            time.sleep(0.2)
            monitor.record_latency("retrieval", 150.5)
            
            # Simular operación de generación
            time.sleep(0.3)
            monitor.record_latency("generation", 350.2)
            
            # Simular uso de recursos
            data = b"x" * (10 * 1024 * 1024)  # 10MB
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupción manual detectada")
    
    finally:
        # Detener monitoreo y generar reporte
        monitor.stop_monitoring()
        report = monitor.generate_report()
        
        print("\n📊 REPORTE DE RENDIMIENTO")
        print("="*50)
        
        for section, data in report.items():
            print(f"\n{section}:")
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, float):
                        print(f"  • {k}: {v:.2f}")
                    else:
                        print(f"  • {k}: {v}")
            else:
                print(f"  {data}")
        
        # Guardar métricas
        monitor.save_metrics()