import time
import psutil
import GPUtil
from PyQt5.QtCore import QThread, pyqtSignal

class PerformanceOptimizer(QThread):
    optimization_signal = pyqtSignal(dict)

    def __init__(self, update_interval=1):
        super().__init__()
        self.update_interval = update_interval
        self._run_flag = True

    def run(self):
        while self._run_flag:
            metrics = self.get_metrics()
            optimizations = self.suggest_optimizations(metrics)
            self.optimization_signal.emit(optimizations)
            time.sleep(self.update_interval)

    def stop(self):
        self._run_flag = False
        self.wait()

    def get_metrics(self):
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        gpu_metrics = {}
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming we're using the first GPU
                gpu_metrics = {
                    "gpu_load": gpu.load * 100,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                }
        except:
            pass  # GPU metrics not available

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            **gpu_metrics
        }

    def suggest_optimizations(self, metrics):
        optimizations = {}
        
        if metrics["cpu_percent"] > 90:
            optimizations["cpu"] = "High CPU usage detected. Consider reducing video resolution or frame rate."
        
        if metrics["memory_percent"] > 90:
            optimizations["memory"] = "High memory usage detected. Consider closing unnecessary applications."
        
        if "gpu_load" in metrics and metrics["gpu_load"] > 90:
            optimizations["gpu"] = "High GPU usage detected. Consider reducing the number of active effects or lowering video quality."
        
        if "gpu_memory_used" in metrics and metrics["gpu_memory_used"] / metrics["gpu_memory_total"] > 0.9:
            optimizations["gpu_memory"] = "GPU memory is almost full. Try processing fewer video streams simultaneously."
        
        return optimizations