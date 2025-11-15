"""
Performance Benchmarking System
Part of SuperShader Project - Phase 8: Testing and Quality Assurance

This module implements performance benchmarking for individual modules,
profiling tools, performance regression testing, and GPU utilization monitoring.
"""

import time
import psutil
import GPUtil
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import json
import threading
import statistics


@dataclass
class BenchmarkResult:
    """Represents the result of a performance benchmark"""
    test_name: str
    execution_time: float
    fps: float
    memory_usage_mb: float
    gpu_usage_percent: float
    params: Dict[str, Any]
    timestamp: float


class ShaderModuleProfiler:
    """
    Profiling tools for individual shader modules
    """
    
    def __init__(self):
        self.profiling_results: List[BenchmarkResult] = []
        self.monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.system_stats: List[Dict[str, float]] = []
    
    def start_monitoring(self) -> None:
        """Start system resource monitoring in a background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system_resources)
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop system resource monitoring"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitor_system_resources(self) -> None:
        """Monitor system resources in a background thread"""
        while self.monitoring:
            stats = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
            
            # Get GPU stats if available
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                stats['gpu_percent'] = gpu.load * 100
                stats['gpu_memory_percent'] = gpu.memoryUtil * 100
                stats['gpu_temperature'] = gpu.temperature
            else:
                stats['gpu_percent'] = 0
                stats['gpu_memory_percent'] = 0
                stats['gpu_temperature'] = 0
            
            self.system_stats.append(stats)
            time.sleep(0.1)  # Monitor every 100ms
    
    def profile_shader_module(self, shader_func: Callable, params: Dict[str, Any], 
                            iterations: int = 1000) -> BenchmarkResult:
        """
        Profile a shader module function with given parameters
        """
        # Start monitoring
        self.start_monitoring()
        
        # Store initial resource usage
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        
        # Execute the shader function multiple times
        for _ in range(iterations):
            # In a real implementation, this would execute the actual shader
            # For simulation, we'll just call the function
            if callable(shader_func):
                shader_func(params)
        
        end_time = time.time()
        
        # Stop monitoring
        self.stop_monitoring()
        
        execution_time = end_time - start_time
        fps = iterations / execution_time if execution_time > 0 else float('inf')
        
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        memory_used = final_memory - initial_memory
        
        # Get GPU usage from monitoring data
        if self.system_stats:
            avg_gpu_usage = statistics.mean([s['gpu_percent'] for s in self.system_stats])
        else:
            avg_gpu_usage = 0
        
        result = BenchmarkResult(
            test_name=f"profiling_{shader_func.__name__ if callable(shader_func) else 'unknown'}",
            execution_time=execution_time,
            fps=fps,
            memory_usage_mb=memory_used,
            gpu_usage_percent=avg_gpu_usage,
            params=params,
            timestamp=time.time()
        )
        
        self.profiling_results.append(result)
        return result
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get a summary of all profiling results"""
        if not self.profiling_results:
            return {"message": "No profiling data available"}
        
        execution_times = [r.execution_time for r in self.profiling_results]
        fps_values = [r.fps for r in self.profiling_results if r.fps != float('inf')]
        memory_usages = [r.memory_usage_mb for r in self.profiling_results]
        gpu_usages = [r.gpu_usage_percent for r in self.profiling_results]
        
        return {
            "total_tests": len(self.profiling_results),
            "execution_time_stats": {
                "min": min(execution_times) if execution_times else 0,
                "max": max(execution_times) if execution_times else 0,
                "avg": statistics.mean(execution_times) if execution_times else 0,
                "median": statistics.median(execution_times) if execution_times else 0
            },
            "fps_stats": {
                "min": min(fps_values) if fps_values else 0,
                "max": max(fps_values) if fps_values else float('inf'),
                "avg": statistics.mean(fps_values) if fps_values else 0,
                "median": statistics.median(fps_values) if fps_values else 0
            },
            "memory_stats": {
                "min": min(memory_usages) if memory_usages else 0,
                "max": max(memory_usages) if memory_usages else 0,
                "avg": statistics.mean(memory_usages) if memory_usages else 0
            },
            "gpu_stats": {
                "min": min(gpu_usages) if gpu_usages else 0,
                "max": max(gpu_usages) if gpu_usages else 0,
                "avg": statistics.mean(gpu_usages) if gpu_usages else 0
            },
            "recent_results": [r.test_name for r in self.profiling_results[-5:]]  # Last 5
        }


class PerformanceRegressionTester:
    """
    System for performance regression testing
    """
    
    def __init__(self):
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        self.regression_results: List[Dict[str, Any]] = []
    
    def establish_baseline(self, test_name: str, result: BenchmarkResult) -> None:
        """Establish a performance baseline for a test"""
        self.baseline_results[test_name] = result
    
    def check_regression(self, test_name: str, result: BenchmarkResult, 
                        threshold_percent: float = 10.0) -> Dict[str, Any]:
        """
        Check if there's a performance regression compared to baseline
        """
        if test_name not in self.baseline_results:
            # No baseline, establish this as baseline
            self.establish_baseline(test_name, result)
            return {
                "regression_detected": False,
                "message": "No baseline available, establishing current result as baseline",
                "baseline": None,
                "current": result.execution_time
            }
        
        baseline = self.baseline_results[test_name]
        
        # Check for performance degradation (higher execution time is worse)
        time_diff = result.execution_time - baseline.execution_time
        time_change_percent = (time_diff / baseline.execution_time) * 100 if baseline.execution_time > 0 else 0
        
        regression_detected = time_change_percent > threshold_percent
        
        regression_info = {
            "regression_detected": regression_detected,
            "time_change_percent": time_change_percent,
            "threshold_used": threshold_percent,
            "baseline": baseline.execution_time,
            "current": result.execution_time,
            "fps_change": result.fps - baseline.fps,
            "memory_change_mb": result.memory_usage_mb - baseline.memory_usage_mb
        }
        
        if regression_detected:
            self.regression_results.append({
                "test_name": test_name,
                "regression_info": regression_info,
                "timestamp": time.time()
            })
        
        return regression_info
    
    def run_regression_test(self, test_name: str, shader_func: Callable, 
                           params: Dict[str, Any], profiler: ShaderModuleProfiler,
                           threshold_percent: float = 10.0) -> Dict[str, Any]:
        """
        Run a complete regression test for a shader function
        """
        # Profile the shader
        result = profiler.profile_shader_module(shader_func, params)
        
        # Check for regression
        regression_info = self.check_regression(test_name, result, threshold_percent)
        
        return regression_info


class GPUMonitor:
    """
    GPU utilization and memory monitoring system
    """
    
    def __init__(self):
        self.gpu_stats_history: List[Dict[str, Any]] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self) -> None:
        """Start GPU monitoring in a background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpu)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_gpu(self) -> None:
        """Monitor GPU in a background thread"""
        while self.monitoring:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    stats = {
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_percent': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature,
                        'timestamp': time.time()
                    }
                    self.gpu_stats_history.append(stats)
            except Exception:
                # GPUtil not available or other error, just continue
                pass
            
            time.sleep(1)  # Monitor every second
    
    def get_gpu_summary(self) -> Dict[str, Any]:
        """Get a summary of GPU usage statistics"""
        if not self.gpu_stats_history:
            return {"message": "No GPU monitoring data available"}
        
        # Group by GPU ID
        gpu_data = {}
        for stat in self.gpu_stats_history:
            gpu_id = stat['id']
            if gpu_id not in gpu_data:
                gpu_data[gpu_id] = {
                    'name': stat['name'],
                    'loads': [],
                    'memory_percents': [],
                    'temperatures': []
                }
            
            gpu_data[gpu_id]['loads'].append(stat['load'])
            gpu_data[gpu_id]['memory_percents'].append(stat['memory_percent'])
            gpu_data[gpu_id]['temperatures'].append(stat['temperature'])
        
        summary = {}
        for gpu_id, data in gpu_data.items():
            summary[gpu_id] = {
                'name': data['name'],
                'load_stats': {
                    'min': min(data['loads']),
                    'max': max(data['loads']),
                    'avg': statistics.mean(data['loads'])
                },
                'memory_stats': {
                    'min': min(data['memory_percents']),
                    'max': max(data['memory_percents']),
                    'avg': statistics.mean(data['memory_percents'])
                },
                'temperature_stats': {
                    'min': min(data['temperatures']),
                    'max': max(data['temperatures']),
                    'avg': statistics.mean(data['temperatures'])
                }
            }
        
        return summary


class ComprehensivePerformanceTestSuite:
    """
    Main system for comprehensive performance testing
    """
    
    def __init__(self):
        self.profiler = ShaderModuleProfiler()
        self.regression_tester = PerformanceRegressionTester()
        self.gpu_monitor = GPUMonitor()
        self.test_results: List[Dict[str, Any]] = []
    
    def add_shader_module_test(self, name: str, shader_func: Callable, 
                              params: Dict[str, Any], iterations: int = 1000) -> Dict[str, Any]:
        """Add and run a test for a shader module"""
        # Profile the shader
        profile_result = self.profiler.profile_shader_module(shader_func, params, iterations)
        
        # Check for regressions
        regression_result = self.regression_tester.run_regression_test(
            name, shader_func, params, self.profiler
        )
        
        test_result = {
            "test_name": name,
            "profile_result": profile_result,
            "regression_result": regression_result,
            "completed_at": time.time()
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def run_comprehensive_test(self, shader_modules: List[Tuple[str, Callable, Dict[str, Any]]], 
                              iterations: int = 1000) -> Dict[str, Any]:
        """
        Run comprehensive performance tests on multiple shader modules
        """
        print("Starting comprehensive performance test...")
        
        # Start GPU monitoring
        self.gpu_monitor.start_monitoring()
        
        results = []
        for name, shader_func, params in shader_modules:
            print(f"Testing: {name}")
            result = self.add_shader_module_test(name, shader_func, params, iterations)
            results.append(result)
        
        # Stop GPU monitoring
        self.gpu_monitor.stop_monitoring()
        
        # Compile report
        profile_summary = self.profiler.get_profiling_summary()
        gpu_summary = self.gpu_monitor.get_gpu_summary()
        
        return {
            "test_count": len(results),
            "profile_summary": profile_summary,
            "gpu_summary": gpu_summary,
            "regressions_detected": len(self.regression_tester.regression_results),
            "regression_details": self.regression_tester.regression_results[-5:],  # Last 5
            "completed_tests": [r["test_name"] for r in results]
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        return {
            "profiling_summary": self.profiler.get_profiling_summary(),
            "regression_summary": {
                "total_regressions": len(self.regression_tester.regression_results),
                "recent_regressions": self.regression_tester.regression_results[-5:]
            },
            "gpu_summary": self.gpu_monitor.get_gpu_summary(),
            "total_tests_run": len(self.test_results)
        }


def simulate_shader_function(params: Dict[str, Any]) -> None:
    """Simulate a shader function for testing purposes"""
    import math
    
    # Simulate some computational work based on parameters
    detail_level = params.get('detail_level', 1.0)
    complexity = params.get('complexity', 1.0)
    
    # Perform some calculations to simulate shader work
    result = 0
    for i in range(int(100 * detail_level * complexity)):
        result += math.sin(i * 0.1) * math.cos(i * 0.15)


def main():
    """
    Example usage of the Performance Benchmarking System
    """
    print("Performance Benchmarking System")
    print("Part of SuperShader Project - Phase 8")
    
    # Create the comprehensive test suite
    test_suite = ComprehensivePerformanceTestSuite()
    
    # Define test shaders with parameters
    test_modules = [
        ("simple_lighting", simulate_shader_function, {"detail_level": 0.5, "complexity": 0.5}),
        ("medium_shading", simulate_shader_function, {"detail_level": 1.0, "complexity": 1.0}),
        ("complex_rendering", simulate_shader_function, {"detail_level": 2.0, "complexity": 1.5}),
    ]
    
    # Run comprehensive tests
    print("\n--- Running Comprehensive Performance Tests ---")
    comprehensive_results = test_suite.run_comprehensive_test(test_modules, iterations=500)
    
    print(f"Tested {comprehensive_results['test_count']} shader modules")
    print(f"Regressions detected: {comprehensive_results['regressions_detected']}")
    
    if comprehensive_results['profile_summary']['total_tests'] > 0:
        avg_fps = comprehensive_results['profile_summary']['fps_stats']['avg']
        print(f"Average FPS across tests: {avg_fps:.2f}")
    
    # Show GPU summary if available
    if comprehensive_results['gpu_summary'].get('message') != "No GPU monitoring data available":
        print("GPU utilization summary available")
    
    # Generate and print final report
    print("\n--- Performance Report ---")
    report = test_suite.generate_performance_report()
    
    print("Profiling Summary:")
    print(f"  Total tests: {report['profiling_summary']['total_tests']}")
    if report['profiling_summary']['total_tests'] > 0:
        print(f"  Avg execution time: {report['profiling_summary']['execution_time_stats']['avg']:.4f}s")
        print(f"  Avg FPS: {report['profiling_summary']['fps_stats']['avg']:.2f}")
    
    print(f"Regressions detected: {report['regression_summary']['total_regressions']}")


if __name__ == "__main__":
    main()