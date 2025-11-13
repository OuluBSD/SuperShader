# Benchmarking System for SuperShader

import time
import statistics
from typing import List, Callable, Any
import performance_profiler

class BenchmarkSuite:
    def __init__(self):
        self.results = {}
    
    def benchmark_function(self, func: Callable, *args, name: str = None, iterations: int = 10) -> dict:
        """Benchmark a function and return statistics."""
        func_name = name or func.__name__
        times = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            result = func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        stats = {
            "function": func_name,
            "iterations": iterations,
            "times": times,
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times)
        }
        
        self.results[func_name] = stats
        return stats
    
    def benchmark_module_combination(self, module_count_range: range) -> dict:
        """Benchmark module combination performance."""
        def test_combination(count):
            # Simulate combining modules (simplified)
            time.sleep(0.001 * count)  # Simulate some work proportional to module count
            return f"Combined {count} modules"
        
        results = {}
        for count in module_count_range:
            stats = self.benchmark_function(test_combination, count, name=f"combine_{count}_modules", iterations=5)
            results[count] = stats
        
        return results
    
    def get_performance_report(self) -> str:
        """Generate a performance report."""
        if not self.results:
            return "No benchmarking data available."
        
        report = ["Benchmarking Report", "=" * 20]
        
        for func_name, stats in self.results.items():
            report.append(f"\n{func_name}:")
            report.append(f"  Iterations: {stats['iterations']}")
            report.append(f"  Mean Time: {stats['mean']:.6f}s")
            report.append(f"  Median Time: {stats['median']:.6f}s")
            report.append(f"  Std Dev: {stats['stdev']:.6f}s")
            report.append(f"  Min Time: {stats['min']:.6f}s")
            report.append(f"  Max Time: {stats['max']:.6f}s")
        
        return "\n".join(report)
    
    def compare_implementations(self, implementations: List[Callable], test_input: Any, name: str) -> dict:
        """Compare performance of different implementations."""
        comparison = {}
        
        for impl in implementations:
            stats = self.benchmark_function(impl, test_input, name=f"{name}_{impl.__name__}")
            comparison[impl.__name__] = stats
        
        return comparison


def main():
    profiler = performance_profiler.profiler
    
    # Create benchmark suite
    benchmark_suite = BenchmarkSuite()
    
    print("Running basic benchmarks...")
    
    # Benchmark some simple operations
    def simple_addition(n):
        result = 0
        for i in range(n):
            result += i
        return result
    
    def list_comprehension_addition(n):
        return sum([i for i in range(n)])
    
    # Benchmark the functions
    add_stats = benchmark_suite.benchmark_function(simple_addition, 1000, name="simple_addition")
    comp_stats = benchmark_suite.benchmark_function(list_comprehension_addition, 1000, name="list_comprehension_addition")
    
    print("Benchmark completed!")
    print(benchmark_suite.get_performance_report())
    
    # Also run through the profiler
    @profiler.profile("benchmark_test")
    def benchmark_test():
        simple_addition(1000)
        list_comprehension_addition(1000)
    
    benchmark_test()
    
    print("\nProfiler report:")
    print(profiler.get_report())


if __name__ == "__main__":
    main()
