#!/usr/bin/env python3
"""
Performance Profiler for SuperShader System
Tests performance and optimization of the module system
"""

import time
import cProfile
import pstats
from io import StringIO
from create_module_engine import ModuleEngine
from create_module_registry import ModuleRegistry


def profile_module_engine():
    """Profile the module engine performance"""
    print("Profiling Module Engine Performance...")
    
    # Create a profiler
    pr = cProfile.Profile()
    
    # Run operations to profile
    pr.enable()
    
    # Test module engine performance
    start_time = time.time()
    
    for i in range(100):  # Run 100 iterations to get meaningful data
        engine = ModuleEngine()
        
        # Add some modules without dependency conflicts
        engine.add_module('lighting/point_light/basic_point_light')
        engine.add_module('lighting/diffuse/diffuse_lighting')
        engine.add_module('lighting/normal_mapping/normal_mapping')
        # Only add specular if it doesn't cause dependency issues
        try:
            engine.add_module('lighting/specular/specular_lighting')
        except:
            # If there's a dependency issue, just continue
            pass
        
        # Validate the combination
        validation = engine.validate_combination()
        
        # Generate a shader
        shader = engine.generate_shader()
    
    end_time = time.time()
    
    pr.disable()
    
    # Get profiling stats
    s = StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(10)  # Top 10 functions
    
    print(f"Total time for 100 iterations: {end_time - start_time:.4f} seconds")
    print(f"Average time per iteration: {(end_time - start_time)/100:.6f} seconds")
    print("\nTop 10 performance bottlenecks:")
    print(s.getvalue())
    
    return end_time - start_time


def test_memory_usage():
    """Test memory usage of the system"""
    print("\nTesting Memory Usage...")
    print("(Memory profiling requires the 'psutil' package)")
    print("To get detailed memory usage, install with: pip install psutil")
    
    # Create several module engines to test general behavior
    engines = []
    for i in range(10):
        engine = ModuleEngine()
        engine.add_module('lighting/point_light/basic_point_light')
        engines.append(engine)
    
    print(f"Created 10 engine instances successfully")
    print(f"Each engine has {len(engines[0].selected_modules)} selected modules")


def benchmark_registry():
    """Benchmark registry performance"""
    print("\nBenchmarking Registry Performance...")
    
    start_time = time.time()
    
    # Test registry creation and search
    registry = ModuleRegistry()
    
    # Search for different patterns multiple times
    for _ in range(1000):
        modules = registry.search_modules(pattern="light")
        modules = registry.search_modules(pattern="normal")
        modules = registry.search_modules(genre="lighting")
    
    end_time = time.time()
    
    print(f"Registry benchmark (1000 searches): {end_time - start_time:.4f} seconds")
    print(f"Average search time: {(end_time - start_time) / 3000:.6f} seconds per search")
    
    return end_time - start_time


def optimize_module_loading():
    """Suggest optimizations for module loading"""
    print("\nOptimization Suggestions:")
    print("1. Module registry could implement lazy loading for modules")
    print("2. Cache frequently used module combinations")
    print("3. Implement module pre-compilation for faster shader generation")
    print("4. Use more efficient data structures for dependency resolution")
    print("5. Implement asynchronous module loading for UI applications")


def run_complete_performance_analysis():
    """Run complete performance analysis"""
    print("SuperShader Performance Analysis")
    print("=" * 40)
    
    # Profile module engine
    engine_time = profile_module_engine()
    
    # Test memory usage
    test_memory_usage()
    
    # Benchmark registry
    registry_time = benchmark_registry()
    
    # Provide optimization suggestions
    optimize_module_loading()
    
    print(f"\nPerformance Summary:")
    print(f"  - Module engine (100 iterations): {engine_time:.4f}s")
    print(f"  - Registry benchmark (1000 searches): {registry_time:.4f}s")
    print(f"  - Combined performance index: {(engine_time + registry_time)/2:.4f}s")
    
    print(f"\n✓ Performance analysis complete!")


if __name__ == "__main__":
    run_complete_performance_analysis()