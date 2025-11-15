#!/usr/bin/env python3
"""
Performance Profiler for SuperShader System
Profiles critical paths and identifies performance bottlenecks
"""

import cProfile
import pstats
import time
import sys
from io import StringIO
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Callable

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))


class PerformanceProfiler:
    def __init__(self):
        self.results_dir = Path("profiles")
        self.results_dir.mkdir(exist_ok=True)
        self.profiles = {}
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a single function and return profiling results."""
        pr = cProfile.Profile()
        pr.enable()
        
        # Execute the function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        pr.disable()
        
        # Get profiling stats
        s = StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 most time-consuming functions
        
        return {
            'result': result,
            'execution_time': execution_time,
            'profile_stats': s.getvalue(),
            'function_name': func.__name__
        }
    
    def profile_module_combination(self):
        """Profile the module combination process."""
        print("Profiling module combination process...")
        
        from optimized_module_combiner import OptimizedModuleCombiner
        
        def run_combination():
            combiner = OptimizedModuleCombiner()
            # Use some common modules for testing
            modules = ['perlin_noise', 'verlet_integration']  # These should be available
            result = combiner.combine_modules(modules)
            return result
        
        return self.profile_function(run_combination)
    
    def profile_shader_generation(self):
        """Profile the shader generation process."""
        print("Profiling shader generation process...")
        
        from efficient_shader_generator import EfficientShaderGenerator
        
        def run_shader_generation():
            generator = EfficientShaderGenerator()
            # Use some common modules for testing
            modules = ['diffuse_lighting', 'specular_lighting']  # Placeholder names
            config = {'shader_type': 'fragment'}
            result = generator.generate_shader_with_validation(modules, config)
            return result
        
        return self.profile_function(run_shader_generation)
    
    def profile_pseudocode_translation(self):
        """Profile the pseudocode translation process."""
        print("Profiling pseudocode translation process...")
        
        from optimized_pseudocode_translator import OptimizedPseudocodeTranslator
        
        sample_pseudocode = """
        // Sample pseudocode for performance testing
        vec3 sampleFunction(vec2 coord) {
            return length(coord);
        }
        """
        
        def run_translation():
            translator = OptimizedPseudocodeTranslator()
            # Test multiple translations to get meaningful profiling
            for _ in range(100):
                glsl_result = translator.translate_to_glsl(sample_pseudocode)
                metal_result = translator.translate(sample_pseudocode, 'metal')
                wgsl_result = translator.translate(sample_pseudocode, 'wgsl')
            return glsl_result
        
        return self.profile_function(run_translation)
    
    def profile_module_loading(self):
        """Profile the module loading process."""
        print("Profiling module loading process...")
        
        def run_module_loading():
            from management.module_combiner import ModuleCombiner
            combiner = ModuleCombiner()
            
            # Find and load a few modules
            modules_to_find = ['perlin_noise', 'verlet_integration', 'uv_mapping']
            for module_name in modules_to_find:
                module_path = combiner.find_module_file(module_name)
                if module_path:
                    combiner.load_module(module_path)
        
        return self.profile_function(run_module_loading)
    
    def profile_all_critical_paths(self) -> Dict[str, Any]:
        """Profile all critical paths in the SuperShader system."""
        print("Starting comprehensive performance profiling...")
        print("=" * 60)
        
        results = {}
        
        # Profile module combination
        results['module_combination'] = self.profile_module_combination()
        print(f"Module combination: {results['module_combination']['execution_time']:.3f}s")
        
        # Profile shader generation
        results['shader_generation'] = self.profile_shader_generation()
        print(f"Shader generation: {results['shader_generation']['execution_time']:.3f}s")
        
        # Profile pseudocode translation
        results['pseudocode_translation'] = self.profile_pseudocode_translation()
        print(f"Pseudocode translation: {results['pseudocode_translation']['execution_time']:.3f}s")
        
        # Profile module loading
        results['module_loading'] = self.profile_module_loading()
        print(f"Module loading: {results['module_loading']['execution_time']:.3f}s")
        
        print("=" * 60)
        print("Profiling completed!")
        
        return results
    
    def analyze_profiles(self, profile_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze profile results to identify bottlenecks and optimization opportunities."""
        analysis = {}
        
        for component, result in profile_results.items():
            execution_time = result['execution_time']
            profile_stats = result['profile_stats']
            
            # Analyze the profile stats to find hotspots
            lines = profile_stats.split('\\n')
            top_functions = []
            
            # Skip header lines and get the function timing info
            for line in lines:
                if line.strip() and not line.startswith('   ncalls') and not line.startswith('         '):
                    # Extract function name and time if possible
                    parts = line.split()
                    if len(parts) >= 4 and parts[0].replace('/', '').replace('.', '').isdigit():
                        try:
                            cumtime = float(parts[3])
                            func_name = ' '.join(parts[4:])  # The function name is after the numeric columns
                            top_functions.append({
                                'function': func_name,
                                'cumulative_time': cumtime
                            })
                        except ValueError:
                            continue
            
            analysis[component] = {
                'execution_time': execution_time,
                'top_hotspots': top_functions[:5],  # Top 5 hotspots
                'recommendations': self._generate_recommendations(top_functions[:5], execution_time, component)
            }
        
        return analysis
    
    def _generate_recommendations(self, hotspots: List[Dict], execution_time: float, component: str) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        if execution_time > 1.0:  # If execution is taking more than 1 second
            recommendations.append(f"Consider optimizing {component} - execution time: {execution_time:.3f}s is high")
        
        # Analyze hotspots for specific optimization opportunities
        for hotspot in hotspots[:3]:  # Look at top 3 hotspots
            func_name = hotspot['function']
            cumtime = hotspot['cumulative_time']
            
            if 'regex' in func_name.lower():
                recommendations.append(f"Regex operations in {func_name} may be optimized with compiled patterns")
            elif 'json' in func_name.lower() or 'load' in func_name.lower():
                recommendations.append(f"JSON operations in {func_name} may benefit from caching or streaming")
            elif 'string' in func_name.lower() or func_name.count('(') > 2:  # Multiple nested funcs
                recommendations.append(f"String operations in {func_name} may be optimized with join() or other methods")
            elif cumtime > 0.1:  # If function takes more than 0.1s cumulatively
                recommendations.append(f"Function {func_name} is taking {cumtime:.3f}s - consider algorithmic optimization")
        
        # Add general recommendations based on component
        if component == 'pseudocode_translation':
            recommendations.append("Consider using LRU caching for translation results")
            recommendations.append("Use batch operations when translating multiple pseudocode blocks")
        elif component == 'module_combination':
            recommendations.append("Use parallel processing for independent module operations")
            recommendations.append("Implement more efficient dependency resolution algorithms")
        elif component == 'shader_generation':
            recommendations.append("Cache shader templates and skeleton generation")
            recommendations.append("Optimize string concatenation operations")
        elif component == 'module_loading':
            recommendations.append("Implement module index for faster lookups")
            recommendations.append("Use lazy loading where possible")
        
        return recommendations
    
    def save_profiling_results(self, profile_results: Dict[str, Any], analysis: Dict[str, Any]):
        """Save profiling results and analysis to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw profiling results
        raw_results_file = self.results_dir / f"profile_raw_results_{timestamp}.json"
        with open(raw_results_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = {}
            for key, value in profile_results.items():
                serializable_results[key] = {
                    'execution_time': value['execution_time'],
                    'function_name': value['function_name']
                }
            json.dump(serializable_results, f, indent=2)
        
        # Save detailed analysis
        analysis_file = self.results_dir / f"profile_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save detailed profile stats to text files for each component
        for component, result in profile_results.items():
            stats_file = self.results_dir / f"profile_stats_{component}_{timestamp}.txt"
            with open(stats_file, 'w') as f:
                f.write(f"Profiling results for {component}\\n")
                f.write("=" * 50 + "\\n")
                f.write(result['profile_stats'])
                f.write("\\n")
        
        print(f"\\nProfiling results saved to:")
        print(f"  - Raw results: {raw_results_file}")
        print(f"  - Analysis: {analysis_file}")
        for component in profile_results.keys():
            print(f"  - Stats for {component}: {self.results_dir}/profile_stats_{component}_{timestamp}.txt")
    
    def run_complete_profiling(self):
        """Run complete profiling and analysis."""
        print("Running Complete Performance Profiling for SuperShader System")
        print("=" * 70)
        
        # Profile all critical paths
        raw_results = self.profile_all_critical_paths()
        
        # Analyze results to identify bottlenecks
        analysis = self.analyze_profiles(raw_results)
        
        # Print analysis summary
        print("\\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        for component, comp_analysis in analysis.items():
            print(f"\\n{component.upper()}:")
            print(f"  Execution time: {comp_analysis['execution_time']:.3f}s")
            
            if comp_analysis['top_hotspots']:
                print("  Top bottlenecks:")
                for i, hotspot in enumerate(comp_analysis['top_hotspots'][:3], 1):
                    print(f"    {i}. {hotspot['function']}: {hotspot['cumulative_time']:.3f}s")
            
            if comp_analysis['recommendations']:
                print("  Recommendations:")
                for i, rec in enumerate(comp_analysis['recommendations'][:3], 1):  # Top 3 recommendations
                    print(f"    {i}. {rec}")
        
        # Save results
        self.save_profiling_results(raw_results, analysis)
        
        return raw_results, analysis


def main():
    """Main entry point for performance profiling."""
    profiler = PerformanceProfiler()
    
    print("SuperShader Performance Profiler")
    print("This tool will analyze critical paths in the SuperShader system")
    print("to identify performance bottlenecks and optimization opportunities.")
    print()
    
    raw_results, analysis = profiler.run_complete_profiling()
    
    print("\\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)
    print("The profiler has identified performance characteristics of the system.")
    print("Review the recommendations and consider implementing optimizations")
    print("for the identified bottlenecks.")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)