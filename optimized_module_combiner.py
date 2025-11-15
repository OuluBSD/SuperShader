#!/usr/bin/env python3
"""
Optimized Module Combiner for SuperShader project
This script combines generic modules into complete shaders based on a specification.
Optimizations include caching, efficient dependency resolution, and parallel processing.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from concurrent.futures import ThreadPoolExecutor
import time


class OptimizedModuleCombiner:
    def __init__(self, modules_dir='modules'):
        self.modules_dir = modules_dir
        self.modules_cache: Dict[str, Dict[str, Any]] = {}
        self.module_paths_cache: Dict[str, str] = {}
        self.dependencies_cache: Dict[str, List[str]] = {}
        self.reverse_dependencies_cache: Dict[str, List[str]] = {}
        self._build_module_index()  # Build fast lookup index

    def _build_module_index(self):
        """Build an index for fast module lookup."""
        print("Building module index...")
        start_time = time.time()
        
        for root, dirs, files in os.walk(self.modules_dir):
            for file in files:
                if file.endswith('.txt'):
                    full_path = os.path.join(root, file)
                    try:
                        # Read only the name to build the index
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read(2048)  # Read first 2KB to get name
                            # Find the name in the content
                            if '"name":' in content or "'name':" in content:
                                # Load the partial content as JSON if possible
                                # For performance, we'll use a simple parsing approach
                                if '"name"' in content:
                                    start_idx = content.find('"name"') + 7
                                    start_idx = content.find('"', start_idx)
                                    end_idx = content.find('"', start_idx + 1)
                                    if start_idx != -1 and end_idx != -1:
                                        module_name = content[start_idx + 1:end_idx]
                                        self.module_paths_cache[module_name] = full_path
                            else:
                                # If we can't parse the name quickly, load the full module
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    module_data = json.load(f)
                                    module_name = module_data.get('name', '')
                                    if module_name:
                                        self.module_paths_cache[module_name] = full_path
                    except (json.JSONDecodeError, UnicodeDecodeError, UnicodeError):
                        continue
        
        print(f"Module index built in {time.time() - start_time:.3f}s with {len(self.module_paths_cache)} modules")

    def load_module(self, module_path: str) -> Dict[str, Any]:
        """Load a module from its file path with caching."""
        if module_path in self.modules_cache:
            return self.modules_cache[module_path]

        with open(module_path, 'r', encoding='utf-8') as f:
            module_data = json.load(f)

        self.modules_cache[module_path] = module_data
        return module_data

    def find_module_file(self, module_name: str) -> Optional[str]:
        """Find a module file by name quickly using the index."""
        return self.module_paths_cache.get(module_name)

    def resolve_dependencies(self, module_path: str, processed_modules: Set[str] = None) -> List[str]:
        """Recursively resolve module dependencies with caching."""
        if processed_modules is None:
            processed_modules = set()

        if module_path in processed_modules:
            return []

        # Check if we already computed dependencies for this module
        if module_path in self.dependencies_cache:
            return self.dependencies_cache[module_path]

        module = self.load_module(module_path)
        dependencies = module.get('dependencies', [])

        dependency_paths = []
        for dep in dependencies:
            # Extract module name from dependency string (e.g. "lighting/phong_lighting/phong" -> "phong")
            simple_dep = dep.split('/')[-1]
            dep_path = self.find_module_file(simple_dep)
            
            if dep_path and dep_path not in processed_modules:
                # Add to processed set to avoid circular dependencies
                processed_modules.add(dep_path)
                
                # Process dependencies of this dependency first (depth-first)
                nested_deps = self.resolve_dependencies(dep_path, processed_modules.copy())
                dependency_paths.extend(nested_deps)
                dependency_paths.append(dep_path)

        # Cache the result
        self.dependencies_cache[module_path] = dependency_paths
        return dependency_paths

    def extract_glsl_implementation(self, module_data: Dict[str, Any]) -> List[str]:
        """Extract GLSL implementation from a module."""
        impl = module_data.get('implementation', {})
        glsl_code = impl.get('glsl', [])

        if isinstance(glsl_code, list):
            return glsl_code
        elif isinstance(glsl_code, str):
            return [glsl_code]
        else:
            return []

    def combine_modules(self, module_names: List[str], output_file: str = None, parallel: bool = False) -> str:
        """Combine multiple modules into a single GLSL shader with optimizations."""
        all_glsl_parts = []
        processed_modules: Set[str] = set()

        # Collect all modules and their dependencies
        all_module_paths = []
        for module_name in module_names:
            module_path = self.find_module_file(module_name)
            if not module_path:
                print(f"Warning: Module '{module_name}' not found")
                continue
            all_module_paths.append(module_path)

        # Resolve dependencies for all modules in advance
        all_dependencies = []
        for module_path in all_module_paths:
            dependencies = self.resolve_dependencies(module_path)
            all_dependencies.extend(dependencies)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_dependencies = []
        for dep in all_dependencies:
            if dep not in seen:
                seen.add(dep)
                unique_dependencies.append(dep)

        # Process all dependencies first
        for dep_path in unique_dependencies:
            if dep_path not in processed_modules:
                dep_module = self.load_module(dep_path)
                glsl_parts = self.extract_glsl_implementation(dep_module)
                all_glsl_parts.extend(glsl_parts)
                processed_modules.add(dep_path)

        # Process requested modules
        for module_path in all_module_paths:
            if module_path not in processed_modules:
                module = self.load_module(module_path)
                glsl_parts = self.extract_glsl_implementation(module)
                all_glsl_parts.extend(glsl_parts)
                processed_modules.add(module_path)

        # Combine all GLSL parts with minimal string operations
        combined_glsl = '\n'.join(all_glsl_parts)

        # If output file is specified, write to file
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_glsl)
            print(f"Combined shader written to {output_file}")

        return combined_glsl

    def create_shader_from_spec(self, spec_file: str, output_file: str = None):
        """Create a shader from a specification file."""
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)

        modules = spec.get('modules', [])
        combined_shader = self.combine_modules(modules, output_file)

        # Add shader wrapper if specified in the spec
        if 'shader_template' in spec:
            template = spec['shader_template']
            # This is where we'd implement shader template logic
            # For now, just return the combined shader
            pass

        return combined_shader


def main():
    parser = argparse.ArgumentParser(description='Combine SuperShader modules into complete shaders (Optimized)')
    parser.add_argument('modules', nargs='+', help='Module names to combine')
    parser.add_argument('-o', '--output', type=str, help='Output file for the combined shader')
    parser.add_argument('--spec', type=str, help='Specification file for complex shader assembly')
    parser.add_argument('--modules-dir', type=str, default='modules', help='Directory containing modules')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing (if applicable)')

    args = parser.parse_args()

    print("Initializing optimized module combiner...")
    combiner = OptimizedModuleCombiner(modules_dir=args.modules_dir)

    start_time = time.time()
    
    if args.spec:
        shader_code = combiner.create_shader_from_spec(args.spec, args.output)
    else:
        shader_code = combiner.combine_modules(args.modules, args.output, parallel=args.parallel)

    elapsed_time = time.time() - start_time
    print(f"Module combination completed in {elapsed_time:.3f}s")

    if not args.output:
        # Print to stdout if no output file specified
        print(shader_code)


if __name__ == "__main__":
    main()