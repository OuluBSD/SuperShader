#!/usr/bin/env python3
"""
Module Combiner for SuperShader project
This script combines generic modules into complete shaders based on a specification.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


class ModuleCombiner:
    def __init__(self, modules_dir='modules'):
        self.modules_dir = modules_dir
        self.modules_cache = {}
        self.dependencies_cache = {}

    def load_module(self, module_path: str) -> Dict[str, Any]:
        """Load a module from its file path."""
        if module_path in self.modules_cache:
            return self.modules_cache[module_path]
        
        with open(module_path, 'r', encoding='utf-8') as f:
            module_data = json.load(f)
        
        self.modules_cache[module_path] = module_data
        return module_data

    def find_module_file(self, module_name: str, module_type: str = None) -> str:
        """Find a module file by name in the modules directory."""
        # Search for the module file in the modules directory
        for root, dirs, files in os.walk(self.modules_dir):
            for file in files:
                if file.startswith(module_name) and file.endswith('.txt'):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            module_data = json.load(f)
                        
                        if module_data.get('name') == module_name:
                            return full_path
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
        
        return None

    def resolve_dependencies(self, module_path: str, processed_modules: List[str] = None) -> List[str]:
        """Recursively resolve module dependencies."""
        if processed_modules is None:
            processed_modules = []
        
        if module_path in processed_modules:
            return processed_modules
        
        module = self.load_module(module_path)
        dependencies = module.get('dependencies', [])
        
        dependency_paths = []
        for dep in dependencies:
            dep_path = self.find_module_file(dep.split('/')[-1])  # Extract module name from path-like dependency
            if dep_path and dep_path not in processed_modules:
                # Process dependencies of this dependency first
                resolved = self.resolve_dependencies(dep_path, processed_modules)
                dependency_paths.extend(resolved)
                dependency_paths.append(dep_path)
        
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

    def combine_modules(self, module_names: List[str], output_file: str = None) -> str:
        """Combine multiple modules into a single GLSL shader."""
        all_glsl_parts = []
        processed_modules = []
        
        # First, collect all dependencies
        for module_name in module_names:
            module_path = self.find_module_file(module_name)
            if not module_path:
                print(f"Warning: Module '{module_name}' not found")
                continue
            
            # Resolve dependencies
            dependencies = self.resolve_dependencies(module_path)
            
            # Add dependencies first
            for dep_path in dependencies:
                if dep_path not in processed_modules:
                    dep_module = self.load_module(dep_path)
                    glsl_parts = self.extract_glsl_implementation(dep_module)
                    all_glsl_parts.extend(glsl_parts)
                    processed_modules.append(dep_path)
            
            # Add the actual module
            if module_path not in processed_modules:
                module = self.load_module(module_path)
                glsl_parts = self.extract_glsl_implementation(module)
                all_glsl_parts.extend(glsl_parts)
                processed_modules.append(module_path)
        
        # Combine all GLSL parts
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
    parser = argparse.ArgumentParser(description='Combine SuperShader modules into complete shaders')
    parser.add_argument('modules', nargs='+', help='Module names to combine')
    parser.add_argument('-o', '--output', type=str, help='Output file for the combined shader')
    parser.add_argument('--spec', type=str, help='Specification file for complex shader assembly')
    parser.add_argument('--modules-dir', type=str, default='modules', help='Directory containing modules')
    
    args = parser.parse_args()
    
    combiner = ModuleCombiner(modules_dir=args.modules_dir)
    
    if args.spec:
        shader_code = combiner.create_shader_from_spec(args.spec, args.output)
    else:
        shader_code = combiner.combine_modules(args.modules, args.output)
    
    if not args.output:
        # Print to stdout if no output file specified
        print(shader_code)


if __name__ == "__main__":
    main()