#!/usr/bin/env python3
"""
Raymarching Module Registry
Registers all available raymarching modules and their metadata
"""

import importlib.util
import os

def load_module_function(module_path, function_name):
    """Dynamically load a function from a module file"""
    spec = importlib.util.spec_from_file_location(os.path.basename(module_path)[:-3], module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

# Register all raymarching modules
RAYMARCHING_MODULES = {
    'raymarching_core': {
        'path': 'modules/raymarching/core/raymarching_core.py',
        'get_pseudocode': lambda: load_module_function('modules/raymarching/core/raymarching_core.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/raymarching/core/raymarching_core.py', 'get_metadata')()
    },
    'sdf_primitives': {
        'path': 'modules/raymarching/distance_functions/sdf_primitives.py',
        'get_pseudocode': lambda: load_module_function('modules/raymarching/distance_functions/sdf_primitives.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/raymarching/distance_functions/sdf_primitives.py', 'get_metadata')()
    },
    'ray_generation': {
        'path': 'modules/raymarching/ray_generation/ray_generation.py',
        'get_pseudocode': lambda: load_module_function('modules/raymarching/ray_generation/ray_generation.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/raymarching/ray_generation/ray_generation.py', 'get_metadata')()
    },
    'raymarching_lighting': {
        'path': 'modules/raymarching/lighting/raymarching_lighting.py',
        'get_pseudocode': lambda: load_module_function('modules/raymarching/lighting/raymarching_lighting.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/raymarching/lighting/raymarching_lighting.py', 'get_metadata')()
    },
    'raymarching_advanced_branching': {
        'path': 'modules/raymarching/standardized/advanced_raymarching_branching.py',
        'get_pseudocode': lambda: load_module_function('modules/raymarching/standardized/advanced_raymarching_branching.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/raymarching/standardized/advanced_raymarching_branching.py', 'get_metadata')()
    }
}

def get_all_modules():
    """Return all registered raymarching modules"""
    modules = []
    for name, module_data in RAYMARCHING_MODULES.items():
        metadata = module_data['get_metadata']()
        modules.append({
            'name': name,
            'metadata': metadata
        })
    return modules

def get_module_by_name(name):
    """Get a specific module by name"""
    if name in RAYMARCHING_MODULES:
        module_data = RAYMARCHING_MODULES[name]
        return {
            'name': name,
            'pseudocode': module_data['get_pseudocode'](),
            'metadata': module_data['get_metadata']()
        }
    return None

def search_modules_by_pattern(pattern):
    """Search for modules that implement a specific pattern"""
    results = []
    for name, module_data in RAYMARCHING_MODULES.items():
        metadata = module_data['get_metadata']()
        if pattern.lower() in [p.lower() for p in metadata.get('patterns', [])]:
            results.append({
                'name': name,
                'metadata': metadata
            })
    return results

def get_module_dependencies(name):
    """Get dependencies for a specific module"""
    module = get_module_by_name(name)
    if module:
        return module['metadata'].get('dependencies', [])
    return []

def get_module_conflicts(name):
    """Get conflicts for a specific module"""
    module = get_module_by_name(name)
    if module:
        return module['metadata'].get('conflicts', [])
    return []