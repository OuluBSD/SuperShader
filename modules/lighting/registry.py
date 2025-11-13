#!/usr/bin/env python3
"""
Lighting Module Registry
Registers all available lighting modules and their metadata
"""

import importlib.util
import os

def load_module_function(module_path, function_name):
    """Dynamically load a function from a module file"""
    spec = importlib.util.spec_from_file_location(os.path.basename(module_path)[:-3], module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

# Register all lighting modules
LIGHTING_MODULES = {
    'basic_point_light': {
        'path': 'modules/lighting/point_light/basic_point_light.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/point_light/basic_point_light.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/point_light/basic_point_light.py', 'get_metadata')()
    },
    'normal_mapping': {
        'path': 'modules/lighting/normal_mapping/normal_mapping.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/normal_mapping/normal_mapping.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/normal_mapping/normal_mapping.py', 'get_metadata')()
    },
    'specular_lighting': {
        'path': 'modules/lighting/specular/specular_lighting.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/specular/specular_lighting.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/specular/specular_lighting.py', 'get_metadata')()
    },
    'pbr_lighting': {
        'path': 'modules/lighting/pbr/pbr_lighting.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/pbr/pbr_lighting.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/pbr/pbr_lighting.py', 'get_metadata')()
    },
    'diffuse_lighting': {
        'path': 'modules/lighting/diffuse/diffuse_lighting.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/diffuse/diffuse_lighting.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/diffuse/diffuse_lighting.py', 'get_metadata')()
    },
    'shadow_mapping': {
        'path': 'modules/lighting/shadow_mapping/shadow_mapping.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/shadow_mapping/shadow_mapping.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/shadow_mapping/shadow_mapping.py', 'get_metadata')()
    },
    'directional_light': {
        'path': 'modules/lighting/directional_light/directional_light.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/directional_light/directional_light.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/directional_light/directional_light.py', 'get_metadata')()
    },
    'raymarching_lighting': {
        'path': 'modules/lighting/raymarching_lighting/raymarching_lighting.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/raymarching_lighting/raymarching_lighting.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/raymarching_lighting/raymarching_lighting.py', 'get_metadata')()
    },
    'cel_shading': {
        'path': 'modules/lighting/cel_shading/cel_shading.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/cel_shading/cel_shading.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/cel_shading/cel_shading.py', 'get_metadata')()
    },
    'spot_light': {
        'path': 'modules/lighting/spot_light/spot_light.py',
        'get_pseudocode': lambda: load_module_function('modules/lighting/spot_light/spot_light.py', 'get_pseudocode')(),
        'get_metadata': lambda: load_module_function('modules/lighting/spot_light/spot_light.py', 'get_metadata')()
    }
}

def get_all_modules():
    """Return all registered lighting modules"""
    modules = []
    for name, module_data in LIGHTING_MODULES.items():
        metadata = module_data['get_metadata']()
        modules.append({
            'name': name,
            'metadata': metadata
        })
    return modules

def get_module_by_name(name):
    """Get a specific module by name"""
    if name in LIGHTING_MODULES:
        module_data = LIGHTING_MODULES[name]
        return {
            'name': name,
            'pseudocode': module_data['get_pseudocode'](),
            'metadata': module_data['get_metadata']()
        }
    return None

def search_modules_by_pattern(pattern):
    """Search for modules that implement a specific pattern"""
    results = []
    for name, module_data in LIGHTING_MODULES.items():
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