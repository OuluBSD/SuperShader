#!/usr/bin/env python3
"""
Raymarching Shader Generator
Generates complete raymarching shaders from module combinations
"""

from modules.raymarching.combiner import combine_raymarching_modules
from create_pseudocode_translator import PseudocodeTranslator
from modules.raymarching.registry import get_all_modules
import json


class RaymarchingShaderGenerator:
    def __init__(self):
        self.translator = PseudocodeTranslator()
        self.available_modules = {m['name'] for m in get_all_modules()}

    def generate_shader(self, module_names, shader_config=None):
        """
        Generate a complete raymarching shader from module names
        
        Args:
            module_names (list): List of module names to include
            shader_config (dict): Configuration options for the shader
            
        Returns:
            str: Generated GLSL shader code
        """
        if shader_config is None:
            shader_config = {}
        
        # Validate modules
        invalid_modules = [m for m in module_names if m not in self.available_modules]
        if invalid_modules:
            raise ValueError(f"Invalid modules: {invalid_modules}")
        
        # Combine modules
        shader_code = combine_raymarching_modules(module_names)
        
        # Apply configuration
        if shader_config.get('version'):
            shader_code = shader_code.replace('#version 330 core', f"#version {shader_config['version']}")
        
        # Add custom uniforms if specified
        if 'custom_uniforms' in shader_config:
            custom_uniforms = '\n'.join(shader_config['custom_uniforms'])
            shader_code = shader_code.replace('// Common raymarching uniforms', 
                                            f"// Common raymarching uniforms\n{custom_uniforms}\n")
        
        # Add custom constants if specified
        if 'custom_constants' in shader_config:
            custom_constants = '\n'.join(shader_config['custom_constants'])
            shader_code = shader_code.replace('// PI constant', 
                                            f"// PI constant\n{custom_constants}\n")
        
        return shader_code
    
    def generate_scene_shader(self, scene_modules, lighting_modules, raygen_modules):
        """
        Generate a complete scene shader with specific modules
        """
        all_modules = scene_modules + lighting_modules + raygen_modules
        
        config = {
            'custom_constants': [
                'const int MAX_STEPS = 100;',
                'const float MIN_DIST = 0.001;',
                'const float MAX_DIST = 100.0;',
                'const float SURFACE_DIST = 0.0001;'
            ]
        }
        
        return self.generate_shader(all_modules, config)
    
    def create_scene_config(self, scene_type='spheres', lighting='basic', raygen='perspective'):
        """
        Create a predefined scene configuration
        """
        config_map = {
            'spheres': ['sdf_primitives'],
            'complex_scene': ['sdf_primitives', 'raymarching_core'],
            'advanced': ['sdf_primitives', 'raymarching_core', 'raymarching_lighting']
        }
        
        lighting_map = {
            'basic': ['raymarching_lighting'],
            'advanced': ['raymarching_lighting']
        }
        
        raygen_map = {
            'perspective': ['ray_generation'],
            'orthographic': ['ray_generation']
        }
        
        scene_modules = config_map.get(scene_type, ['sdf_primitives'])
        lighting_modules = lighting_map.get(lighting, ['raymarching_lighting'])
        raygen_modules = raygen_map.get(raygen, ['ray_generation'])
        
        return {
            'scene_modules': scene_modules,
            'lighting_modules': lighting_modules,
            'raygen_modules': raygen_modules
        }
    
    def generate_from_config(self, config):
        """
        Generate shader from a configuration dictionary
        """
        return self.generate_scene_shader(
            config['scene_modules'],
            config['lighting_modules'], 
            config['raygen_modules']
        )


def generate_demo_shaders():
    """
    Generate example raymarching shaders for demonstration
    """
    generator = RaymarchingShaderGenerator()
    
    # Example 1: Basic raymarching scene
    basic_modules = ['raymarching_core', 'sdf_primitives', 'ray_generation', 'raymarching_lighting']
    basic_shader = generator.generate_shader(basic_modules)
    
    with open('demo_raymarching_basic.glsl', 'w') as f:
        f.write(basic_shader)
    print("✓ Generated basic raymarching shader: demo_raymarching_basic.glsl")
    
    # Example 2: Advanced scene using predefined config
    scene_config = generator.create_scene_config('advanced', 'advanced', 'perspective')
    advanced_shader = generator.generate_from_config(scene_config)
    
    with open('demo_raymarching_advanced.glsl', 'w') as f:
        f.write(advanced_shader)
    print("✓ Generated advanced raymarching shader: demo_raymarching_advanced.glsl")
    
    # Example 3: Custom configuration
    custom_config = {
        'version': '410 core',
        'custom_uniforms': [
            'uniform vec3 lightPosition;',
            'uniform vec3 lightColor;',
            'uniform float materialRoughness;'
        ],
        'custom_constants': [
            'const float PI = 3.14159265359;',
            'const int MAX_STEPS = 128;',
            'const float MIN_DIST = 0.0005;',
            'const float MAX_DIST = 50.0;'
        ]
    }
    
    custom_shader = generator.generate_shader(basic_modules, custom_config)
    
    with open('demo_raymarching_custom.glsl', 'w') as f:
        f.write(custom_shader)
    print("✓ Generated custom raymarching shader: demo_raymarching_custom.glsl")
    
    # Save configuration for reference
    with open('raymarching_config_example.json', 'w') as f:
        json.dump({
            'basic': basic_modules,
            'advanced': scene_config,
            'custom': custom_config
        }, f, indent=2)
    print("✓ Generated configuration example: raymarching_config_example.json")


if __name__ == "__main__":
    print("Generating Raymarching Shaders...")
    generate_demo_shaders()
    print("Raymarching shader generation completed!")