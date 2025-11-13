#!/usr/bin/env python3
"""
Raymarching Module Combiner
Combines multiple raymarching modules into a complete shader
"""

import os
import importlib.util
from modules.raymarching.registry import get_module_by_name, get_all_modules
from create_pseudocode_translator import PseudocodeTranslator


def combine_raymarching_modules(module_names):
    """
    Combine multiple raymarching modules into a single GLSL shader
    """
    shader_parts = {
        'header': '#version 330 core\n\n',
        'uniforms': '',
        'inputs': '',
        'outputs': '',
        'common_functions': '',
        'module_functions': [],
        'main_function': ''
    }

    # Add common raymarching uniforms
    shader_parts['uniforms'] = '''// Common raymarching uniforms
uniform vec3 cameraPos;
uniform vec3 cameraTarget;
uniform vec2 resolution;
uniform float time;
uniform vec2 mouse;

'''

    # Add common inputs/outputs for raymarching
    shader_parts['inputs'] = '''// Input variables for raymarching
in vec2 fragCoord;

'''

    shader_parts['outputs'] = '''// Output
out vec4 FragColor;

'''

    # Add common functions needed for raymarching
    shader_parts['common_functions'] = '''// Common raymarching functions
// Placeholder for distance function that will be defined by the scene module
vec2 map(vec3 p) {
    // This function will be replaced by the selected scene module
    return vec2(0.0, 0.0);
}

// PI constant
const float PI = 3.14159265359;

'''

    # Collect functions from selected modules
    all_pseudocodes = []
    for module_name in module_names:
        module = get_module_by_name(module_name)
        if module:
            pseudocode = module['pseudocode']
            shader_parts['module_functions'].append(pseudocode)
            all_pseudocodes.append(pseudocode)

    # Create the main function based on selected modules
    main_func = create_main_function(module_names)
    shader_parts['main_function'] = main_func

    # Combine all parts
    shader = shader_parts['header']
    shader += shader_parts['uniforms']
    shader += shader_parts['inputs']
    shader += shader_parts['outputs']
    shader += shader_parts['common_functions']
    
    for func in shader_parts['module_functions']:
        shader += func
        shader += "\n"
    
    shader += shader_parts['main_function']

    return shader


def create_main_function(module_names):
    """
    Create the main function based on selected modules
    """
    main_func = '''void main() {
    // Set up ray
    vec2 uv = (fragCoord - 0.5 * resolution.xy) / resolution.y;
    
    // Generate ray direction
    vec3 ro = cameraPos;  // Ray origin
    vec3 rd = normalize(vec3(uv, -1.0));  // Ray direction
    
    // Apply camera transformation based on target
    vec3 forward = normalize(cameraTarget - cameraPos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));
    
    rd = normalize(forward + uv.x * right + uv.y * up);
    
    // Perform raymarching
    vec2 result = raymarch(ro, rd, 20.0, 64);  // Default values
    float dist = result.x;
    float material_id = result.y;
    
    // Calculate color based on distance
    vec3 color = vec3(0.0);
    
    if (dist < 20.0) {  // Hit something
        vec3 pos = ro + rd * dist;
        vec3 normal = calculateNormal(pos, 0.001);
        
        // Apply lighting
        vec3 viewDir = normalize(cameraPos - pos);
        color = raymarchingLighting(pos, normal, viewDir, vec3(0.5, 0.7, 1.0), 0.2, 0.0);
    } else {  // Background
        color = vec3(0.05);
    }
    
    // Apply post-processing effects if available
    color = pow(color, vec3(0.4545));  // Gamma correction
    
    FragColor = vec4(color, 1.0);
}
'''

    # For now, return a basic main function
    # In a more advanced implementation, we'd customize this based on selected modules
    return main_func


def generate_complete_raymarching_shader(module_names):
    """
    Generate a complete raymarching shader from selected modules
    """
    print(f"Generating raymarching shader with modules: {module_names}")
    
    # Combine modules
    shader_code = combine_raymarching_modules(module_names)
    
    # Translate if needed (though in this case pseudocode is already GLSL-like)
    translator = PseudocodeTranslator()
    translated_shader = translator.translate_to_glsl(shader_code)
    
    return translated_shader


if __name__ == "__main__":
    # Example usage
    print("Testing Raymarching Module Combiner...")
    
    # Combine some raymarching modules
    test_modules = ['raymarching_core', 'sdf_primitives', 'raymarching_lighting', 'ray_generation']
    shader = combine_raymarching_modules(test_modules)
    
    print("Generated shader (first 30 lines):")
    lines = shader.split('\n')
    for i, line in enumerate(lines[:30]):
        print(f"{i+1:2d}: {line}")
    if len(lines) > 30:
        print(f"... and {len(lines) - 30} more lines")
    
    # Save the shader
    with open('generated_raymarching_shader.glsl', 'w') as f:
        f.write(shader)
    
    print("\nShader saved to generated_raymarching_shader.glsl")
    
    # Also try generating from selected modules
    complete_shader = generate_complete_raymarching_shader(test_modules)
    with open('complete_raymarching_shader.glsl', 'w') as f:
        f.write(complete_shader)
    
    print("Complete shader saved to complete_raymarching_shader.glsl")