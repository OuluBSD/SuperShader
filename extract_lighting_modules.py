#!/usr/bin/env python3
"""
Improved script to extract complete lighting modules from shaders.
"""

import json
import os
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def find_matching_brace(code, start_pos):
    """
    Find the matching closing brace for an opening brace at start_pos.
    
    Args:
        code (str): The code string
        start_pos (int): Position of the opening brace
    
    Returns:
        int: Position of the matching closing brace, or -1 if not found
    """
    brace_count = 1
    pos = start_pos + 1
    
    while pos < len(code) and brace_count > 0:
        if code[pos] == '{':
            brace_count += 1
        elif code[pos] == '}':
            brace_count -= 1
        pos += 1
    
    return pos - 1 if brace_count == 0 else -1


def extract_complete_functions(code, pattern):
    """
    Extract complete functions that match a given pattern.
    
    Args:
        code (str): GLSL code to search in
        pattern (str): Base pattern to match (like 'ambient', 'diffuse', etc.)
    
    Returns:
        list: List of complete function definitions
    """
    functions = []
    
    # Updated regex to find function declarations with name containing the pattern
    func_patterns = [
        rf'(\w+)\s+[\w\d_]*{pattern}[\w\d_]*\s*\([^)]*\)\s*\{{',
        rf'(\w+)\s+[\w\d_]*{pattern}[\w\d_]*\s*\([^)]*\)\s+[\w\d_]*\s*\{{',  # With qualifier like 'const'
        rf'(\w+)\s+[\w\d_]*{pattern}[\w\d_]*\s*\([^)]*\)\s*\n\s*\{{',  # With newline
    ]
    
    for func_pattern in func_patterns:
        matches = re.finditer(func_pattern, code, re.IGNORECASE)
        
        for match in matches:
            # Find the opening brace position
            open_brace_pos = match.end() - 1  # Position of the opening brace
            
            # Find the matching closing brace
            close_brace_pos = find_matching_brace(code, open_brace_pos)
            
            if close_brace_pos != -1:
                # Extract the complete function
                func_start = match.start()
                func_end = close_brace_pos + 1
                function_code = code[func_start:func_end]
                
                # Clean up and add to results if it's not already there
                function_code = function_code.strip()
                if function_code and function_code not in functions:
                    functions.append(function_code)
    
    return functions


def find_lighting_shaders(json_dir='json'):
    """
    Find all JSON files that contain lighting-related tags.
    """
    print("Finding lighting-related shaders...")
    
    lighting_keywords = [
        'lighting', 'light', 'shadow', 'specular', 'diffuse', 'ambient', 
        'phong', 'pbr', 'illumination', 'lightning', 'reflection', 
        'refraction', 'fresnel', 'brdf', 'bdrf', 'radiance', 'irradiance'
    ]
    
    lighting_shaders = []
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} JSON files for lighting-related tags...")
    
    for i, filepath in enumerate(json_files):
        if i % 1000 == 0:
            print(f"Scanned {i}/{len(json_files)} files...")
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict) and 'info' in data:
                info = data['info']
                tags = [tag.lower() for tag in info.get('tags', [])]
                name = info.get('name', '').lower()
                description = info.get('description', '').lower()
                
                # Check if this shader is lighting-related based on tags, name, or description
                is_lighting_related = False
                
                # Check tags
                for tag in tags:
                    if any(keyword in tag for keyword in lighting_keywords):
                        is_lighting_related = True
                        break
                
                # Check name
                if not is_lighting_related:
                    for keyword in lighting_keywords:
                        if keyword in name:
                            is_lighting_related = True
                            break
                
                # Check description
                if not is_lighting_related:
                    for keyword in lighting_keywords:
                        if keyword in description:
                            is_lighting_related = True
                            break
                
                if is_lighting_related:
                    shader_info = {
                        'id': info.get('id', os.path.basename(filepath).replace('.json', '')),
                        'name': info.get('name', ''),
                        'tags': tags,
                        'username': info.get('username', ''),
                        'description': info.get('description', ''),
                        'filepath': filepath
                    }
                    lighting_shaders.append((filepath, shader_info))
                    
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue

    print(f"Found {len(lighting_shaders)} lighting-related shaders")
    return lighting_shaders


def extract_shader_code(filepath):
    """
    Extract GLSL code from a JSON shader file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        shader_data = json.load(f)

    glsl_code = []

    # Extract GLSL code based on render passes
    if 'renderpass' in shader_data:
        for i, pass_data in enumerate(shader_data['renderpass']):
            if 'code' in pass_data:
                code = pass_data['code']
                pass_type = pass_data.get('type', 'fragment')
                name = pass_data.get('name', f'Pass {i}')
                
                glsl_code.append(f"// {name} ({pass_type})")
                glsl_code.append(code)
                glsl_code.append("")  # Empty line separator
    else:
        # Try to find shader code in other possible fields
        possible_fields = ['fragment_shader', 'vertex_shader', 'shader', 'code', 'main']
        for field in possible_fields:
            if field in shader_data and isinstance(shader_data[field], str):
                code = shader_data[field]
                glsl_code.append(f"// From field: {field}")
                glsl_code.append(code)
                glsl_code.append("")
    
    return "\n".join(glsl_code)


def extract_specific_lighting_functions(shader_codes):
    """
    Extract specific types of lighting functions from shader codes.
    
    Args:
        shader_codes (list): List of shader data from analysis
    
    Returns:
        dict: Dictionary of extracted lighting modules by type
    """
    print("Extracting specific lighting functions...")
    
    modules = {
        'ambient': set(),
        'diffuse': set(),
        'specular': set(),
        'phong': set(),
        'blinn_phong': set(),
        'pbr': set(),
        'shadow': set(),
        'light_calculation': set()
    }
    
    total_processed = 0
    
    for shader_data in shader_codes:
        code = shader_data['code']
        
        # Extract ambient functions
        ambient_funcs = extract_complete_functions(code, 'ambient')
        modules['ambient'].update(ambient_funcs)
        
        # Extract diffuse functions
        diffuse_funcs = extract_complete_functions(code, 'diffuse')
        modules['diffuse'].update(diffuse_funcs)
        
        # Extract specular functions
        specular_funcs = extract_complete_functions(code, 'specular')
        modules['specular'].update(specular_funcs)
        
        # Extract phong functions
        phong_funcs = extract_complete_functions(code, 'phong')
        modules['phong'].update(phong_funcs)
        
        # Extract blinn-phong functions
        blinn_phong_funcs = extract_complete_functions(code, 'blinn')
        modules['blinn_phong'].update(blinn_phong_funcs)
        
        # Extract PBR functions
        pbr_funcs = extract_complete_functions(code, 'pbr')
        modules['pbr'].update(pbr_funcs)
        
        # Extract shadow functions
        shadow_funcs = extract_complete_functions(code, 'shadow')
        modules['shadow'].update(shadow_funcs)
        
        # Extract general light calculation functions
        light_funcs = extract_complete_functions(code, 'light')
        modules['light_calculation'].update(light_funcs)
        
        total_processed += 1
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(shader_codes)} shaders...")
    
    print(f"Extraction complete! Found:")
    for module_type, funcs in modules.items():
        print(f"  {module_type}: {len(funcs)} functions")
    
    return modules


def save_improved_modules(modules):
    """
    Save the improved lighting modules to organized files.
    """
    os.makedirs('modules/lighting', exist_ok=True)
    
    for module_type, func_list in modules.items():
        if func_list:  # Only save if there are modules of this type
            with open(f'modules/lighting/{module_type}_functions.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} Lighting Functions\n")
                f.write("// Automatically extracted from lighting-related shaders\n\n")
                
                for i, func in enumerate(func_list, 1):
                    f.write(f"// Function {i}\n")
                    f.write(func)
                    f.write("\n\n")


def create_standardized_modules(modules):
    """
    Create standardized lighting modules based on patterns found.
    """
    print("Creating standardized lighting modules...")
    
    # Define standardized module templates
    standardized_modules = {
        'pbr_lit.glsl': {
            'description': 'Physically Based Rendering lighting calculations',
            'pseudocode': '''PBR Lighting Model:
    Input: 
        - surface properties (albedo, metallic, roughness, normal)
        - light properties (direction, color, intensity)
        - view direction
    
    Process:
        1. Calculate microfacet distribution (D)
        2. Calculate geometry function (G) 
        3. Calculate Fresnel term (F)
        4. Combine using Cook-Torrance BRDF
        5. Apply energy conservation
        
    Output: Final lit color
            '''
        },
        
        'diffuse_lit.glsl': {
            'description': 'Standard diffuse lighting calculations',
            'pseudocode': '''Diffuse Lighting Model:
    Input: 
        - surface normal
        - light direction
        - light color
    
    Process:
        1. Calculate dot product of normal and light direction
        2. Clamp to [0, 1] range
        3. Multiply by light color
    
    Output: Diffuse contribution to final color
            '''
        },
        
        'specular_blinn_phong.glsl': {
            'description': 'Blinn-Phong specular lighting model',
            'pseudocode': '''Blinn-Phong Specular Model:
    Input: 
        - surface normal
        - view direction
        - light direction
        - material properties (shininess)
    
    Process:
        1. Calculate half vector between view and light directions
        2. Compute dot product of normal and half vector
        3. Raise to power of shininess
        4. Clamp to [0, 1] range
    
    Output: Specular highlight contribution
            '''
        },
        
        'shadow_mapping.glsl': {
            'description': 'Shadow mapping calculations',
            'pseudocode': '''Shadow Mapping:
    Input:
        - fragment position in light space
        - shadow map texture
        - light properties
    
    Process:
        1. Transform fragment position to light space
        2. Sample shadow map
        3. Compare depth values
        4. Apply shadow factor
    
    Output: Shadow attenuation factor
            '''
        },
        
        'light_attenuation.glsl': {
            'description': 'Light attenuation calculations',
            'pseudocode': '''Light Attenuation:
    Input:
        - distance from light source
        - light attenuation parameters
    
    Process:
        1. Calculate distance to light
        2. Apply quadratic, linear, or constant attenuation
        3. Calculate final attenuation factor
    
    Output: Attenuation factor (0 to 1)
            '''
        }
    }
    
    os.makedirs('modules/lighting/standardized', exist_ok=True)
    
    # Create standardized modules with pseudocode
    for filename, data in standardized_modules.items():
        filepath = f'modules/lighting/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"// {data['description']}\n")
            f.write("// Pseudocode definition for standardized lighting module\n\n")
            f.write("// PSEUDOCODE:\n")
            for line in data['pseudocode'].split('\n'):
                f.write(f"// {line}\n")
            f.write("\n")
            f.write("// Implementation would go here based on the pseudocode above\n")
    
    print(f"Created {len(standardized_modules)} standardized lighting modules")


def main():
    # Find lighting shaders
    lighting_shaders = find_lighting_shaders()
    
    # Extract shader codes
    shader_codes = []
    for filepath, shader_info in lighting_shaders[:500]:  # Limit to first 500 for efficiency
        shader_code = extract_shader_code(filepath)
        shader_codes.append({
            'info': shader_info,
            'code': shader_code
        })
    
    # Extract specific lighting functions
    modules = extract_specific_lighting_functions(shader_codes)
    
    # Save the improved modules
    save_improved_modules(modules)
    
    # Create standardized modules
    create_standardized_modules(modules)
    
    print("Improved lighting modules extraction and standardization completed!")


if __name__ == "__main__":
    main()