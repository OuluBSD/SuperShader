#!/usr/bin/env python3
"""
Process lighting-related shaders from JSON files to identify common patterns
and extract reusable lighting modules.
"""

import json
import os
import glob
from collections import Counter, defaultdict
from pathlib import Path


def find_lighting_shaders(json_dir='json'):
    """
    Find all JSON files that contain lighting-related tags.

    Args:
        json_dir (str): Directory containing JSON shader files

    Returns:
        list: List of tuples (filepath, shader_info) for lighting-related shaders
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

    Args:
        filepath (str): Path to the JSON shader file

    Returns:
        str: GLSL code extracted from the file
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


def identify_lighting_patterns(shader_code):
    """
    Identify common lighting patterns in shader code.

    Args:
        shader_code (str): GLSL code to analyze

    Returns:
        dict: Dictionary of identified lighting patterns
    """
    patterns = {
        'blinn_phong': 'Blinn-Phong' in shader_code or 'blinnPhong' in shader_code or 'blinn-phong' in shader_code.lower(),
        'phong': 'phong' in shader_code.lower() and not 'blinn' in shader_code.lower(),
        'lambert': 'lambert' in shader_code.lower() or 'diffuse' in shader_code.lower() and 'lambert' in shader_code,
        'pbr': 'pbr' in shader_code.lower() or 'metallic' in shader_code.lower() or 'roughness' in shader_code.lower() or 'ao' in shader_code.lower(),
        'specular': 'specular' in shader_code.lower(),
        'diffuse': 'diffuse' in shader_code.lower(),
        'ambient': 'ambient' in shader_code.lower(),
        'shadow_mapping': 'shadow' in shader_code.lower() and 'map' in shader_code.lower(),
        'normal_mapping': 'normal' in shader_code.lower() and 'map' in shader_code.lower(),
        'fresnel': 'fresnel' in shader_code.lower(),
        'toon_shading': 'toon' in shader_code.lower(),
        'cel_shading': 'cel' in shader_code.lower(),
        'ray_tracing': 'ray' in shader_code.lower() and ('trace' in shader_code.lower() or 'tracing' in shader_code.lower()),
        'ray_marching': 'raymarch' in shader_code.lower() or 'ray marching' in shader_code.lower(),
        'light_direction': 'lightDir' in shader_code or 'lightDirection' in shader_code or 'light_dir' in shader_code,
        'light_color': 'lightColor' in shader_code or 'light.color' in shader_code,
        'light_attenuation': 'attenuation' in shader_code.lower(),
        'multiple_lights': 'light[' in shader_code or 'lights[' in shader_code,
        'point_light': 'pointLight' in shader_code or ('point' in shader_code.lower() and 'light' in shader_code.lower()),
        'spot_light': 'spotLight' in shader_code or ('spot' in shader_code.lower() and 'light' in shader_code.lower()),
        'directional_light': 'dirLight' in shader_code or ('directional' in shader_code.lower() and 'light' in shader_code.lower()),
    }
    
    # Filter only the patterns that were found
    active_patterns = {k: v for k, v in patterns.items() if v}
    return active_patterns


def analyze_lighting_shaders():
    """
    Main function to analyze lighting shaders.
    """
    print("Analyzing lighting shaders...")
    
    lighting_shaders = find_lighting_shaders()
    
    # Store shader codes and identified patterns
    shader_codes = []
    all_patterns = []
    pattern_counts = Counter()
    
    print("\nAnalyzing lighting patterns in shaders...")
    for i, (filepath, shader_info) in enumerate(lighting_shaders):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(lighting_shaders)} lighting shaders...")
        
        shader_code = extract_shader_code(filepath)
        patterns = identify_lighting_patterns(shader_code)
        
        shader_codes.append({
            'info': shader_info,
            'code': shader_code,
            'patterns': patterns
        })
        
        all_patterns.append(patterns)
        
        # Update pattern counts
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    print(f"\nAnalysis complete! Found {len(lighting_shaders)} lighting shaders.")
    
    # Print pattern distribution
    print(f"\nCommon lighting patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count} shaders")
    
    # Save analysis results
    save_lighting_analysis(shader_codes, pattern_counts)
    
    return shader_codes, pattern_counts


def save_lighting_analysis(shader_codes, pattern_counts):
    """
    Save the lighting analysis results to files.
    """
    os.makedirs('analysis/lighting', exist_ok=True)
    
    # Save pattern statistics
    with open('analysis/lighting/pattern_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Lighting Pattern Statistics\n")
        f.write("=" * 50 + "\n")
        for pattern, count in pattern_counts.most_common():
            f.write(f"{pattern.replace('_', ' ').title()}: {count}\n")
    
    # Save detailed shader analysis
    with open('analysis/lighting/shader_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Lighting Shader Analysis\n")
        f.write("=" * 50 + "\n")
        for shader_data in shader_codes:
            info = shader_data['info']
            patterns = shader_data['patterns']
            
            f.write(f"\nShader ID: {info['id']}\n")
            f.write(f"Name: {info['name']}\n")
            f.write(f"Author: {info['username']}\n")
            f.write(f"Tags: {', '.join(info['tags'])}\n")
            f.write(f"Patterns: {', '.join([p.replace('_', ' ').title() for p in patterns])}\n")
            f.write("-" * 30 + "\n")
    
    print("Lighting shader analysis saved to analysis/lighting/ directory")


def extract_lighting_modules(shader_codes):
    """
    Extract reusable lighting modules from analyzed shaders.

    Args:
        shader_codes (list): List of shader data from analysis

    Returns:
        dict: Dictionary of extracted lighting modules
    """
    print("\nExtracting reusable lighting modules...")
    
    modules = {
        'ambient': [],
        'diffuse': [],
        'specular': [],
        'phong_blinns': [],
        'pbr': [],
        'shadow_mapping': [],
        'light_types': [],
        'light_attenuation': []
    }
    
    for shader_data in shader_codes:
        code = shader_data['code']
        patterns = shader_data['patterns']
        
        # Extract ambient lighting functions
        if patterns.get('ambient', False):
            # Look for ambient lighting calculation functions
            import re
            ambient_matches = re.findall(r'float\s+\w*ambient\w*\s*\([^)]*\)\s*\{[^}]*\}', code, re.IGNORECASE)
            ambient_matches += re.findall(r'vec[34]\s+\w*ambient\w*\s*\([^)]*\)\s*\{[^}]*\}', code, re.IGNORECASE)
            if ambient_matches:
                modules['ambient'].extend(ambient_matches)
        
        # Extract diffuse lighting functions
        if patterns.get('diffuse', False):
            # Look for diffuse lighting calculation functions
            diffuse_matches = re.findall(r'float\s+\w*diffuse\w*\s*\([^)]*\)\s*\{[^}]*\}', code, re.IGNORECASE)
            diffuse_matches += re.findall(r'vec[34]\s+\w*diffuse\w*\s*\([^)]*\)\s*\{[^}]*\}', code, re.IGNORECASE)
            if diffuse_matches:
                modules['diffuse'].extend(diffuse_matches)
        
        # Extract specular lighting functions
        if patterns.get('specular', False):
            # Look for specular lighting calculation functions
            specular_matches = re.findall(r'float\s+\w*specular\w*\s*\([^)]*\)\s*\{[^}]*\}', code, re.IGNORECASE)
            specular_matches += re.findall(r'vec[34]\s+\w*specular\w*\s*\{[^}]*\}', code, re.IGNORECASE)
            if specular_matches:
                modules['specular'].extend(specular_matches)
    
    # Remove duplicates
    for module_type in modules:
        modules[module_type] = list(set(modules[module_type]))
    
    print(f"Extracted {len(modules['ambient'])} ambient, {len(modules['diffuse'])} diffuse, "
          f"{len(modules['specular'])} specular lighting modules")
    
    # Save modules
    save_lighting_modules(modules)
    
    return modules


def save_lighting_modules(modules):
    """
    Save extracted lighting modules to files.
    """
    os.makedirs('modules', exist_ok=True)
    os.makedirs('modules/lighting', exist_ok=True)
    
    for module_type, module_list in modules.items():
        if module_list:  # Only save if there are modules of this type
            with open(f'modules/lighting/{module_type}.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} Lighting Modules\n\n")
                for i, module in enumerate(module_list):
                    f.write(f"// Module {i+1}\n")
                    f.write(module)
                    f.write("\n\n")
    
    print("Lighting modules saved to modules/lighting/ directory")


def main():
    # Analyze lighting shaders
    shader_codes, pattern_counts = analyze_lighting_shaders()
    
    # Extract reusable modules
    modules = extract_lighting_modules(shader_codes)
    
    print("\nLighting shader analysis and module extraction completed!")


if __name__ == "__main__":
    main()