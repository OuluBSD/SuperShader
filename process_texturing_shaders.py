#!/usr/bin/env python3
"""
Process texturing/mapping shaders from JSON files to identify common patterns
and extract reusable modules.
"""

import json
import os
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def find_texturing_mapping_shaders(json_dir='json'):
    """
    Find all JSON files that contain texturing/mapping related tags.

    Args:
        json_dir (str): Directory containing JSON shader files

    Returns:
        list: List of tuples (filepath, shader_info) for texturing/mapping shaders
    """
    print("Finding texturing/mapping related shaders...")
    
    keywords = [
        'texture', 'texturing', 'uv', 'mapping', 'sampling', 'sampler', 'texel',
        'coordinate', 'coordinates', 'projection', 'cubemap', 'cubemap', 'envmap',
        'normal', 'normalmap', 'bump', 'bumpmap', 'parallax', 'displacement',
        'triplanar', 'planar', 'cylindrical', 'spherical', 'projection',
        'tiled', 'tiling', 'repeat', 'wrap', 'seamless', 'atlas', 'mipmap',
        'anisotropic', 'filtering', 'bilinear', 'trilinear', 'nearest', 'mip',
        'packing', 'unpacking', 'blend', 'blending', 'alpha', 'opacity',
        'color', 'rgb', 'rgba', 'channels', 'swizzle', 'packing', 'unpacking'
    ]
    
    texture_shaders = []
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} JSON files for texturing/mapping tags...")
    
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
                
                # Check if this shader is texturing/mapping related 
                is_texture_related = False
                
                # Check tags
                for tag in tags:
                    if any(keyword in tag for keyword in keywords):
                        is_texture_related = True
                        break
                
                # Check name
                if not is_texture_related:
                    for keyword in keywords:
                        if keyword in name:
                            is_texture_related = True
                            break
                
                # Check description
                if not is_texture_related:
                    for keyword in keywords:
                        if keyword in description:
                            is_texture_related = True
                            break
                
                if is_texture_related:
                    shader_info = {
                        'id': info.get('id', os.path.basename(filepath).replace('.json', '')),
                        'name': info.get('name', ''),
                        'tags': tags,
                        'username': info.get('username', ''),
                        'description': info.get('description', ''),
                        'filepath': filepath
                    }
                    texture_shaders.append((filepath, shader_info))
                    
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue

    print(f"Found {len(texture_shaders)} texturing/mapping related shaders")
    return texture_shaders


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


def identify_texture_patterns(shader_code):
    """
    Identify common texturing/mapping patterns in shader code.

    Args:
        shader_code (str): GLSL code to analyze

    Returns:
        dict: Dictionary of identified texture patterns
    """
    patterns = {
        # UV mapping
        'uv_mapping': 'uv' in shader_code.lower(),
        'uv_coordinates': 'fragcoord' in shader_code.lower() or 'uv' in shader_code.lower() and 'coord' in shader_code.lower(),
        'uv_manipulation': 'uv' in shader_code.lower() and ('*' in shader_code or '+' in shader_code or '-' in shader_code or '/' in shader_code),
        
        # Texture sampling
        'texture_sampling': 'texture' in shader_code.lower() or 'texel' in shader_code.lower(),
        'texture2d_sampling': 'texture2d' in shader_code.lower() or 'texture' in shader_code.lower() and '2d' in shader_code.lower(),
        'texture3d_sampling': 'texture3d' in shader_code.lower() or 'texture' in shader_code.lower() and '3d' in shader_code.lower(),
        'cubemap_sampling': 'texturecube' in shader_code.lower() or 'cube' in shader_code.lower() and 'map' in shader_code.lower(),
        
        # Normal mapping
        'normal_mapping': 'normal' in shader_code.lower() and 'map' in shader_code.lower(),
        'tangent_space': 'tangent' in shader_code.lower() or 'binormal' in shader_code.lower() or 'bitangent' in shader_code.lower(),
        
        # Procedural texturing
        'procedural_texture': 'procedural' in shader_code.lower() and 'texture' in shader_code.lower(),
        'triplanar_mapping': 'triplanar' in shader_code.lower() or ('x' in shader_code and 'y' in shader_code and 'z' in shader_code and 'xyz' in shader_code),
        'cylindrical_mapping': 'cylindrical' in shader_code.lower() and 'map' in shader_code.lower(),
        'spherical_mapping': 'spherical' in shader_code.lower() and 'map' in shader_code.lower(),
        
        # Filtering
        'bilinear_filtering': 'bilinear' in shader_code.lower(),
        'trilinear_filtering': 'trilinear' in shader_code.lower(),
        'anisotropic_filtering': 'anisotropic' in shader_code.lower(),
        
        # Texture blending
        'texture_blend': 'blend' in shader_code.lower() and 'texture' in shader_code.lower(),
        'alphablend': 'alpha' in shader_code.lower() and 'blend' in shader_code.lower(),
        'multiply_blend': 'multiply' in shader_code.lower() or ('*' in shader_code and 'color' in shader_code.lower()),
        'additive_blend': 'add' in shader_code.lower() or '+' in shader_code,
        
        # Tiling
        'tiling': 'tile' in shader_code.lower() or 'repeat' in shader_code.lower(),
        'seamless_texture': 'seamless' in shader_code.lower() and 'texture' in shader_code.lower(),
        
        # Texture transformations
        'texture_offset': 'offset' in shader_code.lower() and 'uv' in shader_code.lower(),
        'texture_scale': 'scale' in shader_code.lower() and 'uv' in shader_code.lower(),
        'texture_rotation': 'rotate' in shader_code.lower() and 'uv' in shader_code.lower(),
        
        # Special texture operations
        'mipmapping': 'mip' in shader_code.lower() or 'mipmap' in shader_code.lower(),
        'texture_atlas': 'atlas' in shader_code.lower(),
        'packing_unpacking': 'pack' in shader_code.lower() or 'unpack' in shader_code.lower(),
    }
    
    # Filter only the patterns that were found
    active_patterns = {k: v for k, v in patterns.items() if v}
    return active_patterns


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
        pattern (str): Base pattern to match (like 'texture', 'uv', 'normal', etc.)
    
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


def analyze_texture_shaders():
    """
    Main function to analyze texturing/mapping shaders.
    """
    print("Analyzing texturing/mapping shaders...")
    
    texture_shaders = find_texturing_mapping_shaders()
    
    # Store shader codes and identified patterns
    shader_codes = []
    all_patterns = []
    pattern_counts = Counter()
    
    print("\nAnalyzing texture patterns in shaders...")
    for i, (filepath, shader_info) in enumerate(texture_shaders):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(texture_shaders)} texture shaders...")
        
        shader_code = extract_shader_code(filepath)
        patterns = identify_texture_patterns(shader_code)
        
        shader_codes.append({
            'info': shader_info,
            'code': shader_code,
            'patterns': patterns
        })
        
        all_patterns.append(patterns)
        
        # Update pattern counts
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    print(f"\nAnalysis complete! Found {len(texture_shaders)} texturing/mapping shaders.")
    
    # Print pattern distribution
    print(f"\nCommon texturing patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count} shaders")
    
    # Save analysis results
    save_texture_analysis(shader_codes, pattern_counts)
    
    return shader_codes, pattern_counts


def save_texture_analysis(shader_codes, pattern_counts):
    """
    Save the texture analysis results to files.
    """
    os.makedirs('analysis/texturing', exist_ok=True)
    
    # Save pattern statistics
    with open('analysis/texturing/pattern_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Texturing Pattern Statistics\n")
        f.write("=" * 50 + "\n")
        for pattern, count in pattern_counts.most_common():
            f.write(f"{pattern.replace('_', ' ').title()}: {count}\n")
    
    # Save detailed shader analysis
    with open('analysis/texturing/shader_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Texturing/Mapping Shader Analysis\n")
        f.write("=" * 50 + "\n")
        for shader_data in shader_codes[:50]:  # Only first 50 for file size
            info = shader_data['info']
            patterns = shader_data['patterns']
            
            f.write(f"\nShader ID: {info['id']}\n")
            f.write(f"Name: {info['name']}\n")
            f.write(f"Author: {info['username']}\n")
            f.write(f"Tags: {', '.join(info['tags'])}\n")
            f.write(f"Patterns: {', '.join([p.replace('_', ' ').title() for p in patterns])}\n")
            f.write("-" * 30 + "\n")
    
    print("Texturing/mapping shader analysis saved to analysis/texturing/ directory")


def extract_texture_modules(shader_codes):
    """
    Extract reusable texturing modules from analyzed shaders.

    Args:
        shader_codes (list): List of shader data from analysis

    Returns:
        dict: Dictionary of extracted texturing modules
    """
    print("\nExtracting reusable texturing modules...")
    
    modules = {
        'uv_mapping': set(),
        'texture_sampling': set(),
        'normal_mapping': set(),
        'procedural_textures': set(),
        'texture_filtering': set(),
        'texture_blending': set()
    }
    
    total_processed = 0
    
    for shader_data in shader_codes:
        code = shader_data['code']
        
        # Extract UV mapping functions
        uv_funcs = extract_complete_functions(code, 'uv')
        uv_funcs += extract_complete_functions(code, 'map')
        modules['uv_mapping'].update(uv_funcs)
        
        # Extract texture sampling functions
        texture_funcs = extract_complete_functions(code, 'texture')
        texture_funcs += extract_complete_functions(code, 'sample')
        modules['texture_sampling'].update(texture_funcs)
        
        # Extract normal mapping functions
        normal_funcs = extract_complete_functions(code, 'normal')
        normal_funcs += extract_complete_functions(code, 'tangent')
        modules['normal_mapping'].update(normal_funcs)
        
        # Extract procedural texture functions
        proc_funcs = extract_complete_functions(code, 'procedural')
        proc_funcs += extract_complete_functions(code, 'noise')
        modules['procedural_textures'].update(proc_funcs)
        
        # Extract filtering functions
        filter_funcs = extract_complete_functions(code, 'filter')
        filter_funcs += extract_complete_functions(code, 'mip')
        modules['texture_filtering'].update(filter_funcs)
        
        # Extract blending functions
        blend_funcs = extract_complete_functions(code, 'blend')
        blend_funcs += extract_complete_functions(code, 'mix')
        modules['texture_blending'].update(blend_funcs)
        
        total_processed += 1
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(shader_codes)} shaders...")
    
    print(f"Extraction complete! Found:")
    for module_type, funcs in modules.items():
        print(f"  {module_type}: {len(funcs)} functions")
    
    # Save modules
    save_texture_modules(modules)
    
    return modules


def save_texture_modules(modules):
    """
    Save extracted texturing modules to files.
    """
    os.makedirs('modules/texturing', exist_ok=True)
    
    for module_type, func_list in modules.items():
        if func_list:  # Only save if there are modules of this type
            with open(f'modules/texturing/{module_type}_functions.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} Texturing Functions\n")
                f.write("// Automatically extracted from texturing/mapping-related shaders\n\n")
                
                for i, func in enumerate(func_list, 1):
                    f.write(f"// Function {i}\n")
                    f.write(func)
                    f.write("\n\n")
    
    print("Texturing modules saved to modules/texturing/ directory")


def create_standardized_texture_modules():
    """
    Create standardized texturing modules based on patterns found.
    """
    print("Creating standardized texturing modules...")
    
    # Define standardized module templates with actual GLSL implementations
    standardized_modules = {
        'uv_mapping.glsl': generate_uv_mapping_glsl(),
        'texture_sampling.glsl': generate_texture_sampling_glsl(),
        'normal_mapping.glsl': generate_normal_mapping_glsl(),
        'procedural_textures.glsl': generate_procedural_textures_glsl(),
        'texture_blending.glsl': generate_texture_blending_glsl(),
        'triplanar_mapping.glsl': generate_triplanar_mapping_glsl()
    }
    
    os.makedirs('modules/texturing/standardized', exist_ok=True)
    
    # Create standardized modules
    for filename, code in standardized_modules.items():
        filepath = f'modules/texturing/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Created {len(standardized_modules)} standardized texturing modules")


def generate_uv_mapping_glsl():
    """Generate GLSL implementation for UV mapping."""
    return """// UV mapping module
// Standardized UV mapping implementations

// Basic UV coordinate calculation
vec2 calculateUV(vec2 fragCoord, vec2 resolution) {
    return fragCoord.xy / resolution.xy;
}

// UV with aspect ratio correction
vec2 calculateUVCorrected(vec2 fragCoord, vec2 resolution) {
    vec2 uv = fragCoord.xy / resolution.xy;
    uv.x *= resolution.x / resolution.y;
    return uv;
}

// Tiled UV mapping with repeat
vec2 tileUV(vec2 uv, float repeat) {
    return fract(uv * repeat);
}

// Offset UV mapping
vec2 offsetUV(vec2 uv, vec2 offset) {
    return uv + offset;
}

// Scale UV mapping
vec2 scaleUV(vec2 uv, vec2 scale) {
    return uv * scale;
}

// Rotate UV coordinates around center
vec2 rotateUV(vec2 uv, float angle) {
    uv -= 0.5; // Center origin
    float s = sin(angle);
    float c = cos(angle);
    uv = mat2(c, -s, s, c) * uv;
    uv += 0.5; // Return to [0,1] space
    return uv;
}

// Polar UV mapping (radial)
vec2 cartesianToPolar(vec2 uv) {
    uv -= 0.5; // Center origin
    float r = length(uv);
    float theta = atan(uv.y, uv.x);
    return vec2(r, theta);
}

// Cylindrical UV mapping
vec3 cartesianToCylindrical(vec3 pos) {
    float r = length(pos.xz);
    float theta = atan(pos.z, pos.x);
    return vec3(r, pos.y, theta);
}

// Spherical UV mapping
vec3 cartesianToSpherical(vec3 pos) {
    float r = length(pos);
    float theta = atan(pos.z, pos.x);
    float phi = acos(pos.y / r);
    return vec3(r, theta, phi);
}
"""


def generate_texture_sampling_glsl():
    """Generate GLSL implementation for texture sampling."""
    return """// Texture sampling module
// Standardized texture sampling implementations

// Basic texture sampling with 2D UV
vec4 sampleTexture(sampler2D tex, vec2 uv) {
    return texture2D(tex, uv);
}

// Texture sampling with triplanar blending
vec4 sampleTriplanar(sampler2D tex, vec3 worldPos, vec3 normal) {
    // Project world position onto each axis
    vec2 uvX = worldPos.yz;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, uvX);
    vec4 texY = texture2D(tex, uvY);
    vec4 texZ = texture2D(tex, uvZ);
    
    // Use normal to determine blending weights
    vec3 blend = abs(normal);
    blend = normalize(max(blend, 0.00001)); // Avoid division by zero
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}

// Sample texture with triplanar using 3D coordinates
vec4 sampleTriplanar3D(sampler2D tex, vec3 pos, vec3 normal, float scale) {
    // Project world position onto each axis
    vec3 projX = vec3(pos.y, pos.z, 0.0) * scale;
    vec3 projY = vec3(pos.x, pos.z, 0.0) * scale;
    vec3 projZ = vec3(pos.x, pos.y, 0.0) * scale;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, projX.xy);
    vec4 texY = texture2D(tex, projY.xy);
    vec4 texZ = texture2D(tex, projZ.xy);
    
    // Use normal to determine blending weights
    vec3 blend = pow(abs(normal), vec3(2.0));
    blend = normalize(max(blend, 0.00001));
    blend /= (blend.x + blend.y + blend.z);
    
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}

// Sample texture with custom filtering
vec4 sampleTextureFiltered(sampler2D tex, vec2 uv, vec2 texelSize, float filterRadius) {
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    
    // Sample in a 5x5 area around the center
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 offset = vec2(float(x), float(y)) * filterRadius * texelSize;
            float weight = exp(-(x*x + y*y) / (2.0 * filterRadius * filterRadius));
            result += texture2D(tex, uv + offset) * weight;
            totalWeight += weight;
        }
    }
    
    return result / totalWeight;
}

// Sample texture with anisotropic filtering effect
vec4 sampleTextureAnisotropic(sampler2D tex, vec2 uv, vec2 dx, vec2 dy, float maxSamples) {
    vec4 color = vec4(0.0);
    float numSamples = min(max(length(dx) / length(dy), length(dy) / length(dx)), maxSamples);
    
    for(float i = 0.0; i < numSamples; i += 1.0) {
        float t = i / numSamples - 0.5;
        color += texture2D(tex, uv + dx * t);
    }
    
    return color / numSamples;
}
"""


def generate_normal_mapping_glsl():
    """Generate GLSL implementation for normal mapping."""
    return """// Normal mapping module
// Standardized normal mapping implementations

// Calculate tangent space matrix from world position and UV coordinates
mat3 calculateTangentSpace(vec3 worldPos, vec3 worldNormal, vec2 uv) {
    // Calculate tangent and bitangent
    vec3 dp1 = dFdx(worldPos);
    vec3 dp2 = dFdy(worldPos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    
    vec3 tangent = normalize(dp2 * duv1.x - dp1 * duv2.x);
    vec3 bitangent = normalize(dp1 * duv2.y - dp2 * duv1.y);
    
    // Create TBN matrix
    mat3 TBN = mat3(tangent, bitangent, worldNormal);
    return TBN;
}

// Sample normal map and convert from [0,1] to [-1,1] range
vec3 sampleNormalMap(sampler2D normalMap, vec2 uv) {
    vec3 tangentNormal = texture2D(normalMap, uv).xyz;
    tangentNormal = normalize(tangentNormal * 2.0 - 1.0); // Convert from [0,1] to [-1,1]
    return tangentNormal;
}

// Apply normal map in world space
vec3 applyNormalMap(sampler2D normalMap, vec2 uv, vec3 worldNormal, mat3 TBN) {
    vec3 tangentNormal = sampleNormalMap(normalMap, uv);
    vec3 worldNormalMapped = normalize(TBN * tangentNormal);
    return worldNormalMapped;
}

// Perturb normal with normal map in fragment shader
vec3 perturbNormal(vec3 pos, vec3 normal, vec2 uv, sampler2D normalMap) {
    mat3 TBN = calculateTangentSpace(pos, normal, uv);
    vec3 newNormal = applyNormalMap(normalMap, uv, normal, TBN);
    return newNormal;
}

// Parallax mapping effect
vec2 parallaxMapping(sampler2D heightMap, vec2 uv, vec3 viewDir) {
    float height = texture2D(heightMap, uv).r;
    vec3 viewDirTangent = normalize(viewDir);
    vec2 p = viewDirTangent.xy / viewDirTangent.z * (height * 0.02);
    return uv - p;
}

// Steep parallax mapping with multiple samples
vec2 steepParallaxMapping(sampler2D heightMap, vec2 uv, vec3 viewDir, int numLayers) {
    float layerDepth = 1.0 / float(numLayers);
    float currentLayerDepth = 0.0;
    
    vec2 deltaUV = viewDir.xy * 0.02 / (viewDir.z * float(numLayers));
    
    vec2 currentUV = uv;
    float currentDepthValue = texture2D(heightMap, currentUV).r;
    
    while(currentLayerDepth < currentDepthValue) {
        currentUV -= deltaUV;
        currentDepthValue = texture2D(heightMap, currentUV).r;
        currentLayerDepth += layerDepth;
    }
    
    return currentUV;
}
"""


def generate_procedural_textures_glsl():
    """Generate GLSL implementation for procedural textures."""
    return """// Procedural textures module
// Standardized procedural texture implementations

// Checkerboard pattern
float checkerPattern(vec2 uv, float scale) {
    uv *= scale;
    return mod(floor(uv.x) + floor(uv.y), 2.0);
}

// Stripes pattern
float stripesPattern(vec2 uv, float scale, float width) {
    return mod(uv.x * scale, 1.0) < width ? 1.0 : 0.0;
}

// Radial pattern
float radialPattern(vec2 uv, float rings) {
    uv -= 0.5;
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    return mod(angle * rings / (2.0 * 3.14159), 1.0);
}

// Diamond pattern
float diamondPattern(vec2 uv, float scale) {
    uv = fract(uv * scale) - 0.5;
    return 1.0 - abs(uv.x) - abs(uv.y);
}

// Hexagonal pattern
vec2 hexagonalPattern(vec2 uv, float scale) {
    uv *= scale;
    float hexSize = 1.0;
    
    float q = uv.x * 0.57735 * 2.0;
    float r = (uv.y - uv.x * 0.5) / 0.57735;
    
    float x = q;
    float z = r;
    float y = -x - z;
    
    // Round to nearest hex
    float rx = floor(x + 0.5);
    float ry = floor(y + 0.5);
    float rz = floor(z + 0.5);
    
    float x_diff = abs(rx - x);
    float y_diff = abs(ry - y);
    float z_diff = abs(rz - z);
    
    if (x_diff > y_diff && x_diff > z_diff) {
        rx = -ry - rz;
    } else if (y_diff > z_diff) {
        ry = -rx - rz;
    } else {
        rz = -rx - ry;
    }
    
    return vec2(rx, ry);
}

// Generate procedural texture based on pattern
vec3 proceduralTexture(vec2 uv, float patternType) {
    uv *= 10.0; // Scale for visibility
    
    if (patternType < 1.0) {
        // Checkerboard
        float checker = checkerPattern(uv, 4.0);
        return vec3(checker);
    } else if (patternType < 2.0) {
        // Stripes
        float stripes = stripesPattern(uv, 8.0, 0.3);
        return vec3(stripes);
    } else if (patternType < 3.0) {
        // Radial
        float radial = radialPattern(uv, 6.0);
        return vec3(radial);
    } else if (patternType < 4.0) {
        // Diamond
        float diamond = diamondPattern(uv, 4.0);
        diamond = max(diamond, 0.0);
        return vec3(diamond);
    } else {
        // Hexagonal
        vec2 hex = hexagonalPattern(uv, 4.0);
        float hexPattern = fract(length(hex));
        return vec3(hexPattern);
    }
}
"""


def generate_texture_blending_glsl():
    """Generate GLSL implementation for texture blending."""
    return """// Texture blending module
// Standardized texture blending implementations

// Alpha blending between two textures
vec4 alphaBlend(vec4 base, vec4 blend, float alpha) {
    return base * (1.0 - alpha) + blend * alpha;
}

// Multiply blending
vec4 multiplyBlend(vec4 base, vec4 blend) {
    return base * blend;
}

// Additive blending
vec4 additiveBlend(vec4 base, vec4 blend) {
    return min(base + blend, vec4(1.0));
}

// Screen blending
vec4 screenBlend(vec4 base, vec4 blend) {
    return vec4(1.0) - (vec4(1.0) - base) * (vec4(1.0) - blend);
}

// Overlay blending
vec4 overlayBlend(vec4 base, vec4 blend) {
    vec4 result;
    result.r = base.r < 0.5 ? 2.0 * base.r * blend.r : 1.0 - 2.0 * (1.0 - base.r) * (1.0 - blend.r);
    result.g = base.g < 0.5 ? 2.0 * base.g * blend.g : 1.0 - 2.0 * (1.0 - base.g) * (1.0 - blend.g);
    result.b = base.b < 0.5 ? 2.0 * base.b * blend.b : 1.0 - 2.0 * (1.0 - base.b) * (1.0 - blend.b);
    result.a = base.a;
    return result;
}

// Soft light blending
vec4 softLightBlend(vec4 base, vec4 blend) {
    vec4 result;
    result.r = blend.r < 0.5 ? 
        2.0 * base.r * blend.r + base.r * base.r * (1.0 - 2.0 * blend.r) : 
        2.0 * base.r * (1.0 - blend.r) + sqrt(base.r) * (2.0 * blend.r - 1.0);
    result.g = blend.g < 0.5 ? 
        2.0 * base.g * blend.g + base.g * base.g * (1.0 - 2.0 * blend.g) : 
        2.0 * base.g * (1.0 - blend.g) + sqrt(base.g) * (2.0 * blend.g - 1.0);
    result.b = blend.b < 0.5 ? 
        2.0 * base.b * blend.b + base.b * base.b * (1.0 - 2.0 * blend.b) : 
        2.0 * base.b * (1.0 - blend.b) + sqrt(base.b) * (2.0 * blend.b - 1.0);
    result.a = base.a;
    return result;
}

// Blend multiple textures based on a mask
vec4 blendMultipleTextures(sampler2D tex1, sampler2D tex2, sampler2D mask, vec2 uv) {
    vec4 color1 = texture2D(tex1, uv);
    vec4 color2 = texture2D(tex2, uv);
    float maskValue = texture2D(mask, uv).r;
    
    return mix(color1, color2, maskValue);
}

// Blend textures with triplanar
vec4 blendTriplanarTextures(sampler2D tex1, sampler2D tex2, vec3 worldPos, vec3 normal) {
    vec4 color1 = sampleTriplanar(tex1, worldPos, normal);
    vec4 color2 = sampleTriplanar(tex2, worldPos, normal);
    
    // Use normal to determine blend weights
    vec3 blendWeights = abs(normal);
    blendWeights = normalize(max(blendWeights, 0.00001));
    blendWeights /= (blendWeights.x + blendWeights.y + blendWeights.z);
    
    return color1 * blendWeights.x + color2 * blendWeights.y + mix(color1, color2, 0.5) * blendWeights.z;
}
"""


def generate_triplanar_mapping_glsl():
    """Generate GLSL implementation for triplanar mapping."""
    return """// Triplanar mapping module
// Standardized triplanar mapping implementations

// Basic triplanar mapping for a single texture
vec4 triplanarSample(sampler2D tex, vec3 worldPos, vec3 normal, float blendSharpness) {
    // Project world position onto each axis
    vec2 uvX = worldPos.yz;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, uvX);
    vec4 texY = texture2D(tex, uvY);
    vec4 texZ = texture2D(tex, uvZ);
    
    // Use normal to determine blending weights
    vec3 blend = pow(abs(normal), vec3(blendSharpness));
    blend = normalize(max(blend, 0.00001)); // Avoid division by zero
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}

// Triplanar mapping with separate normal and color textures
vec4 triplanarSampleTextured(sampler2D colorTex, sampler2D normalTex, vec3 worldPos, vec3 normal, float blendSharpness) {
    // Use the same triplanar approach for both textures
    vec2 uvX = worldPos.yz;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;
    
    // Sample color textures
    vec4 colX = texture2D(colorTex, uvX);
    vec4 colY = texture2D(colorTex, uvY);
    vec4 colZ = texture2D(colorTex, uvZ);
    
    // Sample normal textures
    vec3 normX = sampleNormalMap(normalTex, uvX);
    vec3 normY = sampleNormalMap(normalTex, uvY);
    vec3 normZ = sampleNormalMap(normalTex, uvZ);
    
    // Use normal to determine blending weights
    vec3 blend = pow(abs(normal), vec3(blendSharpness));
    blend = normalize(max(blend, 0.00001));
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    vec4 blendedColor = colX * blend.x + colY * blend.y + colZ * blend.z;
    
    return blendedColor;
}

// Triplanar mapping with scaling factor
vec4 triplanarSampleScaled(sampler2D tex, vec3 worldPos, vec3 normal, float scale, float blendSharpness) {
    // Project world position onto each axis with scaling
    vec2 uvX = worldPos.yz * scale;
    vec2 uvY = worldPos.xz * scale;
    vec2 uvZ = worldPos.xy * scale;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, uvX);
    vec4 texY = texture2D(tex, uvY);
    vec4 texZ = texture2D(tex, uvZ);
    
    // Use normal to determine blending weights
    vec3 blend = pow(abs(normal), vec3(blendSharpness));
    blend = normalize(max(blend, 0.00001)); // Avoid division by zero
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}

// Blended triplanar with smooth transitions
vec4 triplanarSmooth(sampler2D tex, vec3 worldPos, vec3 normal) {
    // Use the smooth version of the triplanar mapping
    vec2 uvX = worldPos.yz;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, uvX);
    vec4 texY = texture2D(tex, uvY);
    vec4 texZ = texture2D(tex, uvZ);
    
    // Use smooth blending with absolute normal values
    vec3 blend = abs(normal);
    blend = normalize(max(blend, 0.00001));
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}
"""


def main():
    # Find texturing/mapping shaders
    texture_shaders = find_texturing_mapping_shaders()
    
    # Extract shader codes for a subset (first 500) for efficiency
    shader_codes = []
    for filepath, shader_info in texture_shaders[:500]:  # Limit to first 500 for efficiency
        shader_code = extract_shader_code(filepath)
        shader_codes.append({
            'info': shader_info,
            'code': shader_code
        })
    
    # Analyze shader patterns
    analyzed_shaders, pattern_counts = analyze_texture_shaders()
    
    # Extract specific texturing functions
    modules = extract_texture_modules(analyzed_shaders)
    
    # Create standardized modules
    create_standardized_texture_modules()
    
    print("Texturing/mapping shader analysis and module extraction completed!")


if __name__ == "__main__":
    main()