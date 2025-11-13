#!/usr/bin/env python3
"""
Process raymarching/raytracing shaders from JSON files to identify common patterns
and extract reusable modules.
"""

import json
import os
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def find_raymarching_raytracing_shaders(json_dir='json'):
    """
    Find all JSON files that contain raymarching/raytracing related tags.

    Args:
        json_dir (str): Directory containing JSON shader files

    Returns:
        list: List of tuples (filepath, shader_info) for raymarching/raytracing shaders
    """
    print("Finding raymarching/raytracing related shaders...")
    
    keywords = [
        'raymarching', 'ray', 'raymarch', 'sdf', 'distance', 'field', 'raytracing',
        'marching', 'sphere', 'box', 'torus', 'cylinder', 'plane', 'cone',
        'distance_function', 'scene', 'hit', 'cast', 'raycast', 'geometry',
        'signed', 'distance', 'field', 'map', 'sphere_trace', 'tracing',
        'path', 'path_tracing', 'global', 'illumination', 'gi',
        'ray_sphere', 'ray_box', 'ray_plane', 'intersection'
    ]
    
    ray_shaders = []
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} JSON files for raymarching/raytracing tags...")
    
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
                
                # Check if this shader is raymarching/raytracing related 
                is_ray_related = False
                
                # Check tags
                for tag in tags:
                    if any(keyword in tag for keyword in keywords):
                        is_ray_related = True
                        break
                
                # Check name
                if not is_ray_related:
                    for keyword in keywords:
                        if keyword in name:
                            is_ray_related = True
                            break
                
                # Check description
                if not is_ray_related:
                    for keyword in keywords:
                        if keyword in description:
                            is_ray_related = True
                            break
                
                if is_ray_related:
                    shader_info = {
                        'id': info.get('id', os.path.basename(filepath).replace('.json', '')),
                        'name': info.get('name', ''),
                        'tags': tags,
                        'username': info.get('username', ''),
                        'description': info.get('description', ''),
                        'filepath': filepath
                    }
                    ray_shaders.append((filepath, shader_info))
                    
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue

    print(f"Found {len(ray_shaders)} raymarching/raytracing related shaders")
    return ray_shaders


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


def identify_raymarching_patterns(shader_code):
    """
    Identify common raymarching/raytracing patterns in shader code.

    Args:
        shader_code (str): GLSL code to analyze

    Returns:
        dict: Dictionary of identified raymarching patterns
    """
    patterns = {
        # Core raymarching components
        'raymarching_loop': 'for' in shader_code and ('march' in shader_code.lower() or 'ray' in shader_code.lower()) and ('i <' in shader_code or 'step' in shader_code.lower() or 'max' in shader_code.lower()),
        'distance_function': 'map' in shader_code.lower() or 'dist' in shader_code.lower() or 'distance' in shader_code.lower() and 'func' in shader_code.lower(),
        'signed_distance_field': 'sdf' in shader_code.lower() or ('signed' in shader_code.lower() and 'distance' in shader_code.lower()),
        
        # Ray generation
        'ray_generation': 'ray' in shader_code.lower() and ('dir' in shader_code.lower() or 'origin' in shader_code.lower() or 'cast' in shader_code.lower()),
        'camera_ray': 'cam' in shader_code.lower() and 'ray' in shader_code.lower(),
        
        # Primitives
        'sphere_primitive': 'sphere' in shader_code.lower() and ('sd' in shader_code.lower() or 'dist' in shader_code.lower()),
        'box_primitive': 'box' in shader_code.lower() and ('sd' in shader_code.lower() or 'dist' in shader_code.lower()),
        'plane_primitive': 'plane' in shader_code.lower() and ('sd' in shader_code.lower() or 'dist' in shader_code.lower()),
        'torus_primitive': 'torus' in shader_code.lower() and ('sd' in shader_code.lower() or 'dist' in shader_code.lower()),
        'cylinder_primitive': 'cylinder' in shader_code.lower() and ('sd' in shader_code.lower() or 'dist' in shader_code.lower()),
        'cone_primitive': 'cone' in shader_code.lower() and ('sd' in shader_code.lower() or 'dist' in shader_code.lower()),
        
        # Ray operations
        'ray_intersection': 'intersect' in shader_code.lower(),
        'ray_normal': 'normal' in shader_code.lower() and ('ray' in shader_code.lower() or 'calc' in shader_code.lower()),
        'ray_reflection': 'reflect' in shader_code.lower(),
        'ray_refraction': 'refract' in shader_code.lower(),
        
        # Lighting in raymarching
        'raymarching_lighting': 'light' in shader_code.lower() and ('ray' in shader_code.lower() or 'march' in shader_code.lower()),
        'shadow_ray': 'shadow' in shader_code.lower() and ('ray' in shader_code.lower() or 'march' in shader_code.lower()),
        'ao_ray': 'ao' in shader_code.lower() and ('ray' in shader_code.lower() or 'march' in shader_code.lower()),
        
        # Raytracing
        'raytracing': 'trace' in shader_code.lower() and ('ray' in shader_code.lower() or ('cast' in shader_code.lower() and 'primary' in shader_code.lower())),
        'path_tracing': 'path' in shader_code.lower() and 'trace' in shader_code.lower(),
        'global_illumination': 'global' in shader_code.lower() and 'illumination' in shader_code.lower(),
        
        # Optimizations
        'adaptive_stepping': 'adaptive' in shader_code.lower() and ('step' in shader_code.lower() or 'march' in shader_code.lower()),
        'early_termination': 'break' in shader_code.lower() and ('hit' in shader_code.lower() or 'dist' in shader_code.lower() or 'epsilon' in shader_code.lower()),
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
        pattern (str): Base pattern to match (like 'map', 'ray', 'sdf', etc.)
    
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


def analyze_raymarching_shaders():
    """
    Main function to analyze raymarching/raytracing shaders.
    """
    print("Analyzing raymarching/raytracing shaders...")
    
    ray_shaders = find_raymarching_raytracing_shaders()
    
    # Store shader codes and identified patterns
    shader_codes = []
    all_patterns = []
    pattern_counts = Counter()
    
    print("\nAnalyzing raymarching patterns in shaders...")
    for i, (filepath, shader_info) in enumerate(ray_shaders):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(ray_shaders)} raymarching shaders...")
        
        shader_code = extract_shader_code(filepath)
        patterns = identify_raymarching_patterns(shader_code)
        
        shader_codes.append({
            'info': shader_info,
            'code': shader_code,
            'patterns': patterns
        })
        
        all_patterns.append(patterns)
        
        # Update pattern counts
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    print(f"\nAnalysis complete! Found {len(ray_shaders)} raymarching/raytracing shaders.")
    
    # Print pattern distribution
    print(f"\nCommon raymarching patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count} shaders")
    
    # Save analysis results
    save_raymarching_analysis(shader_codes, pattern_counts)
    
    return shader_codes, pattern_counts


def save_raymarching_analysis(shader_codes, pattern_counts):
    """
    Save the raymarching analysis results to files.
    """
    os.makedirs('analysis/raymarching', exist_ok=True)
    
    # Save pattern statistics
    with open('analysis/raymarching/pattern_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Raymarching Pattern Statistics\n")
        f.write("=" * 50 + "\n")
        for pattern, count in pattern_counts.most_common():
            f.write(f"{pattern.replace('_', ' ').title()}: {count}\n")
    
    # Save detailed shader analysis
    with open('analysis/raymarching/shader_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Raymarching Shader Analysis\n")
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
    
    print("Raymarching shader analysis saved to analysis/raymarching/ directory")


def extract_raymarching_modules(shader_codes):
    """
    Extract reusable raymarching modules from analyzed shaders.

    Args:
        shader_codes (list): List of shader data from analysis

    Returns:
        dict: Dictionary of extracted raymarching modules
    """
    print("\nExtracting reusable raymarching modules...")
    
    modules = {
        'distance_functions': set(),
        'ray_generations': set(),
        'primitives': set(),
        'ray_operations': set(),
        'lighting': set(),
        'optimizations': set()
    }
    
    total_processed = 0
    
    for shader_data in shader_codes:
        code = shader_data['code']
        
        # Extract distance functions (the core of raymarching)
        dist_funcs = extract_complete_functions(code, 'map')
        dist_funcs += extract_complete_functions(code, 'dist')
        dist_funcs += extract_complete_functions(code, 'sdf')
        dist_funcs += extract_complete_functions(code, 'scene')
        modules['distance_functions'].update(dist_funcs)
        
        # Extract ray generation functions
        ray_gen_funcs = extract_complete_functions(code, 'getRay')
        ray_gen_funcs += extract_complete_functions(code, 'rayDir')
        ray_gen_funcs += extract_complete_functions(code, 'camRay')
        modules['ray_generations'].update(ray_gen_funcs)
        
        # Extract primitive functions
        prim_funcs = extract_complete_functions(code, 'sphere')
        prim_funcs += extract_complete_functions(code, 'box')
        prim_funcs += extract_complete_functions(code, 'plane')
        prim_funcs += extract_complete_functions(code, 'torus')
        prim_funcs += extract_complete_functions(code, 'cylinder')
        prim_funcs += extract_complete_functions(code, 'cone')
        modules['primitives'].update(prim_funcs)
        
        # Extract ray operations
        ray_op_funcs = extract_complete_functions(code, 'ray')
        ray_op_funcs += extract_complete_functions(code, 'normal')
        ray_op_funcs += extract_complete_functions(code, 'reflect')
        ray_op_funcs += extract_complete_functions(code, 'refract')
        modules['ray_operations'].update(ray_op_funcs)
        
        # Extract lighting functions
        light_funcs = extract_complete_functions(code, 'light')
        light_funcs += extract_complete_functions(code, 'shade')
        light_funcs += extract_complete_functions(code, 'shadow')
        light_funcs += extract_complete_functions(code, 'ao')
        modules['lighting'].update(light_funcs)
        
        # Extract optimization functions
        opt_funcs = extract_complete_functions(code, 'step')
        opt_funcs += extract_complete_functions(code, 'march')
        modules['optimizations'].update(opt_funcs)
        
        total_processed += 1
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(shader_codes)} shaders...")
    
    print(f"Extraction complete! Found:")
    for module_type, funcs in modules.items():
        print(f"  {module_type}: {len(funcs)} functions")
    
    # Save modules
    save_raymarching_modules(modules)
    
    return modules


def save_raymarching_modules(modules):
    """
    Save extracted raymarching modules to files.
    """
    os.makedirs('modules/raymarching', exist_ok=True)
    
    for module_type, func_list in modules.items():
        if func_list:  # Only save if there are modules of this type
            with open(f'modules/raymarching/{module_type}_functions.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} Raymarching Functions\n")
                f.write("// Automatically extracted from raymarching/raytracing-related shaders\n\n")
                
                for i, func in enumerate(func_list, 1):
                    f.write(f"// Function {i}\n")
                    f.write(func)
                    f.write("\n\n")
    
    print("Raymarching modules saved to modules/raymarching/ directory")


def create_standardized_raymarching_modules():
    """
    Create standardized raymarching modules based on patterns found.
    """
    print("Creating standardized raymarching modules...")
    
    # Define standardized module templates with actual GLSL implementations
    standardized_modules = {
        'distance_functions.glsl': generate_distance_functions_glsl(),
        'ray_generators.glsl': generate_ray_generators_glsl(),
        'primitives.glsl': generate_primitives_glsl(),
        'ray_operations.glsl': generate_ray_operations_glsl(),
        'raymarching_core.glsl': generate_raymarching_core_glsl(),
        'lighting.glsl': generate_lighting_glsl()
    }
    
    os.makedirs('modules/raymarching/standardized', exist_ok=True)
    
    # Create standardized modules
    for filename, code in standardized_modules.items():
        filepath = f'modules/raymarching/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Created {len(standardized_modules)} standardized raymarching modules")


def generate_distance_functions_glsl():
    """Generate GLSL implementation for distance functions."""
    return """// Distance functions module
// Standardized distance function implementations for raymarching

// Sphere distance function
float sdSphere(vec3 p, float radius) {
    return length(p) - radius;
}

// Box distance function
float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

// Rounded box distance function
float sdRoundedBox(vec3 p, vec3 b, float r) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0)) - r;
}

// Torus distance function
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Cylinder distance function
float sdCylinder(vec3 p, vec2 h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Cone distance function
float sdCone(vec3 p, vec2 c) {
    // c must be normalized
    float q = length(p.xz);
    return dot(c, vec2(q, p.y));
}

// Plane distance function
float sdPlane(vec3 p, vec4 n) {
    // n must be normalized
    return dot(p, n.xyz) + n.w;
}

// Union operation for combining shapes
float opUnion(float d1, float d2) {
    return min(d1, d2);
}

// Subtraction operation
float opSubtraction(float d1, float d2) {
    return max(-d1, d2);
}

// Intersection operation
float opIntersection(float d1, float d2) {
    return max(d1, d2);
}

// Smooth union operation
float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}
"""


def generate_ray_generators_glsl():
    """Generate GLSL implementation for ray generation."""
    return """// Ray generation module
// Standardized ray generation functions for raymarching

// Generate a ray from camera position through a screen pixel
vec3 getRayDirection(vec2 uv, vec3 cameraPos, vec3 cameraTarget) {
    vec3 forward = normalize(cameraTarget - cameraPos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));
    
    vec3 rayDir = normalize(forward + uv.x * right + uv.y * up);
    return rayDir;
}

// Generate a ray with FOV consideration
vec3 getRayDirectionWithFOV(vec2 uv, vec3 rd, float fov) {
    rd = normalize(rd + fov * uv.x * vec3(1, 0, 0) + fov * uv.y * vec3(0, 1, 0));
    return rd;
}

// Generate primary ray with aspect ratio correction
vec3 generatePrimaryRay(vec2 fragCoord, vec2 resolution, vec3 cameraPos, vec3 cameraTarget) {
    vec2 uv = (fragCoord - 0.5 * resolution.xy) / resolution.y;
    return getRayDirection(uv, cameraPos, cameraTarget);
}
"""


def generate_primitives_glsl():
    """Generate GLSL implementation for raymarching primitives."""
    return """// Primitives module
// Standardized primitive shapes for raymarching

// Transform primitive by rotation
vec3 rotateX(vec3 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(1.0, 0.0, 0.0,
                0.0, c, -s,
                0.0, s, c) * p;
}

vec3 rotateY(vec3 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(c, 0.0, s,
                0.0, 1.0, 0.0,
                -s, 0.0, c) * p;
}

vec3 rotateZ(vec3 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(c, -s, 0.0,
                s, c, 0.0,
                0.0, 0.0, 1.0) * p;
}

// Transform primitive by translation
vec3 translate(vec3 p, vec3 t) {
    return p - t;
}

// Transform primitive by scaling
vec3 scale(vec3 p, float s) {
    return p / s;
}

// Repeat space for creating instances
vec3 repeat(vec3 p, vec3 c) {
    return mod(p, c) - 0.5 * c;
}

// Domain warping
vec3 warp(vec3 p) {
    p.x += sin(p.y * 0.5) * 0.5;
    p.y += cos(p.x * 0.5) * 0.5;
    p.z += sin(p.x * 0.3) * 0.3;
    return p;
}
"""


def generate_ray_operations_glsl():
    """Generate GLSL implementation for ray operations."""
    return """// Ray operations module
// Standardized ray operation functions for raymarching

// Calculate normal using gradient of distance field
vec3 calcNormal(vec3 p, float epsilon) {
    vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x
    ));
}

// Calculate normal with adaptive epsilon
vec3 calcNormalAdaptive(vec3 p) {
    float dx = map(p + vec3(0.001, 0.0, 0.0)).x - map(p - vec3(0.001, 0.0, 0.0)).x;
    float dy = map(p + vec3(0.0, 0.001, 0.0)).x - map(p - vec3(0.0, 0.001, 0.0)).x;
    float dz = map(p + vec3(0.0, 0.0, 0.001)).x - map(p - vec3(0.0, 0.0, 0.001)).x;
    return normalize(vec3(dx, dy, dz));
}

// Calculate ambient occlusion based on distance field
float calcAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for(int i = 0; i < 5; i++) {
        float h = 0.01 + 0.15 * float(i) / 4.0;
        float d = map(p + h * n).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

// Calculate soft shadows
float calcSoftShadow(vec3 ro, vec3 rd, float mint, float tmax, float k) {
    float res = 1.0;
    float t = mint;
    for(int i = 0; i < 32; i++) {
        float h = map(ro + rd * t).x;
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.10);
        if(res < 0.001 || t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}
"""


def generate_raymarching_core_glsl():
    """Generate GLSL implementation for core raymarching."""
    return """// Core raymarching module
// Standardized raymarching algorithm implementations

// Basic raymarching function
vec2 raymarch(vec3 ro, vec3 rd, float maxDist, int maxSteps) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled
    
    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        d = map(p).x;
        if(d < 0.001 || t > maxDist) break;
        t += d;
    }
    
    return vec2(t, d);
}

// Raymarching with adaptive step size
vec2 raymarchAdaptive(vec3 ro, vec3 rd, float maxDist, int maxSteps) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled
    float f = 1.0; // Adaptive factor
    
    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        d = map(p).x;
        if(d < 0.001 || t > maxDist) break;
        
        // Adaptive step size based on distance
        float adaptiveStep = d * f;
        t += max(0.01, adaptiveStep);
        
        // Reduce the factor as we get closer to surface
        f = 0.5 + 0.5 * min(1.0, d * 4.0);
    }
    
    return vec2(t, d);
}

// Raymarching with early termination optimization
vec2 raymarchOptimized(vec3 ro, vec3 rd, float maxDist, int maxSteps) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled
    
    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        d = map(p).x;
        
        // Early termination if we're very close to surface
        if(d < 0.0001) return vec2(t, d);
        
        // Stop if we've gone too far
        if(t > maxDist) return vec2(maxDist, d);
        
        t += d;
    }
    
    return vec2(t, d);
}
"""


def generate_lighting_glsl():
    """Generate GLSL implementation for raymarching lighting."""
    return """// Lighting module
// Standardized lighting functions for raymarching scenes

// Basic Phong lighting model for raymarching
vec3 phongLighting(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 lightColor, vec3 materialColor) {
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    
    vec3 ambient = 0.1 * materialColor;
    vec3 diffuse = diff * lightColor * materialColor;
    vec3 specular = spec * lightColor;
    
    return ambient + diffuse + specular;
}

// Calculate multiple light sources
vec3 multiLighting(vec3 pos, vec3 normal, vec3 viewDir, vec3 materialColor) {
    vec3 color = vec3(0.0);
    
    // Light 1
    vec3 lightPos1 = vec3(5.0, 5.0, 5.0);
    vec3 lightDir1 = normalize(lightPos1 - pos);
    vec3 lightColor1 = vec3(1.0, 0.9, 0.8);
    color += phongLighting(normal, viewDir, lightDir1, lightColor1, materialColor);
    
    // Light 2
    vec3 lightPos2 = vec3(-5.0, 3.0, -2.0);
    vec3 lightDir2 = normalize(lightPos2 - pos);
    vec3 lightColor2 = vec3(0.2, 0.5, 1.0);
    color += phongLighting(normal, viewDir, lightDir2, lightColor2, materialColor);
    
    return color;
}

// Ambient occlusion for raymarching scenes
float ambientOcclusion(vec3 pos, vec3 normal) {
    float occ = 0.0;
    float sca = 1.0;
    for(int i = 0; i < 5; i++) {
        float h = 0.01 + 0.15 * float(i) / 4.0;
        float d = map(pos + h * normal).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

// Fresnel effect for raymarching
float fresnel(vec3 viewDir, vec3 normal, float power) {
    return pow(1.0 - clamp(dot(normal, viewDir), 0.0, 1.0), power);
}
"""


def main():
    # Find raymarching/raytracing shaders
    ray_shaders = find_raymarching_raytracing_shaders()
    
    # Extract shader codes for a subset (first 500) for efficiency
    shader_codes = []
    for filepath, shader_info in ray_shaders[:500]:  # Limit to first 500 for efficiency
        shader_code = extract_shader_code(filepath)
        shader_codes.append({
            'info': shader_info,
            'code': shader_code
        })
    
    # Analyze shader patterns
    analyzed_shaders, pattern_counts = analyze_raymarching_shaders()
    
    # Extract specific raymarching functions
    modules = extract_raymarching_modules(analyzed_shaders)
    
    # Create standardized modules
    create_standardized_raymarching_modules()
    
    print("Raymarching/raytracing shader analysis and module extraction completed!")


if __name__ == "__main__":
    main()