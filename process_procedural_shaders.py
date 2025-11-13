#!/usr/bin/env python3
"""
Process animation/procedural shaders from JSON files to identify common patterns
and extract reusable modules.
"""

import json
import os
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def find_animation_procedural_shaders(json_dir='json'):
    """
    Find all JSON files that contain animation/procedural generation related tags.

    Args:
        json_dir (str): Directory containing JSON shader files

    Returns:
        list: List of tuples (filepath, shader_info) for animation/procedural shaders
    """
    print("Finding animation/procedural generation related shaders...")
    
    keywords = [
        'animation', 'animate', 'moving', 'motion', 'time', 'dynamic', 'sequence',
        'procedural', 'noise', 'fractal', 'generative', 'algorithmic', 'pattern', 'algorithm',
        'perlin', 'simplex', 'value', 'cellular', 'voronoi', 'fBm', 'turbulence',
        'waves', 'oscillation', 'sine', 'cosine', 'trigonometry', 'waveform',
        'particles', 'simulation', 'physics', 'motion', 'flow', 'fluid'
    ]
    
    proc_shaders = []
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} JSON files for animation/procedural tags...")
    
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
                
                # Check if this shader is animation/procedural related 
                is_proc_related = False
                
                # Check tags
                for tag in tags:
                    if any(keyword in tag for keyword in keywords):
                        is_proc_related = True
                        break
                
                # Check name
                if not is_proc_related:
                    for keyword in keywords:
                        if keyword in name:
                            is_proc_related = True
                            break
                
                # Check description
                if not is_proc_related:
                    for keyword in keywords:
                        if keyword in description:
                            is_proc_related = True
                            break
                
                if is_proc_related:
                    shader_info = {
                        'id': info.get('id', os.path.basename(filepath).replace('.json', '')),
                        'name': info.get('name', ''),
                        'tags': tags,
                        'username': info.get('username', ''),
                        'description': info.get('description', ''),
                        'filepath': filepath
                    }
                    proc_shaders.append((filepath, shader_info))
                    
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue

    print(f"Found {len(proc_shaders)} animation/procedural generation related shaders")
    return proc_shaders


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


def identify_procedural_patterns(shader_code):
    """
    Identify common procedural generation patterns in shader code.

    Args:
        shader_code (str): GLSL code to analyze

    Returns:
        dict: Dictionary of identified procedural patterns
    """
    patterns = {
        # Noise functions
        'perlin_noise': 'perlin' in shader_code.lower(),
        'simplex_noise': 'simplex' in shader_code.lower(),
        'value_noise': 'value' in shader_code.lower() and 'noise' in shader_code.lower(),
        'cellular_noise': 'cellular' in shader_code.lower() or 'voronoi' in shader_code.lower(),
        'fractional_brownian_motion': 'fbm' in shader_code.lower() or 'fractal' in shader_code.lower() or ('brownian' in shader_code.lower() and 'motion' in shader_code.lower()),
        'noise_functions': 'noise' in shader_code.lower(),
        
        # Animation patterns
        'time_based_animation': 'time' in shader_code.lower() or 'itime' in shader_code.lower() or 'i_time' in shader_code.lower(),
        'oscillation': 'sin' in shader_code.lower() or 'cos' in shader_code.lower() or 'oscill' in shader_code.lower(),
        'wave_functions': 'wave' in shader_code.lower(),
        
        # Procedural patterns
        'fractals': 'fractal' in shader_code.lower() or 'mandelbrot' in shader_code.lower() or 'julia' in shader_code.lower() or 'burning' in shader_code.lower() or 'sierpinski' in shader_code.lower(),
        'procedural_textures': 'procedural' in shader_code.lower() and 'texture' in shader_code.lower(),
        'pattern_generation': 'pattern' in shader_code.lower(),
        'tiling_patterns': 'tile' in shader_code.lower() or 'repeat' in shader_code.lower(),
        
        # Mathematical functions
        'trigonometric_functions': 'sin' in shader_code.lower() or 'cos' in shader_code.lower() or 'tan' in shader_code.lower(),
        'mathematical_patterns': 'math' in shader_code.lower(),
        
        # Flow and simulation
        'flow_fields': 'flow' in shader_code.lower() and 'field' in shader_code.lower(),
        'particle_systems': 'particle' in shader_code.lower(),
        'physics_simulation': 'physics' in shader_code.lower() or 'simulate' in shader_code.lower(),
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
        pattern (str): Base pattern to match (like 'noise', 'animate', etc.)
    
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


def analyze_procedural_shaders():
    """
    Main function to analyze animation/procedural shaders.
    """
    print("Analyzing animation/procedural shaders...")
    
    proc_shaders = find_animation_procedural_shaders()
    
    # Store shader codes and identified patterns
    shader_codes = []
    all_patterns = []
    pattern_counts = Counter()
    
    print("\nAnalyzing procedural patterns in shaders...")
    for i, (filepath, shader_info) in enumerate(proc_shaders):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(proc_shaders)} procedural shaders...")
        
        shader_code = extract_shader_code(filepath)
        patterns = identify_procedural_patterns(shader_code)
        
        shader_codes.append({
            'info': shader_info,
            'code': shader_code,
            'patterns': patterns
        })
        
        all_patterns.append(patterns)
        
        # Update pattern counts
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    print(f"\nAnalysis complete! Found {len(proc_shaders)} procedural shaders.")
    
    # Print pattern distribution
    print(f"\nCommon procedural patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count} shaders")
    
    # Save analysis results
    save_procedural_analysis(shader_codes, pattern_counts)
    
    return shader_codes, pattern_counts


def save_procedural_analysis(shader_codes, pattern_counts):
    """
    Save the procedural analysis results to files.
    """
    os.makedirs('analysis/procedural', exist_ok=True)
    
    # Save pattern statistics
    with open('analysis/procedural/pattern_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Procedural Pattern Statistics\n")
        f.write("=" * 50 + "\n")
        for pattern, count in pattern_counts.most_common():
            f.write(f"{pattern.replace('_', ' ').title()}: {count}\n")
    
    # Save detailed shader analysis
    with open('analysis/procedural/shader_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Procedural Shader Analysis\n")
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
    
    print("Procedural shader analysis saved to analysis/procedural/ directory")


def extract_procedural_modules(shader_codes):
    """
    Extract reusable procedural modules from analyzed shaders.

    Args:
        shader_codes (list): List of shader data from analysis

    Returns:
        dict: Dictionary of extracted procedural modules
    """
    print("\nExtracting reusable procedural modules...")
    
    modules = {
        'noise': set(),
        'animation': set(),
        'fractals': set(),
        'patterns': set(),
        'mathematical': set(),
        'simulation': set()
    }
    
    total_processed = 0
    
    for shader_data in shader_codes:
        code = shader_data['code']
        
        # Extract noise functions
        noise_funcs = extract_complete_functions(code, 'noise')
        noise_funcs += extract_complete_functions(code, 'perlin')
        noise_funcs += extract_complete_functions(code, 'simplex')
        noise_funcs += extract_complete_functions(code, 'voronoi')
        modules['noise'].update(noise_funcs)
        
        # Extract animation functions
        anim_funcs = extract_complete_functions(code, 'animate')
        anim_funcs += extract_complete_functions(code, 'time')
        anim_funcs += extract_complete_functions(code, 'oscill')
        modules['animation'].update(anim_funcs)
        
        # Extract fractal functions
        fractal_funcs = extract_complete_functions(code, 'fractal')
        fractal_funcs += extract_complete_functions(code, 'mandelbrot')
        fractal_funcs += extract_complete_functions(code, 'julia')
        modules['fractals'].update(fractal_funcs)
        
        # Extract pattern functions
        pattern_funcs = extract_complete_functions(code, 'pattern')
        pattern_funcs += extract_complete_functions(code, 'tile')
        modules['patterns'].update(pattern_funcs)
        
        # Extract mathematical functions
        math_funcs = extract_complete_functions(code, 'math')
        math_funcs += extract_complete_functions(code, 'trig')
        modules['mathematical'].update(math_funcs)
        
        # Extract simulation functions
        sim_funcs = extract_complete_functions(code, 'physics')
        sim_funcs += extract_complete_functions(code, 'simulate')
        sim_funcs += extract_complete_functions(code, 'particle')
        modules['simulation'].update(sim_funcs)
        
        total_processed += 1
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(shader_codes)} shaders...")
    
    print(f"Extraction complete! Found:")
    for module_type, funcs in modules.items():
        print(f"  {module_type}: {len(funcs)} functions")
    
    # Save modules
    save_procedural_modules(modules)
    
    return modules


def save_procedural_modules(modules):
    """
    Save extracted procedural modules to files.
    """
    os.makedirs('modules/procedural', exist_ok=True)
    
    for module_type, func_list in modules.items():
        if func_list:  # Only save if there are modules of this type
            with open(f'modules/procedural/{module_type}_functions.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} Procedural Functions\n")
                f.write("// Automatically extracted from procedural-related shaders\n\n")
                
                for i, func in enumerate(func_list, 1):
                    f.write(f"// Function {i}\n")
                    f.write(func)
                    f.write("\n\n")
    
    print("Procedural modules saved to modules/procedural/ directory")


def create_standardized_procedural_modules():
    """
    Create standardized procedural modules based on patterns found.
    """
    print("Creating standardized procedural modules...")
    
    # Define standardized module templates with actual GLSL implementations
    standardized_modules = {
        'noise_functions.glsl': generate_noise_functions_glsl(),
        'animation_utils.glsl': generate_animation_utils_glsl(),
        'fractal_generation.glsl': generate_fractal_generation_glsl(),
        'procedural_patterns.glsl': generate_procedural_patterns_glsl(),
        'mathematical_functions.glsl': generate_mathematical_functions_glsl(),
        'simulation_kernels.glsl': generate_simulation_kernels_glsl()
    }
    
    os.makedirs('modules/procedural/standardized', exist_ok=True)
    
    # Create standardized modules
    for filename, code in standardized_modules.items():
        filepath = f'modules/procedural/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Created {len(standardized_modules)} standardized procedural modules")


def generate_noise_functions_glsl():
    """Generate GLSL implementation for noise functions."""
    return """// Noise functions module
// Standardized noise function implementations

// Classic Perlin noise by Ken Perlin
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// 2D Perlin noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Four corners in 2D of a tile
    vec2 I = vec2(0.0, 1.0);
    float a = rand(i + I.yy);
    float b = rand(i + I.xy);
    float c = rand(i + I.yx);
    float d = rand(i + I.xx);
    
    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    // Mix 4 coorners percentages
    return mix(a, b, u.x) + 
          (c - a)* u.y * (1.0 - u.x) + 
          (d - b) * u.x * u.y;
}

// Fractional Brownian Motion (fBm)
float fbm(vec2 p, int octaves, float amp, float freq) {
    float total = 0.0;
    float amplitude = amp;
    float frequency = freq;
    
    for(int i = 0; i < octaves; i++) {
        total += noise(p * frequency) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return total;
}

// Turbulence function (absolute value of fBm)
float turbulence(vec2 p, int octaves, float amp, float freq) {
    float total = 0.0;
    float amplitude = amp;
    float frequency = freq;
    
    for(int i = 0; i < octaves; i++) {
        total += abs(noise(p * frequency) * amplitude);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return total;
}
"""


def generate_animation_utils_glsl():
    """Generate GLSL implementation for animation utilities."""
    return """// Animation utilities module
// Standardized animation function implementations

// Linear interpolation with time
float linear(float start, float end, float time, float duration) {
    float t = clamp(time / duration, 0.0, 1.0);
    return mix(start, end, t);
}

// Smoothstep interpolation
float smoothStep(float start, float end, float time, float duration) {
    float t = clamp(time / duration, 0.0, 1.0);
    return mix(start, end, smoothstep(0.0, 1.0, t));
}

// Sine wave oscillation
float sineWave(float frequency, float amplitude, float time, float offset) {
    return sin(time * frequency + offset) * amplitude;
}

// Square wave oscillation
float squareWave(float frequency, float amplitude, float time) {
    return sign(sin(time * frequency)) * amplitude;
}

// Triangle wave oscillation
float triangleWave(float frequency, float amplitude, float time) {
    return abs(fract(time * frequency + 0.25) * 2.0 - 1.0) * 2.0 * amplitude - amplitude;
}

// Sawtooth wave oscillation
float sawtoothWave(float frequency, float amplitude, float time) {
    return (fract(time * frequency) * 2.0 - 1.0) * amplitude;
}

// Ping-pong oscillation (smooth back and forth)
float pingPong(float minVal, float maxVal, float time, float duration) {
    float t = (time / duration);
    float range = maxVal - minVal;
    float cycle = 2.0 * range;
    float position = fract(t) * cycle;
    return position > range ? (2.0 * range - position) + minVal : position + minVal;
}
"""


def generate_fractal_generation_glsl():
    """Generate GLSL implementation for fractal generation."""
    return """// Fractal generation module
// Standardized fractal function implementations

// Mandelbrot set calculation
float mandelbrot(vec2 c, int maxIterations) {
    vec2 z = vec2(0.0);
    float iterations = 0.0;
    
    for(int i = 0; i < maxIterations; i++) {
        if(dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }
    
    // Smooth coloring
    if(iterations < float(maxIterations)) {
        float log_zn = log(dot(z, z)) / 2.0;
        float nu = log(log_zn / log(2.0)) / log(2.0);
        iterations = iterations + 1.0 - nu;
    }
    
    return iterations / float(maxIterations);
}

// Julia set calculation
float julia(vec2 z, vec2 c, int maxIterations) {
    float iterations = 0.0;
    
    for(int i = 0; i < maxIterations; i++) {
        if(dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }
    
    return iterations / float(maxIterations);
}

// Burning Ship fractal
float burningShip(vec2 c, int maxIterations) {
    vec2 z = vec2(0.0);
    float iterations = 0.0;
    
    for(int i = 0; i < maxIterations; i++) {
        if(dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, abs(2.0 * z.x * z.y)) + abs(c);
        iterations++;
    }
    
    return iterations / float(maxIterations);
}
"""


def generate_procedural_patterns_glsl():
    """Generate GLSL implementation for procedural patterns."""
    return """// Procedural patterns module
// Standardized procedural pattern implementations

// Checkerboard pattern
float checker(vec2 p) {
    vec2 i = floor(p);
    return mod(i.x + i.y, 2.0);
}

// Stripes pattern
float stripes(vec2 p) {
    return abs(fract(p.x) - 0.5);
}

// Radial pattern
float radial(vec2 p) {
    return length(p);
}

// Angle pattern
float angle(vec2 p) {
    return atan(p.y, p.x);
}

// Hexagonal tiling pattern
float hexTile(vec2 p) {
    vec2 q = vec2(p.x * 1.1547, p.y + p.x * 0.57735);
    vec2 i = floor(q);
    vec2 f = fract(q);
    
    if(f.y < 1.0 - f.x) {
        return dot(i, vec2(1.0, 0.0));
    } else {
        return dot(i + vec2(1.0, 1.0), vec2(1.0, 0.0));
    }
}

// Voronoi pattern (simplified)
float voronoi(vec2 p) {
    vec2 g = floor(p);
    vec2 f = fract(p);
    
    float distance = 1.0;
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash2(g + neighbor) * 0.5 + 0.25; // Assume hash2 function exists
            vec2 diff = neighbor + point - f;
            float dist = length(diff);
            distance = min(distance, dist);
        }
    }
    
    return distance;
}
"""


def generate_mathematical_functions_glsl():
    """Generate GLSL implementation for mathematical functions."""
    return """// Mathematical functions module
// Standardized mathematical function implementations

// Smooth minimum function
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
    return mix(a, b, h) - k * h * (1.0 - h);
}

// Smooth maximum function
float smax(float a, float b, float k) {
    return -smin(-a, -b, k);
}

// Power function with smooth transition
float smoothPow(float base, float exponent, float smoothness) {
    return pow(base, exponent + smoothness * sin(exponent));
}

// Smooth absolute value
float smoothAbs(float x, float k) {
    return k * log(exp(x / k) + exp(-x / k));
}

// Sigmoid function
float sigmoid(float x, float sharpness) {
    return 1.0 / (1.0 + exp(-x * sharpness));
}

// Smooth pulse function
float smoothPulse(float edge0, float edge1, float x) {
    return smoothstep(edge0, edge0 + (edge1 - edge0) * 0.1, x) * 
           (1.0 - smoothstep(edge1 - (edge1 - edge0) * 0.1, edge1, x));
}
"""


def generate_simulation_kernels_glsl():
    """Generate GLSL implementation for simulation kernels."""
    return """// Simulation kernels module
// Standardized simulation function implementations

// Basic particle update
vec2 particleUpdate(vec2 position, vec2 velocity, float deltaTime) {
    return position + velocity * deltaTime;
}

// Simple Euler integration
vec2 eulerIntegration(vec2 position, vec2 velocity, vec2 acceleration, float deltaTime) {
    velocity += acceleration * deltaTime;
    position += velocity * deltaTime;
    return position;
}

// Velocity damping
vec2 applyDamping(vec2 velocity, float dampingFactor) {
    return velocity * (1.0 - dampingFactor);
}

// Constrain to bounds
vec2 constrainToBounds(vec2 position, vec2 minBounds, vec2 maxBounds) {
    return clamp(position, minBounds, maxBounds);
}

// Distance constraint for springs
vec2 springConstraint(vec2 posA, vec2 posB, float restLength) {
    vec2 delta = posB - posA;
    float distance = length(delta);
    if(distance > 0.0) {
        float diff = (distance - restLength) / distance;
        vec2 offset = delta * diff * 0.5;
        return offset;
    }
    return vec2(0.0);
}

// Simple collision detection with response
vec2 collisionResponse(vec2 position, vec2 size, vec2 minBounds, vec2 maxBounds, vec2 velocity) {
    vec2 newPos = position;
    
    if(position.x - size.x < minBounds.x) {
        newPos.x = minBounds.x + size.x;
        velocity.x = -velocity.x * 0.8; // 80% bounce
    } else if(position.x + size.x > maxBounds.x) {
        newPos.x = maxBounds.x - size.x;
        velocity.x = -velocity.x * 0.8;
    }
    
    if(position.y - size.y < minBounds.y) {
        newPos.y = minBounds.y + size.y;
        velocity.y = -velocity.y * 0.8;
    } else if(position.y + size.y > maxBounds.y) {
        newPos.y = maxBounds.y - size.y;
        velocity.y = -velocity.y * 0.8;
    }
    
    return newPos;
}
"""


def main():
    # Find animation/procedural shaders
    proc_shaders = find_animation_procedural_shaders()
    
    # Extract shader codes for a subset (first 500) for efficiency
    shader_codes = []
    for filepath, shader_info in proc_shaders[:500]:  # Limit to first 500 for efficiency
        shader_code = extract_shader_code(filepath)
        shader_codes.append({
            'info': shader_info,
            'code': shader_code
        })
    
    # Analyze shader patterns
    analyzed_shaders, pattern_counts = analyze_procedural_shaders()
    
    # Extract specific procedural functions
    modules = extract_procedural_modules(analyzed_shaders)
    
    # Create standardized modules
    create_standardized_procedural_modules()
    
    print("Animation/procedural shader analysis and module extraction completed!")


if __name__ == "__main__":
    main()