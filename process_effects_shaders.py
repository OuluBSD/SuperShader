#!/usr/bin/env python3
"""
Process effects/post-processing shaders from JSON files to identify common patterns
and extract reusable effect modules.
"""

import json
import os
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def find_effect_shaders(json_dir='json'):
    """
    Find all JSON files that contain effects/post-processing related tags.

    Args:
        json_dir (str): Directory containing JSON shader files

    Returns:
        list: List of tuples (filepath, shader_info) for effect-related shaders
    """
    print("Finding effects/post-processing related shaders...")
    
    effect_keywords = [
        'effect', 'effects', 'post', 'filter', 'blur', 'glow', 'fx', 
        'distortion', 'vignette', 'glitch', 'processing', 'composite',
        'color', 'tone', 'contrast', 'saturation', 'brightness', 'gamma',
        'bloom', 'hdr', 'ssao', 'ssr', 'depth', 'dof', 'motion', 'anti-aliasing'
    ]
    
    effect_shaders = []
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} JSON files for effects-related tags...")
    
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
                
                # Check if this shader is effects-related based on tags, name, or description
                is_effect_related = False
                
                # Check tags
                for tag in tags:
                    if any(keyword in tag for keyword in effect_keywords):
                        is_effect_related = True
                        break
                
                # Check name
                if not is_effect_related:
                    for keyword in effect_keywords:
                        if keyword in name:
                            is_effect_related = True
                            break
                
                # Check description
                if not is_effect_related:
                    for keyword in effect_keywords:
                        if keyword in description:
                            is_effect_related = True
                            break
                
                if is_effect_related:
                    shader_info = {
                        'id': info.get('id', os.path.basename(filepath).replace('.json', '')),
                        'name': info.get('name', ''),
                        'tags': tags,
                        'username': info.get('username', ''),
                        'description': info.get('description', ''),
                        'filepath': filepath
                    }
                    effect_shaders.append((filepath, shader_info))
                    
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue

    print(f"Found {len(effect_shaders)} effects-related shaders")
    return effect_shaders


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


def identify_effect_patterns(shader_code):
    """
    Identify common effect patterns in shader code.

    Args:
        shader_code (str): GLSL code to analyze

    Returns:
        dict: Dictionary of identified effect patterns
    """
    patterns = {
        # Color processing effects
        'color_adjustments': 'color' in shader_code.lower() and ('adjust' in shader_code.lower() or 'correct' in shader_code.lower() or 'process' in shader_code.lower()),
        'gamma_correction': 'gamma' in shader_code.lower(),
        'brightness_contrast': 'brightness' in shader_code.lower() or 'contrast' in shader_code.lower(),
        'hue_saturation': 'hue' in shader_code.lower() or 'saturation' in shader_code.lower(),
        'color_grading': 'color' in shader_code.lower() and ('grade' in shader_code.lower() or 'lut' in shader_code.lower()),
        
        # Blur effects
        'gaussian_blur': 'gaussian' in shader_code.lower() and 'blur' in shader_code.lower(),
        'box_blur': 'box' in shader_code.lower() and 'blur' in shader_code.lower(),
        'bilateral_blur': 'bilateral' in shader_code.lower() and 'blur' in shader_code.lower(),
        'motion_blur': 'motion' in shader_code.lower() and 'blur' in shader_code.lower(),
        
        # Distortion effects
        'distortion': 'distort' in shader_code.lower(),
        'warping': 'warp' in shader_code.lower(),
        'displacement': 'displace' in shader_code.lower(),
        
        # Visual effects
        'bloom': 'bloom' in shader_code.lower(),
        'glow': 'glow' in shader_code.lower(),
        'vignette': 'vignette' in shader_code.lower(),
        'chromatic_aberration': 'chromatic' in shader_code.lower() and 'aberration' in shader_code.lower(),
        'lens_flare': 'lens' in shader_code.lower() and 'flare' in shader_code.lower(),
        'film_grain': 'film' in shader_code.lower() and 'grain' in shader_code.lower(),
        
        # Post-processing effects
        'fxaa': 'fxaa' in shader_code.lower(),
        'taa': 'taa' in shader_code.lower() or ('temporal' in shader_code.lower() and 'anti' in shader_code.lower()),
        'ssao': 'ssao' in shader_code.lower() or ('screen' in shader_code.lower() and 'space' in shader_code.lower() and 'ambient' in shader_code.lower()),
        'ssr': 'ssr' in shader_code.lower() or ('screen' in shader_code.lower() and 'space' in shader_code.lower() and 'reflection' in shader_code.lower()),
        'dof': 'dof' in shader_code.lower() or ('depth' in shader_code.lower() and 'field' in shader_code.lower()),
        
        # Filters and compositing
        'filter': 'filter' in shader_code.lower(),
        'composite': 'composite' in shader_code.lower() or 'compositing' in shader_code.lower(),
        'threshold': 'threshold' in shader_code.lower(),
        
        # Edge detection
        'edge_detection': 'edge' in shader_code.lower() and ('detect' in shader_code.lower() or 'sobel' in shader_code.lower() or 'prewitt' in shader_code.lower()),
        
        # Special effects
        'glitch': 'glitch' in shader_code.lower(),
        'pixelation': 'pixel' in shader_code.lower() and 'size' in shader_code.lower(),
        'scanlines': 'scanline' in shader_code.lower(),
        'glitch_artifacts': 'glitch' in shader_code.lower() and ('shift' in shader_code.lower() or 'noise' in shader_code.lower()),
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
        pattern (str): Base pattern to match (like 'blur', 'filter', etc.)
    
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


def analyze_effect_shaders():
    """
    Main function to analyze effect shaders.
    """
    print("Analyzing effects shaders...")
    
    effect_shaders = find_effect_shaders()
    
    # Store shader codes and identified patterns
    shader_codes = []
    all_patterns = []
    pattern_counts = Counter()
    
    print("\nAnalyzing effect patterns in shaders...")
    for i, (filepath, shader_info) in enumerate(effect_shaders):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(effect_shaders)} effect shaders...")
        
        shader_code = extract_shader_code(filepath)
        patterns = identify_effect_patterns(shader_code)
        
        shader_codes.append({
            'info': shader_info,
            'code': shader_code,
            'patterns': patterns
        })
        
        all_patterns.append(patterns)
        
        # Update pattern counts
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    print(f"\nAnalysis complete! Found {len(effect_shaders)} effect shaders.")
    
    # Print pattern distribution
    print(f"\nCommon effect patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count} shaders")
    
    # Save analysis results
    save_effect_analysis(shader_codes, pattern_counts)
    
    return shader_codes, pattern_counts


def save_effect_analysis(shader_codes, pattern_counts):
    """
    Save the effect analysis results to files.
    """
    os.makedirs('analysis/effects', exist_ok=True)
    
    # Save pattern statistics
    with open('analysis/effects/pattern_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Effect Pattern Statistics\n")
        f.write("=" * 50 + "\n")
        for pattern, count in pattern_counts.most_common():
            f.write(f"{pattern.replace('_', ' ').title()}: {count}\n")
    
    # Save detailed shader analysis
    with open('analysis/effects/shader_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Effect Shader Analysis\n")
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
    
    print("Effect shader analysis saved to analysis/effects/ directory")


def extract_effect_modules(shader_codes):
    """
    Extract reusable effect modules from analyzed shaders.

    Args:
        shader_codes (list): List of shader data from analysis

    Returns:
        dict: Dictionary of extracted effect modules
    """
    print("\nExtracting reusable effect modules...")
    
    modules = {
        'blur': set(),
        'color_adjustments': set(),
        'distortion': set(),
        'compositing': set(),
        'filters': set(),
        'post_processing': set(),
        'edge_detection': set()
    }
    
    total_processed = 0
    
    for shader_data in shader_codes:
        code = shader_data['code']
        
        # Extract blur functions
        blur_funcs = extract_complete_functions(code, 'blur')
        modules['blur'].update(blur_funcs)
        
        # Extract color adjustment functions
        color_funcs = extract_complete_functions(code, 'color')
        color_funcs += extract_complete_functions(code, 'adjust')
        modules['color_adjustments'].update(color_funcs)
        
        # Extract distortion functions
        distortion_funcs = extract_complete_functions(code, 'distort')
        distortion_funcs += extract_complete_functions(code, 'warp')
        modules['distortion'].update(distortion_funcs)
        
        # Extract compositing functions
        composite_funcs = extract_complete_functions(code, 'composite')
        modules['compositing'].update(composite_funcs)
        
        # Extract filter functions
        filter_funcs = extract_complete_functions(code, 'filter')
        modules['filters'].update(filter_funcs)
        
        # Extract post-processing functions
        pp_funcs = extract_complete_functions(code, 'post')
        pp_funcs += extract_complete_functions(code, 'process')
        modules['post_processing'].update(pp_funcs)
        
        # Extract edge detection functions
        edge_funcs = extract_complete_functions(code, 'edge')
        edge_funcs += extract_complete_functions(code, 'sobel')
        edge_funcs += extract_complete_functions(code, 'canny')
        modules['edge_detection'].update(edge_funcs)
        
        total_processed += 1
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(shader_codes)} shaders...")
    
    print(f"Extraction complete! Found:")
    for module_type, funcs in modules.items():
        print(f"  {module_type}: {len(funcs)} functions")
    
    # Save modules
    save_effect_modules(modules)
    
    return modules


def save_effect_modules(modules):
    """
    Save extracted effect modules to files.
    """
    os.makedirs('modules/effects', exist_ok=True)
    
    for module_type, func_list in modules.items():
        if func_list:  # Only save if there are modules of this type
            with open(f'modules/effects/{module_type}_functions.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} Effect Functions\n")
                f.write("// Automatically extracted from effect-related shaders\n\n")
                
                for i, func in enumerate(func_list, 1):
                    f.write(f"// Function {i}\n")
                    f.write(func)
                    f.write("\n\n")
    
    print("Effect modules saved to modules/effects/ directory")


def create_standardized_effect_modules():
    """
    Create standardized effect modules based on patterns found.
    """
    print("Creating standardized effect modules...")
    
    # Define standardized module templates with actual GLSL implementations
    standardized_modules = {
        'blur.glsl': generate_blur_glsl(),
        'color_adjustment.glsl': generate_color_adjustment_glsl(),
        'bloom_effect.glsl': generate_bloom_effect_glsl(),
        'composite.glsl': generate_composite_glsl(),
        'edge_detection.glsl': generate_edge_detection_glsl(),
        'vignette_effect.glsl': generate_vignette_effect_glsl()
    }
    
    os.makedirs('modules/effects/standardized', exist_ok=True)
    
    # Create standardized modules
    for filename, code in standardized_modules.items():
        filepath = f'modules/effects/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Created {len(standardized_modules)} standardized effect modules")


def generate_blur_glsl():
    """Generate GLSL implementation for blur effects."""
    return """// Blur effect module
// Standardized blur functions

// Gaussian blur implementation
vec4 GaussianBlur(sampler2D texture, vec2 texCoord, vec2 texSize, float radius) {
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    
    // Sample in a 5x5 area around the center
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 offset = vec2(float(x), float(y)) * radius / texSize;
            float weight = exp(-(x*x + y*y) / (2.0 * radius * radius));
            result += texture2D(texture, texCoord + offset) * weight;
            totalWeight += weight;
        }
    }
    
    return result / totalWeight;
}

// Simple box blur implementation
vec4 BoxBlur(sampler2D texture, vec2 texCoord, vec2 texSize, float radius) {
    vec4 result = vec4(0.0);
    int count = 0;
    
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 offset = vec2(float(x), float(y)) * radius / texSize;
            result += texture2D(texture, texCoord + offset);
            count++;
        }
    }
    
    return result / float(count);
}

// Motion blur in a specific direction
vec4 MotionBlur(sampler2D texture, vec2 texCoord, vec2 motionVector, int samples) {
    vec4 result = vec4(0.0);
    vec2 sampleStep = motionVector / float(max(samples, 1));
    
    for(int i = 0; i < samples; i++) {
        vec2 offset = texCoord + sampleStep * float(i) - motionVector * 0.5;
        result += texture2D(texture, offset);
    }
    
    return result / float(max(samples, 1));
}
"""


def generate_color_adjustment_glsl():
    """Generate GLSL implementation for color adjustments."""
    return """// Color adjustment module
// Standardized color adjustment functions

// Adjust brightness, contrast and saturation
vec3 AdjustColor(vec3 color, float brightness, float contrast, float saturation) {
    // Brightness
    color += brightness;
    
    // Contrast
    color = (color - 0.5) * contrast + 0.5;
    
    // Saturation
    vec3 grey = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
    color = mix(grey, color, saturation);
    
    return clamp(color, 0.0, 1.0);
}

// Apply gamma correction
vec3 ApplyGamma(vec3 color, float gamma) {
    return pow(color, vec3(1.0 / gamma));
}

// Adjust hue, saturation and lightness
vec3 HSLAdjust(vec3 color, float hueAdjust, float satAdjust, float lightAdjust) {
    // Convert RGB to HSL
    vec3 hsl = RGBToHSL(color);
    
    // Adjust HSL values
    hsl.x += hueAdjust;
    hsl.y = clamp(hsl.y * satAdjust, 0.0, 1.0);
    hsl.z = clamp(hsl.z + lightAdjust, 0.0, 1.0);
    
    // Convert back to RGB
    return HSLToRGB(hsl);
}

// Convert RGB to HSL
vec3 RGBToHSL(vec3 color) {
    float minVal = min(min(color.r, color.g), color.b);
    float maxVal = max(max(color.r, color.g), color.b);
    float delta = maxVal - minVal;
    
    vec3 hsl = vec3(0.0);
    hsl.z = (maxVal + minVal) / 2.0;
    
    if(delta == 0.0) {
        hsl.x = 0.0;
        hsl.y = 0.0;
    } else {
        if(hsl.z < 0.5)
            hsl.y = delta / (maxVal + minVal);
        else
            hsl.y = delta / (2.0 - maxVal - minVal);
        
        if(color.r == maxVal)
            hsl.x = (color.g - color.b) / delta;
        else if(color.g == maxVal)
            hsl.x = 2.0 + (color.b - color.r) / delta;
        else
            hsl.x = 4.0 + (color.r - color.g) / delta;
        
        hsl.x = hsl.x / 6.0;
        if(hsl.x < 0.0)
            hsl.x += 1.0;
    }
    
    return hsl;
}

// Convert HSL to RGB
vec3 HSLToRGB(vec3 hsl) {
    vec3 rgb;
    
    if(hsl.y == 0.0) {
        rgb = vec3(hsl.z);
    } else {
        float q = hsl.z < 0.5 ? 
            hsl.z * (1.0 + hsl.y) : 
            hsl.z + hsl.y - hsl.z * hsl.y;
        float p = 2.0 * hsl.z - q;
        
        rgb.r = HueToRGB(p, q, hsl.x + (1.0/3.0));
        rgb.g = HueToRGB(p, q, hsl.x);
        rgb.b = HueToRGB(p, q, hsl.x - (1.0/3.0));
    }
    
    return rgb;
}

// Helper for HSL to RGB conversion
float HueToRGB(float p, float q, float t) {
    if(t < 0.0) t += 1.0;
    if(t > 1.0) t -= 1.0;
    if(t < 1.0/6.0) return p + (q - p) * 6.0 * t;
    if(t < 1.0/2.0) return q;
    if(t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
    return p;
}
"""


def generate_bloom_effect_glsl():
    """Generate GLSL implementation for bloom effect."""
    return """// Bloom effect module
// Standardized bloom implementation

// Extract bright areas from the scene
vec3 ExtractBloom(vec3 color, float threshold) {
    vec3 bloom = max(color - vec3(threshold), 0.0);
    return bloom / (bloom + vec3(0.5));
}

// Apply bloom effect to the original color
vec3 ApplyBloom(vec3 original, vec3 bloom, float intensity) {
    return original + bloom * intensity;
}
"""


def generate_composite_glsl():
    """Generate GLSL implementation for compositing operations."""
    return """// Compositing module
// Standardized compositing functions

// Standard alpha blending
vec4 AlphaBlend(vec4 src, vec4 dst) {
    return src + dst * (1.0 - src.a);
}

// Multiplicative blending
vec3 MultiplyBlend(vec3 src, vec3 dst) {
    return src * dst;
}

// Additive blending
vec3 AddBlend(vec3 src, vec3 dst) {
    return min(src + dst, 1.0);
}

// Screen blending
vec3 ScreenBlend(vec3 src, vec3 dst) {
    return 1.0 - (1.0 - src) * (1.0 - dst);
}

// Overlay blending
vec3 OverlayBlend(vec3 src, vec3 dst) {
    vec3 result;
    result.r = dst.r < 0.5 ? 2.0 * dst.r * src.r : 1.0 - 2.0 * (1.0 - dst.r) * (1.0 - src.r);
    result.g = dst.g < 0.5 ? 2.0 * dst.g * src.g : 1.0 - 2.0 * (1.0 - dst.g) * (1.0 - src.g);
    result.b = dst.b < 0.5 ? 2.0 * dst.b * src.b : 1.0 - 2.0 * (1.0 - dst.b) * (1.0 - src.b);
    return result;
}

// Linear interpolation blend
vec3 LerpBlend(vec3 src, vec3 dst, float factor) {
    return mix(dst, src, factor);
}
"""


def generate_edge_detection_glsl():
    """Generate GLSL implementation for edge detection."""
    return """// Edge detection module
// Standardized edge detection functions

// Sobel edge detection
float SobelEdge(sampler2D texture, vec2 texCoord, vec2 texSize) {
    vec2 texelSize = 1.0 / texSize;
    
    // Sample 3x3 neighborhood
    float tl = texture2D(texture, texCoord + vec2(-texelSize.x, -texelSize.y)).r;
    float tm = texture2D(texture, texCoord + vec2(0.0, -texelSize.y)).r;
    float tr = texture2D(texture, texCoord + vec2(texelSize.x, -texelSize.y)).r;
    float ml = texture2D(texture, texCoord + vec2(-texelSize.x, 0.0)).r;
    float mm = texture2D(texture, texCoord).r;
    float mr = texture2D(texture, texCoord + vec2(texelSize.x, 0.0)).r;
    float bl = texture2D(texture, texCoord + vec2(-texelSize.x, texelSize.y)).r;
    float bm = texture2D(texture, texCoord + vec2(0.0, texelSize.y)).r;
    float br = texture2D(texture, texCoord + vec2(texelSize.x, texelSize.y)).r;
    
    // Sobel X kernel
    float x = (-1.0 * tl) + (1.0 * tr) +
              (-2.0 * ml) + (2.0 * mr) +
              (-1.0 * bl) + (1.0 * br);
    
    // Sobel Y kernel
    float y = (-1.0 * tl) + (-2.0 * tm) + (-1.0 * tr) +
              ( 1.0 * bl) + ( 2.0 * bm) + ( 1.0 * br);
    
    return sqrt(x * x + y * y);
}

// Prewitt edge detection
float PrewittEdge(sampler2D texture, vec2 texCoord, vec2 texSize) {
    vec2 texelSize = 1.0 / texSize;
    
    // Sample 3x3 neighborhood
    float tl = texture2D(texture, texCoord + vec2(-texelSize.x, -texelSize.y)).r;
    float tm = texture2D(texture, texCoord + vec2(0.0, -texelSize.y)).r;
    float tr = texture2D(texture, texCoord + vec2(texelSize.x, -texelSize.y)).r;
    float ml = texture2D(texture, texCoord + vec2(-texelSize.x, 0.0)).r;
    float mm = texture2D(texture, texCoord).r;
    float mr = texture2D(texture, texCoord + vec2(texelSize.x, 0.0)).r;
    float bl = texture2D(texture, texCoord + vec2(-texelSize.x, texelSize.y)).r;
    float bm = texture2D(texture, texCoord + vec2(0.0, texelSize.y)).r;
    float br = texture2D(texture, texCoord + vec2(texelSize.x, texelSize.y)).r;
    
    // Prewitt X kernel
    float x = (-1.0 * tl) + (0.0 * tm) + (1.0 * tr) +
              (-1.0 * ml) + (0.0 * mm) + (1.0 * mr) +
              (-1.0 * bl) + (0.0 * bm) + (1.0 * br);
    
    // Prewitt Y kernel
    float y = (-1.0 * tl) + (-1.0 * tm) + (-1.0 * tr) +
              ( 0.0 * ml) + ( 0.0 * mm) + ( 0.0 * mr) +
              ( 1.0 * bl) + ( 1.0 * bm) + ( 1.0 * br);
    
    return sqrt(x * x + y * y);
}
"""


def generate_vignette_effect_glsl():
    """Generate GLSL implementation for vignette effect."""
    return """// Vignette effect module
// Standardized vignette implementation

// Create a vignette effect based on distance from center
vec3 ApplyVignette(vec3 color, vec2 uv, float strength, float radius) {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    float vignette = 1.0 - pow(dist / radius, strength);
    return color * vignette;
}

// Radial vignette with smooth falloff
vec3 RadialVignette(vec3 color, vec2 uv, vec2 center, float softness, float intensity) {
    float dist = distance(uv, center);
    dist = smoothstep(0.0, softness, dist * intensity);
    return color * (1.0 - dist);
}
"""


def main():
    # Find effect shaders
    effect_shaders = find_effect_shaders()
    
    # Extract shader codes for a subset (first 500) for efficiency
    shader_codes = []
    for filepath, shader_info in effect_shaders[:500]:  # Limit to first 500 for efficiency
        shader_code = extract_shader_code(filepath)
        shader_codes.append({
            'info': shader_info,
            'code': shader_code
        })
    
    # Analyze shader patterns
    analyzed_shaders, pattern_counts = analyze_effect_shaders()
    
    # Extract specific effect functions
    modules = extract_effect_modules(analyzed_shaders)
    
    # Create standardized modules
    create_standardized_effect_modules()
    
    print("Effects shader analysis and module extraction completed!")


if __name__ == "__main__":
    main()