#!/usr/bin/env python3
"""
Process UI/2D graphics shaders from JSON files to identify common patterns
and extract reusable modules.
"""

import json
import os
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def find_ui_2d_graphics_shaders(json_dir='json'):
    """
    Find all JSON files that contain UI/2D graphics related tags.

    Args:
        json_dir (str): Directory containing JSON shader files

    Returns:
        list: List of tuples (filepath, shader_info) for UI/2D graphics shaders
    """
    print("Finding UI/2D graphics related shaders...")
    
    keywords = [
        'ui', 'interface', '2d', '2d graphics', 'graphics', 'gui', 'hud', 'overlay',
        'screen', 'text', 'font', 'label', 'button', 'panel', 'window', 'menu',
        'icon', 'sprite', 'canvas', 'widget', 'control', 'element', 'layout',
        'frame', 'border', 'corner', 'round', 'rect', 'rectangle', 'shape',
        'line', 'circle', 'ellipse', 'triangle', 'polygon', 'fill', 'stroke',
        'antialias', 'smooth', 'gradient', 'linear', 'radial', 'box', 'blur',
        'shadow', 'outline', 'mask', 'clip', 'alpha', 'blend', 'composite',
        'draw', 'render', 'pixel', 'coordinate', 'position'
    ]
    
    ui_shaders = []
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} JSON files for UI/2D graphics tags...")
    
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
                
                # Check if this shader is UI/2D graphics related 
                is_ui_related = False
                
                # Check tags
                for tag in tags:
                    if any(keyword in tag for keyword in keywords):
                        is_ui_related = True
                        break
                
                # Check name
                if not is_ui_related:
                    for keyword in keywords:
                        if keyword in name:
                            is_ui_related = True
                            break
                
                # Check description
                if not is_ui_related:
                    for keyword in keywords:
                        if keyword in description:
                            is_ui_related = True
                            break
                
                if is_ui_related:
                    shader_info = {
                        'id': info.get('id', os.path.basename(filepath).replace('.json', '')),
                        'name': info.get('name', ''),
                        'tags': tags,
                        'username': info.get('username', ''),
                        'description': info.get('description', ''),
                        'filepath': filepath
                    }
                    ui_shaders.append((filepath, shader_info))
                    
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue

    print(f"Found {len(ui_shaders)} UI/2D graphics related shaders")
    return ui_shaders


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


def identify_ui_2d_patterns(shader_code):
    """
    Identify common UI/2D graphics patterns in shader code.

    Args:
        shader_code (str): GLSL code to analyze

    Returns:
        dict: Dictionary of identified UI/2D patterns
    """
    patterns = {
        # Basic 2D shapes
        'rectangle_draw': 'rect' in shader_code.lower() or 'box' in shader_code.lower(),
        'circle_draw': 'circle' in shader_code.lower() or 'round' in shader_code.lower(),
        'line_draw': 'line' in shader_code.lower(),
        'triangle_draw': 'triangle' in shader_code.lower(),
        'ellipse_draw': 'ellipse' in shader_code.lower(),
        
        # UI elements
        'button_draw': 'button' in shader_code.lower(),
        'panel_draw': 'panel' in shader_code.lower(),
        'window_draw': 'window' in shader_code.lower(),
        'menu_draw': 'menu' in shader_code.lower(),
        'icon_draw': 'icon' in shader_code.lower(),
        
        # Graphics operations
        'gradient_fill': 'gradient' in shader_code.lower(),
        'linear_gradient': 'linear' in shader_code.lower() and 'gradient' in shader_code.lower(),
        'radial_gradient': 'radial' in shader_code.lower() and 'gradient' in shader_code.lower(),
        'antialiasing': 'anti' in shader_code.lower() and 'alias' in shader_code.lower(),
        'stroke_draw': 'stroke' in shader_code.lower(),
        'fill_draw': 'fill' in shader_code.lower(),
        
        # Text rendering
        'text_render': 'text' in shader_code.lower(),
        'font_render': 'font' in shader_code.lower(),
        'glyph_render': 'glyph' in shader_code.lower(),
        
        # Effects
        'shadow_effect': 'shadow' in shader_code.lower(),
        'outline_effect': 'outline' in shader_code.lower(),
        'blur_effect': 'blur' in shader_code.lower(),
        
        # Transformations
        'coordinate_transform': 'coord' in shader_code.lower() or 'transform' in shader_code.lower(),
        'positioning': 'pos' in shader_code.lower(),
        
        # Pixel operations
        'pixel_shader': 'frag' in shader_code.lower() or 'pixel' in shader_code.lower(),
        'color_blend': 'blend' in shader_code.lower(),
        'alpha_composite': 'alpha' in shader_code.lower() and 'comp' in shader_code.lower(),
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
        pattern (str): Base pattern to match (like 'draw', 'ui', 'shape', etc.)
    
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


def analyze_ui_shaders():
    """
    Main function to analyze UI/2D graphics shaders.
    """
    print("Analyzing UI/2D graphics shaders...")
    
    ui_shaders = find_ui_2d_graphics_shaders()
    
    # Store shader codes and identified patterns
    shader_codes = []
    all_patterns = []
    pattern_counts = Counter()
    
    print("\nAnalyzing UI/2D patterns in shaders...")
    for i, (filepath, shader_info) in enumerate(ui_shaders):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(ui_shaders)} UI/2D shaders...")
        
        shader_code = extract_shader_code(filepath)
        patterns = identify_ui_2d_patterns(shader_code)
        
        shader_codes.append({
            'info': shader_info,
            'code': shader_code,
            'patterns': patterns
        })
        
        all_patterns.append(patterns)
        
        # Update pattern counts
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    print(f"\nAnalysis complete! Found {len(ui_shaders)} UI/2D graphics shaders.")
    
    # Print pattern distribution
    print(f"\nCommon UI/2D patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count} shaders")
    
    # Save analysis results
    save_ui_analysis(shader_codes, pattern_counts)
    
    return shader_codes, pattern_counts


def save_ui_analysis(shader_codes, pattern_counts):
    """
    Save the UI/2D analysis results to files.
    """
    os.makedirs('analysis/ui', exist_ok=True)
    
    # Save pattern statistics
    with open('analysis/ui/pattern_stats.txt', 'w', encoding='utf-8') as f:
        f.write("UI/2D Pattern Statistics\n")
        f.write("=" * 50 + "\n")
        for pattern, count in pattern_counts.most_common():
            f.write(f"{pattern.replace('_', ' ').title()}: {count}\n")
    
    # Save detailed shader analysis
    with open('analysis/ui/shader_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed UI/2D Graphics Shader Analysis\n")
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
    
    print("UI/2D graphics shader analysis saved to analysis/ui/ directory")


def extract_ui_modules(shader_codes):
    """
    Extract reusable UI/2D graphics modules from analyzed shaders.

    Args:
        shader_codes (list): List of shader data from analysis

    Returns:
        dict: Dictionary of extracted UI/2D graphics modules
    """
    print("\nExtracting reusable UI/2D graphics modules...")
    
    modules = {
        'basic_shapes': set(),
        'ui_elements': set(),
        'graphics_operations': set(),
        'text_rendering': set(),
        'effects': set(),
        'utilities': set()
    }
    
    total_processed = 0
    
    for shader_data in shader_codes:
        code = shader_data['code']
        
        # Extract basic shape drawing functions
        shape_funcs = extract_complete_functions(code, 'rect')
        shape_funcs += extract_complete_functions(code, 'circle')
        shape_funcs += extract_complete_functions(code, 'line')
        shape_funcs += extract_complete_functions(code, 'draw')
        modules['basic_shapes'].update(shape_funcs)
        
        # Extract UI element functions
        ui_funcs = extract_complete_functions(code, 'ui')
        ui_funcs += extract_complete_functions(code, 'button')
        ui_funcs += extract_complete_functions(code, 'panel')
        ui_funcs += extract_complete_functions(code, 'window')
        modules['ui_elements'].update(ui_funcs)
        
        # Extract graphics operations
        graphics_funcs = extract_complete_functions(code, 'gradient')
        graphics_funcs += extract_complete_functions(code, 'fill')
        graphics_funcs += extract_complete_functions(code, 'stroke')
        modules['graphics_operations'].update(graphics_funcs)
        
        # Extract text rendering functions
        text_funcs = extract_complete_functions(code, 'text')
        text_funcs += extract_complete_functions(code, 'font')
        text_funcs += extract_complete_functions(code, 'glyph')
        modules['text_rendering'].update(text_funcs)
        
        # Extract effect functions
        effect_funcs = extract_complete_functions(code, 'effect')
        effect_funcs += extract_complete_functions(code, 'blur')
        effect_funcs += extract_complete_functions(code, 'shadow')
        modules['effects'].update(effect_funcs)
        
        # Extract utility functions
        util_funcs = extract_complete_functions(code, 'util')
        util_funcs += extract_complete_functions(code, 'coord')
        util_funcs += extract_complete_functions(code, 'transform')
        modules['utilities'].update(util_funcs)
        
        total_processed += 1
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(shader_codes)} shaders...")
    
    print(f"Extraction complete! Found:")
    for module_type, funcs in modules.items():
        print(f"  {module_type}: {len(funcs)} functions")
    
    # Save modules
    save_ui_modules(modules)
    
    return modules


def save_ui_modules(modules):
    """
    Save extracted UI/2D graphics modules to files.
    """
    os.makedirs('modules/ui', exist_ok=True)
    
    for module_type, func_list in modules.items():
        if func_list:  # Only save if there are modules of this type
            with open(f'modules/ui/{module_type}_functions.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} UI/2D Functions\n")
                f.write("// Automatically extracted from UI/2D graphics-related shaders\n\n")
                
                for i, func in enumerate(func_list, 1):
                    f.write(f"// Function {i}\n")
                    f.write(func)
                    f.write("\n\n")
    
    print("UI/2D graphics modules saved to modules/ui/ directory")


def create_standardized_ui_modules():
    """
    Create standardized UI/2D graphics modules based on patterns found.
    """
    print("Creating standardized UI/2D graphics modules...")
    
    # Define standardized module templates with actual GLSL implementations
    standardized_modules = {
        'basic_shapes.glsl': generate_basic_shapes_glsl(),
        'ui_elements.glsl': generate_ui_elements_glsl(),
        'graphics_operations.glsl': generate_graphics_operations_glsl(),
        'primitives.glsl': generate_primitives_glsl(),
        'effects.glsl': generate_effects_glsl(),
        'utilities.glsl': generate_utilities_glsl()
    }
    
    os.makedirs('modules/ui/standardized', exist_ok=True)
    
    # Create standardized modules
    for filename, code in standardized_modules.items():
        filepath = f'modules/ui/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Created {len(standardized_modules)} standardized UI/2D graphics modules")


def generate_basic_shapes_glsl():
    """Generate GLSL implementation for basic shapes."""
    return """// Basic shapes module
// Standardized 2D shape implementations

// Draw a rectangle
float rect(vec2 st, vec2 pos, vec2 size) {
    vec2 adjustedST = st - pos;
    float horizontal = step(0.0, adjustedST.x) * step(0.0, size.x - adjustedST.x);
    float vertical = step(0.0, adjustedST.y) * step(0.0, size.y - adjustedST.y);
    return horizontal * vertical;
}

// Draw a rectangle with rounded corners
float roundedRect(vec2 st, vec2 pos, vec2 size, float radius) {
    vec2 adjustedST = st - pos - size * 0.5;
    size -= vec2(radius * 2.0);
    vec2 corner = vec2(radius);
    
    return rect(st, pos + vec2(radius), size) +
           rect(st, pos + vec2(0, size.y * 0.5), vec2(radius * 2.0, size.y)) +
           rect(st, pos + vec2(size.x, size.y * 0.5), vec2(radius * 2.0, size.y)) +
           rect(st, pos + vec2(size.x * 0.5, 0), vec2(size.x, radius * 2.0)) +
           rect(st, pos + vec2(size.x * 0.5, size.y), vec2(size.x, radius * 2.0)) +
           circle(st, pos + vec2(radius, radius), radius) +
           circle(st, pos + vec2(size.x + radius, radius), radius) +
           circle(st, pos + vec2(radius, size.y + radius), radius) +
           circle(st, pos + size + vec2(radius, radius), radius);
}

// Draw a circle/ellipse
float circle(vec2 st, vec2 center, float radius) {
    float d = distance(st, center);
    d -= radius;
    return 1.0 - smoothstep(0.0, 1.0, d);
}

// Draw an ellipse
float ellipse(vec2 st, vec2 center, vec2 axes) {
    vec2 d = (st - center) / axes;
    return 1.0 - smoothstep(0.9, 1.0, dot(d, d));
}

// Draw a line
float line(vec2 st, vec2 a, vec2 b, float thickness) {
    vec2 ba = b - a;
    vec2 pa = st - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    float d = length(pa - ba * h);
    return 1.0 - smoothstep(thickness * 0.5, thickness * 0.5 + 0.01, d);
}

// Draw a triangle
float triangle(vec2 st, vec2 a, vec2 b, vec2 c) {
    vec3 bary = vec3(
        (b.y - c.y) * (st.x - c.x) + (c.x - b.x) * (st.y - c.y),
        (c.y - a.y) * (st.x - c.x) + (a.x - c.x) * (st.y - c.y),
        (a.y - b.y) * (st.x - c.x) + (b.x - a.x) * (st.y - c.y)
    );
    
    // If p is on the same side of all edges, return 1
    return step(0.0, bary.x) * step(0.0, bary.y) * step(0.0, bary.z);
}

// Draw a polygon (simplified for quadrilateral)
float quad(vec2 st, vec2 a, vec2 b, vec2 c, vec2 d) {
    float result = 0.0;
    
    // Check if point is inside the quad by testing against each edge
    vec2 edges[4];
    edges[0] = b - a;
    edges[1] = c - b;
    edges[2] = d - c;
    edges[3] = a - d;
    
    vec2 points[4];
    points[0] = a;
    points[1] = b;
    points[2] = c;
    points[3] = d;
    
    for (int i = 0; i < 4; i++) {
        vec2 edge = edges[i];
        vec2 point = points[i];
        vec2 perp = vec2(-edge.y, edge.x);
        vec2 toPoint = st - point;
        
        float side = dot(toPoint, perp);
        result = i == 0 ? step(0.0, side) : min(result, step(0.0, side));
    }
    
    return result;
}
"""


def generate_ui_elements_glsl():
    """Generate GLSL implementation for UI elements."""
    return """// UI elements module
// Standardized UI element implementations

// Draw a button with optional text
vec3 drawButton(vec2 uv, vec2 pos, vec2 size, vec3 color, bool isPressed) {
    vec2 localUV = (uv - pos) / size;
    
    // Check if UV is within button bounds
    float inButton = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                     (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    if (inButton < 0.5) return vec3(0.0);
    
    // Create button appearance
    vec3 buttonColor = color;
    
    // Add pressed effect
    if (isPressed) {
        buttonColor *= 0.7; // Darken when pressed
    }
    
    // Add bevel effect
    float bevel = 0.1 * (1.0 - localUV.y); // Lighter at top
    buttonColor += bevel;
    
    // Add border
    float borderSize = 0.03;
    float border = (localUV.x < borderSize || localUV.x > 1.0 - borderSize || 
                    localUV.y < borderSize || localUV.y > 1.0 - borderSize) ? 1.0 : 0.0;
    buttonColor = mix(buttonColor * 0.8, color * 1.5, border);
    
    return buttonColor * inButton;
}

// Draw a progress bar
vec3 drawProgressBar(vec2 uv, vec2 pos, vec2 size, float progress, vec3 fillColor, vec3 emptyColor) {
    vec2 localUV = (uv - pos) / size;
    
    // Background
    float background = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                       (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    // Progress fill
    float fill = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                 step(localUV.x, progress) * (1.0 - step(1.0, localUV.y));
    
    return mix(emptyColor, fillColor, fill) * background;
}

// Draw a slider
vec3 drawSlider(vec2 uv, vec2 pos, vec2 size, float value, vec3 trackColor, vec3 thumbColor) {
    vec2 localUV = (uv - pos) / size;
    
    // Track
    float track = step(0.4, localUV.y) * step(0.6, 1.0 - localUV.y) * 
                  step(0.0, localUV.x) * (1.0 - step(1.0, localUV.x));
    
    // Thumb position
    float thumbPos = value * size.x;
    vec2 thumbUV = (uv - vec2(pos.x + thumbPos, pos.y + size.y * 0.5)) / vec2(size.y * 0.8, size.y * 0.8);
    thumbUV += vec2(0.0, 0.0);
    float thumb = (abs(thumbUV.x) < 0.5 && abs(thumbUV.y) < 0.5) ? 1.0 : 0.0;
    
    vec3 result = trackColor * track;
    result = mix(result, thumbColor, thumb);
    
    return result;
}

// Draw a checkbox
vec3 drawCheckbox(vec2 uv, vec2 pos, vec2 size, bool isChecked, bool isHovered) {
    vec2 localUV = (uv - pos) / size;
    
    // Background square
    float background = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                       (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    vec3 color = vec3(0.8); // Default color
    
    if (isHovered) {
        color = vec3(1.0); // Highlight when hovered
    }
    
    // Draw checkmark if checked
    if (isChecked) {
        float check = 0.0;
        // Draw simple checkmark
        vec2 checkUV = localUV * 2.0 - 1.0; // Center UV in [-1,1]
        
        // Checkmark lines
        float line1 = 1.0 - smoothstep(0.05, 0.06, abs(checkUV.x - checkUV.y));
        float line2 = 1.0 - smoothstep(0.05, 0.06, abs(checkUV.x + checkUV.y + 0.3));
        check = max(line1, line2);
        
        color = mix(color, vec3(0.2, 0.8, 0.2), check); // Green checkmark
    }
    
    return color * background;
}

// Draw a text character (simplified)
float drawCharacter(vec2 uv, vec2 pos, int charIndex) {
    // This is a simplified character drawing function
    // A full implementation would require a font texture or signed distance fields
    vec2 localUV = (uv - pos) / vec2(0.05, 0.07); // Character size
    localUV -= vec2(0.5);
    
    // Draw a simple box
    float box = (step(-0.4, localUV.x) - step(0.4, localUV.x)) * 
                (step(-0.4, localUV.y) - step(0.4, localUV.y));
    
    return box;
}

// Draw a panel/frame
vec3 drawPanel(vec2 uv, vec2 pos, vec2 size, vec3 color) {
    vec2 localUV = (uv - pos) / size;
    
    // Background
    float background = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                       (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    // Border
    float borderSize = 0.02;
    float border = 0.0;
    border += step(0.0, localUV.x) * step(0.0, localUV.y) * 
              step(borderSize, localUV.x) * (1.0 - step(1.0 - borderSize, localUV.x)) * 
              step(0.0, localUV.y) * (1.0 - step(1.0, localUV.y));
    border += step(0.0, localUV.x) * step(0.0, localUV.y) * 
              step(0.0, localUV.x) * (1.0 - step(1.0, localUV.x)) * 
              step(borderSize, localUV.y) * (1.0 - step(1.0 - borderSize, localUV.y));
    
    vec3 borderColor = color * 0.5; // Darker border
    vec3 panelColor = color * 0.8;  // Slightly darker inside
    
    return mix(panelColor, borderColor, border) * background;
}
"""


def generate_graphics_operations_glsl():
    """Generate GLSL implementation for graphics operations."""
    return """// Graphics operations module
// Standardized graphics operation implementations

// Apply linear gradient
vec3 linearGradient(vec2 st, vec2 startPoint, vec2 endPoint, vec3 startColor, vec3 endColor) {
    vec2 direction = endPoint - startPoint;
    float lengthSquared = dot(direction, direction);
    
    if (lengthSquared == 0.0) {
        return startColor;
    }
    
    vec2 fromStart = st - startPoint;
    float projection = dot(fromStart, direction) / lengthSquared;
    projection = clamp(projection, 0.0, 1.0);
    
    return mix(startColor, endColor, projection);
}

// Apply radial gradient
vec3 radialGradient(vec2 st, vec2 center, float radius, vec3 innerColor, vec3 outerColor) {
    float dist = distance(st, center);
    float t = clamp(dist / radius, 0.0, 1.0);
    return mix(innerColor, outerColor, t);
}

// Apply box blur
vec3 boxBlur(sampler2D texture, vec2 uv, vec2 resolution, float radius) {
    vec3 color = vec3(0.0);
    float total = 0.0;
    
    float radiusInUV = radius / min(resolution.x, resolution.y);
    
    for (float x = -radius; x <= radius; x++) {
        for (float y = -radius; y <= radius; y++) {
            vec2 offset = vec2(x, y) * radiusInUV;
            color += texture2D(texture, uv + offset).rgb;
            total += 1.0;
        }
    }
    
    return color / total;
}

// Apply anti-aliased line
float antiAliasedLine(vec2 st, vec2 a, vec2 b, float thickness) {
    vec2 ba = b - a;
    vec2 pa = st - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    float d = length(pa - ba * h);
    
    // Anti-aliasing using smoothstep over a small range
    return 1.0 - smoothstep(thickness * 0.5 - 0.01, thickness * 0.5 + 0.01, d);
}

// Apply fill with optional stroke
vec3 fillAndStroke(vec2 shape, vec2 st, vec2 center, float strokeWidth, vec3 fillColor, vec3 strokeColor) {
    float filled = shape;
    float outline = shape - circle(st, center, strokeWidth);  // Simplified outline
    
    return mix(fillColor, strokeColor, outline);
}

// Apply alpha blending
vec4 alphaBlend(vec4 src, vec4 dst) {
    float alpha = src.a + dst.a * (1.0 - src.a);
    if (alpha == 0.0) {
        return vec4(0.0);
    }
    
    vec3 color = (src.rgb * src.a + dst.rgb * dst.a * (1.0 - src.a)) / alpha;
    return vec4(color, alpha);
}

// Apply color tint
vec3 applyTint(vec3 color, vec3 tint, float intensity) {
    return mix(color, color * tint, intensity);
}

// Apply color adjustment (brightness, contrast, saturation)
vec3 applyColorAdjustment(vec3 color, float brightness, float contrast, float saturation) {
    // Apply brightness
    color += brightness;
    
    // Apply contrast
    color = (color - 0.5) * contrast + 0.5;
    
    // Apply saturation
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(gray), color, saturation);
    
    return clamp(color, 0.0, 1.0);
}
"""


def generate_primitives_glsl():
    """Generate GLSL implementation for additional primitives."""
    return """// Primitives module
// Additional 2D primitive implementations

// Draw a rounded rectangle with adjustable corner radius
float roundedRectCustom(vec2 st, vec2 pos, vec2 size, vec4 radii) {
    // radii: x=top-left, y=top-right, z=bottom-right, w=bottom-left
    vec2 adjustedST = st - pos;
    vec2 halfSize = size * 0.5;
    vec2 center = pos + halfSize;
    
    // Top-left corner
    float tl = circle(adjustedST, vec2(radii.x, radii.x), radii.x);
    if (adjustedST.x < radii.x && adjustedST.y < radii.x) return tl;
    
    // Top-right corner  
    float tr = circle(adjustedST, vec2(size.x - radii.y, radii.y), radii.y);
    if (adjustedST.x > size.x - radii.y && adjustedST.y < radii.y) return tr;
    
    // Bottom-right corner
    float br = circle(adjustedST, vec2(size.x - radii.z, size.y - radii.z), radii.z);
    if (adjustedST.x > size.x - radii.z && adjustedST.y > size.y - radii.z) return br;
    
    // Bottom-left corner
    float bl = circle(adjustedST, vec2(radii.w, size.y - radii.w), radii.w);
    if (adjustedST.x < radii.w && adjustedST.y > size.y - radii.w) return bl;
    
    // Center rectangle
    float centerRect = rect(adjustedST, vec2(radii.x, 0.0), vec2(size.x - radii.x - radii.y, radii.x)) + 
                       rect(adjustedST, vec2(0.0, radii.w), vec2(radii.w, size.y - radii.w - radii.z)) + 
                       rect(adjustedST, vec2(radii.y, radii.y), vec2(size.x - radii.x - radii.y, size.y - radii.w - radii.z));
    
    return max(max(max(tl, tr), max(br, bl)), centerRect);
}

// Draw a polygon using ray casting method (simplified for triangle and quad)
float polygon(vec2 st, vec2[] points, int numPoints) {
    if (numPoints < 3) return 0.0;
    
    // Simplified code for triangle
    if (numPoints == 3) {
        return triangle(st, points[0], points[1], points[2]);
    }
    
    // For quads we can break into triangles
    if (numPoints == 4) {
        float tri1 = triangle(st, points[0], points[1], points[2]);
        float tri2 = triangle(st, points[0], points[2], points[3]);
        return max(tri1, tri2);
    }
    
    // For other polygons, implement ray casting (simplified version)
    int crossings = 0;
    for (int i = 0; i < numPoints; i++) {
        int next = (i + 1) % numPoints;
        vec2 p1 = points[i];
        vec2 p2 = points[next];
        
        if (((p1.y > st.y) != (p2.y > st.y)) &&
            (st.x < (p2.x - p1.x) * (st.y - p1.y) / (p2.y - p1.y) + p1.x)) {
            crossings++;
        }
    }
    
    return float(crossings % 2);
}

// Draw a star
float star(vec2 st, vec2 center, float outerRadius, float innerRadius, int numPoints) {
    st -= center;
    
    float angle = atan(st.y, st.x);
    float radius = length(st);
    float angleStep = 3.14159 * 2.0 / float(numPoints);
    
    // Determine which section of the star we're in
    float section = mod(angle, angleStep);
    section = min(section, angleStep - section);
    
    // Calculate the distance to the star edge in this section
    float maxRadius = mix(innerRadius, outerRadius, 
                         abs(2.0 * section / angleStep - 1.0));
    
    return 1.0 - smoothstep(maxRadius - 0.01, maxRadius + 0.01, radius);
}

// Draw an arc
float arc(vec2 st, vec2 center, float radius, float startAngle, float endAngle, float thickness) {
    st -= center;
    float angle = atan(st.y, st.x);
    float dist = length(st);
    
    // Normalize angles to [0, 2*PI]
    startAngle = mod(startAngle, 3.14159 * 2.0);
    endAngle = mod(endAngle, 3.14159 * 2.0);
    
    if (startAngle > endAngle) endAngle += 3.14159 * 2.0;
    
    float inArc = 0.0;
    if (endAngle > 3.14159 * 2.0) {
        // Handle case where arc crosses 0 angle
        float normalizedAngle = mod(angle + 3.14159 * 2.0, 3.14159 * 2.0);
        inArc = step(startAngle, normalizedAngle) * step(normalizedAngle, mod(endAngle, 3.14159 * 2.0));
    } else {
        float normalizedAngle = mod(angle + 3.14159 * 2.0, 3.14159 * 2.0);
        float start = mod(startAngle + 3.14159 * 2.0, 3.14159 * 2.0);
        float end = mod(endAngle + 3.14159 * 2.0, 3.14159 * 2.0);
        if (start <= end) {
            inArc = step(start, normalizedAngle) * step(normalizedAngle, end);
        } else {
            inArc = (step(start, normalizedAngle) + step(0.0, normalizedAngle) * step(normalizedAngle, end));
        }
    }
    
    float distToEdge = abs(dist - radius);
    float inThickness = 1.0 - smoothstep(thickness * 0.5 - 0.01, thickness * 0.5 + 0.01, distToEdge);
    
    return inArc * inThickness;
}

// Draw a pie slice
float pie(vec2 st, vec2 center, float radius, float startAngle, float endAngle) {
    st -= center;
    float angle = atan(st.y, st.x);
    float dist = length(st);
    
    // Normalize angles to [0, 2*PI]
    float normStart = mod(startAngle + 3.14159 * 2.0, 3.14159 * 2.0);
    float normEnd = mod(endAngle + 3.14159 * 2.0, 3.14159 * 2.0);
    
    float inArc = 0.0;
    if (normStart <= normEnd) {
        float normalizedAngle = mod(angle + 3.14159 * 2.0, 3.14159 * 2.0);
        inArc = step(normStart, normalizedAngle) * step(normalizedAngle, normEnd);
    } else {
        // Handle wrap-around case
        float normalizedAngle = mod(angle + 3.14159 * 2.0, 3.14159 * 2.0);
        inArc = step(normStart, normalizedAngle) + step(0.0, normalizedAngle) * step(normalizedAngle, normEnd);
        inArc = clamp(inArc, 0.0, 1.0);
    }
    
    float inCircle = 1.0 - smoothstep(radius - 0.01, radius + 0.01, dist);
    
    return inArc * inCircle;
}
"""


def generate_effects_glsl():
    """Generate GLSL implementation for effects."""
    return """// Effects module
// Standardized UI/2D graphics effect implementations

// Apply drop shadow
vec3 applyDropShadow(vec2 st, float shape, vec2 offset, float blur, vec3 shadowColor) {
    // Create a blurred version of the shape at an offset position
    float shadow = 0.0;
    vec2 sampleOffset = vec2(0.0);
    float total = 0.0;
    
    for (float x = -blur; x <= blur; x += 0.5) {
        for (float y = -blur; y <= blur; y += 0.5) {
            vec2 samplePoint = st - offset + vec2(x, y) * 0.01;
            shadow += 1.0 - smoothstep(0.0, 0.1, shape);
            total += 1.0;
        }
    }
    
    shadow /= total;
    return shadowColor * shadow;
}

// Apply inner shadow
vec3 applyInnerShadow(vec2 st, float shape, vec2 offset, float blur, vec3 shadowColor) {
    // Inner shadow is the shadow within the shape
    float edge = fwidth(shape) * 2.0;
    float innerArea = 1.0 - shape;
    float shadowEffect = 1.0 - smoothstep(-edge, edge, innerArea - 0.5);
    return shadowColor * shadowEffect;
}

// Apply glow effect
vec3 applyGlow(vec2 st, float shape, float intensity, float size, vec3 glowColor) {
    float dist = 1.0 - shape;
    float glow = intensity * (1.0 - smoothstep(0.0, size, dist));
    return glowColor * glow;
}

// Apply outline
vec3 applyOutline(vec2 st, float shape, float thickness, vec3 outlineColor) {
    float outline = shape - rect(st, 
                                vec2(thickness, thickness), 
                                vec2(1.0 - 2.0 * thickness, 1.0 - 2.0 * thickness)); // Simplified
    return outlineColor * outline;
}

// Apply bevel effect
vec3 applyBevel(vec2 st, float shape, float size, vec3 lightColor, vec3 darkColor) {
    float dx = fwidth(st.x) * size;
    float dy = fwidth(st.y) * size;
    
    // Create a bevel by sampling the shape at different offsets
    float left = 1.0 - shape;  // Simplified - in reality would need to sample shape at offset
    float right = shape;
    float top = 1.0 - shape;
    float bottom = shape;
    
    // Calculate lighting based on height
    float light = (top + right) * 0.5;
    float dark = (bottom + left) * 0.5;
    
    return mix(darkColor, lightColor, light);
}

// Apply gradient overlay
vec3 applyGradientOverlay(vec2 st, vec3 baseColor, vec3 gradientColor, float intensity) {
    vec3 gradient = linearGradient(st, vec2(0.0), vec2(1.0, 1.0), vec3(0.0), vec3(1.0));
    return mix(baseColor, gradientColor, intensity * gradient.x);
}

// Apply noise
vec3 applyNoise(vec2 st, vec3 color, float intensity) {
    // Simple noise function
    float noise = fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453);
    return color + vec3(noise) * intensity;
}

// Apply saturation adjustment
vec3 saturate(vec3 color, float saturation) {
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    return mix(vec3(gray), color, saturation);
}
"""


def generate_utilities_glsl():
    """Generate GLSL implementation for utilities."""
    return """// Utilities module
// Standardized UI/2D graphics utility functions

// Remap a value from one range to another
float remap(float value, float inputMin, float inputMax, float outputMin, float outputMax) {
    return outputMin + (outputMax - outputMin) * (value - inputMin) / (inputMax - inputMin);
}

// Clamp with float3
vec3 clamp3(vec3 value, vec3 minVal, vec3 maxVal) {
    return vec3(clamp(value.x, minVal.x, maxVal.x),
                clamp(value.y, minVal.y, maxVal.y),
                clamp(value.z, minVal.z, maxVal.z));
}

// Linear interpolation for vec3
vec3 lerp3(vec3 a, vec3 b, float t) {
    return a + (b - a) * t;
}

// Calculate UV for a specific element in a layout
vec2 calculateElementUV(vec2 globalUV, vec2 elementPos, vec2 elementSize) {
    return (globalUV - elementPos) / elementSize;
}

// Convert screen coordinates to normalized coordinates [0, 1]
vec2 screenToNormalized(vec2 screenPos, vec2 resolution) {
    return screenPos / resolution;
}

// Convert normalized coordinates [0, 1] to screen coordinates
vec2 normalizedToScreen(vec2 normalizedPos, vec2 resolution) {
    return normalizedPos * resolution;
}

// Calculate aspect ratio
float calculateAspectRatio(vec2 resolution) {
    return resolution.x / resolution.y;
}

// Apply aspect ratio correction
vec2 applyAspectRatio(vec2 uv, vec2 resolution) {
    float aspectRatio = resolution.x / resolution.y;
    if (aspectRatio > 1.0) {
        uv.x *= aspectRatio;
        uv.x -= (aspectRatio - 1.0) * 0.5;
    } else {
        uv.y /= aspectRatio;
        uv.y -= (1.0 / aspectRatio - 1.0) * 0.5;
    }
    return uv;
}

// Create a checkerboard pattern
vec3 checkerboard(vec2 uv, float scale) {
    vec2 c = floor(uv * scale);
    return mod(c.x + c.y, 2.0) > 0.0 ? vec3(1.0) : vec3(0.8);
}

// Apply transformation matrix to UV
vec2 applyTransform(vec2 uv, mat2 transformMatrix, vec2 translation) {
    return (transformMatrix * uv) + translation;
}

// Get the distance field for multiple shapes combined
float combineShapes(float shape1, float shape2, int operation) {
    // operation: 0 = union, 1 = intersection, 2 = difference
    if (operation == 0) { // Union
        return max(shape1, shape2);
    } else if (operation == 1) { // Intersection
        return min(shape1, shape2);
    } else if (operation == 2) { // Difference
        return min(shape1, 1.0 - shape2);
    }
    return shape1;
}

// Smoothly combine shapes
float smoothCombine(float shape1, float shape2, float k, int operation) {
    // Using smooth minimum/maximum functions
    if (operation == 0) { // Union
        return -k * log(exp(-shape1/k) + exp(-shape2/k));
    } else if (operation == 1) { // Intersection
        return k * log(exp(shape1/k) + exp(shape2/k));
    }
    return shape1; // Default to first shape
}
"""


def main():
    # Find UI/2D graphics shaders
    ui_shaders = find_ui_2d_graphics_shaders()
    
    # Extract shader codes for a subset (first 500) for efficiency
    shader_codes = []
    for filepath, shader_info in ui_shaders[:500]:  # Limit to first 500 for efficiency
        shader_code = extract_shader_code(filepath)
        shader_codes.append({
            'info': shader_info,
            'code': shader_code
        })
    
    # Analyze shader patterns
    analyzed_shaders, pattern_counts = analyze_ui_shaders()
    
    # Extract specific UI functions
    modules = extract_ui_modules(analyzed_shaders)
    
    # Create standardized modules
    create_standardized_ui_modules()
    
    print("UI/2D graphics shader analysis and module extraction completed!")


if __name__ == "__main__":
    main()