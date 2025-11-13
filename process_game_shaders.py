#!/usr/bin/env python3
"""
Process game/interactive shaders from JSON files to identify common patterns
and extract reusable modules.
"""

import json
import os
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def find_game_interactive_shaders(json_dir='json'):
    """
    Find all JSON files that contain game/interactive related tags.

    Args:
        json_dir (str): Directory containing JSON shader files

    Returns:
        list: List of tuples (filepath, shader_info) for game/interactive shaders
    """
    print("Finding game/interactive related shaders...")
    
    keywords = [
        'game', 'interactive', 'interaction', 'input', 'mouse', 'keyboard', 'control',
        'player', 'character', 'avatar', 'controller', 'ui', 'interface', 'hud',
        'menu', 'button', 'click', 'select', 'choose', 'option', 'gameplay',
        'interactive', 'user', 'touch', 'screen', 'feedback', 'response',
        'navigation', 'map', 'minimap', 'crosshair', 'reticle', 'aim',
        'health', 'life', 'score', 'points', 'lives', 'ammo', 'inventory',
        'level', 'stage', 'world', 'environment', 'scene', 'transition'
    ]
    
    game_shaders = []
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} JSON files for game/interactive tags...")
    
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
                
                # Check if this shader is game/interactive related 
                is_game_related = False
                
                # Check tags
                for tag in tags:
                    if any(keyword in tag for keyword in keywords):
                        is_game_related = True
                        break
                
                # Check name
                if not is_game_related:
                    for keyword in keywords:
                        if keyword in name:
                            is_game_related = True
                            break
                
                # Check description
                if not is_game_related:
                    for keyword in keywords:
                        if keyword in description:
                            is_game_related = True
                            break
                
                if is_game_related:
                    shader_info = {
                        'id': info.get('id', os.path.basename(filepath).replace('.json', '')),
                        'name': info.get('name', ''),
                        'tags': tags,
                        'username': info.get('username', ''),
                        'description': info.get('description', ''),
                        'filepath': filepath
                    }
                    game_shaders.append((filepath, shader_info))
                    
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue

    print(f"Found {len(game_shaders)} game/interactive related shaders")
    return game_shaders


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


def identify_game_patterns(shader_code):
    """
    Identify common game/interactive patterns in shader code.

    Args:
        shader_code (str): GLSL code to analyze

    Returns:
        dict: Dictionary of identified game patterns
    """
    patterns = {
        # Input handling
        'mouse_input': 'mouse' in shader_code.lower(),
        'keyboard_input': 'key' in shader_code.lower() or 'keyboard' in shader_code.lower(),
        'input_handling': 'input' in shader_code.lower(),
        'touch_input': 'touch' in shader_code.lower(),
        
        # UI elements
        'hud_elements': 'hud' in shader_code.lower(),
        'health_bar': 'health' in shader_code.lower() or 'life' in shader_code.lower(),
        'score_display': 'score' in shader_code.lower() or 'points' in shader_code.lower(),
        'ui_elements': 'ui' in shader_code.lower() or 'interface' in shader_code.lower(),
        'button_elements': 'button' in shader_code.lower(),
        'crosshair': 'crosshair' in shader_code.lower() or 'reticle' in shader_code.lower() or 'aim' in shader_code.lower(),
        
        # Game elements
        'player_indicator': 'player' in shader_code.lower() or 'character' in shader_code.lower(),
        'inventory_display': 'inventory' in shader_code.lower(),
        'minimap_display': 'minimap' in shader_code.lower() or 'map' in shader_code.lower(),
        'ammo_display': 'ammo' in shader_code.lower() or 'bullets' in shader_code.lower(),
        
        # Interaction
        'click_detection': 'click' in shader_code.lower(),
        'select_highlight': 'select' in shader_code.lower() or 'highlight' in shader_code.lower(),
        'hover_effect': 'hover' in shader_code.lower(),
        'feedback_visual': 'feedback' in shader_code.lower(),
        
        # Game state
        'level_transition': 'level' in shader_code.lower() or 'transition' in shader_code.lower(),
        'game_state': 'state' in shader_code.lower() or 'gamestate' in shader_code.lower(),
        'menu_system': 'menu' in shader_code.lower(),
        
        # Visual feedback
        'pulse_effect': 'pulse' in shader_code.lower(),
        'glow_effect': 'glow' in shader_code.lower(),
        'selection_effect': 'select' in shader_code.lower() and 'effect' in shader_code.lower(),
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
        pattern (str): Base pattern to match (like 'ui', 'input', 'game', etc.)
    
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


def analyze_game_shaders():
    """
    Main function to analyze game/interactive shaders.
    """
    print("Analyzing game/interactive shaders...")
    
    game_shaders = find_game_interactive_shaders()
    
    # Store shader codes and identified patterns
    shader_codes = []
    all_patterns = []
    pattern_counts = Counter()
    
    print("\nAnalyzing game patterns in shaders...")
    for i, (filepath, shader_info) in enumerate(game_shaders):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(game_shaders)} game shaders...")
        
        shader_code = extract_shader_code(filepath)
        patterns = identify_game_patterns(shader_code)
        
        shader_codes.append({
            'info': shader_info,
            'code': shader_code,
            'patterns': patterns
        })
        
        all_patterns.append(patterns)
        
        # Update pattern counts
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    print(f"\nAnalysis complete! Found {len(game_shaders)} game/interactive shaders.")
    
    # Print pattern distribution
    print(f"\nCommon game patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count} shaders")
    
    # Save analysis results
    save_game_analysis(shader_codes, pattern_counts)
    
    return shader_codes, pattern_counts


def save_game_analysis(shader_codes, pattern_counts):
    """
    Save the game analysis results to files.
    """
    os.makedirs('analysis/game', exist_ok=True)
    
    # Save pattern statistics
    with open('analysis/game/pattern_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Game Pattern Statistics\n")
        f.write("=" * 50 + "\n")
        for pattern, count in pattern_counts.most_common():
            f.write(f"{pattern.replace('_', ' ').title()}: {count}\n")
    
    # Save detailed shader analysis
    with open('analysis/game/shader_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Game/Interactive Shader Analysis\n")
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
    
    print("Game/interactive shader analysis saved to analysis/game/ directory")


def extract_game_modules(shader_codes):
    """
    Extract reusable game/interactive modules from analyzed shaders.

    Args:
        shader_codes (list): List of shader data from analysis

    Returns:
        dict: Dictionary of extracted game/interactive modules
    """
    print("\nExtracting reusable game/interactive modules...")
    
    modules = {
        'ui_elements': set(),
        'input_handling': set(),
        'game_visuals': set(),
        'interaction_feedback': set(),
        'hud_components': set(),
        'game_state_visuals': set()
    }
    
    total_processed = 0
    
    for shader_data in shader_codes:
        code = shader_data['code']
        
        # Extract UI elements functions
        ui_funcs = extract_complete_functions(code, 'ui')
        ui_funcs += extract_complete_functions(code, 'interface')
        ui_funcs += extract_complete_functions(code, 'button')
        modules['ui_elements'].update(ui_funcs)
        
        # Extract input handling functions
        input_funcs = extract_complete_functions(code, 'input')
        input_funcs += extract_complete_functions(code, 'mouse')
        input_funcs += extract_complete_functions(code, 'key')
        modules['input_handling'].update(input_funcs)
        
        # Extract game visual elements
        game_funcs = extract_complete_functions(code, 'game')
        game_funcs += extract_complete_functions(code, 'player')
        game_funcs += extract_complete_functions(code, 'character')
        modules['game_visuals'].update(game_funcs)
        
        # Extract interaction feedback functions
        feedback_funcs = extract_complete_functions(code, 'feedback')
        feedback_funcs += extract_complete_functions(code, 'select')
        feedback_funcs += extract_complete_functions(code, 'highlight')
        modules['interaction_feedback'].update(feedback_funcs)
        
        # Extract HUD components
        hud_funcs = extract_complete_functions(code, 'hud')
        hud_funcs += extract_complete_functions(code, 'health')
        hud_funcs += extract_complete_functions(code, 'score')
        modules['hud_components'].update(hud_funcs)
        
        # Extract game state visual functions
        state_funcs = extract_complete_functions(code, 'state')
        state_funcs += extract_complete_functions(code, 'menu')
        state_funcs += extract_complete_functions(code, 'transition')
        modules['game_state_visuals'].update(state_funcs)
        
        total_processed += 1
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(shader_codes)} shaders...")
    
    print(f"Extraction complete! Found:")
    for module_type, funcs in modules.items():
        print(f"  {module_type}: {len(funcs)} functions")
    
    # Save modules
    save_game_modules(modules)
    
    return modules


def save_game_modules(modules):
    """
    Save extracted game/interactive modules to files.
    """
    os.makedirs('modules/game', exist_ok=True)
    
    for module_type, func_list in modules.items():
        if func_list:  # Only save if there are modules of this type
            with open(f'modules/game/{module_type}_functions.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} Game Functions\n")
                f.write("// Automatically extracted from game/interactive-related shaders\n\n")
                
                for i, func in enumerate(func_list, 1):
                    f.write(f"// Function {i}\n")
                    f.write(func)
                    f.write("\n\n")
    
    print("Game/interactive modules saved to modules/game/ directory")


def create_standardized_game_modules():
    """
    Create standardized game/interactive modules based on patterns found.
    """
    print("Creating standardized game/interactive modules...")
    
    # Define standardized module templates with actual GLSL implementations
    standardized_modules = {
        'ui_elements.glsl': generate_ui_elements_glsl(),
        'input_handling.glsl': generate_input_handling_glsl(),
        'hud_components.glsl': generate_hud_components_glsl(),
        'game_visuals.glsl': generate_game_visuals_glsl(),
        'interaction_feedback.glsl': generate_interaction_feedback_glsl(),
        'game_state_visuals.glsl': generate_game_state_visuals_glsl()
    }
    
    os.makedirs('modules/game/standardized', exist_ok=True)
    
    # Create standardized modules
    for filename, code in standardized_modules.items():
        filepath = f'modules/game/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Created {len(standardized_modules)} standardized game modules")


def generate_ui_elements_glsl():
    """Generate GLSL implementation for UI elements."""
    return """// UI elements module
// Standardized UI element implementations

// Draw a simple button
float drawButton(vec2 uv, vec2 pos, vec2 size) {
    vec2 center = pos + size * 0.5;
    vec2 buttonUV = (uv - pos) / size;
    
    // Check if UV is within button bounds
    float inButton = step(0.0, buttonUV.x) * step(0.0, buttonUV.y) * 
                     (1.0 - step(1.0, buttonUV.x)) * (1.0 - step(1.0, buttonUV.y));
    
    return inButton;
}

// Draw a rectangular UI panel
float drawPanel(vec2 uv, vec2 pos, vec2 size) {
    vec2 panelUV = (uv - pos) / size;
    return step(0.0, panelUV.x) * step(0.0, panelUV.y) * 
           (1.0 - step(1.0, panelUV.x)) * (1.0 - step(1.0, panelUV.y));
}

// Draw a simple checkbox
float drawCheckbox(vec2 uv, vec2 center, float size, bool checked) {
    vec2 localUV = (uv - center) / size;
    localUV += vec2(0.5);
    
    // Border
    float border = (step(0.0, localUV.x) - step(1.0, localUV.x)) * 
                   (step(0.0, localUV.y) - step(1.0, localUV.y));
    
    // Fill if checked
    float fill = 0.0;
    if(checked) {
        fill = (step(0.1, localUV.x) - step(0.9, localUV.x)) * 
               (step(0.1, localUV.y) - step(0.9, localUV.y));
    }
    
    return border + fill * 0.7;
}

// Draw a slider
float drawSlider(vec2 uv, vec2 pos, float width, float height, float value) {
    // Draw track
    vec2 trackUV = (uv - vec2(pos.x, pos.y - height * 0.5)) / vec2(width, height);
    float track = step(0.0, trackUV.x) * step(0.0, trackUV.y) * 
                  (1.0 - step(1.0, trackUV.x)) * (1.0 - step(1.0, trackUV.y));
    
    // Draw thumb
    vec2 thumbPos = vec2(pos.x + width * value, pos.y);
    float thumb = drawCircle(uv, thumbPos, height * 0.7);
    
    return track + thumb;
}

// Draw a simple text character (simplified)
float drawChar(vec2 uv, vec2 pos, float charIndex) {
    // Simplified character drawing - in practice you'd use a font texture
    vec2 charUV = (uv - pos) / vec2(0.05, 0.1); // Character size
    charUV -= vec2(0.5);
    
    // Simple box character
    return (step(-0.4, charUV.x) - step(0.4, charUV.x)) * 
           (step(-0.4, charUV.y) - step(0.4, charUV.y));
}

// Draw a circle/round element
float drawCircle(vec2 uv, vec2 center, float radius) {
    float dist = distance(uv, center);
    return 1.0 - smoothstep(radius - 0.005, radius, dist);
}

// Create a progress bar
float drawProgressBar(vec2 uv, vec2 pos, vec2 size, float progress) {
    // Background
    vec2 bgUV = (uv - pos) / size;
    float background = step(0.0, bgUV.x) * step(0.0, bgUV.y) * 
                       (1.0 - step(1.0, bgUV.x)) * (1.0 - step(1.0, bgUV.y));
    
    // Foreground (progress)
    vec2 fgUV = (uv - pos) / size;
    float foreground = step(0.0, fgUV.x) * step(0.0, fgUV.y) * 
                       step(fgUV.x, progress) * (1.0 - step(1.0, fgUV.y));
    
    return background * 0.3 + foreground;
}
"""


def generate_input_handling_glsl():
    """Generate GLSL implementation for input handling."""
    return """// Input handling module
// Standardized input handling implementations

// Get mouse position in UV coordinates
vec2 getMousePosition(sampler2D mouseChannel) {
    return texture2D(mouseChannel, vec2(0.0, 0.0)).xy;
}

// Get mouse click state
bool isMouseClicked(sampler2D mouseChannel) {
    return texture2D(mouseChannel, vec2(0.0, 0.0)).z > 0.0;
}

// Get mouse button state
bool isMouseButtonDown(sampler2D mouseChannel, int buttonIndex) {
    // Assuming button states are stored in different channels
    float buttonState = texture2D(mouseChannel, vec2(float(buttonIndex) / 4.0, 0.0)).x;
    return buttonState > 0.5;
}

// Get keyboard key state
bool isKeyDown(sampler2D keyboardChannel, int keyCode) {
    // Assuming each key state is stored at a specific position in the texture
    float keyState = texture2D(keyboardChannel, vec2(float(keyCode) / 256.0, 0.0)).x;
    return keyState > 0.5;
}

// Check if mouse is over an area
bool isMouseOver(vec2 mousePos, vec2 elementPos, vec2 elementSize) {
    return (mousePos.x > elementPos.x && mousePos.x < elementPos.x + elementSize.x &&
            mousePos.y > elementPos.y && mousePos.y < elementPos.y + elementSize.y);
}

// Get input-based selection
int getSelection(vec2 mousePos, vec2[] elementPositions, vec2[] elementSizes, int numElements) {
    for(int i = 0; i < numElements; i++) {
        if(isMouseOver(mousePos, elementPositions[i], elementSizes[i])) {
            return i;
        }
    }
    return -1; // No selection
}

// Handle joystick input (simplified)
vec2 getJoystickInput(sampler2D joystickChannel, int stickIndex) {
    vec2 offset = vec2(float(stickIndex) / 4.0, 0.0);
    return texture2D(joystickChannel, offset).xy;
}

// Get input direction from D-pad or WASD
vec2 getDirectionInput(sampler2D inputChannel) {
    vec4 keys = texture2D(inputChannel, vec2(0.0, 0.0)); // Assuming W, A, S, D in rgba
    vec2 direction = vec2(0.0);
    
    direction.x = (keys.y > 0.5 ? -1.0 : 0.0) + (keys.w > 0.5 ? 1.0 : 0.0); // A/D
    direction.y = (keys.x > 0.5 ? 1.0 : 0.0) + (keys.z > 0.5 ? -1.0 : 0.0); // W/S
    
    return normalize(direction);
}
"""


def generate_hud_components_glsl():
    """Generate GLSL implementation for HUD components."""
    return """// HUD components module
// Standardized HUD component implementations

// Draw a health bar
vec3 drawHealthBar(vec2 uv, vec2 pos, vec2 size, float health, vec3 fillColor, vec3 emptyColor) {
    // Background
    vec2 bgUV = (uv - pos) / size;
    float background = step(0.0, bgUV.x) * step(0.0, bgUV.y) * 
                       (1.0 - step(1.0, bgUV.x)) * (1.0 - step(1.0, bgUV.y));
    
    // Health fill
    vec2 fillUV = (uv - pos) / size;
    float fill = step(0.0, fillUV.x) * step(0.0, fillUV.y) * 
                 step(fillUV.x, health) * (1.0 - step(1.0, fillUV.y));
    
    return mix(emptyColor, fillColor, fill) * background;
}

// Draw an ammo counter
vec3 drawAmmoCounter(vec2 uv, vec2 pos, vec2 size, int currentAmmo, int maxAmmo) {
    // Draw background
    vec2 bgUV = (uv - pos) / size;
    float background = step(0.0, bgUV.x) * step(0.0, bgUV.y) * 
                       (1.0 - step(1.0, bgUV.x)) * (1.0 - step(1.0, bgUV.y));
    
    // Draw ammo count (simplified as bars)
    float ammoPercent = float(currentAmmo) / float(maxAmmo);
    vec2 fillUV = (uv - pos) / size;
    float ammo = step(0.0, fillUV.x) * step(0.0, fillUV.y) * 
                 step(fillUV.x, ammoPercent) * (1.0 - step(1.0, fillUV.y));
    
    // Draw ammo boxes based on count
    vec3 color = vec3(0.0);
    if(background > 0.5) {
        color = vec3(0.2, 0.2, 0.2); // Background color
        if(ammo > 0.5) {
            color = vec3(1.0, 1.0, 0.0); // Ammo color
        }
    }
    
    return color;
}

// Draw a score display
vec3 drawScore(vec2 uv, vec2 pos, vec2 charSize, int score) {
    // Simple digit drawing
    vec3 color = vec3(0.0);
    int remainingScore = score;
    int digitCount = 0;
    
    // Count digits
    int tempScore = max(1, score);  // Ensure at least 1 digit
    while(tempScore > 0) {
        tempScore /= 10;
        digitCount++;
    }
    
    // Draw each digit
    vec2 digitPos = pos;
    for(int i = 0; i < 6; i++) { // Max 6 digits
        if(i < digitCount) {
            int digit = int(mod(float(remainingScore), 10.0));
            remainingScore /= 10;
            
            // Draw digit (simplified)
            vec2 localUV = (uv - digitPos) / charSize;
            localUV -= vec2(0.5);
            
            // Simple box for digit
            float digitShape = (step(-0.4, localUV.x) - step(0.4, localUV.x)) * 
                               (step(-0.4, localUV.y) - step(0.4, localUV.y));
            
            color += vec3(digitShape * float(digit + 1) / 10.0);
            digitPos.x += charSize.x * 1.2;  // Space between digits
        }
    }
    
    return color;
}

// Draw a minimap
vec3 drawMinimap(vec2 uv, vec2 pos, vec2 size, sampler2D worldTexture, vec2 playerPos) {
    // Background
    vec2 bgUV = (uv - pos) / size;
    float background = step(0.0, bgUV.x) * step(0.0, bgUV.y) * 
                       (1.0 - step(1.0, bgUV.x)) * (1.0 - step(1.0, bgUV.y));
    
    if(background < 0.5) return vec3(0.0);
    
    // Sample the world at the relative position
    vec2 worldUV = bgUV; // Simple mapping from minimap UV to world UV
    vec3 worldSample = texture2D(worldTexture, worldUV).rgb;
    
    // Draw player indicator
    vec2 playerUV = (playerPos - pos) / size;
    float playerIndicator = drawCircle(uv, pos + playerUV * size, 0.01);
    
    return worldSample * (1.0 - playerIndicator) + vec3(1.0, 0.0, 0.0) * playerIndicator;
}

// Draw a crosshair
vec3 drawCrosshair(vec2 uv, vec2 center, float size, vec3 color, float thickness) {
    float horizontal = (abs(uv.y - center.y) < thickness) * 
                       (abs(uv.x - center.x) < size ? 1.0 : 0.0);
    float vertical = (abs(uv.x - center.x) < thickness) * 
                     (abs(uv.y - center.y) < size ? 1.0 : 0.0);
    
    // Don't double count the center
    float centerPixel = (abs(uv.x - center.x) < thickness) * 
                        (abs(uv.y - center.y) < thickness);
    
    return color * (horizontal + vertical - centerPixel);
}

// Draw a compass
vec3 drawCompass(vec2 uv, vec2 center, float radius, float rotation) {
    float dist = distance(uv, center);
    float angle = atan(uv.y - center.y, uv.x - center.x);
    
    // Only draw within the circle
    float circle = step(dist, radius) - step(dist, radius * 0.9);
    
    // Calculate angle relative to player direction
    float adjustedAngle = angle - rotation;
    
    // Draw N, E, S, W markers
    float north = step(0.95, cos(adjustedAngle)) * step(dist, radius * 0.95);
    float east = step(0.95, sin(adjustedAngle)) * step(dist, radius * 0.95);
    float south = step(0.95, cos(adjustedAngle + 3.14159)) * step(dist, radius * 0.95);
    float west = step(0.95, sin(adjustedAngle + 3.14159)) * step(dist, radius * 0.95);
    
    vec3 compassColor = vec3(0.0);
    compassColor += vec3(1.0, 0.0, 0.0) * north;  // N = Red
    compassColor += vec3(0.0, 1.0, 0.0) * east;   // E = Green
    compassColor += vec3(0.0, 0.0, 1.0) * south;  // S = Blue
    compassColor += vec3(1.0, 1.0, 0.0) * west;   // W = Yellow
    
    return compassColor * circle;
}
"""


def generate_game_visuals_glsl():
    """Generate GLSL implementation for game visuals."""
    return """// Game visuals module
// Standardized game visual implementations

// Draw a player character indicator
vec3 drawPlayer(vec2 uv, vec2 pos, float size, float rotation) {
    vec2 localUV = (uv - pos) / size;
    localUV = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation)) * localUV;
    
    // Draw a simple arrow or triangle for the player
    float playerShape = 1.0 - step(0.0, -localUV.y) * 
                        step(abs(localUV.x), -localUV.y * 0.5);
    
    return vec3(playerShape);
}

// Draw an enemy indicator
vec3 drawEnemy(vec2 uv, vec2 pos, float size) {
    vec2 localUV = (uv - pos) / size;
    localUV -= vec2(0.5);
    
    // Draw a simple diamond or cross shape for enemies
    float distToCenter = length(localUV);
    float enemyShape = step(distToCenter, 0.3) * (1.0 - step(distToCenter, 0.1));
    
    // Add cross shape
    float cross = max(abs(localUV.x), abs(localUV.y)) < 0.2 ? 1.0 : 0.0;
    enemyShape = max(enemyShape, cross);
    
    return vec3(enemyShape * 0.8, enemyShape * 0.2, enemyShape * 0.2);  // Red enemy
}

// Draw an item/pickup indicator
vec3 drawItem(vec2 uv, vec2 pos, float size) {
    vec2 localUV = (uv - pos) / size;
    localUV -= vec2(0.5);
    
    // Draw a simple circle with pulsing effect
    float dist = length(localUV);
    float pulse = 0.8 + 0.2 * sin(iTime * 5.0); // Pulsing effect
    float itemShape = (1.0 - smoothstep(0.3 * pulse - 0.02, 0.3 * pulse, dist)) * 
                      smoothstep(0.2 * pulse - 0.02, 0.2 * pulse, dist);
    
    // Add shine effect
    vec2 shinePos = vec2(-0.1, 0.1);
    float shine = (1.0 - smoothstep(0.05 - 0.01, 0.05, distance(localUV, shinePos))) * 0.5;
    
    return vec3(itemShape * 0.8 + shine, itemShape * 0.9 + shine, itemShape * 0.2 + shine);  // Yellow/golden item
}

// Draw a weapon indicator
vec3 drawWeapon(vec2 uv, vec2 pos, float size, float rotation) {
    vec2 localUV = (uv - pos) / size;
    localUV = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation)) * localUV;
    
    // Draw a simple gun shape
    float barrel = (abs(localUV.x - 0.3) < 0.05) * (abs(localUV.y) < 0.15) * step(0.0, localUV.x);
    float body = (abs(localUV.x) < 0.25) * (abs(localUV.y) < 0.2);
    float grip = (abs(localUV.x + 0.15) < 0.05) * (abs(localUV.y - 0.15) < 0.15) * step(localUV.y - 0.15, 0.0);
    
    float weaponShape = max(max(barrel, body), grip);
    return vec3(weaponShape * 0.5, weaponShape * 0.5, weaponShape * 0.6);  // Gray weapon
}

// Draw a health pickup
vec3 drawHealthPickup(vec2 uv, vec2 pos, float size) {
    vec2 localUV = (uv - pos) / size;
    localUV -= vec2(0.5);
    
    // Draw a simple cross shape for health
    float horizontal = (abs(localUV.y) < 0.2) * (abs(localUV.x) < 0.05);
    float vertical = (abs(localUV.x) < 0.2) * (abs(localUV.y) < 0.05);
    float cross = max(horizontal, vertical);
    
    return vec3(cross * 0.8, cross * 0.2, cross * 0.2);  // Red health
}

// Create a damage indicator effect
vec3 drawDamageIndicator(vec2 uv, float damageTime, vec2 damagePos) {
    float elapsed = iTime - damageTime;
    if(elapsed > 0.5) return vec3(0.0); // Effect only lasts 0.5 seconds
    
    // Radial red flash
    float dist = distance(uv, damagePos);
    float flash = (1.0 - smoothstep(0.0, 0.2, dist)) * (1.0 - elapsed / 0.5);
    
    return vec3(flash, flash * 0.2, flash * 0.2);  // Red damage indicator
}
"""


def generate_interaction_feedback_glsl():
    """Generate GLSL implementation for interaction feedback."""
    return """// Interaction feedback module
// Standardized interaction feedback implementations

// Create a selection highlight
vec3 createSelectionHighlight(vec2 uv, vec2 elementPos, vec2 elementSize, float time) {
    vec2 localUV = (uv - elementPos) / elementSize;
    
    // Check if UV is within element bounds
    float inElement = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                      (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    // Create animated border
    float borderWidth = 0.02;
    float border = 
        (step(borderWidth, localUV.x) * (1.0 - step(1.0 - borderWidth, localUV.x)) *
         step(0.0, localUV.y) * (1.0 - step(1.0, localUV.y))) +
        (step(borderWidth, localUV.y) * (1.0 - step(1.0 - borderWidth, localUV.y)) *
         step(0.0, localUV.x) * (1.0 - step(1.0, localUV.x)));
    
    // Animate the border color
    float pulse = 0.7 + 0.3 * sin(time * 10.0);
    vec3 highlightColor = vec3(1.0, pulse, 0.0); // Animated yellow border
    
    return highlightColor * border * inElement;
}

// Create a hover effect
vec3 createHoverEffect(vec2 uv, vec2 elementPos, vec2 elementSize, vec2 mousePos, float intensity) {
    vec2 localUV = (uv - elementPos) / elementSize;
    
    // Check if UV is within element bounds
    float inElement = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                      (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    // Check if mouse is hovering over element
    float isHovered = isMouseOver(mousePos, elementPos, elementSize) ? 1.0 : 0.0;
    
    // Create glow effect when hovered
    float glow = 0.0;
    if(isHovered > 0.5) {
        float distToEdge = min(min(localUV.x, 1.0 - localUV.x), min(localUV.y, 1.0 - localUV.y));
        glow = (1.0 - smoothstep(0.0, 0.1, distToEdge)) * intensity;
    }
    
    return vec3(glow, glow * 0.8, glow * 0.2) * inElement;
}

// Create a click feedback effect
vec3 createClickFeedback(vec2 uv, vec2 elementPos, vec2 elementSize, vec2 clickPos, float clickTime) {
    vec2 localUV = (uv - elementPos) / elementSize;
    
    // Check if UV is within element bounds
    float inElement = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                      (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    // Create ripple effect from click position
    float timeSinceClick = iTime - clickTime;
    if(timeSinceClick > 1.0) return vec3(0.0); // Effect lasts 1 second
    
    // Calculate distance from click position
    vec2 elementUV = elementPos + localUV * elementSize;
    float distFromClick = distance(elementUV, clickPos);
    
    // Create expanding circle
    float ripple = (1.0 - smoothstep(0.0, timeSinceClick * 0.5, distFromClick));
    ripple *= (1.0 - clamp(timeSinceClick * 2.0, 0.0, 1.0)); // Fade over time
    
    return vec3(ripple, ripple * 0.5, 0.0) * inElement; // Orange ripple
}

// Create a pulse feedback effect
vec3 createPulseEffect(vec2 uv, vec2 elementPos, vec2 elementSize, float pulseSpeed) {
    vec2 localUV = (uv - elementPos) / elementSize;
    
    // Check if UV is within element bounds
    float inElement = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                      (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    // Create pulsing effect
    float pulse = 0.5 + 0.5 * sin(iTime * pulseSpeed);
    
    return vec3(pulse, pulse * 0.7, pulse * 0.3) * inElement; // Orange pulse
}

// Create a progress feedback effect
vec3 createProgressEffect(vec2 uv, vec2 elementPos, vec2 elementSize, float progress) {
    vec2 localUV = (uv - elementPos) / elementSize;
    
    // Check if UV is within element bounds
    float inElement = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                      (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    // Create progress indicator
    float progressIndicator = step(0.0, localUV.y) * step(0.0, localUV.x) * 
                              step(localUV.x, progress) * (1.0 - step(1.0, localUV.y));
    
    vec3 color = vec3(0.2, 1.0, 0.2); // Green progress
    
    return color * progressIndicator * inElement;
}

// Create a drag feedback effect
vec3 createDragFeedback(vec2 uv, vec2 elementPos, vec2 elementSize, bool isDragging) {
    vec2 localUV = (uv - elementPos) / elementSize;
    
    // Check if UV is within element bounds
    float inElement = step(0.0, localUV.x) * step(0.0, localUV.y) * 
                      (1.0 - step(1.0, localUV.x)) * (1.0 - step(1.0, localUV.y));
    
    // Create different appearance when dragging
    if(isDragging) {
        float offset = 0.01 * sin(iTime * 20.0); // Vibrate when dragging
        vec3 dragColor = vec3(1.0, 0.8, 0.0); // Yellow when dragging
        return dragColor * inElement;
    }
    
    return vec3(0.8) * inElement; // Normal appearance
}
"""


def generate_game_state_visuals_glsl():
    """Generate GLSL implementation for game state visuals."""
    return """// Game state visuals module
// Standardized game state visual implementations

// Draw a menu background
vec3 drawMenuBackground(vec2 uv) {
    // Create a subtle animated background
    float pattern = sin(uv.x * 10.0 + iTime) * cos(uv.y * 8.0 + iTime * 0.5);
    pattern = abs(pattern) * 0.1 + 0.1;
    
    return vec3(pattern, pattern * 0.8, pattern * 1.2);
}

// Draw a pause screen overlay
vec3 drawPauseOverlay(vec2 uv) {
    // Semi-transparent dark overlay
    float overlay = 0.7;
    
    // Draw pause symbol (two vertical bars)
    float pauseSymbol = 0.0;
    float barWidth = 0.02;
    float barHeight = 0.1;
    vec2 center = vec2(0.5, 0.5);
    
    // Left bar
    pauseSymbol += (step(center.x - 0.03, uv.x) - step(center.x - 0.03 + barWidth, uv.x)) * 
                   (step(center.y - barHeight * 0.5, uv.y) - step(center.y + barHeight * 0.5, uv.y));
    
    // Right bar
    pauseSymbol += (step(center.x + 0.01, uv.x) - step(center.x + 0.01 + barWidth, uv.x)) * 
                   (step(center.y - barHeight * 0.5, uv.y) - step(center.y + barHeight * 0.5, uv.y));
    
    return vec3(overlay * 0.7 + pauseSymbol * 0.3);
}

// Draw a level transition effect
vec3 drawLevelTransition(vec2 uv, float progress) {
    // Radial transition - starts from center and expands
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    
    // Transition from 0 to 1 based on progress
    float transition = smoothstep(progress - 0.1, progress, dist);
    
    // Add a bright edge at the transition line
    float edge = 1.0 - smoothstep(progress - 0.02, progress + 0.02, abs(dist - progress));
    vec3 edgeColor = vec3(1.0, 1.0, 0.8) * edge;
    
    return vec3(transition) + edgeColor;
}

// Draw a game over screen
vec3 drawGameOver(vec2 uv, float time) {
    // Dark background
    vec3 bg = vec3(0.1, 0.05, 0.05);
    
    // "GAME OVER" text (simplified)
    vec2 textSize = vec2(0.1, 0.05);
    vec2 textPos = vec2(0.5 - 0.3, 0.5);
    
    // Simple character drawing for "GAME OVER"
    float text = 0.0;
    for(int i = 0; i < 4; i++) {
        vec2 charPos = textPos + vec2(float(i) * textSize.x * 1.2, 0.0);
        text += drawChar(uv, charPos, float(i));
    }
    for(int i = 0; i < 4; i++) {
        vec2 charPos = textPos + vec2(float(i) * textSize.x * 1.2, textSize.y * 1.5);
        text += drawChar(uv, charPos, float(i + 4));
    }
    
    // Pulsing effect
    float pulse = 0.8 + 0.2 * sin(time * 5.0);
    
    return bg * (1.0 - text) + vec3(1.0, pulse, pulse) * text;
}

// Draw a loading screen
vec3 drawLoadingScreen(vec2 uv, float progress) {
    // Dark background
    vec3 bg = vec3(0.1);
    
    // Progress bar
    vec2 barPos = vec2(0.25, 0.7);
    vec2 barSize = vec2(0.5, 0.03);
    
    float loading = drawProgressBar(uv, barPos, barSize, progress);
    
    // Add spinning indicator
    vec2 center = vec2(0.5, 0.5);
    float angle = atan(uv.y - center.y, uv.x - center.x);
    float dist = distance(uv, center);
    
    float spinner = (1.0 - smoothstep(0.05, 0.07, dist)) * 
                    step(0.7, sin((angle + iTime * 5.0) * 8.0));
    
    return bg * (1.0 - loading - spinner) + 
           vec3(0.2, 0.6, 1.0) * loading + 
           vec3(1.0, 0.8, 0.2) * spinner;
}

// Draw a win screen
vec3 drawWinScreen(vec2 uv, float time) {
    // Colorful animated background
    vec3 bg = vec3(
        sin(uv.x * 5.0 + time * 2.0) * 0.2 + 0.3,
        sin(uv.y * 4.0 + time * 1.5) * 0.2 + 0.4,
        sin((uv.x + uv.y) * 3.0 + time * 1.0) * 0.2 + 0.5
    );
    
    // "YOU WIN" text (simplified)
    vec2 textSize = vec2(0.1, 0.05);
    vec2 textPos = vec2(0.5 - 0.25, 0.5);
    
    float text = 0.0;
    for(int i = 0; i < 3; i++) { // "YOU"
        vec2 charPos = textPos + vec2(float(i) * textSize.x * 1.2, 0.0);
        text += drawChar(uv, charPos, float(i));
    }
    for(int i = 0; i < 3; i++) { // "WIN"
        vec2 charPos = textPos + vec2(float(i) * textSize.x * 1.2, textSize.y * 1.5);
        text += drawChar(uv, charPos, float(i + 3));
    }
    
    // Add celebration effects
    float confetti = sin(uv.x * 20.0 + time * 10.0) * sin(uv.y * 15.0 + time * 8.0);
    confetti = abs(confetti) * 0.2;
    
    return (bg + vec3(confetti)) * (1.0 - text) + vec3(1.0) * text;
}
"""


def main():
    # Find game/interactive shaders
    game_shaders = find_game_interactive_shaders()
    
    # Extract shader codes for a subset (first 500) for efficiency
    shader_codes = []
    for filepath, shader_info in game_shaders[:500]:  # Limit to first 500 for efficiency
        shader_code = extract_shader_code(filepath)
        shader_codes.append({
            'info': shader_info,
            'code': shader_code
        })
    
    # Analyze shader patterns
    analyzed_shaders, pattern_counts = analyze_game_shaders()
    
    # Extract specific game functions
    modules = extract_game_modules(analyzed_shaders)
    
    # Create standardized modules
    create_standardized_game_modules()
    
    print("Game/interactive shader analysis and module extraction completed!")


if __name__ == "__main__":
    main()