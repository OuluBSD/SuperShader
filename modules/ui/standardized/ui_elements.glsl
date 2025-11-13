// UI elements module
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
