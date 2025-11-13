# ui_elements

**Category:** game
**Type:** standardized

## Dependencies
texture_sampling

## Tags
texturing, game

## Code
```glsl
// UI elements module
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

```