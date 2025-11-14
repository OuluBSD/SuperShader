// Input Handling Module
// Implements input handling and interactive elements for games

// Mouse hover detection
float detectHover(vec2 fragCoord, vec2 mousePos, vec4 hoverArea) {
    // Check if fragment is within hover area
    bool inX = fragCoord.x > hoverArea.x && fragCoord.x < hoverArea.x + hoverArea.z;
    bool inY = fragCoord.y > hoverArea.y && fragCoord.y < hoverArea.y + hoverArea.w;
    
    return inX && inY ? 1.0 : 0.0;
}

// Click detection
float detectClick(vec2 fragCoord, vec2 mousePos, vec4 hoverArea, float clickState) {
    // First check if in hover area
    float hover = detectHover(fragCoord, mousePos, hoverArea);
    
    // Then check if clicked
    return hover * clickState;
}

// Interactive element highlighting
float highlightElement(vec2 fragCoord, vec2 mousePos, vec4 hoverArea, float clickState, float time) {
    float hover = detectHover(fragCoord, mousePos, hoverArea);
    float click = detectClick(fragCoord, mousePos, hoverArea, clickState);
    
    // Create highlight effect
    float highlight = hover * 0.2 + click * 0.4;
    
    // Add pulsing effect when hovered
    highlight += hover * 0.1 * sin(time * 5.0);
    
    return highlight;
}

// Button element with visual feedback
vec4 buttonElement(vec2 fragCoord, vec2 mousePos, vec2 buttonPos, vec2 buttonSize, float clickState, float time) {
    // Calculate normalized position within button
    vec2 pos = (fragCoord - buttonPos) / buttonSize;
    
    // Check bounds
    float inButton = step(0.0, pos.x) * step(0.0, pos.y) * 
                     (1.0 - step(1.0, pos.x)) * (1.0 - step(1.0, pos.y));
    
    // Determine visual state
    vec2 centerPos = (pos - 0.5) * 2.0;
    float dist = length(centerPos);
    float roundness = 1.0 - smoothstep(0.9, 1.0, dist);
    
    // Calculate hover and click
    vec2 absPos = buttonPos + pos * buttonSize;
    float hover = detectHover(fragCoord, mousePos, vec4(buttonPos, buttonSize));
    float click = hover * clickState;
    
    // Create button color based on state
    vec3 buttonColor = vec3(0.3, 0.4, 0.6);  // Default
    buttonColor += vec3(0.2, 0.2, 0.2) * hover;  // Highlight when hovered
    buttonColor -= vec3(0.1, 0.1, 0.1) * click;  // Pressed effect
    
    // Add glow when hovered
    float glow = hover * 0.3 * (1.0 - smoothstep(0.2, 0.8, dist));
    
    return vec4(buttonColor + vec3(glow), inButton * roundness);
}

// Crosshair generation
vec3 generateCrosshair(vec2 fragCoord, vec2 mousePos, float time) {
    vec2 center = mousePos;
    vec2 halfSize = fragCoord - center;
    
    // Create vertical line
    float vLine = 1.0 - smoothstep(0.0, 2.0, abs(halfSize.x));
    vLine *= 1.0 - smoothstep(5.0, 20.0, abs(halfSize.y));
    
    // Create horizontal line
    float hLine = 1.0 - smoothstep(0.0, 2.0, abs(halfSize.y));
    hLine *= 1.0 - smoothstep(5.0, 20.0, abs(halfSize.x));
    
    // Combine lines
    float crosshair = max(vLine, hLine);
    
    // Add pulsing effect
    crosshair += 0.2 * sin(time * 8.0) * crosshair;
    
    return vec3(crosshair);
}

// Health bar visualization
vec3 healthBar(vec2 fragCoord, float health, vec2 barPos, vec2 barSize, float time) {
    vec2 pos = (fragCoord - barPos) / barSize;
    
    // Check bounds
    float inBar = step(0.0, pos.x) * step(0.0, pos.y) * 
                  (1.0 - step(1.0, pos.x)) * (1.0 - step(1.0, pos.y));
    
    // Current fill level based on health
    float fill = step(0.0, pos.x - (1.0 - health));
    
    // Create color gradient (red to green based on health)
    vec3 healthColor = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), health);
    
    // Add damage flash effect
    float damageFlash = step(health, 0.3) * 0.5 * (sin(time * 20.0) * 0.5 + 0.5);
    healthColor += vec3(damageFlash);
    
    return healthColor * fill * inBar;
}

// Score display
vec3 scoreDisplay(vec2 fragCoord, int score, vec2 displayPos, vec2 displaySize) {
    vec2 pos = (fragCoord - displayPos) / displaySize;
    
    // Check bounds
    float inDisplay = step(0.0, pos.x) * step(0.0, pos.y) * 
                      (1.0 - step(1.0, pos.x)) * (1.0 - step(1.0, pos.y));
    
    // This is a simplified version
    // In a real implementation, you'd map each digit to a texture
    // or draw numbers with distance fields
    float scoreVisual = inDisplay * 0.7;
    
    return vec3(scoreVisual);
}

// Selection highlight
vec3 selectionHighlight(vec2 fragCoord, float isSelected, float time) {
    // Create a pulsing border effect
    vec2 borderSize = vec2(0.98);
    vec2 innerSize = vec2(0.96);
    
    vec2 normCoord = abs(fragCoord - 0.5) * 2.0;  // Normalize to 0-1 around center
    
    float outer = max(normCoord.x, normCoord.y);
    float inner = max(normCoord.x * borderSize.x, normCoord.y * borderSize.y);
    float inner2 = max(normCoord.x * innerSize.x, normCoord.y * innerSize.y);
    
    float border = (1.0 - step(inner, outer)) * step(inner2, outer);
    
    // Pulsing effect when selected
    float pulse = isSelected * (0.5 + 0.5 * sin(time * 4.0));
    
    vec3 highlightColor = vec3(1.0, 1.0, 0.0);  // Yellow highlight
    
    return highlightColor * border * pulse;
}

// Touch input visualization
vec3 touchVisualization(vec2 fragCoord, vec2 touchPos, float touchRadius, float time) {
    float dist = distance(fragCoord, touchPos);
    
    // Visualize touch as a growing circle
    float touchCircle = 1.0 - smoothstep(touchRadius - 2.0, touchRadius, dist);
    
    // Add ripple effect
    float ripple = sin((dist - time * 50.0) * 0.2) * 0.5 + 0.5;
    float rippleEffect = ripple * (1.0 - smoothstep(touchRadius, touchRadius + 10.0, dist));
    
    vec3 touchColor = vec3(0.0, 0.8, 1.0);  // Cyan touch indicator
    touchColor += vec3(rippleEffect * 0.3);
    
    return touchColor * touchCircle;
}

// UI glow effect
float uiGlow(vec2 fragCoord, vec2 elementPos, vec2 elementSize, float time) {
    vec2 center = elementPos + elementSize * 0.5;
    float dist = distance(fragCoord, center);
    float elementDist = length((fragCoord - elementPos) / elementSize - vec2(0.5));
    
    // Only glow outside the element
    float outside = 1.0 - step(0.5, elementDist);
    
    // Create pulsing glow
    float glow = outside * (sin(time * 3.0) * 0.1 + 0.1);
    glow *= 1.0 - smoothstep(0.0, 20.0, dist);
    
    return glow;
}