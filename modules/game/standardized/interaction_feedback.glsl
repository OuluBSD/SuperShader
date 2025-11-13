// Interaction feedback module
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
