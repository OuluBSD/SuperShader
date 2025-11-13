// Effects module
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
