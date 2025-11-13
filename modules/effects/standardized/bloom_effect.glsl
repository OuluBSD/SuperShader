// Bloom effect module
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
