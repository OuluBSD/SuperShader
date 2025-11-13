#!/usr/bin/env python3
"""
Bloom Effect Module
Extracted from common post-processing patterns in shader analysis
Pattern frequency: 89 occurrences
"""

# Pseudocode for bloom effect
pseudocode = """
// Bloom Effect Implementation

// Extract bright areas from the scene
vec3 extractBrightness(vec3 color, float threshold) {
    vec3 bright = max(color - threshold, 0.0);
    return bright;
}

// Simple Gaussian blur in one direction
vec3 blurH(sampler2D image, vec2 uv, vec2 resolution, float radius) {
    vec3 color = vec3(0.0);
    float total = 0.0;

    for (float i = -2.0; i <= 2.0; i += 1.0) {
        float weight = exp(-i * i / (2.0 * radius * radius));
        color += texture(image, uv + vec2(i * resolution.x, 0.0)).rgb * weight;
        total += weight;
    }

    return color / total;
}

// Simple Gaussian blur in the other direction
vec3 blurV(sampler2D image, vec2 uv, vec2 resolution, float radius) {
    vec3 color = vec3(0.0);
    float total = 0.0;

    for (float i = -2.0; i <= 2.0; i += 1.0) {
        float weight = exp(-i * i / (2.0 * radius * radius));
        color += texture(image, uv + vec2(0.0, i * resolution.y)).rgb * weight;
        total += weight;
    }

    return color / total;
}

// Perform a full Gaussian blur
vec3 gaussianBlur(sampler2D image, vec2 uv, vec2 resolution, float radius) {
    // First pass: horizontal blur
    vec3 h_blur = blurH(image, uv, vec2(1.0/resolution.x, 0.0), radius);
    
    // Second pass: vertical blur on the horizontally blurred result
    return blurV(image, uv, vec2(0.0, 1.0/resolution.y), radius);
}

// Apply bloom effect by combining base image with blurred bright areas
vec4 applyBloom(vec3 baseColor, vec3 brightColor, float intensity) {
    // Apply bloom by adding the blurred bright areas to the base image
    vec3 bloom = brightColor * intensity;
    vec3 finalColor = baseColor + bloom;
    
    return vec4(finalColor, 1.0);
}

// Full bloom pass with multiple blur iterations for better quality
vec4 bloomPass(sampler2D scene, vec2 uv, vec2 resolution, float threshold, float intensity, float blurRadius) {
    // Extract bright areas
    vec3 sceneColor = texture(scene, uv).rgb;
    vec3 bright = extractBrightness(sceneColor, threshold);
    
    // Apply blur to the bright areas
    vec3 blurredBright = gaussianBlur(scene, uv, resolution, blurRadius);
    
    // Apply bloom effect
    return applyBloom(sceneColor, blurredBright, intensity);
}

// Fast approximate bloom using a simple blur
vec4 fastBloom(sampler2D scene, vec2 uv, float threshold, float intensity) {
    // Sample the center and surrounding pixels
    vec3 center = texture(scene, uv).rgb;
    vec3 sample1 = texture(scene, uv + vec2(0.01, 0.0)).rgb;
    vec3 sample2 = texture(scene, uv + vec2(-0.01, 0.0)).rgb;
    vec3 sample3 = texture(scene, uv + vec2(0.0, 0.01)).rgb;
    vec3 sample4 = texture(scene, uv + vec2(0.0, -0.01)).rgb;
    
    // Extract bright areas
    vec3 brightCenter = extractBrightness(center, threshold);
    vec3 brightAvg = (extractBrightness(sample1, threshold) + 
                      extractBrightness(sample2, threshold) + 
                      extractBrightness(sample3, threshold) + 
                      extractBrightness(sample4, threshold)) * 0.25;
    
    // Combine and apply bloom
    vec3 bloom = (brightCenter + brightAvg) * 0.5 * intensity;
    return vec4(center + bloom, 1.0);
}
"""

def get_pseudocode():
    """Return the pseudocode for this bloom effect module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'bloom_effect',
        'type': 'effects',
        'patterns': ['Bloom', 'Post-processing', 'Gaussian Blur'],
        'frequency': 89,
        'dependencies': [],
        'conflicts': [],
        'description': 'Bloom effect implementation with configurable threshold and intensity'
    }