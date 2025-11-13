#!/usr/bin/env python3
"""
Post-Processing Effects Module
Extracted from common post-processing patterns in shader analysis
Pattern frequency: 142 occurrences
"""

# Pseudocode for post-processing effects
pseudocode = """
// Post-Processing Effects Implementation

// Basic tone mapping using Reinhard operator
vec3 reinhardToneMapping(vec3 color, float exposure) {
    color *= exposure;
    return color / (color + 1.0);
}

// ACES tone mapping (approximation)
vec3 acesToneMapping(vec3 color) {
    color *= 0.6;  // Scale down to prevent clipping
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

// Gamma correction
vec3 gammaCorrection(vec3 color, float gamma) {
    return pow(color, vec3(1.0 / gamma));
}

// Color grading with RGB scale factors
vec3 colorGrade(vec3 color, vec3 redScale, vec3 greenScale, vec3 blueScale) {
    // Apply individual channel scaling
    vec3 graded = color;
    graded.r = dot(graded, redScale);
    graded.g = dot(graded, greenScale);
    graded.b = dot(graded, blueScale);
    return graded;
}

// Simple color grading with lift, gamma, gain
vec3 liftGammaGain(vec3 color, vec3 lift, vec3 gamma, vec3 gain) {
    color = color * gain + lift;
    color = pow(color, 1.0 / gamma);
    return clamp(color, 0.0, 1.0);
}

// Vignette effect
vec3 vignette(vec2 uv, vec3 color, float strength, float falloff) {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    float vig = 1.0 - dist * strength;
    vig = pow(vig, falloff);
    return color * vig;
}

// Chromatic aberration
vec3 chromaticAberration(sampler2D source, vec2 uv, vec2 offset) {
    float r = texture(source, uv + offset).r;
    float g = texture(source, uv).g;  // No offset for green (center)
    float b = texture(source, uv - offset).b;
    return vec3(r, g, b);
}

// Film grain effect
vec3 filmGrain(vec3 color, vec2 uv, float time, float intensity) {
    // Simple noise function
    float noise = sin(uv.x * 100.0) * sin(uv.y * 77.0) * sin(time) * 0.1;
    noise = fract(noise);
    return color + noise * intensity;
}

// Depth of field effect (simplified)
vec3 depthOfField(sampler2D source, vec2 uv, sampler2D depthTexture, 
                  float focusPoint, float focusRange) {
    float depth = texture(depthTexture, uv).r;
    float coc = abs(depth - focusPoint) / focusRange;  // Circle of confusion
    coc = clamp(coc, 0.0, 1.0);
    
    // Simplified CoC-based blur
    vec3 color = texture(source, uv).rgb;
    if (coc > 0.1) {
        // Apply more blur for out-of-focus areas
        vec2 blurOffset = vec2(0.01, 0.0);
        color = (texture(source, uv).rgb + 
                 texture(source, uv + blurOffset).rgb + 
                 texture(source, uv - blurOffset).rgb) / 3.0;
    }
    return color;
}

// Complete post-processing pass combining multiple effects
vec4 postProcess(vec3 baseColor, vec2 uv, float time, float exposure, 
                 float gamma, float vignetteStrength) {
    vec3 color = baseColor;
    
    // Apply tone mapping
    color = reinhardToneMapping(color, exposure);
    
    // Apply gamma correction
    color = gammaCorrection(color, gamma);
    
    // Apply vignette
    color = vignette(uv, color, vignetteStrength, 2.0);
    
    // Apply film grain
    color = filmGrain(color, uv, time, 0.02);
    
    return vec4(color, 1.0);
}

// Full screen effect with multiple post-processing operations
vec4 fullScreenEffect(sampler2D source, vec2 uv, vec2 resolution, float time) {
    // Sample the original color
    vec3 color = texture(source, uv).rgb;
    
    // Apply post-processing effects
    color = reinhardToneMapping(color, 1.2);  // Tone mapping
    color = gammaCorrection(color, 2.2);      // Gamma correction
    color = vignette(uv, color, 0.3, 2.0);    // Vignette
    color = filmGrain(color, uv, time, 0.01); // Film grain
    
    return vec4(color, 1.0);
}
"""

def get_pseudocode():
    """Return the pseudocode for this post-processing effects module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'post_processing',
        'type': 'effects',
        'patterns': ['Post-Processing', 'Tone Mapping', 'Color Grading', 'Effects'],
        'frequency': 142,
        'dependencies': [],
        'conflicts': [],
        'description': 'Comprehensive post-processing effects including tone mapping, color grading, and film effects'
    }