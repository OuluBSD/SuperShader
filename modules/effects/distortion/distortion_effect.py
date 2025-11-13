#!/usr/bin/env python3
"""
Distortion Effect Module
Extracted from common distortion patterns in shader analysis
Pattern frequency: 76 occurrences
"""

# Pseudocode for distortion effect
pseudocode = """
// Distortion Effect Implementation

// Simple barrel distortion
vec2 barrelDistortion(vec2 uv, float strength) {
    vec2 center = uv - 0.5;
    float dist = length(center);
    float distortedDist = dist + strength * dist * dist * dist;
    vec2 distortedCenter = normalize(center) * distortedDist;
    return distortedCenter + 0.5;
}

// Pincushion distortion
vec2 pincushionDistortion(vec2 uv, float strength) {
    vec2 center = uv - 0.5;
    float dist = length(center);
    float distortedDist = dist - strength * dist * dist * dist;
    vec2 distortedCenter = normalize(center) * distortedDist;
    return distortedCenter + 0.5;
}

// Wave distortion effect
vec2 waveDistortion(vec2 uv, float time, float amplitude) {
    float waveX = sin(uv.y * 10.0 + time) * amplitude;
    float waveY = cos(uv.x * 10.0 + time) * amplitude;
    return uv + vec2(waveX, waveY);
}

// Ripple distortion effect
vec2 rippleDistortion(vec2 uv, vec2 center, float time, float frequency, float amplitude) {
    vec2 offset = uv - center;
    float distance = length(offset);
    float ripple = sin(distance * frequency - time * 3.0) * amplitude / (distance + 0.5);
    return uv + normalize(offset) * ripple;
}

// Heat shimmer distortion
vec2 heatShimmer(vec2 uv, float time, float intensity) {
    float noise1 = sin(uv.y * 20.0 + time * 2.0) * 0.01 * intensity;
    float noise2 = cos(uv.x * 15.0 + time * 1.5) * 0.01 * intensity;
    return uv + vec2(noise1, noise2);
}

// Refraction distortion (simulating glass/water effect)
vec2 refractionDistortion(sampler2D normalMap, vec2 uv, vec2 distortionOffset) {
    // Sample the normal map to get distortion direction
    vec3 normalData = texture(normalMap, uv).rgb;
    vec2 normalXY = normalData.rg * 2.0 - 1.0;  // Convert from [0,1] to [-1,1]
    
    // Apply the distortion based on the normal
    vec2 distortedUV = uv + normalXY * distortionOffset;
    return distortedUV;
}

// Apply general distortion to a texture
vec4 applyDistortion(sampler2D source, vec2 uv, vec2 distortionVector) {
    vec2 distortedUV = uv + distortionVector;
    return texture(source, distortedUV);
}

// Multiple distortion effects combined
vec2 combinedDistortions(vec2 uv, float time, float strength) {
    vec2 result = uv;
    
    // Apply barrel distortion
    result = barrelDistortion(result, strength * 0.1);
    
    // Apply wave distortion
    result = waveDistortion(result, time, strength * 0.02);
    
    // Apply heat shimmer
    result = heatShimmer(result, time, strength * 0.5);
    
    return result;
}

// Chromatic aberration effect (color separation)
vec3 chromaticAberration(sampler2D source, vec2 uv, vec2 offset) {
    float r = texture(source, uv + offset).r;
    float g = texture(source, uv).g;  // No offset for green (center)
    float b = texture(source, uv - offset).b;
    
    return vec3(r, g, b);
}

// Time-based animated distortion
vec2 animatedDistortion(vec2 uv, float time, float speed, float amplitude) {
    float t = time * speed;
    
    // Create complex distortion pattern
    float distortionX = sin(uv.y * 5.0 + t) * cos(uv.x * 3.0 + t * 0.5) * amplitude;
    float distortionY = cos(uv.x * 4.0 + t * 1.5) * sin(uv.y * 6.0 + t * 0.3) * amplitude;
    
    return uv + vec2(distortionX, distortionY);
}
"""

def get_pseudocode():
    """Return the pseudocode for this distortion effect module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'distortion_effect',
        'type': 'effects',
        'patterns': ['Distortion', 'Barrel', 'Pincushion', 'Waves'],
        'frequency': 76,
        'dependencies': [],
        'conflicts': [],
        'description': 'Various distortion effects including barrel, wave, and animated distortions'
    }