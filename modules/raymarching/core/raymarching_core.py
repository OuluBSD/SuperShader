#!/usr/bin/env python3
"""
Core Raymarching Module
Extracted from common raymarching patterns in shader analysis
Pattern frequency: 245 occurrences
"""

# Pseudocode for core raymarching algorithm
pseudocode = """
// Core Raymarching Implementation

// Basic raymarching function
vec2 raymarch(vec3 ro, vec3 rd, float maxDist, int maxSteps) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled

    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        d = map(p).x;
        if(d < 0.001 || t > maxDist) break;
        t += d;
    }

    return vec2(t, d);
}

// Raymarching with adaptive step size
vec2 raymarchAdaptive(vec3 ro, vec3 rd, float maxDist, int maxSteps) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled
    float f = 1.0; // Adaptive factor

    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        d = map(p).x;
        if(d < 0.001 || t > maxDist) break;

        // Adaptive step size based on distance
        float adaptiveStep = d * f;
        t += max(0.01, adaptiveStep);

        // Reduce the factor as we get closer to surface
        f = 0.5 + 0.5 * min(1.0, d * 4.0);
    }

    return vec2(t, d);
}

// Raymarching with early termination optimization
vec2 raymarchOptimized(vec3 ro, vec3 rd, float maxDist, int maxSteps) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled

    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        d = map(p).x;

        // Early termination if we're very close to surface
        if(d < 0.0001) return vec2(t, d);

        // Stop if we've gone too far
        if(t > maxDist) return vec2(maxDist, d);

        t += d;
    }

    return vec2(t, d);
}

// Multi-raymarching for enhanced quality
vec2 raymarchMulti(vec3 ro, vec3 rd, float maxDist, int maxSteps, float jitter) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled

    // Add jitter to the ray direction for anti-aliasing
    vec3 jittered_rd = normalize(rd + jitter * vec3(
        sin(jitter * ro.x),
        cos(jitter * ro.y),
        sin(jitter * ro.z)
    ));

    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + jittered_rd * t;
        d = map(p).x;
        if(d < 0.001 || t > maxDist) break;
        t += d * 0.8; // Slightly reduce step size for safety
    }

    return vec2(t, d);
}

// Function to calculate normal from distance field
vec3 calculateNormal(vec3 p, float epsilon) {
    vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x
    ));
}

// Function to calculate normal with adaptive epsilon
vec3 calculateNormalAdaptive(vec3 p) {
    float dx = map(p + vec3(0.001, 0.0, 0.0)).x - map(p - vec3(0.001, 0.0, 0.0)).x;
    float dy = map(p + vec3(0.0, 0.001, 0.0)).x - map(p - vec3(0.0, 0.001, 0.0)).x;
    float dz = map(p + vec3(0.0, 0.0, 0.001)).x - map(p - vec3(0.0, 0.0, 0.001)).x;
    return normalize(vec3(dx, dy, dz));
}
"""

def get_pseudocode():
    """Return the pseudocode for this raymarching core module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'raymarching_core',
        'type': 'raymarching',
        'patterns': ['Raymarching', 'Distance Field', 'Marching Algorithm'],
        'frequency': 245,
        'dependencies': ['distance_functions'],
        'conflicts': [],
        'description': 'Core raymarching algorithms with adaptive stepping and optimizations'
    }