#!/usr/bin/env python3
"""
Ray Generation Module for Raymarching
Extracted from common raymarching patterns in shader analysis
Pattern frequency: 189 occurrences
"""

# Pseudocode for ray generation
pseudocode = """
// Ray Generation Implementation

// Generate a ray from camera position through a screen pixel
vec3 getRayDirection(vec2 uv, vec3 cameraPos, vec3 cameraTarget) {
    vec3 forward = normalize(cameraTarget - cameraPos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));

    vec3 rayDir = normalize(forward + uv.x * right + uv.y * up);
    return rayDir;
}

// Generate a ray with FOV consideration
vec3 getRayDirectionWithFOV(vec2 uv, vec3 rd, float fov) {
    rd = normalize(rd + fov * uv.x * vec3(1, 0, 0) + fov * uv.y * vec3(0, 1, 0));
    return rd;
}

// Generate primary ray with aspect ratio correction
vec3 generatePrimaryRay(vec2 fragCoord, vec2 resolution, vec3 cameraPos, vec3 cameraTarget) {
    vec2 uv = (fragCoord - 0.5 * resolution.xy) / resolution.y;
    return getRayDirection(uv, cameraPos, cameraTarget);
}

// Generate ray with lens distortion
vec3 generateDistortedRay(vec2 uv, vec3 cameraPos, vec3 cameraTarget, float distortion) {
    vec3 forward = normalize(cameraTarget - cameraPos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));

    // Apply radial distortion
    float r2 = dot(uv, uv);
    float distortionFactor = 1.0 + r2 * distortion;
    vec2 distortedUV = uv * distortionFactor;

    vec3 rayDir = normalize(forward + distortedUV.x * right + distortedUV.y * up);
    return rayDir;
}

// Generate ray for fisheye effect
vec3 generateFisheyeRay(vec2 uv, vec3 cameraPos, vec3 cameraTarget) {
    vec3 forward = normalize(cameraTarget - cameraPos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));

    // Convert to polar coordinates for fisheye
    float r = length(uv);
    float phi = atan(uv.y, uv.x);
    
    // Apply fisheye mapping
    float theta = r * 3.14159;
    vec3 rayDir = normalize(
        sin(theta) * (cos(phi) * right + sin(phi) * up) + 
        cos(theta) * forward
    );
    
    return rayDir;
}

// Generate multiple rays for anti-aliasing
vec3[] generateAARays(vec2 fragCoord, vec2 resolution, vec3 cameraPos, vec3 cameraTarget, int samples) {
    vec3[] rays = new vec3[samples];
    vec2 pixelSize = 1.0 / resolution;
    
    for (int i = 0; i < samples; i++) {
        // Jitter sample position within the pixel
        vec2 jitter = vec2(
            (float(i) * 1.618033988749) - float(int(float(i) * 1.618033988749)),
            float(i + 1) / float(samples)
        );
        
        vec2 sampleCoord = fragCoord + (jitter - 0.5) * pixelSize;
        rays[i] = getRayDirection(
            (sampleCoord - 0.5 * resolution.xy) / resolution.y,
            cameraPos,
            cameraTarget
        );
    }
    
    return rays;
}

// Generate ray with motion blur effect
vec3 generateMotionBlurRay(vec2 uv, vec3 cameraPos, vec3 cameraTarget, float motionVector) {
    vec3 forward = normalize(cameraTarget - cameraPos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));
    
    // Apply motion vector for motion blur
    vec2 motionUV = uv + motionVector;
    vec3 rayDir = normalize(forward + motionUV.x * right + motionUV.y * up);
    return rayDir;
}
"""

def get_pseudocode():
    """Return the pseudocode for this ray generation module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'ray_generation',
        'type': 'raymarching',
        'patterns': ['Ray Generation', 'Camera', 'Perspective'],
        'frequency': 189,
        'dependencies': [],
        'conflicts': [],
        'description': 'Ray generation functions for creating primary and special effect rays'
    }