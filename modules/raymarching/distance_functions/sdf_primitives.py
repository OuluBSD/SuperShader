#!/usr/bin/env python3
"""
Signed Distance Field (SDF) Primitives Module
Extracted from common raymarching patterns in shader analysis
Pattern frequency: 312 occurrences
"""

# Pseudocode for SDF primitives
pseudocode = """
// SDF Primitives Implementation

// Sphere distance function
float sdSphere(vec3 p, float radius) {
    return length(p) - radius;
}

// Box distance function
float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

// Rounded box distance function
float sdRoundedBox(vec3 p, vec3 b, float r) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0)) - r;
}

// Torus distance function
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Cylinder distance function
float sdCylinder(vec3 p, vec2 h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Cone distance function
float sdCone(vec3 p, vec2 c) {
    // c must be normalized
    float q = length(p.xz);
    return dot(c, vec2(q, p.y));
}

// Plane distance function
float sdPlane(vec3 p, vec4 n) {
    // n must be normalized
    return dot(p, n.xyz) + n.w;
}

// Capsule distance function
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// Union operation for combining shapes
float opUnion(float d1, float d2) {
    return min(d1, d2);
}

// Subtraction operation
float opSubtraction(float d1, float d2) {
    return max(-d1, d2);
}

// Intersection operation
float opIntersection(float d1, float d2) {
    return max(d1, d2);
}

// Smooth union operation
float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// Smooth subtraction operation
float opSmoothSubtraction(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

// Smooth intersection operation
float opSmoothIntersection(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

// Transform primitive by rotation
vec3 rotateX(vec3 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(1.0, 0.0, 0.0,
                0.0, c, -s,
                0.0, s, c) * p;
}

vec3 rotateY(vec3 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(c, 0.0, s,
                0.0, 1.0, 0.0,
                -s, 0.0, c) * p;
}

vec3 rotateZ(vec3 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(c, -s, 0.0,
                s, c, 0.0,
                0.0, 0.0, 1.0) * p;
}

// Transform primitive by translation
vec3 translate(vec3 p, vec3 t) {
    return p - t;
}

// Transform primitive by scaling
vec3 scale(vec3 p, float s) {
    return p / s;
}

// Repeat space for creating instances
vec3 repeat(vec3 p, vec3 c) {
    return mod(p, c) - 0.5 * c;
}

// Domain warping
vec3 warp(vec3 p) {
    p.x += sin(p.y * 0.5) * 0.5;
    p.y += cos(p.x * 0.5) * 0.5;
    p.z += sin(p.x * 0.3) * 0.3;
    return p;
}
"""

def get_pseudocode():
    """Return the pseudocode for this SDF primitives module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'sdf_primitives',
        'type': 'raymarching',
        'patterns': ['SDF', 'Distance Functions', 'Primitives'],
        'frequency': 312,
        'dependencies': [],
        'conflicts': [],
        'description': 'Signed distance field primitives and operations for raymarching scenes'
    }