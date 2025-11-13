#!/usr/bin/env python3
"""
Procedural Texturing Module
Extracted from common procedural generation patterns in shader analysis
Pattern frequency: 178 occurrences
"""

# Pseudocode for procedural texturing
pseudocode = """
// Procedural Texturing Implementation

// Classic Perlin noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Four corners in 2D of a tile
    vec2 a = vec2(i.x, i.y);
    vec2 b = vec2(i.x + 1.0, i.y);
    vec2 c = vec2(i.x, i.y + 1.0);
    vec2 d = vec2(i.x + 1.0, i.y + 1.0);

    // Random values at the corners
    float va = dot(vec2(cos(dot(a, vec2(12.9898, 78.233))), sin(dot(a, vec2(12.9898, 78.233)))), f - vec2(0.0, 0.0));
    float vb = dot(vec2(cos(dot(b, vec2(12.9898, 78.233))), sin(dot(b, vec2(12.9898, 78.233)))), f - vec2(1.0, 0.0));
    float vc = dot(vec2(cos(dot(c, vec2(12.9898, 78.233))), sin(dot(c, vec2(12.9898, 78.233)))), f - vec2(0.0, 1.0));
    float vd = dot(vec2(cos(dot(d, vec2(12.9898, 78.233))), sin(dot(d, vec2(12.9898, 78.233)))), f - vec2(1.0, 1.0));

    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
}

// Fractional Brownian Motion (fBm)
float fbm(vec2 p, int octaves, float lacunarity, float gain) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise(p * frequency);
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return value;
}

// Simplex noise (simplified version)
float simplexNoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Helper function for permutations (needed for simplex noise)
vec3 permute(vec3 x) {
    return mod(((x * 34.0) + 1.0) * x, 289.0);
}

// Checkerboard pattern
float checker(vec2 p) {
    vec2 i = floor(p);
    return mod(i.x + i.y, 2.0);
}

// Gradient pattern
float gradient(vec2 p, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    mat2 rot = mat2(c, -s, s, c);
    vec2 rotated = rot * p;
    return fract(rotated.x * 5.0);
}

// Radial pattern
float radial(vec2 p, float segments) {
    float angle = atan(p.y, p.x);
    float sector = floor(angle * segments / (2.0 * 3.14159));
    return fract(sector / segments);
}

// Cellular/F1 noise (simplified)
float cellular(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float min_dist = 1.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(x, y);
            vec2 point = i + neighbor + 0.5 + 0.5 * sin(vec2(i.x + x, i.y + y) * 10.0);
            vec2 diff = neighbor + 0.5 + 0.5 * sin(vec2(i.x + x, i.y + y) * 10.0) - f;
            float dist = length(diff);
            min_dist = min(min_dist, dist);
        }
    }

    return min_dist;
}

// Wood texture pattern
float wood(vec2 p) {
    float angle = atan(p.y, p.x);
    float radius = length(p);
    float rings = radius * 10.0;
    float ring_noise = noise(vec2(rings, angle * 5.0));
    return fract(rings + ring_noise);
}

// Marble texture pattern
float marble(vec2 p, float scale) {
    float noise_val = fbm(p * scale, 4, 2.0, 0.5);
    return abs(sin((p.x + noise_val) * 3.0));
}

// Procedural brick pattern
vec3 brick(vec2 p, vec2 brickSize, vec2 mortarSize) {
    vec2 c = mod(p, brickSize) - brickSize * 0.5;
    vec2 m = step(c, mortarSize);
    float f = m.x * m.y;
    return vec3(f);
}

// Multiple procedural patterns combined
vec3 proceduralTexture(vec2 uv, float time) {
    // Create a complex procedural texture by combining multiple patterns
    float n1 = noise(uv * 5.0);
    float n2 = fbm(uv * 3.0, 4, 2.0, 0.5);
    float n3 = cellular(uv * 4.0);
    
    // Combine with time animation
    float pattern = mix(n1, n2, 0.5) + 0.3 * sin(n3 * 10.0 + time);
    
    // Convert to color
    return vec3(pattern * 0.5 + 0.5);
}
"""

def get_pseudocode():
    """Return the pseudocode for this procedural texturing module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'procedural_texturing',
        'type': 'texturing',
        'patterns': ['Procedural', 'Noise', 'fBm', 'Patterns'],
        'frequency': 178,
        'dependencies': [],
        'conflicts': [],
        'description': 'Procedural texturing functions including noise, patterns, and procedural generation'
    }