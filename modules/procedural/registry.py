#!/usr/bin/env python3
"""
Registry for procedural modules
Provides access to procedural modules with their interfaces and metadata
"""

def get_module_by_name(module_name):
    """
    Get a procedural module by name
    """
    # This is a simplified registry - in a full implementation, 
    # this would load from actual module files
    modules = {
        'perlin_noise': {
            'name': 'perlin_noise',
            'type': 'procedural',
            'patterns': ['Perlin Noise', 'Fractal Brownian Motion'],
            'frequency': 200,
            'dependencies': [],
            'conflicts': [],
            'description': 'Perlin noise generation with FBM',
            'pseudocode': '''
// Perlin Noise Implementation
float perlinNoise(vec2 coord, float scale, float time) {
    // Scale the coordinates
    vec2 scaledCoord = coord * scale;

    // Calculate integer and fractional parts
    vec2 i = floor(scaledCoord);
    vec2 f = fract(scaledCoord);

    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);

    // Generate random values at the four corners
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Interpolate between the values
    float value = mix(a, b, u.x) +
                  (c - a) * u.y * (1.0 - u.x) +
                  (d - b) * u.x * u.y;

    return value;
}

// Random function for noise generation
float random(vec2 coord) {
    return fract(sin(dot(coord, vec2(12.9898, 78.233))) * 43758.5453);
}

// Fractal Brownian Motion combining multiple octaves
float fbm(vec2 coord, float scale, float time) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < 4; i++) {
        value += amplitude * perlinNoise(coord * frequency, scale, time);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}
'''
        }
    }

    return modules.get(module_name)