#!/usr/bin/env python3
'''
Perlin Noise Module with Interface Definition
Extracted from common procedural generation patterns in shader analysis
Pattern frequency: 200 occurrences
'''

# Interface definition
INTERFACE = {
    'inputs': [
        {'name': 'uv', 'type': 'vec2', 'direction': 'in', 'semantic': 'texture_coordinates'},
        {'name': 'scale', 'type': 'float', 'direction': 'uniform', 'semantic': 'noise_scale'},
        {'name': 'time', 'type': 'float', 'direction': 'uniform', 'semantic': 'time_parameter'}
    ],
    'outputs': [
        {'name': 'noiseValue', 'type': 'float', 'direction': 'out', 'semantic': 'noise_output'}
    ],
    'uniforms': [
        {'name': 'scale', 'type': 'float', 'semantic': 'noise_scale'},
        {'name': 'time', 'type': 'float', 'semantic': 'time_parameter'}
    ]
}

# Pseudocode for Perlin noise generation
pseudocode = '''
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

def get_interface():
    '''Return the interface definition for this module'''
    return INTERFACE

def get_pseudocode():
    '''Return the pseudocode for this procedural module'''
    return pseudocode

def get_metadata():
    '''Return metadata about this module'''
    return {
        'name': 'perlin_noise',
        'type': 'procedural',
        'patterns': ['Perlin Noise', 'Fractal Brownian Motion'],
        'frequency': 200,
        'dependencies': [],
        'conflicts': [],
        'description': 'Perlin noise generation with FBM',
        'interface': INTERFACE
    }