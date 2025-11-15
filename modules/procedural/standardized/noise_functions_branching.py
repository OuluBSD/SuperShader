#!/usr/bin/env python3
'''
Procedural Noise Module with Branching for Conflicting Features
This module demonstrates different noise generation algorithms with branching for conflicting features
'''

# Interface definition with branching options
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
    ],
    'branches': {
        'noise_algorithm': {
            'perlin': {
                'name': 'Perlin Noise',
                'description': 'Classic Perlin noise with smooth interpolation',
                'requires': [],
                'conflicts': ['simplex', 'value']
            },
            'simplex': {
                'name': 'Simplex Noise',
                'description': 'Improved Simplex noise with better performance',
                'requires': [],
                'conflicts': ['perlin', 'value']
            },
            'value': {
                'name': 'Value Noise',
                'description': 'Simple value noise based on random gradients',
                'requires': [],
                'conflicts': ['perlin', 'simplex']
            }
        },
        'octave_mode': {
            'fbm': {
                'name': 'Fractal Brownian Motion',
                'description': 'Standard FBM combining multiple octaves',
                'requires': [],
                'conflicts': ['ridged', 'turbulence']
            },
            'ridged': {
                'name': 'Ridged Multi-Fractal',
                'description': 'Creates ridged, mountain-like patterns',
                'requires': [],
                'conflicts': ['fbm', 'turbulence']
            },
            'turbulence': {
                'name': 'Turbulence',
                'description': 'Abs-based turbulence pattern',
                'requires': [],
                'conflicts': ['fbm', 'ridged']
            }
        }
    }
}

# Pseudocode for different noise algorithms
pseudocode = {
    'perlin': '''
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
    ''',
    
    'simplex': '''
// Simplex Noise Implementation
float simplexNoise(vec2 coord, float scale, float time) {
    // Scale the coordinates
    vec2 scaledCoord = coord * scale;

    // Skew the input space to determine which simplex cell we're in
    const vec2 F2 = vec2(0.366025403, 0.211324865); // (sqrt(3) - 1) / 2
    const vec2 G2 = vec2(0.211324865, 0.133974596); // (3 - sqrt(3)) / 2
    
    vec2 skewed = (scaledCoord.x + scaledCoord.y) * F2;
    vec2 i = floor(scaledCoord + skewed);
    
    vec2 unskewed = (i.x + i.y) * G2;
    vec2 origin = i - unskewed;
    vec2 delta0 = scaledCoord - origin;
    
    // Determine which simplex we're in
    vec2 i1, i2;
    if (delta0.x > delta0.y) {
        i1 = vec2(1.0, 0.0);
        i2 = vec2(1.0, 1.0);
    } else {
        i1 = vec2(0.0, 1.0);
        i2 = vec2(1.0, 1.0);
    }
    
    // Calculate the contribution from the three corners
    vec2 x1 = delta0 - i1 + G2;
    vec2 x2 = delta0 - i2 + 2.0 * G2;
    
    float n0 = gradCoord(i, delta0);
    float n1 = gradCoord(i + i1, x1);
    float n2 = gradCoord(i + i2, x2);
    
    // Compute the fade curve value for x
    float t0 = 0.5 - dot(delta0, delta0);
    if (t0 < 0.0) t0 = 0.0;
    t0 *= t0;
    float n0_contrib = t0 * t0 * n0;
    
    float t1 = 0.5 - dot(x1, x1);
    if (t1 < 0.0) t1 = 0.0;
    t1 *= t1;
    float n1_contrib = t1 * t1 * n1;
    
    float t2 = 0.5 - dot(x2, x2);
    if (t2 < 0.0) t2 = 0.0;
    t2 *= t2;
    float n2_contrib = t2 * t2 * n2;
    
    return 40.0 * (n0_contrib + n1_contrib + n2_contrib);
}

// Gradient function for Simplex noise
float gradCoord(vec2 i, vec2 coord) {
    float hash = random(i);
    float gradient = dot(coord, vec2(cos(hash * 6.28318531), sin(hash * 6.28318531)));
    return gradient;
}
    ''',
    
    'value': '''
// Value Noise Implementation
float valueNoise(vec2 coord, float scale, float time) {
    // Scale the coordinates
    vec2 scaledCoord = coord * scale;

    // Calculate integer and fractional parts
    vec2 i = floor(scaledCoord);
    vec2 f = fract(scaledCoord);

    // Linear interpolation
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Interpolate between the values
    float ix0 = mix(a, b, f.x);
    float ix1 = mix(c, d, f.x);
    float value = mix(ix0, ix1, f.y);

    return value;
}

// Random function for noise generation
float random(vec2 coord) {
    return fract(sin(dot(coord, vec2(12.9898, 78.233))) * 43758.5453);
}
    ''',
    
    'fbm': '''
// Fractal Brownian Motion combining multiple octaves
float fbmNoise(vec2 coord, float scale, float time, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * baseNoise(coord * frequency, scale, time);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}
    ''',
    
    'ridged': '''
// Ridged Multi-Fractal combining multiple octaves
float ridgedNoise(vec2 coord, float scale, float time, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float offset = 1.0;
    float gain = 2.0;

    for (int i = 0; i < octaves; i++) {
        float temp = baseNoise(coord * frequency, scale, time);
        temp = offset - abs(temp);  // Make a "ridged" shape
        temp = temp * temp;         // Square it
        value += temp * amplitude;
        amplitude *= gain;
        frequency *= 2.0;
    }

    return value;
}
    ''',
    
    'turbulence': '''
// Turbulence function using absolute value of noise
float turbulenceNoise(vec2 coord, float scale, float time, int octaves) {
    float value = 0.0;
    float amplitude = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * abs(baseNoise(coord, scale, time));
        amplitude *= 0.5;
        coord *= 2.0;
        scale *= 2.0;
    }

    return value;
}
    '''
}

def get_interface():
    '''Return the interface definition for this module'''
    return INTERFACE

def get_pseudocode(branch_name=None):
    '''Return the pseudocode for this procedural module or specific branch'''
    if branch_name and branch_name in pseudocode:
        return pseudocode[branch_name]
    else:
        # Return all pseudocodes
        return pseudocode

def get_metadata():
    '''Return metadata about this module'''
    return {
        'name': 'noise_functions_branching',
        'type': 'procedural',
        'patterns': ['Perlin Noise', 'Simplex Noise', 'Value Noise', 'Fractal Brownian Motion', 'Ridged Multi-Fractal', 'Turbulence'],
        'frequency': 200,
        'dependencies': [],
        'conflicts': [],
        'description': 'Procedural noise generation with branching for different algorithms and octave modes',
        'interface': INTERFACE,
        'branches': INTERFACE['branches']
    }

def validate_branches(selected_branches):
    '''Validate that the selected branches don't have conflicts'''
    # Check for conflicts between noise algorithm and octave mode
    if 'noise_algorithm' in selected_branches and 'octave_mode' in selected_branches:
        noise_algo = selected_branches['noise_algorithm']
        octave_mode = selected_branches['octave_mode']
        
        # Basic validation - in a real implementation we'd have more complex rules
        return True
    return False