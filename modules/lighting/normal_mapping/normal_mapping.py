
#!/usr/bin/env python3
'''
Normal Mapping Module with Interface Definition
Extracted from common lighting patterns in shader analysis
Pattern frequency: 533 occurrences
'''

# Interface definition
INTERFACE = {
    'inputs': [
        {'name': 'TexCoords', 'type': 'vec2', 'direction': 'in', 'semantic': 'tex_coords'},
        {'name': 'FragPos', 'type': 'vec3', 'direction': 'in', 'semantic': 'position'},
        {'name': 'Normal', 'type': 'vec3', 'direction': 'in', 'semantic': 'normal'},
        {'name': 'Tangent', 'type': 'vec3', 'direction': 'in', 'semantic': 'tangent'},
        {'name': 'normalMap', 'type': 'sampler2D', 'direction': 'uniform', 'semantic': 'normal_texture'}
    ],
    'outputs': [
        {'name': 'normalOut', 'type': 'vec3', 'direction': 'out', 'semantic': 'normal_world_space'}
    ],
    'uniforms': [
        {'name': 'normalMap', 'type': 'sampler2D', 'semantic': 'normal_texture'}
    ]
}

# Pseudocode for normal mapping
pseudocode = '''
// Normal Mapping Implementation
vec3 getNormalFromMap(sampler2D normalMap, vec2 uv, vec3 pos, vec3 normal, vec3 tangent) {
    // Sample the normal map
    vec3 tangentNormal = texture(normalMap, uv).xyz * 2.0 - 1.0;
    
    // Create TBN matrix
    vec3 T = normalize(tangent);
    vec3 N = normalize(normal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    
    // Transform normal from tangent space to world space
    vec3 finalNormal = normalize(TBN * tangentNormal);
    
    return finalNormal;
}

// Alternative: Simple normal mapping with normal map sampling
vec3 sampleNormalMap(sampler2D normalMap, vec2 uv) {
    vec3 normal = texture(normalMap, uv).xyz * 2.0 - 1.0;
    normal.xy *= -1.0;  // Flip X and Y for correct orientation
    return normalize(normal);
}
'''

def get_interface():
    '''Return the interface definition for this module'''
    return INTERFACE

def get_pseudocode():
    '''Return the pseudocode for this lighting module'''
    return pseudocode

def get_metadata():
    '''Return metadata about this module'''
    return {
        'name': 'normal_mapping',
        'type': 'lighting',
        'patterns': ['Normal Mapping'],
        'frequency': 533,
        'dependencies': [],
        'conflicts': [],
        'description': 'Normal mapping implementation with TBN matrix',
        'interface': INTERFACE
    }
