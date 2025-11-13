#!/usr/bin/env python3
"""
Normal Mapping Module
Extracted from common lighting patterns in shader analysis
Pattern frequency: 533 occurrences
"""

# Pseudocode for normal mapping
pseudocode = """
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
"""

def get_pseudocode():
    """Return the pseudocode for this lighting module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'normal_mapping',
        'type': 'lighting',
        'patterns': ['Normal Mapping'],
        'frequency': 533,
        'dependencies': [],
        'conflicts': [],
        'description': 'Normal mapping implementation with TBN matrix'
    }