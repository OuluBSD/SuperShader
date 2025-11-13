#!/usr/bin/env python3
"""
Shadow Mapping Module
Extracted from common lighting patterns in shader analysis
Pattern frequency: 282 occurrences
"""

# Pseudocode for shadow mapping
pseudocode = """
// Shadow Mapping Implementation

// Basic shadow calculation
float calculateShadow(sampler2D shadowMap, vec4 fragPosLightSpace, float bias) {
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get closest depth value from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    
    // Check whether current frag pos is in shadow
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    
    return shadow;
}

// Percentage-Closer Filtering for smoother shadows
float calculatePCFShadow(sampler2D shadowMap, vec4 fragPosLightSpace, float bias) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }
    }
    return shadow / 9.0;
}

// Exponential Shadow Maps
float calculateExponentialShadow(sampler2D shadowMap, vec4 fragPosLightSpace, float k) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float depth = texture(shadowMap, projCoords.xy).r;
    float w = texture(shadowMap, projCoords.xy).g;  // Weight or moment
    
    float currentDepth = projCoords.z;
    
    // Calculate probability of being lit
    float p = (currentDepth <= depth) ? 1.0 : w / (w + currentDepth - depth);
    return 1.0 - p;
}
"""

def get_pseudocode():
    """Return the pseudocode for this lighting module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'shadow_mapping',
        'type': 'lighting',
        'patterns': ['Shadow Mapping'],
        'frequency': 282,
        'dependencies': [],
        'conflicts': [],
        'description': 'Shadow mapping implementations with PCF and exponential methods'
    }