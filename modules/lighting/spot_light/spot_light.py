#!/usr/bin/env python3
"""
Spot Light Module
Extracted from common lighting patterns in shader analysis
Pattern frequency: 63 occurrences
"""

# Pseudocode for spot light
pseudocode = """
// Spot Light Implementation

// Basic spot light calculation
vec3 calculateSpotLight(vec3 fragPos, vec3 normal, 
                       vec3 lightPos, vec3 lightDir, 
                       vec3 lightColor, float cutOff, float outerCutOff) {
    vec3 lightDirToFragment = normalize(lightPos - fragPos);
    
    // Check if fragment is inside the spotlight cone
    float theta = dot(lightDirToFragment, normalize(-lightDir)); 
    float epsilon = cutOff - outerCutOff;
    float intensity = clamp((theta - outerCutOff) / epsilon, 0.0, 1.0);
    
    // Only calculate lighting if fragment is inside the light cone
    if(theta < outerCutOff) {
        intensity = 0.0;
    }
    
    // Diffuse lighting
    float diff = max(dot(normal, lightDirToFragment), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Attenuate with distance
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    diffuse *= attenuation * intensity;
    
    return diffuse;
}

// Smooth spotlight edge
vec3 calculateSmoothSpotLight(vec3 fragPos, vec3 normal, 
                             vec3 lightPos, vec3 lightDir, 
                             vec3 lightColor, float cutOff, float outerCutOff) {
    vec3 lightDirToFragment = normalize(lightPos - fragPos);
    
    float theta = dot(lightDirToFragment, normalize(-lightDir)); 
    float epsilon = cutOff - outerCutOff;
    float intensity = smoothstep(outerCutOff, cutOff, theta);
    
    // Diffuse lighting
    float diff = max(dot(normal, lightDirToFragment), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Attenuate with distance
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    diffuse *= attenuation * intensity;
    
    return diffuse;
}

// Multiple spot lights
vec3 calculateMultipleSpotLights(vec3 fragPos, vec3 normal, 
                                vec3[4] lightPositions, vec3[4] lightDirections, 
                                vec3[4] lightColors, float cutOff, float outerCutOff) {
    vec3 result = vec3(0.0);
    
    for(int i = 0; i < 4; i++) {
        result += calculateSpotLight(fragPos, normal, 
                                   lightPositions[i], lightDirections[i], 
                                   lightColors[i], cutOff, outerCutOff);
    }
    
    return result;
}
"""

def get_pseudocode():
    """Return the pseudocode for this lighting module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'spot_light',
        'type': 'lighting',
        'patterns': ['Spot Light'],
        'frequency': 63,
        'dependencies': ['diffuse_lighting', 'basic_point_light'],
        'conflicts': [],
        'description': 'Spot light implementations with cutoff angles and smooth edges'
    }