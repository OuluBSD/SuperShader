#!/usr/bin/env python3
"""
Directional Light Module
Extracted from common lighting patterns in shader analysis
Pattern frequency: 60 occurrences
"""

# Pseudocode for directional light
pseudocode = """
// Directional Light Implementation

// Simple directional light calculation
vec3 calculateDirectionalLight(vec3 normal, vec3 lightDir, vec3 lightColor, float ambientStrength) {
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 ambient = ambientStrength * lightColor;
    vec3 diffuse = diff * lightColor;
    
    return ambient + diffuse;
}

// Directional light with shadow mapping
vec3 calculateDirectionalLightWithShadow(vec3 normal, vec3 lightDir, vec3 lightColor, 
                                        float ambientStrength, float shadow) {
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 ambient = ambientStrength * lightColor;
    vec3 diffuse = diff * lightColor;
    
    // Apply shadow
    vec3 lighting = ambient + (1.0 - shadow) * diffuse;
    
    return lighting;
}

// Multiple directional lights
vec3 calculateMultipleDirectionalLights(vec3 normal, 
                                       vec3[4] lightDirs, 
                                       vec3[4] lightColors,
                                       float ambientStrength) {
    vec3 result = vec3(0.0);
    
    for(int i = 0; i < 4; i++) {
        float diff = max(dot(normal, normalize(lightDirs[i])), 0.0);
        result += diff * lightColors[i] + ambientStrength * lightColors[i];
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
        'name': 'directional_light',
        'type': 'lighting',
        'patterns': ['Directional Light', 'Light Direction'],
        'frequency': 60,
        'dependencies': ['diffuse_lighting'],
        'conflicts': [],
        'description': 'Directional light calculations'
    }