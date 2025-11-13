#!/usr/bin/env python3
"""
Specular Lighting Module
Extracted from common lighting patterns in shader analysis
Pattern frequency: 310 occurrences
"""

# Pseudocode for specular lighting
pseudocode = """
// Phong Specular Lighting
vec3 calculateSpecularPhong(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 specularColor, float shininess) {
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    return specularColor * spec;
}

// Blinn-Phong Specular Lighting
vec3 calculateSpecularBlinnPhong(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 specularColor, float shininess) {
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    return specularColor * spec;
}

// Fresnel Effect
float fresnelSchlick(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Complete specular calculation with Fresnel
vec3 calculateSpecularFresnel(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 F0) {
    vec3 H = normalize(lightDir + viewDir);
    float cosTheta = max(dot(viewDir, H), 0.0);
    return fresnelSchlick(cosTheta, F0);
}
"""

def get_pseudocode():
    """Return the pseudocode for this lighting module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'specular_lighting',
        'type': 'lighting',
        'patterns': ['Specular', 'Phong', 'Blinn Phong', 'Fresnel'],
        'frequency': 310,
        'dependencies': ['normal_mapping'],
        'conflicts': [],
        'description': 'Specular lighting calculations including Phong, Blinn-Phong and Fresnel'
    }