#!/usr/bin/env python3
"""
Diffuse Lighting Module
Extracted from common lighting patterns in shader analysis
Pattern frequency: 378 occurrences
"""

# Pseudocode for diffuse lighting
pseudocode = """
// Lambert Diffuse Lighting
vec3 calculateDiffuseLambert(vec3 lightDir, vec3 normal, vec3 diffuseColor) {
    float diff = max(dot(normal, lightDir), 0.0);
    return diff * diffuseColor;
}

// Oren-Nayar Diffuse (for rough surfaces)
vec3 calculateDiffuseOrenNayar(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 diffuseColor, float roughness) {
    float NdotL = dot(normal, lightDir);
    float NdotV = dot(normal, viewDir);
    
    float sigma2 = roughness * roughness;
    float A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
    float B = 0.45 * sigma2 / (sigma2 + 0.09);
    
    float angleLV = acos(dot(lightDir, viewDir));
    float alpha = max(NdotL, NdotV);
    float beta = min(NdotL, NdotV);
    
    float C = sin(angleLV) * tan(angleLV);
    
    float orenNayar = A + B * C * max(0.0, alpha) * beta;
    return max(0.0, NdotL) * diffuseColor * orenNayar;
}

// Ambient lighting
vec3 calculateAmbient(vec3 ambientColor, float ambientStrength) {
    return ambientStrength * ambientColor;
}

// Combined diffuse and ambient
vec3 calculateDiffuseAndAmbient(vec3 position, vec3 normal, vec3 lightPos, vec3 lightColor, 
                               vec3 ambientColor, float ambientStrength) {
    vec3 lightDir = normalize(lightPos - position);
    vec3 diffuse = calculateDiffuseLambert(lightDir, normal, lightColor);
    vec3 ambient = calculateAmbient(ambientColor, ambientStrength);
    
    return diffuse + ambient;
}
"""

def get_pseudocode():
    """Return the pseudocode for this lighting module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'diffuse_lighting',
        'type': 'lighting',
        'patterns': ['Diffuse', 'Lambert', 'Ambient'],
        'frequency': 378,
        'dependencies': [],
        'conflicts': [],
        'description': 'Diffuse lighting calculations including Lambert and Oren-Nayar models'
    }