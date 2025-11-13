#!/usr/bin/env python3
"""
Cel Shading Module
Extracted from common lighting patterns in shader analysis
Pattern frequency: 181 occurrences
"""

# Pseudocode for cel shading
pseudocode = """
// Cel Shading Implementation

// Basic cel shading (toon shading)
vec3 calculateCelShading(vec3 normal, vec3 lightDir, vec3 diffuseColor, vec3 specularColor) {
    float NdotL = dot(normal, lightDir);
    
    // Quantize the lighting to create bands
    float intensity = smoothstep(0.0, 0.01, NdotL);
    intensity += step(0.5, NdotL);
    intensity += step(0.8, NdotL);
    
    // Limit to 0-1 range
    intensity = min(intensity, 1.0);
    
    // Apply the quantized intensity to the diffuse color
    vec3 toonDiffuse = diffuseColor * vec3(intensity);
    
    // Add specular if intensity is high enough
    float spec = step(0.9, NdotL) * step(0.2, dot(reflect(-lightDir, normal), viewDir));
    vec3 toonSpecular = spec * specularColor;
    
    return toonDiffuse + toonSpecular;
}

// Multiple-toned cel shading
vec3 calculateMultiToneCelShading(vec3 normal, vec3 lightDir, vec3 lightColor, 
                                 vec3 darkColor, vec3 midColor, vec3 lightColorOut) {
    float NdotL = dot(normal, lightDir);
    
    // Define transition thresholds
    float darkThreshold = 0.3;
    float midThreshold = 0.6;
    
    // Determine tone based on lighting
    vec3 finalColor;
    if (NdotL < darkThreshold) {
        finalColor = darkColor;
    } else if (NdotL < midThreshold) {
        finalColor = midColor;
    } else {
        finalColor = lightColorOut;
    }
    
    return finalColor * lightColor;
}

// Outline detection for cel shading
float calculateOutline(vec3 normal, float edgeThreshold) {
    float edge = 1.0 - abs(dot(normal, vec3(0.0, 0.0, 1.0)));
    edge = smoothstep(edgeThreshold, 1.0, edge);
    return edge;
}

// Complete cel shading with outline
vec4 calculateCompleteCelShading(vec3 position, vec3 normal, vec3 viewDir, 
                                vec3 lightPos, vec3 diffuseColor, vec3 specularColor) {
    vec3 lightDir = normalize(lightPos - position);
    
    // Calculate toon shading
    vec3 toonColor = calculateCelShading(normal, lightDir, diffuseColor, specularColor);
    
    // Calculate outline
    float outline = calculateOutline(normal, 0.8);
    
    // Blend with outline
    vec3 finalColor = mix(toonColor, vec3(0.0), outline);
    
    return vec4(finalColor, 1.0);
}
"""

def get_pseudocode():
    """Return the pseudocode for this lighting module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'cel_shading',
        'type': 'lighting',
        'patterns': ['Cel Shading', 'Toon Shading'],
        'frequency': 181,
        'dependencies': ['diffuse_lighting'],
        'conflicts': ['pbr_lighting'],  # Cel shading conflicts with realistic PBR
        'description': 'Cel/toon shading implementations with outline detection'
    }