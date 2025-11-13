#!/usr/bin/env python3
"""
Raymarching Lighting Module
Extracted from common raymarching patterns in shader analysis
Pattern frequency: 156 occurrences
"""

# Pseudocode for raymarching lighting
pseudocode = """
// Raymarching Lighting Implementation

// Calculate normal using gradient of distance field
vec3 calcNormal(vec3 p, float epsilon) {
    vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x
    ));
}

// Calculate normal with adaptive epsilon
vec3 calcNormalAdaptive(vec3 p) {
    float dx = map(p + vec3(0.001, 0.0, 0.0)).x - map(p - vec3(0.001, 0.0, 0.0)).x;
    float dy = map(p + vec3(0.0, 0.001, 0.0)).x - map(p - vec3(0.0, 0.001, 0.0)).x;
    float dz = map(p + vec3(0.0, 0.0, 0.001)).x - map(p - vec3(0.0, 0.0, 0.001)).x;
    return normalize(vec3(dx, dy, dz));
}

// Calculate ambient occlusion based on distance field
float calcAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for(int i = 0; i < 5; i++) {
        float h = 0.01 + 0.15 * float(i) / 4.0;
        float d = map(p + h * n).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

// Calculate soft shadows
float calcSoftShadow(vec3 ro, vec3 rd, float mint, float tmax, float k) {
    float res = 1.0;
    float t = mint;
    for(int i = 0; i < 32; i++) {
        float h = map(ro + rd * t).x;
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.10);
        if(res < 0.001 || t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Basic Phong lighting model for raymarching
vec3 phongLighting(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 lightColor, vec3 materialColor) {
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);

    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

    vec3 ambient = 0.1 * materialColor;
    vec3 diffuse = diff * lightColor * materialColor;
    vec3 specular = spec * lightColor;

    return ambient + diffuse + specular;
}

// Calculate multiple light sources
vec3 multiLighting(vec3 pos, vec3 normal, vec3 viewDir, vec3 materialColor) {
    vec3 color = vec3(0.0);

    // Light 1
    vec3 lightPos1 = vec3(5.0, 5.0, 5.0);
    vec3 lightDir1 = normalize(lightPos1 - pos);
    vec3 lightColor1 = vec3(1.0, 0.9, 0.8);
    color += phongLighting(normal, viewDir, lightDir1, lightColor1, materialColor);

    // Light 2
    vec3 lightPos2 = vec3(-5.0, 3.0, -2.0);
    vec3 lightDir2 = normalize(lightPos2 - pos);
    vec3 lightColor2 = vec3(0.2, 0.5, 1.0);
    color += phongLighting(normal, viewDir, lightDir2, lightColor2, materialColor);

    return color;
}

// Ambient occlusion for raymarching scenes
float ambientOcclusion(vec3 pos, vec3 normal) {
    float occ = 0.0;
    float sca = 1.0;
    for(int i = 0; i < 5; i++) {
        float h = 0.01 + 0.15 * float(i) / 4.0;
        float d = map(pos + h * normal).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

// Fresnel effect for raymarching
float fresnel(vec3 viewDir, vec3 normal, float power) {
    return pow(1.0 - clamp(dot(normal, viewDir), 0.0, 1.0), power);
}

// Complete lighting calculation for raymarching
vec3 raymarchingLighting(vec3 pos, vec3 normal, vec3 viewDir, vec3 materialColor, float materialRoughness, float materialMetallic) {
    // Calculate ambient occlusion
    float ao = ambientOcclusion(pos, normal);
    
    // Calculate soft shadows
    float shadow = calcSoftShadow(pos, normalize(vec3(1.0, 2.0, 1.0)), 0.01, 20.0, 16.0);
    
    // Calculate lighting
    vec3 lighting = multiLighting(pos, normal, viewDir, materialColor);
    
    // Apply ambient occlusion and shadows
    lighting *= ao * (0.5 + 0.5 * shadow);
    
    // Apply fresnel effect
    float fres = fresnel(viewDir, normal, 5.0);
    lighting = mix(lighting, vec3(1.0), fres * 0.2);
    
    return lighting;
}
"""

def get_pseudocode():
    """Return the pseudocode for this raymarching lighting module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'raymarching_lighting',
        'type': 'raymarching',
        'patterns': ['Raymarching Lighting', 'Normal Calculation', 'AO'],
        'frequency': 156,
        'dependencies': ['core', 'distance_functions'],
        'conflicts': [],
        'description': 'Lighting calculations specifically designed for raymarching scenes'
    }