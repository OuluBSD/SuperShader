#!/usr/bin/env python3
"""
Create proper GLSL implementations for standardized lighting modules.
"""

import os

def generate_pbr_lit_glsl():
    """Generate GLSL implementation for PBR lighting."""
    return """// Physically Based Rendering lighting calculations
// Standardized lighting module

// Input parameters
struct Material {
    vec3 albedo;
    float metallic;
    float roughness;
    float ao;
};

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

// Constants
const float PI = 3.14159265359;

// Normal Distribution Function - GGX
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

// Geometry Function - Smith
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// Geometry Function - Smith
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Fresnel Function - Schlick
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// Cook-Torrance BRDF
vec3 CookTorranceBRDF(vec3 L, vec3 V, vec3 N, vec3 F0, float roughness, vec3 diffuseColor) {
    vec3 H = normalize(L + V);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, L), 0.0) * max(dot(N, V), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 diffuse = (kD * diffuseColor) / PI;

    return (diffuse + specular);
}

// Main PBR lighting function
vec3 CalculatePBRLighting(Material material, Light light, vec3 worldPos, vec3 normal, vec3 viewDir) {
    vec3 L = normalize(light.position - worldPos);
    vec3 V = normalize(viewDir - worldPos);
    vec3 N = normalize(normal);
    
    vec3 F0 = mix(vec3(0.04), material.albedo, material.metallic);
    
    vec3 Lo = CookTorranceBRDF(L, V, N, F0, material.roughness, material.albedo);
    
    float NdotL = max(dot(N, L), 0.0);
    return Lo * light.color * light.intensity * NdotL;
}
"""


def generate_diffuse_lit_glsl():
    """Generate GLSL implementation for diffuse lighting."""
    return """// Standard diffuse lighting calculations
// Standardized lighting module

// Lambert diffuse lighting model
float LambertDiffuse(vec3 normal, vec3 lightDir) {
    return max(dot(normal, lightDir), 0.0);
}

// Basic diffuse lighting calculation
vec3 CalculateDiffuseLighting(vec3 albedo, vec3 normal, vec3 lightDir, vec3 lightColor) {
    float diff = LambertDiffuse(normal, lightDir);
    return diff * albedo * lightColor;
}

// Diffuse lighting with multiple lights
vec3 CalculateMultiDiffuseLighting(vec3 albedo, vec3 normal, vec3 lightDir1, vec3 lightColor1, 
                                   vec3 lightDir2, vec3 lightColor2) {
    float diff1 = LambertDiffuse(normal, lightDir1);
    float diff2 = LambertDiffuse(normal, lightDir2);
    
    vec3 diffuse1 = diff1 * albedo * lightColor1;
    vec3 diffuse2 = diff2 * albedo * lightColor2;
    
    return diffuse1 + diffuse2;
}
"""


def generate_specular_blinn_phong_glsl():
    """Generate GLSL implementation for Blinn-Phong specular lighting."""
    return """// Blinn-Phong specular lighting model
// Standardized lighting module

// Blinn-Phong specular calculation
float BlinnPhongSpecular(vec3 normal, vec3 lightDir, vec3 viewDir, float shininess) {
    vec3 H = normalize(lightDir + viewDir);
    return pow(max(dot(normal, H), 0.0), shininess);
}

// Full Blinn-Phong lighting model
vec3 BlinnPhongLighting(vec3 albedo, vec3 specular, vec3 normal, vec3 lightDir, vec3 viewDir, 
                        vec3 lightColor, float shininess, float ambientFactor) {
    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular component
    float spec = BlinnPhongSpecular(normal, lightDir, viewDir, shininess);
    
    vec3 ambient = ambientFactor * albedo;
    vec3 diffuse = diff * albedo * lightColor;
    vec3 specular = spec * specular * lightColor;
    
    return ambient + diffuse + specular;
}
"""


def generate_shadow_mapping_glsl():
    """Generate GLSL implementation for shadow mapping."""
    return """// Shadow mapping calculations
// Standardized lighting module

uniform sampler2D shadowMap;
uniform mat4 lightSpaceMatrix;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get closest depth value from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    
    // Check whether current frag pos is in shadow
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    float shadow = 0.0;
    
    if (currentDepth - bias > closestDepth) {
        shadow = 1.0;
    }
    
    // PCF (Percentage Closer Filtering) for softer shadows
    float texelSize = 1.0 / textureSize(shadowMap, 0).x;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    
    return shadow;
}
"""


def generate_light_attenuation_glsl():
    """Generate GLSL implementation for light attenuation."""
    return """// Light attenuation calculations
// Standardized lighting module

// Calculate attenuation based on distance
float CalculateAttenuation(float distance, float constant, float linear, float quadratic) {
    float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));
    return attenuation;
}

// Calculate attenuation with range (attenuation becomes 0 at maxDistance)
float CalculateRangeAttenuation(float distance, float maxDistance) {
    float attenuation = 1.0 - pow(distance / maxDistance, 2.0);
    attenuation = max(attenuation, 0.0);
    attenuation = min(attenuation, 1.0);
    return attenuation;
}

// Point light attenuation calculation
float PointLightAttenuation(vec3 lightPos, vec3 fragPos, float constant, float linear, float quadratic) {
    float distance = length(lightPos - fragPos);
    return CalculateAttenuation(distance, constant, linear, quadratic);
}

// Calculate full light contribution with attenuation
vec3 CalculateLightWithAttenuation(vec3 lightPos, vec3 fragPos, vec3 lightColor, 
                                   float constant, float linear, float quadratic) {
    float distance = length(lightPos - fragPos);
    float attenuation = CalculateAttenuation(distance, constant, linear, quadratic);
    return lightColor * attenuation;
}
"""


def main():
    # Create standardized modules directory
    os.makedirs('modules/lighting/standardized', exist_ok=True)
    
    # Generate all standardized modules
    modules = {
        'pbr_lit.glsl': generate_pbr_lit_glsl(),
        'diffuse_lit.glsl': generate_diffuse_lit_glsl(),
        'specular_blinn_phong.glsl': generate_specular_blinn_phong_glsl(),
        'shadow_mapping.glsl': generate_shadow_mapping_glsl(),
        'light_attenuation.glsl': generate_light_attenuation_glsl()
    }
    
    # Write modules to files
    for filename, content in modules.items():
        filepath = f'modules/lighting/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Generated: {filename}")
    
    print("All standardized lighting modules have been generated with proper GLSL implementations!")


if __name__ == "__main__":
    main()