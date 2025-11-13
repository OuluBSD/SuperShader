// Standardized lighting module interface
// Defines the standard inputs, outputs, and structure for lighting modules

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

struct Material {
    vec3 albedo;
    float metallic;
    float roughness;
    float ao;
};

struct SurfaceData {
    vec3 position;
    vec3 normal;
    vec3 viewDir;
    vec2 texCoords;
};

// Standard lighting interface function
vec3 computeStandardLighting(SurfaceData surface, Material material, Light light) {
    // Calculate light direction and distance
    vec3 lightDir = normalize(light.position - surface.position);
    float distance = length(light.position - surface.position);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = light.color * light.intensity * attenuation;
    
    // Cook-Torrance BRDF calculation
    vec3 halfwayDir = normalize(lightDir + surface.viewDir);
    
    // Fresnel
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, material.albedo, material.metallic);
    vec3 F = fresnelSchlick(max(dot(halfwayDir, surface.viewDir), 0.0), F0);
    
    // Normal distribution function
    float NDF = DistributionGGX(surface.normal, halfwayDir, material.roughness);
    
    // Geometry function
    float G = GeometrySmith(surface.normal, surface.viewDir, lightDir, material.roughness);
    
    // Specular BRDF
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - material.metallic;
    
    // Final calculation
    float NdotL = max(dot(surface.normal, lightDir), 0.0);
    
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(surface.normal, surface.viewDir), 0.0) * max(dot(surface.normal, lightDir), 0.0);
    vec3 specular = numerator / max(denominator, 0.001);
    
    vec3 diffuse = kD * material.albedo / PI;
    
    vec3 result = (diffuse + specular) * radiance * NdotL * material.ao;
    
    return result;
}

// Standard function for multiple lights
vec3 computeStandardLightingMultiLight(SurfaceData surface, Material material, Light lights[4], int lightCount) {
    vec3 result = vec3(0.0);
    
    for(int i = 0; i < lightCount; i++) {
        result += computeStandardLighting(surface, material, lights[i]);
    }
    
    return result;
}

// Standard ambient lighting
vec3 computeStandardAmbient(Material material, vec3 ambientLightColor) {
    return ambientLightColor * material.albedo * material.ao;
}

// Standard directional lighting
vec3 computeStandardDirectionalLight(SurfaceData surface, Material material, vec3 lightDir, vec3 lightColor) {
    vec3 normalizedLightDir = normalize(-lightDir);
    
    // Calculate halfway vector
    vec3 halfwayDir = normalize(normalizedLightDir + surface.viewDir);
    
    // Fresnel
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, material.albedo, material.metallic);
    vec3 F = fresnelSchlick(max(dot(halfwayDir, surface.viewDir), 0.0), F0);
    
    // BRDF components
    float NDF = DistributionGGX(surface.normal, halfwayDir, material.roughness);
    float G = GeometrySmith(surface.normal, surface.viewDir, normalizedLightDir, material.roughness);
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - material.metallic;
    
    float NdotL = max(dot(surface.normal, normalizedLightDir), 0.0);
    
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(surface.normal, surface.viewDir), 0.0) * max(dot(surface.normal, normalizedLightDir), 0.0);
    vec3 specular = numerator / max(denominator, 0.001);
    
    vec3 diffuse = kD * material.albedo / PI;
    
    return (diffuse + specular) * lightColor * NdotL * material.ao;
}