// Comprehensive lighting functions module
// Combines different lighting models into unified functions

const float PI = 3.14159265359;

// Ambient lighting
vec3 calculateAmbientLighting(vec3 albedo, float ambientStrength) {
    return ambientStrength * albedo;
}

// Basic Phong lighting model
vec3 phongLighting(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 lightColor, vec3 albedo, float ambientStrength, float diffuseStrength, float specularStrength) {
    // Ambient
    vec3 ambient = calculateAmbientLighting(albedo, ambientStrength);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * albedo * diffuseStrength;
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;
    
    return ambient + diffuse + specular;
}

// Blinn-Phong lighting model
vec3 blinnPhongLighting(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 lightColor, vec3 albedo, float ambientStrength, float diffuseStrength, float specularStrength) {
    // Ambient
    vec3 ambient = ambientStrength * albedo * lightColor;
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo * lightColor * diffuseStrength;
    
    // Specular (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;
    
    return ambient + diffuse + specular;
}

// Unified lighting model function that can handle different approaches
vec3 unifiedLighting(vec3 fragPos, vec3 normal, vec3 viewDir, vec3 lightPos, vec3 lightColor, 
                     vec3 albedo, float metallic, float roughness, float ao) {
    
    // Calculate light direction and distance
    vec3 lightDir = normalize(lightPos - fragPos);
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = lightColor * attenuation;
    
    // Cook-Torrance BRDF
    vec3 halfwayDir = normalize(lightDir + viewDir);
    
    // Fresnel
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    vec3 F = fresnelSchlick(max(dot(halfwayDir, viewDir), 0.0), F0);
    
    // Other half of DFG
    float NDF = DistributionGGX(normal, halfwayDir, roughness);
    float G = GeometrySmith(normal, viewDir, lightDir, roughness);
    
    // Specular BRDF
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
    vec3 specular = numerator / max(denominator, 0.001);
    
    // Diffuse BRDF
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;
    
    float NdotL = max(dot(normal, lightDir), 0.0);
    
    return (kD * albedo / PI + specular) * radiance * NdotL * ao;
}

// Complete lighting function with multiple light types
vec3 computeLighting(vec3 fragPos, vec3 normal, vec3 viewDir, vec3 albedo, float metallic, float roughness, 
                     vec3 lightPositions[4], vec3 lightColors[4], int numLights) {
    
    vec3 output = vec3(0.0);
    
    // Ambient
    vec3 ambient = 0.05 * albedo;
    
    for(int i = 0; i < numLights; i++) {
        // Calculate per-light radiance
        vec3 lightDir = normalize(lightPositions[i] - fragPos);
        float distance = length(lightPositions[i] - fragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;
        
        // Cook-Torrance BRDF
        vec3 halfwayDir = normalize(lightDir + viewDir);
        
        // Fresnel
        vec3 F0 = vec3(0.04);
        F0 = mix(F0, albedo, metallic);
        vec3 F = fresnelSchlick(max(dot(halfwayDir, viewDir), 0.0), F0);
        
        // Other BRDF calculations
        float NDF = DistributionGGX(normal, halfwayDir, roughness);
        float G = GeometrySmith(normal, viewDir, lightDir, roughness);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
        vec3 specular = numerator / max(denominator, 0.001);
        
        float NdotL = max(dot(normal, lightDir), 0.0);
        
        vec3 lightContribution = (kD * albedo / PI + specular) * radiance * NdotL;
        output += lightContribution;
    }
    
    return output + ambient;
}

// Lighting function with shadow consideration
vec3 computeLightingWithShadows(vec3 fragPos, vec3 normal, vec3 viewDir, vec3 albedo, 
                                float metallic, float roughness, vec3 lightPos, vec3 lightColor, 
                                float shadowFactor) {
    vec3 lighting = unifiedLighting(fragPos, normal, viewDir, lightPos, lightColor, albedo, metallic, roughness, 1.0);
    return mix(lighting * 0.2, lighting, shadowFactor); // Apply shadow as a factor between full lighting and 20% of lighting
}

// Simplified lighting function for basic use cases
vec3 simpleLighting(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 lightColor, vec3 albedo, float shininess) {
    // Diffuse
    float NdotL = max(dot(normal, lightDir), 0.0);
    
    // Specular
    vec3 R = reflect(-lightDir, normal);
    float RdotV = max(dot(R, viewDir), 0.0);
    float spec = pow(RdotV, shininess);
    
    return lightColor * albedo * NdotL + lightColor * spec;
}