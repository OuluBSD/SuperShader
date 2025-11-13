// Directional light module
// Implements directional (sun) lighting calculations

// Basic directional light calculation
vec3 calculateDirectionalLight(vec3 normal, vec3 lightDir, vec3 lightColor, vec3 albedo, float specularStrength) {
    // Diffuse
    float diff = max(dot(normal, normalize(-lightDir)), 0.0);
    vec3 diffuse = diff * albedo * lightColor;
    
    // Specular
    vec3 viewDir = normalize(cameraPos - fragPos);
    vec3 reflectDir = reflect(lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;
    
    return diffuse + specular;
}

// Directional light with ambient component
vec3 directionalLightWithAmbient(vec3 normal, vec3 lightDir, vec3 lightColor, vec3 albedo, float ambientStrength) {
    vec3 norm = normalize(normal);
    vec3 lightDirN = normalize(-lightDir);
    
    // Diffuse
    float diff = max(dot(norm, lightDirN), 0.0);
    vec3 diffuse = diff * albedo * lightColor;
    
    // Ambient
    vec3 ambient = ambientStrength * lightColor * albedo;
    
    return diffuse + ambient;
}

// Multi-directional light (for multiple light sources)
struct DirectionalLight {
    vec3 direction;
    vec3 color;
    float intensity;
};

vec3 computeDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir, vec3 albedo, float metallic, float roughness) {
    vec3 lightDir = normalize(-light.direction);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    
    vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfway = normalize(lightDir + viewDir);
    
    // Cook-Torrance BRDF components
    float NDF = DistributionGGX(normal, halfway, roughness);
    float G = GeometrySmith(normal, viewDir, lightDir, roughness);
    vec3 F = F_Schlick(max(dot(halfway, viewDir), 0.0), vec3(0.04));
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;
    
    vec3 nominator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0) + 0.0001;
    vec3 specular = nominator / denominator;
    
    vec3 diffuse = kD * albedo / PI;
    
    float NdotL = max(dot(normal, lightDir), 0.0);
    
    return ((diffuse + specular) * light.color) * light.intensity * NdotL;
}

// Simple directional light function
vec3 simpleDirectionalLight(vec3 normal, vec3 lightDir, vec3 lightColor) {
    float diff = max(dot(normal, normalize(-lightDir)), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 ambient = 0.1 * lightColor;
    return diffuse + ambient;
}

// Directional light with shadow consideration
vec3 directionalLightWithShadow(vec3 normal, vec3 lightDir, vec3 lightColor, float shadowFactor) {
    float diff = max(dot(normal, normalize(-lightDir)), 0.0);
    vec3 diffuse = diff * lightColor * shadowFactor;
    vec3 ambient = 0.1 * lightColor * shadowFactor;
    return diffuse + ambient;
}