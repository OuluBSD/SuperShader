// Basic point light module
// Implements standard point light calculations with attenuation

// Attenuation function for point lights
float calculateAttenuation(vec3 lightPos, vec3 fragPos, vec3 lightParams) {
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (lightParams.x + lightParams.y * distance + lightParams.z * (distance * distance));
    return attenuation;
}

// Point light calculation with attenuation
vec3 calculatePointLight(vec3 fragPos, vec3 normal, vec3 lightPos, vec3 lightColor, vec3 lightParams, vec3 albedo, float specularStrength) {
    vec3 lightDir = normalize(lightPos - fragPos);
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (lightParams.x + lightParams.y * distance + lightParams.z * (distance * distance));
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo * lightColor;
    
    // Specular
    vec3 viewDir = normalize(cameraPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;
    
    // Apply attenuation
    diffuse *= attenuation;
    specular *= attenuation;
    
    return diffuse + specular;
}

// Simplified point light with default attenuation
vec3 pointLightSimple(vec3 fragPos, vec3 normal, vec3 lightPos, vec3 lightColor, vec3 albedo) {
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 viewDir = normalize(cameraPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    
    vec3 ambient = 0.1 * lightColor * albedo;
    vec3 diffuse = diff * lightColor * albedo;
    vec3 specular = spec * lightColor;
    
    return ambient + diffuse + specular;
}

// Point light with radius consideration
vec3 pointLightWithRadius(vec3 fragPos, vec3 normal, vec3 lightPos, float lightRadius, vec3 lightColor, vec3 albedo) {
    vec3 lightDir = lightPos - fragPos;
    float distance = length(lightDir);
    lightDir /= distance;
    
    float attenuation = 1.0 / (distance * distance);
    
    // Account for light radius
    float lightCos = max(0.0, dot(normal, lightDir));
    float radiusFactor = 1.0 - smoothstep(lightRadius * 0.9, lightRadius, distance);
    
    vec3 diffuse = lightCos * albedo * lightColor * attenuation * radiusFactor;
    
    return diffuse;
}

// Spot light cone implementation
vec3 spotLight(vec3 fragPos, vec3 normal, vec3 lightPos, vec3 lightDir, vec3 color, float cutoff, float outerCutoff, vec3 albedo) {
    vec3 lightDirToFragment = normalize(lightPos - fragPos);
    float theta = dot(lightDirToFragment, normalize(-lightDir));
    
    if (theta > outerCutoff) {
        float intensity = smoothstep(cutoff, outerCutoff, theta);
        
        float diff = max(dot(normal, lightDirToFragment), 0.0);
        vec3 diffuse = diff * albedo * color * intensity;
        
        return diffuse;
    }
    
    return vec3(0.0);
}