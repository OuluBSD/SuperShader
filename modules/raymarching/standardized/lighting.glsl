// Lighting module
// Standardized lighting functions for raymarching scenes

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
