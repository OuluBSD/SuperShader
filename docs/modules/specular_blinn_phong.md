# specular_blinn_phong

**Category:** lighting
**Type:** standardized

## Dependencies
normal_mapping, lighting

## Tags
lighting, color

## Code
```glsl
// Blinn-Phong specular lighting model
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

```