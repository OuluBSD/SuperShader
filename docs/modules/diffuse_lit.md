# diffuse_lit

**Category:** lighting
**Type:** standardized

## Dependencies
normal_mapping, lighting

## Tags
lighting, color

## Code
```glsl
// Standard diffuse lighting calculations
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

```