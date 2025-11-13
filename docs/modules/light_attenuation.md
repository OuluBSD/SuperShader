# light_attenuation

**Category:** lighting
**Type:** standardized

## Dependencies
lighting

## Tags
lighting, color

## Code
```glsl
// Light attenuation calculations
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

```