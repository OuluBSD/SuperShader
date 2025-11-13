// Cel shading module
// Implements toon/cel shading techniques for cartoon-like rendering

// Basic cel shading function with quantized lighting
float celIntensity(float lightIntensity, float levels) {
    return floor(lightIntensity * levels) / levels;
}

// Cel shading with diffuse and specular bands
vec3 celShade(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 lightColor, vec3 albedo, 
              float diffuseBands, float specularBands, float shininess) {
    
    // Calculate basic lighting
    float NdotL = max(dot(normal, lightDir), 0.0);
    vec3 R = reflect(-lightDir, normal);
    float RdotV = max(dot(R, viewDir), 0.0);
    float spec = pow(RdotV, shininess);
    
    // Quantize diffuse lighting
    float quantizedDiff = celIntensity(NdotL, diffuseBands);
    
    // Quantize specular lighting
    float quantizedSpec = celIntensity(spec, specularBands);
    
    // Apply toon shading
    vec3 diffuse = quantizedDiff * lightColor * albedo;
    vec3 specular = quantizedSpec * lightColor;
    
    return diffuse + specular;
}

// Edge detection for toon outlines
float edgeDetection(vec3 normal, vec3 viewDir, float edgeThreshold) {
    float NdotV = dot(normal, viewDir);
    return step(edgeThreshold, 1.0 - NdotV);
}

// Toon ramp shading - uses a texture as a lookup for lighting values
float toonRamp(float value, sampler2D rampTexture, float rampScale) {
    float rampValue = clamp(value * rampScale, 0.0, 1.0);
    return texture(rampTexture, vec2(rampValue, 0.5)).r;
}

// Complete cel shading function with outline detection
vec3 completeCelShade(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 lightColor, vec3 albedo, 
                      float diffuseBands, float specularBands, float shininess, 
                      float edgeThreshold, vec3 outlineColor) {
    
    // Calculate basic cel shading
    float NdotL = max(dot(normal, lightDir), 0.0);
    float quantizedDiff = celIntensity(NdotL, diffuseBands);
    
    vec3 R = reflect(-lightDir, normal);
    float RdotV = max(dot(R, viewDir), 0.0);
    float spec = pow(RdotV, shininess);
    float quantizedSpec = celIntensity(spec, specularBands);
    
    vec3 diffuse = quantizedDiff * lightColor * albedo;
    vec3 specular = quantizedSpec * lightColor;
    
    // Calculate edge detection for outline
    float edgeFactor = edgeDetection(normal, viewDir, edgeThreshold);
    
    // Combine lighting
    vec3 lighting = diffuse + specular;
    
    // Apply outline if edge detected
    return mix(lighting, outlineColor, edgeFactor);
}

// Stylized lighting for cartoon rendering
vec3 cartoonLighting(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 lightColor, vec3 albedo) {
    float NdotL = max(dot(normal, lightDir), 0.0);
    
    // Create distinct lighting zones
    if (NdotL > 0.95) {
        return lightColor * albedo;        // Fully lit
    } else if (NdotL > 0.5) {
        return lightColor * albedo * 0.7;  // Mid tone
    } else if (NdotL > 0.25) {
        return lightColor * albedo * 0.4;  // Shadow
    } else {
        return lightColor * albedo * 0.1;  // Deep shadow
    }
}

// Cel shading with multiple lights
vec3 multiLightCelShading(vec3 fragPos, vec3 normal, vec3 viewDir, vec3 albedo, 
                          vec3 lightPositions[4], vec3 lightColors[4], int numLights,
                          float diffuseBands, float specularBands, float shininess) {
    
    vec3 result = vec3(0.0);
    
    for(int i = 0; i < numLights; i++) {
        vec3 lightDir = normalize(lightPositions[i] - fragPos);
        
        // Calculate lighting for this light
        float NdotL = max(dot(normal, lightDir), 0.0);
        float quantizedDiff = celIntensity(NdotL, diffuseBands);
        
        vec3 R = reflect(-lightDir, normal);
        vec3 V = normalize(viewDir);
        float RdotV = max(dot(R, V), 0.0);
        float spec = pow(RdotV, shininess);
        float quantizedSpec = celIntensity(spec, specularBands);
        
        vec3 lightContribution = (quantizedDiff * lightColors[i] * albedo) + 
                                (quantizedSpec * lightColors[i]);
        
        result += lightContribution;
    }
    
    return result;
}