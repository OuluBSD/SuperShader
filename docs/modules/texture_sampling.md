# texture_sampling

**Category:** texturing
**Type:** standardized

## Dependencies
texture_sampling, normal_mapping

## Tags
texturing, color

## Code
```glsl
// Texture sampling module
// Standardized texture sampling implementations

// Basic texture sampling with 2D UV
vec4 sampleTexture(sampler2D tex, vec2 uv) {
    return texture2D(tex, uv);
}

// Texture sampling with triplanar blending
vec4 sampleTriplanar(sampler2D tex, vec3 worldPos, vec3 normal) {
    // Project world position onto each axis
    vec2 uvX = worldPos.yz;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, uvX);
    vec4 texY = texture2D(tex, uvY);
    vec4 texZ = texture2D(tex, uvZ);
    
    // Use normal to determine blending weights
    vec3 blend = abs(normal);
    blend = normalize(max(blend, 0.00001)); // Avoid division by zero
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}

// Sample texture with triplanar using 3D coordinates
vec4 sampleTriplanar3D(sampler2D tex, vec3 pos, vec3 normal, float scale) {
    // Project world position onto each axis
    vec3 projX = vec3(pos.y, pos.z, 0.0) * scale;
    vec3 projY = vec3(pos.x, pos.z, 0.0) * scale;
    vec3 projZ = vec3(pos.x, pos.y, 0.0) * scale;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, projX.xy);
    vec4 texY = texture2D(tex, projY.xy);
    vec4 texZ = texture2D(tex, projZ.xy);
    
    // Use normal to determine blending weights
    vec3 blend = pow(abs(normal), vec3(2.0));
    blend = normalize(max(blend, 0.00001));
    blend /= (blend.x + blend.y + blend.z);
    
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}

// Sample texture with custom filtering
vec4 sampleTextureFiltered(sampler2D tex, vec2 uv, vec2 texelSize, float filterRadius) {
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    
    // Sample in a 5x5 area around the center
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 offset = vec2(float(x), float(y)) * filterRadius * texelSize;
            float weight = exp(-(x*x + y*y) / (2.0 * filterRadius * filterRadius));
            result += texture2D(tex, uv + offset) * weight;
            totalWeight += weight;
        }
    }
    
    return result / totalWeight;
}

// Sample texture with anisotropic filtering effect
vec4 sampleTextureAnisotropic(sampler2D tex, vec2 uv, vec2 dx, vec2 dy, float maxSamples) {
    vec4 color = vec4(0.0);
    float numSamples = min(max(length(dx) / length(dy), length(dy) / length(dx)), maxSamples);
    
    for(float i = 0.0; i < numSamples; i += 1.0) {
        float t = i / numSamples - 0.5;
        color += texture2D(tex, uv + dx * t);
    }
    
    return color / numSamples;
}

```