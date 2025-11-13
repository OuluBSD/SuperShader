// Texture blending module
// Standardized texture blending implementations

// Alpha blending between two textures
vec4 alphaBlend(vec4 base, vec4 blend, float alpha) {
    return base * (1.0 - alpha) + blend * alpha;
}

// Multiply blending
vec4 multiplyBlend(vec4 base, vec4 blend) {
    return base * blend;
}

// Additive blending
vec4 additiveBlend(vec4 base, vec4 blend) {
    return min(base + blend, vec4(1.0));
}

// Screen blending
vec4 screenBlend(vec4 base, vec4 blend) {
    return vec4(1.0) - (vec4(1.0) - base) * (vec4(1.0) - blend);
}

// Overlay blending
vec4 overlayBlend(vec4 base, vec4 blend) {
    vec4 result;
    result.r = base.r < 0.5 ? 2.0 * base.r * blend.r : 1.0 - 2.0 * (1.0 - base.r) * (1.0 - blend.r);
    result.g = base.g < 0.5 ? 2.0 * base.g * blend.g : 1.0 - 2.0 * (1.0 - base.g) * (1.0 - blend.g);
    result.b = base.b < 0.5 ? 2.0 * base.b * blend.b : 1.0 - 2.0 * (1.0 - base.b) * (1.0 - blend.b);
    result.a = base.a;
    return result;
}

// Soft light blending
vec4 softLightBlend(vec4 base, vec4 blend) {
    vec4 result;
    result.r = blend.r < 0.5 ? 
        2.0 * base.r * blend.r + base.r * base.r * (1.0 - 2.0 * blend.r) : 
        2.0 * base.r * (1.0 - blend.r) + sqrt(base.r) * (2.0 * blend.r - 1.0);
    result.g = blend.g < 0.5 ? 
        2.0 * base.g * blend.g + base.g * base.g * (1.0 - 2.0 * blend.g) : 
        2.0 * base.g * (1.0 - blend.g) + sqrt(base.g) * (2.0 * blend.g - 1.0);
    result.b = blend.b < 0.5 ? 
        2.0 * base.b * blend.b + base.b * base.b * (1.0 - 2.0 * blend.b) : 
        2.0 * base.b * (1.0 - blend.b) + sqrt(base.b) * (2.0 * blend.b - 1.0);
    result.a = base.a;
    return result;
}

// Blend multiple textures based on a mask
vec4 blendMultipleTextures(sampler2D tex1, sampler2D tex2, sampler2D mask, vec2 uv) {
    vec4 color1 = texture2D(tex1, uv);
    vec4 color2 = texture2D(tex2, uv);
    float maskValue = texture2D(mask, uv).r;
    
    return mix(color1, color2, maskValue);
}

// Blend textures with triplanar
vec4 blendTriplanarTextures(sampler2D tex1, sampler2D tex2, vec3 worldPos, vec3 normal) {
    vec4 color1 = sampleTriplanar(tex1, worldPos, normal);
    vec4 color2 = sampleTriplanar(tex2, worldPos, normal);
    
    // Use normal to determine blend weights
    vec3 blendWeights = abs(normal);
    blendWeights = normalize(max(blendWeights, 0.00001));
    blendWeights /= (blendWeights.x + blendWeights.y + blendWeights.z);
    
    return color1 * blendWeights.x + color2 * blendWeights.y + mix(color1, color2, 0.5) * blendWeights.z;
}
