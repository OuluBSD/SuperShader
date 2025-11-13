// Triplanar mapping module
// Standardized triplanar mapping implementations

// Basic triplanar mapping for a single texture
vec4 triplanarSample(sampler2D tex, vec3 worldPos, vec3 normal, float blendSharpness) {
    // Project world position onto each axis
    vec2 uvX = worldPos.yz;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, uvX);
    vec4 texY = texture2D(tex, uvY);
    vec4 texZ = texture2D(tex, uvZ);
    
    // Use normal to determine blending weights
    vec3 blend = pow(abs(normal), vec3(blendSharpness));
    blend = normalize(max(blend, 0.00001)); // Avoid division by zero
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}

// Triplanar mapping with separate normal and color textures
vec4 triplanarSampleTextured(sampler2D colorTex, sampler2D normalTex, vec3 worldPos, vec3 normal, float blendSharpness) {
    // Use the same triplanar approach for both textures
    vec2 uvX = worldPos.yz;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;
    
    // Sample color textures
    vec4 colX = texture2D(colorTex, uvX);
    vec4 colY = texture2D(colorTex, uvY);
    vec4 colZ = texture2D(colorTex, uvZ);
    
    // Sample normal textures
    vec3 normX = sampleNormalMap(normalTex, uvX);
    vec3 normY = sampleNormalMap(normalTex, uvY);
    vec3 normZ = sampleNormalMap(normalTex, uvZ);
    
    // Use normal to determine blending weights
    vec3 blend = pow(abs(normal), vec3(blendSharpness));
    blend = normalize(max(blend, 0.00001));
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    vec4 blendedColor = colX * blend.x + colY * blend.y + colZ * blend.z;
    
    return blendedColor;
}

// Triplanar mapping with scaling factor
vec4 triplanarSampleScaled(sampler2D tex, vec3 worldPos, vec3 normal, float scale, float blendSharpness) {
    // Project world position onto each axis with scaling
    vec2 uvX = worldPos.yz * scale;
    vec2 uvY = worldPos.xz * scale;
    vec2 uvZ = worldPos.xy * scale;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, uvX);
    vec4 texY = texture2D(tex, uvY);
    vec4 texZ = texture2D(tex, uvZ);
    
    // Use normal to determine blending weights
    vec3 blend = pow(abs(normal), vec3(blendSharpness));
    blend = normalize(max(blend, 0.00001)); // Avoid division by zero
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}

// Blended triplanar with smooth transitions
vec4 triplanarSmooth(sampler2D tex, vec3 worldPos, vec3 normal) {
    // Use the smooth version of the triplanar mapping
    vec2 uvX = worldPos.yz;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;
    
    // Sample textures for each projection
    vec4 texX = texture2D(tex, uvX);
    vec4 texY = texture2D(tex, uvY);
    vec4 texZ = texture2D(tex, uvZ);
    
    // Use smooth blending with absolute normal values
    vec3 blend = abs(normal);
    blend = normalize(max(blend, 0.00001));
    blend /= (blend.x + blend.y + blend.z);
    
    // Blend the three samples
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}
