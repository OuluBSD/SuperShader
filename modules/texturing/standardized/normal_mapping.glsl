// Normal mapping module
// Standardized normal mapping implementations

// Calculate tangent space matrix from world position and UV coordinates
mat3 calculateTangentSpace(vec3 worldPos, vec3 worldNormal, vec2 uv) {
    // Calculate tangent and bitangent
    vec3 dp1 = dFdx(worldPos);
    vec3 dp2 = dFdy(worldPos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    
    vec3 tangent = normalize(dp2 * duv1.x - dp1 * duv2.x);
    vec3 bitangent = normalize(dp1 * duv2.y - dp2 * duv1.y);
    
    // Create TBN matrix
    mat3 TBN = mat3(tangent, bitangent, worldNormal);
    return TBN;
}

// Sample normal map and convert from [0,1] to [-1,1] range
vec3 sampleNormalMap(sampler2D normalMap, vec2 uv) {
    vec3 tangentNormal = texture2D(normalMap, uv).xyz;
    tangentNormal = normalize(tangentNormal * 2.0 - 1.0); // Convert from [0,1] to [-1,1]
    return tangentNormal;
}

// Apply normal map in world space
vec3 applyNormalMap(sampler2D normalMap, vec2 uv, vec3 worldNormal, mat3 TBN) {
    vec3 tangentNormal = sampleNormalMap(normalMap, uv);
    vec3 worldNormalMapped = normalize(TBN * tangentNormal);
    return worldNormalMapped;
}

// Perturb normal with normal map in fragment shader
vec3 perturbNormal(vec3 pos, vec3 normal, vec2 uv, sampler2D normalMap) {
    mat3 TBN = calculateTangentSpace(pos, normal, uv);
    vec3 newNormal = applyNormalMap(normalMap, uv, normal, TBN);
    return newNormal;
}

// Parallax mapping effect
vec2 parallaxMapping(sampler2D heightMap, vec2 uv, vec3 viewDir) {
    float height = texture2D(heightMap, uv).r;
    vec3 viewDirTangent = normalize(viewDir);
    vec2 p = viewDirTangent.xy / viewDirTangent.z * (height * 0.02);
    return uv - p;
}

// Steep parallax mapping with multiple samples
vec2 steepParallaxMapping(sampler2D heightMap, vec2 uv, vec3 viewDir, int numLayers) {
    float layerDepth = 1.0 / float(numLayers);
    float currentLayerDepth = 0.0;
    
    vec2 deltaUV = viewDir.xy * 0.02 / (viewDir.z * float(numLayers));
    
    vec2 currentUV = uv;
    float currentDepthValue = texture2D(heightMap, currentUV).r;
    
    while(currentLayerDepth < currentDepthValue) {
        currentUV -= deltaUV;
        currentDepthValue = texture2D(heightMap, currentUV).r;
        currentLayerDepth += layerDepth;
    }
    
    return currentUV;
}
