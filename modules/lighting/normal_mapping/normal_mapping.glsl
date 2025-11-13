// Normal mapping module
// Implements tangent space normal mapping for lighting calculations

// Calculate TBN matrix for normal mapping
mat3 calculateTBN(vec3 normal, vec3 tangent, vec3 bitangent) {
    return mat3(tangent, bitangent, normal);
}

// Calculate tangent from texture coordinates
vec3 calculateTangent(vec3 pos1, vec3 pos2, vec3 pos3, vec2 uv1, vec2 uv2, vec2 uv3) {
    vec3 edge1 = pos2 - pos1;
    vec3 edge2 = pos3 - pos1;
    vec2 deltaUV1 = uv2 - uv1;
    vec2 deltaUV2 = uv3 - uv1;
    
    float f = 1.0 / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
    
    vec3 tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
    vec3 bitangent = f * (-deltaUV2.x * edge1 + deltaUV1.x * edge2);
    
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);
    
    return tangent;
}

// Apply normal mapping using tangent space
vec3 applyNormalMapping(vec3 normal, vec3 tangent, vec3 bitangent, vec2 texCoords, sampler2D normalMap) {
    mat3 TBN = calculateTBN(normal, tangent, bitangent);
    vec3 newNormal = texture(normalMap, texCoords).rgb;
    newNormal = newNormal * 2.0 - 1.0; // Convert from [0,1] to [-1,1]
    
    return normalize(TBN * newNormal);
}

// Alternative normal mapping approach using world space
vec3 normalMapWorldSpace(vec3 normal, vec3 pos, vec2 uv, sampler2D normalTex) {
    // Get normal from normal map
    vec3 n = texture(normalTex, uv).rgb;
    n = normalize(n * 2.0 - 1.0);
    
    // Create TBN matrix
    vec3 q1 = dFdx(pos);
    vec3 q2 = dFdy(pos);
    vec2 st1 = dFdx(uv);
    vec2 st2 = dFdy(uv);
    
    vec3 N = normalize(normal);
    vec3 T = normalize(q1 * st2.t - q2 * st1.t);
    vec3 B = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);
    
    return normalize(TBN * n);
}

// Simple normal mapping function
vec3 getNormalFromMap(vec2 uv, vec3 pos, vec3 normal, vec3 tangent) {
    // Get normal from texture
    vec3 texNormal = texture(iChannel0, uv).rgb;
    texNormal = normalize(texNormal * 2.0 - 1.0);
    
    // Create TBN matrix
    vec3 T = normalize(tangent);
    vec3 B = normalize(cross(normal, T));
    vec3 N = normalize(normal);
    
    // Ensure TBN matrix is orthogonal
    T = normalize(T - dot(T, N) * N);
    B = cross(N, T);
    
    mat3 TBN = mat3(T, B, N);
    return normalize(TBN * texNormal);
}

// Perturb normal with normal map for lighting
vec3 perturbNormal(vec3 position, vec3 normal, vec2 uv, sampler2D normalMap) {
    vec3 tangent = dFdx(position);
    vec3 bitangent = dFdy(position);
    
    // Get normal from normal map texture
    vec3 mapN = texture(normalMap, uv).xyz;
    mapN = mapN * 2.0 - 1.0;
    
    // Create TBN matrix
    mat3 TBN = mat3(
        normalize(tangent),
        normalize(bitangent),
        normalize(normal)
    );
    
    return normalize(TBN * mapN);
}