// UV Mapping Texturing Module
// Implements standard UV mapping and texturing functions

// Planar UV mapping
vec2 planarMapping(vec3 position, vec3 axis) {
    if (abs(axis.x) > abs(axis.y) && abs(axis.x) > abs(axis.z)) {
        return position.yz;
    } else if (abs(axis.y) > abs(axis.z)) {
        return position.xz;
    } else {
        return position.xy;
    }
}

// Spherical UV mapping
vec2 sphericalMapping(vec3 normal) {
    float phi = acos(normal.y);
    float theta = atan(normal.x, normal.z);
    return vec2(theta / (2.0 * 3.14159), phi / 3.14159);
}

// Cylindrical UV mapping
vec2 cylindricalMapping(vec3 position) {
    float u = atan(position.x, position.z) / (2.0 * 3.14159);
    float v = position.y;
    return vec2(u, v);
}

// UV tiling and offset
vec2 applyUVTransform(vec2 uv, vec2 offset, vec2 scale, float rotation) {
    // Apply offset
    uv += offset;

    // Apply rotation
    if (rotation != 0.0) {
        float s = sin(rotation);
        float c = cos(rotation);
        mat2 rot = mat2(c, -s, s, c);
        uv = rot * uv;
    }

    // Apply scale
    uv *= scale;

    return uv;
}

// Triplanar texturing for complex objects
vec3 triplanarTexture(sampler2D tex, vec3 worldPos, vec3 normal, float blendStrength) {
    // Get UVs for each axis
    vec2 uvX = worldPos.zy;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;

    // Sample the texture from each axis
    vec3 texX = texture(tex, uvX).rgb;
    vec3 texY = texture(tex, uvY).rgb;
    vec3 texZ = texture(tex, uvZ).rgb;

    // Get blending weights based on normal
    vec3 blend = pow(abs(normal), vec3(blendStrength));
    blend = blend / (blend.x + blend.y + blend.z);

    // Blend the textures
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}

// Texture blending modes
vec3 blendMultiply(vec3 base, vec3 blend) {
    return base * blend;
}

vec3 blendOverlay(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
        step(0.5, base)
    );
}

vec3 blendSoftLight(vec3 base, vec3 blend) {
    return (1.0 - 2.0 * blend) * base * base + 2.0 * blend * base;
}

// Texture coordinate generation with tiling
vec2 generateUV(vec2 position, vec2 tiling, vec2 offset) {
    vec2 uv = position * tiling + offset;
    uv.x = fract(uv.x);  // Wrap coordinates
    uv.y = fract(uv.y);
    return uv;
}

// Parallax mapping for depth effect
vec2 parallaxMapping(sampler2D heightMap, vec2 uv, vec3 viewDir) {
    // Scale view direction to sample range
    vec3 v = normalize(viewDir);
    v.xy /= v.z;

    // Get height from texture
    float height = texture(heightMap, uv).r;

    // Calculate offset
    vec2 offset = -v.xy * height * 0.02;
    return uv + offset;
}

// Normal map UV calculation
vec3 calcNormalFromMap(sampler2D normalMap, vec2 uv, vec3 pos, vec3 normal, vec3 tangent) {
    // Get tangent space normal
    vec3 tangentNormal = texture(normalMap, uv).rgb;
    tangentNormal = normalize(tangentNormal * 2.0 - 1.0);

    // Convert to world space
    vec3 T = normalize(tangent);
    vec3 N = normalize(normal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    return TBN * tangentNormal;
}

// UV animation for scrolling textures
vec2 animatedUV(vec2 uv, vec2 scrollSpeed, float time) {
    return uv + scrollSpeed * time;
}

// Multiple texture blending
vec4 blendTextures(sampler2D tex1, sampler2D tex2, vec2 uv1, vec2 uv2, float blendFactor) {
    vec4 color1 = texture(tex1, uv1);
    vec4 color2 = texture(tex2, uv2);

    return mix(color1, color2, blendFactor);
}