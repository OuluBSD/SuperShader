#version 330 core

precision highp float;

in vec3 position;
in vec3 oldPosition;
in vec3 acceleration;
in float deltaTime;
in vec2 uv;
in vec3 worldPos;
in vec3 normal;
in vec3 tangent;
in float time;
in vec2 tiling;
in vec2 offset;
in float blendFactor;
in vec2 scrollSpeed;

out vec3 newPosition;
out vec2 outputUV;
out vec4 sampledColor;
out vec3 worldNormal;

uniform float deltaTime;
uniform float time;
uniform vec2 tiling;
uniform vec2 offset;
uniform float blendFactor;
uniform vec2 scrollSpeed;

float perlinNoise(vec2 coord, float scale, float time) {
    // Scale the coordinates
    vec2 scaledCoord = coord * scale;

    // Calculate integer and fractional parts
    vec2 i = floor(scaledCoord);
    vec2 f = fract(scaledCoord);

    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);

    // Generate random values at the four corners
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Interpolate between the values
    float value = mix(a, b, u.x) +
                  (c - a) * u.y * (1.0 - u.x) +
                  (d - b) * u.x * u.y;

    return value;
}
float random(vec2 coord) {
    return fract(sin(dot(coord, vec2(12.9898, 78.233))) * 43758.5453);
}
float fbm(vec2 coord, float scale, float time) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < 4; i++) {
        value += amplitude * perlinNoise(coord * frequency, scale, time);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}
vec3 verletIntegration(vec3 position, vec3 oldPosition, vec3 acceleration, float deltaTime) {
    // Calculate velocity from positions
    vec3 velocity = position - oldPosition;

    // Apply acceleration to velocity
    velocity += acceleration * deltaTime * deltaTime;

    // Calculate new position
    vec3 newPosition = position + velocity;

    return newPosition;
}
vec3 verletIntegrationDamped(vec3 position, vec3 oldPosition, vec3 acceleration, float deltaTime, float damping) {
    // Calculate velocity from positions
    vec3 velocity = position - oldPosition;

    // Apply damping to velocity
    velocity *= damping;

    // Apply acceleration to velocity
    velocity += acceleration * deltaTime * deltaTime;

    // Calculate new position
    vec3 newPosition = position + velocity;

    return newPosition;
}
vec3 verletConstrained(vec3 position, vec3 oldPosition, vec3 acceleration, float deltaTime, vec3 constraintCenter, float constraintRadius) {
    // Standard Verlet integration
    vec3 velocity = position - oldPosition;
    velocity += acceleration * deltaTime * deltaTime;
    vec3 newPosition = position + velocity;

    // Apply position constraints
    vec3 toCenter = newPosition - constraintCenter;
    float distance = length(toCenter);

    if (distance > constraintRadius) {
        // Project position back to constraint boundary
        newPosition = constraintCenter + normalize(toCenter) * constraintRadius;
    }

    return newPosition;
}
vec3 verletPhysicsStep(vec3 position, vec3 oldPosition, vec3 gravity, vec3 externalForces, float deltaTime) {
    // Calculate total acceleration (gravity + external forces)
    vec3 totalAcceleration = gravity + externalForces;

    // Apply Verlet integration
    vec3 velocity = position - oldPosition;
    velocity += totalAcceleration * deltaTime * deltaTime;
    vec3 newPosition = position + velocity;

    return newPosition;
}
vec3 verletParticle(vec3 position, vec3 oldPosition, float mass, vec3 totalForce, float deltaTime) {
    // Calculate acceleration using F = ma -> a = F/m
    vec3 acceleration = totalForce / mass;

    // Apply Verlet integration
    vec3 velocity = position - oldPosition;
    velocity += acceleration * deltaTime * deltaTime;
    vec3 newPosition = position + velocity;

    return newPosition;
}
vec2 planarMapping(vec3 position, vec3 axis) {
    if (abs(axis.x) > abs(axis.y) && abs(axis.x) > abs(axis.z)) {
        return position.yz;
    } else if (abs(axis.y) > abs(axis.z)) {
        return position.xz;
    } else {
        return position.xy;
    }
}
vec2 sphericalMapping(vec3 normal) {
    float phi = acos(normal.y);
    float theta = atan(normal.x, normal.z);
    return vec2(theta / (2.0 * 3.14159), phi / 3.14159);
}
vec2 cylindricalMapping(vec3 position) {
    float u = atan(position.x, position.z) / (2.0 * 3.14159);
    float v = position.y;
    return vec2(u, v);
}
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
vec2 generateUV(vec2 position, vec2 tiling, vec2 offset) {
    vec2 uv = position * tiling + offset;
    uv.x = fract(uv.x);  // Wrap coordinates
    uv.y = fract(uv.y);
    return uv;
}
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
vec2 animatedUV(vec2 uv, vec2 scrollSpeed, float time) {
    return uv + scrollSpeed * time;
}
vec4 blendTextures(sampler2D tex1, sampler2D tex2, vec2 uv1, vec2 uv2, float blendFactor) {
    vec4 color1 = texture(tex1, uv1);
    vec4 color2 = texture(tex2, uv2);
    
    return mix(color1, color2, blendFactor);
}
void main() {
    // Main shader function
    // Set default output color
    // TODO: Implement actual shading logic based on modules
    sampledColor = vec4(1.0);  // Default white
}