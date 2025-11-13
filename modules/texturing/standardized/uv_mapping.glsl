// UV mapping module
// Standardized UV mapping implementations

// Basic UV coordinate calculation
vec2 calculateUV(vec2 fragCoord, vec2 resolution) {
    return fragCoord.xy / resolution.xy;
}

// UV with aspect ratio correction
vec2 calculateUVCorrected(vec2 fragCoord, vec2 resolution) {
    vec2 uv = fragCoord.xy / resolution.xy;
    uv.x *= resolution.x / resolution.y;
    return uv;
}

// Tiled UV mapping with repeat
vec2 tileUV(vec2 uv, float repeat) {
    return fract(uv * repeat);
}

// Offset UV mapping
vec2 offsetUV(vec2 uv, vec2 offset) {
    return uv + offset;
}

// Scale UV mapping
vec2 scaleUV(vec2 uv, vec2 scale) {
    return uv * scale;
}

// Rotate UV coordinates around center
vec2 rotateUV(vec2 uv, float angle) {
    uv -= 0.5; // Center origin
    float s = sin(angle);
    float c = cos(angle);
    uv = mat2(c, -s, s, c) * uv;
    uv += 0.5; // Return to [0,1] space
    return uv;
}

// Polar UV mapping (radial)
vec2 cartesianToPolar(vec2 uv) {
    uv -= 0.5; // Center origin
    float r = length(uv);
    float theta = atan(uv.y, uv.x);
    return vec2(r, theta);
}

// Cylindrical UV mapping
vec3 cartesianToCylindrical(vec3 pos) {
    float r = length(pos.xz);
    float theta = atan(pos.z, pos.x);
    return vec3(r, pos.y, theta);
}

// Spherical UV mapping
vec3 cartesianToSpherical(vec3 pos) {
    float r = length(pos);
    float theta = atan(pos.z, pos.x);
    float phi = acos(pos.y / r);
    return vec3(r, theta, phi);
}
