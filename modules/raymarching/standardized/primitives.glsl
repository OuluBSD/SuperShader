// Primitives module
// Standardized primitive shapes for raymarching

// Transform primitive by rotation
vec3 rotateX(vec3 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(1.0, 0.0, 0.0,
                0.0, c, -s,
                0.0, s, c) * p;
}

vec3 rotateY(vec3 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(c, 0.0, s,
                0.0, 1.0, 0.0,
                -s, 0.0, c) * p;
}

vec3 rotateZ(vec3 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(c, -s, 0.0,
                s, c, 0.0,
                0.0, 0.0, 1.0) * p;
}

// Transform primitive by translation
vec3 translate(vec3 p, vec3 t) {
    return p - t;
}

// Transform primitive by scaling
vec3 scale(vec3 p, float s) {
    return p / s;
}

// Repeat space for creating instances
vec3 repeat(vec3 p, vec3 c) {
    return mod(p, c) - 0.5 * c;
}

// Domain warping
vec3 warp(vec3 p) {
    p.x += sin(p.y * 0.5) * 0.5;
    p.y += cos(p.x * 0.5) * 0.5;
    p.z += sin(p.x * 0.3) * 0.3;
    return p;
}
