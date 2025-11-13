// Distance functions module
// Standardized distance function implementations for raymarching

// Sphere distance function
float sdSphere(vec3 p, float radius) {
    return length(p) - radius;
}

// Box distance function
float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

// Rounded box distance function
float sdRoundedBox(vec3 p, vec3 b, float r) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0)) - r;
}

// Torus distance function
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Cylinder distance function
float sdCylinder(vec3 p, vec2 h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Cone distance function
float sdCone(vec3 p, vec2 c) {
    // c must be normalized
    float q = length(p.xz);
    return dot(c, vec2(q, p.y));
}

// Plane distance function
float sdPlane(vec3 p, vec4 n) {
    // n must be normalized
    return dot(p, n.xyz) + n.w;
}

// Union operation for combining shapes
float opUnion(float d1, float d2) {
    return min(d1, d2);
}

// Subtraction operation
float opSubtraction(float d1, float d2) {
    return max(-d1, d2);
}

// Intersection operation
float opIntersection(float d1, float d2) {
    return max(d1, d2);
}

// Smooth union operation
float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}
