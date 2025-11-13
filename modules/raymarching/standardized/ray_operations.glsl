// Ray operations module
// Standardized ray operation functions for raymarching

// Calculate normal using gradient of distance field
vec3 calcNormal(vec3 p, float epsilon) {
    vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x
    ));
}

// Calculate normal with adaptive epsilon
vec3 calcNormalAdaptive(vec3 p) {
    float dx = map(p + vec3(0.001, 0.0, 0.0)).x - map(p - vec3(0.001, 0.0, 0.0)).x;
    float dy = map(p + vec3(0.0, 0.001, 0.0)).x - map(p - vec3(0.0, 0.001, 0.0)).x;
    float dz = map(p + vec3(0.0, 0.0, 0.001)).x - map(p - vec3(0.0, 0.0, 0.001)).x;
    return normalize(vec3(dx, dy, dz));
}

// Calculate ambient occlusion based on distance field
float calcAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for(int i = 0; i < 5; i++) {
        float h = 0.01 + 0.15 * float(i) / 4.0;
        float d = map(p + h * n).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

// Calculate soft shadows
float calcSoftShadow(vec3 ro, vec3 rd, float mint, float tmax, float k) {
    float res = 1.0;
    float t = mint;
    for(int i = 0; i < 32; i++) {
        float h = map(ro + rd * t).x;
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.10);
        if(res < 0.001 || t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}
