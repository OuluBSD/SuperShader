// Standardized Raymarching Primitives and Distance Functions
// Extracted from analysis of real raymarching shaders

#ifndef RAYMARCHING_PRIMITIVES_GL
#define RAYMARCHING_PRIMITIVES_GL

// ---------------------------------------------------------------------------
// BASIC SDF PRIMITIVES
// ---------------------------------------------------------------------------

// Sphere
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

// Box
float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

// Rounded box
float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Torus
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Cylinder
float sdCylinder(vec3 p, vec2 h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Cone
float sdCone(vec3 p, vec2 c) {
    // c must be normalized
    float q = length(p.xz);
    return dot(c, vec2(q, p.y));
}

// Plane
float sdPlane(vec3 p, vec4 n) {
    // n must be normalized
    return dot(p, n.xyz) + n.w;
}

// Hexagonal prism
float sdHexPrism(vec3 p, vec2 h) {
    vec3 q = abs(p);
    return max(q.z-h.y, max((q.x*0.866025+q.y*0.5), q.y) - h.x);
}

// Capsule / Line SDF
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// ---------------------------------------------------------------------------
// SDF OPERATIONS
// ---------------------------------------------------------------------------

// Union
float opUnion(float d1, float d2) {
    return min(d1, d2);
}

// Subtraction
float opSubtraction(float d1, float d2) {
    return max(-d1, d2);
}

// Intersection
float opIntersection(float d1, float d2) {
    return max(d1, d2);
}

// Smooth union (blended)
float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// Smooth subtraction (blended)
float opSmoothSubtraction(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

// Smooth intersection (blended)
float opSmoothIntersection(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

// Transformations
vec3 opTx(vec3 p, mat4 transform) {
    return (transform * vec4(p, 1.0)).xyz;
}

// Repetition
vec3 opRep(vec3 p, vec3 c) {
    return mod(p, c) - 0.5 * c;
}

// Repetition - polar coordinates
vec2 opRepPolar(vec2 p, float r) {
    float angle = 6.28318530718 / r;
    float a = atan(p.y, p.x);
    a = mod(a, angle) - 0.5 * angle;
    return vec2(cos(a), sin(a)) * length(p);
}

// Mirror
vec3 opMirror(vec3 p, vec3 n) {
    return p - 2.0 * min(dot(p, n), 0.0) * n;
}

// ---------------------------------------------------------------------------
// ADVANCED SDF PRIMITIVES
// ---------------------------------------------------------------------------

// Mandelbulb distance estimation
float sdMandelbulb(vec3 pos, float power, float bailout) {
    vec3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    int iterations = 0;
    
    for (int i = 0; i < 16; i++) {
        r = length(z);
        if (r > bailout) break;
        
        // Convert to polar coordinates
        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        
        // Scale and rotate the point
        float zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;
        
        // Convert back to Cartesian coordinates
        z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
        z += pos;
        iterations++;
    }
    
    return 0.5 * log(r) * r / dr;
}

// Fractal - Menger Sponge (approximation)
float sdMengerSponge(vec3 p, int iterations, float scale) {
    vec3 a = abs(p);
    float d = max(a.x, max(a.y, a.z)) - 0.5;
    
    for (int i = 0; i < iterations; i++) {
        if (i > 0) {
            // Menger sponge iteration
            vec3 q = mod(p * scale, 2.0) - 1.0;
            vec3 r = abs(q);
            float a = max(r.x, max(r.y, r.z)) - 0.5;
            float b = max(max(r.x, r.y) - 0.25, max(max(r.y, r.z) - 0.25, max(r.x, r.z) - 0.25)) - 0.125;
            d = max(d, min(a, max(-b, a - 0.125)));
        }
    }
    
    return d;
}

// Octahedron
float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

// Cylinder (infinite in Y-axis)
float sdCylinderVertical(vec3 p, float r) {
    return length(p.xz) - r;
}

// Cylinder (infinite in X-axis)
float sdCylinderHorizontalX(vec3 p, float r) {
    return length(p.yz) - r;
}

// Cylinder (infinite in Z-axis)
float sdCylinderHorizontalZ(vec3 p, float r) {
    return length(p.xy) - r;
}

// ---------------------------------------------------------------------------
// NOISE-BASED SDFS
// ---------------------------------------------------------------------------

// Hash function for noise
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Value noise
float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);  // Smootherstep
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fractional Brownian Motion (for terrain generation)
float fbm(vec2 p, int octaves, float persistence, float frequency) {
    float total = 0.0;
    float amplitude = 1.0;
    float frequency_mult = frequency;
    
    for (int i = 0; i < octaves; i++) {
        total += valueNoise(p * frequency_mult) * amplitude;
        amplitude *= persistence;
        frequency_mult *= 2.0;
    }
    
    return total;
}

// Terrain-like SDF using noise
float sdTerrain(vec2 p, float heightScale, int octaves, float persistence) {
    float noise = fbm(p, octaves, persistence, 1.0);
    return p.y - noise * heightScale;
}

// Ripple effect for SDF
float sdRipple(vec2 p, float frequency, float amplitude, float speed) {
    float r = length(p);
    return p.y - sin(r * frequency - iTime * speed) * amplitude;
}

// ---------------------------------------------------------------------------
// NORMAL CALCULATION
// ---------------------------------------------------------------------------

// Calculate normal using central differences
vec3 calcNormal(vec3 p, float precision) {
    vec2 e = vec2(precision, 0.0);
    
    return normalize(vec3(
        sdScene(p + e.xyy) - sdScene(p - e.xyy),
        sdScene(p + e.yxy) - sdScene(p - e.yxy),
        sdScene(p + e.yyx) - sdScene(p - e.yyx)
    ));
}

// More accurate normal calculation with 6 points
vec3 calcNormal6(vec3 p, float precision) {
    const vec3 v1 = vec3( 1.0, -1.0, -1.0);
    const vec3 v2 = vec3(-1.0, -1.0,  1.0);
    const vec3 v3 = vec3(-1.0,  1.0, -1.0);
    const vec3 v4 = vec3( 1.0,  1.0,  1.0);
    
    float t1 = sdScene(p + v1 * precision);
    float t2 = sdScene(p + v2 * precision);
    float t3 = sdScene(p + v3 * precision);
    float t4 = sdScene(p + v4 * precision);
    
    return normalize(v1 * t1 + v2 * t2 + v3 * t3 + v4 * t4);
}

// ---------------------------------------------------------------------------
// RAYMARCHING UTILITIES
// ---------------------------------------------------------------------------

// Basic raymarching function
float raymarch(vec3 ro, vec3 rd, float maxDist, float precision) {
    float d = 0.0;
    
    for (int i = 0; i < 128; i++) {
        vec3 p = ro + rd * d;
        float res = sdScene(p);
        
        if (res < precision || d > maxDist) {
            break;
        }
        
        d += res * 0.8; // Conservative step to avoid overstepping
    }
    
    return d;
}

// Optimized raymarching with adaptive stepping
float raymarchAdaptive(vec3 ro, vec3 rd, float maxDist, float minPrecision, float maxPrecision) {
    float d = 0.0;
    float precision = minPrecision;
    
    for (int i = 0; i < 128; i++) {
        vec3 p = ro + rd * d;
        float res = sdScene(p);
        
        if (res < precision || d > maxDist) {
            break;
        }
        
        // Adaptive step size based on distance
        float stepSize = res * 0.8;
        d += max(stepSize, minPrecision);
        precision = minPrecision + (maxPrecision - minPrecision) * smoothstep(0.0, maxDist, d);
    }
    
    return d;
}

// Binary search refinement (after initial march)
float refineHit(vec3 ro, vec3 rd, float tmin, float tmax, float precision) {
    for (int i = 0; i < 10; i++) {
        float t = (tmin + tmax) * 0.5;
        float d = sdScene(ro + rd * t);
        
        if (d < precision) {
            tmax = t;
        } else {
            tmin = t;
        }
    }
    
    return (tmin + tmax) * 0.5;
}

// ---------------------------------------------------------------------------
// COMMON SCENE COMPOSITION FUNCTIONS
// ---------------------------------------------------------------------------

// Scene distance function (to be customized per scene)
float sdScene(vec3 p) {
    // This is a placeholder - each scene will define their own version
    // For now, return a simple sphere
    return sdSphere(p, 1.0);
}

// Multiple objects in scene
vec2 opMultiObjSDF(vec3 p) {
    // Return distance and object ID
    float d1 = sdSphere(p - vec3(0.0, 0.0, 0.0), 1.0);
    float d2 = sdBox(p - vec3(2.0, 0.0, 0.0), vec3(0.5));
    float d3 = sdTorus(p - vec3(-2.0, 0.0, 0.0), vec2(0.8, 0.2));
    
    if (d1 < d2 && d1 < d3) return vec2(d1, 1.0);
    else if (d2 < d3) return vec2(d2, 2.0);
    else return vec2(d3, 3.0);
}

// Hollow sphere
float sdHollowSphere(vec3 p, float r, float thickness) {
    float d = length(p) - r;
    return abs(d) - thickness;
}

// Capped cylinder
float sdCappedCylinder(vec3 p, vec2 h) {
    vec2 w = vec2(length(p.xz), abs(p.y));
    return ((w.y > h.y) ? length(w - h) : length(max(w, h)) - h.y);
}

// Elongated primitive
vec3 opElongate(vec3 p, vec3 c) {
    vec3 q = mod(p, 2.0 * c) - c;
    return p - q;
}

// Onion (shell) operator
float opOnion(vec3 p, float thickness) {
    float d = sdScene(p);  // Original SDF
    return abs(d) - thickness;
}

// Shell operator with variable thickness
float opShell(vec3 p, float thickness) {
    float d = sdScene(p);
    return abs(d) - thickness * (0.5 + 0.5 * sin(iTime));  // Animated thickness
}

#endif // RAYMARCHING_PRIMITIVES_GL