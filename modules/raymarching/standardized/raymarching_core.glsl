// Core raymarching module
// Standardized raymarching algorithm implementations

// Basic raymarching function
vec2 raymarch(vec3 ro, vec3 rd, float maxDist, int maxSteps) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled
    
    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        d = map(p).x;
        if(d < 0.001 || t > maxDist) break;
        t += d;
    }
    
    return vec2(t, d);
}

// Raymarching with adaptive step size
vec2 raymarchAdaptive(vec3 ro, vec3 rd, float maxDist, int maxSteps) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled
    float f = 1.0; // Adaptive factor
    
    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        d = map(p).x;
        if(d < 0.001 || t > maxDist) break;
        
        // Adaptive step size based on distance
        float adaptiveStep = d * f;
        t += max(0.01, adaptiveStep);
        
        // Reduce the factor as we get closer to surface
        f = 0.5 + 0.5 * min(1.0, d * 4.0);
    }
    
    return vec2(t, d);
}

// Raymarching with early termination optimization
vec2 raymarchOptimized(vec3 ro, vec3 rd, float maxDist, int maxSteps) {
    float d; // Distance to closest surface
    float t = 0.0; // Total distance traveled
    
    for(int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        d = map(p).x;
        
        // Early termination if we're very close to surface
        if(d < 0.0001) return vec2(t, d);
        
        // Stop if we've gone too far
        if(t > maxDist) return vec2(maxDist, d);
        
        t += d;
    }
    
    return vec2(t, d);
}
