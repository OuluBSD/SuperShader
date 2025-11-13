#!/usr/bin/env python3
"""
Ray Marching Lighting Module
Extracted from common lighting patterns in shader analysis
Pattern frequency: 282 occurrences
"""

# Pseudocode for ray marching lighting
pseudocode = """
// Ray Marching Lighting Implementation

// Standard ray marching with lighting
float rayMarch(vec3 ro, vec3 rd, float maxd, float precis) {
    float d = 0.0;
    for(int i = 0; i < 250; i++) {
        if(abs(d) < precis || d > maxd) break;
        d = map(ro + rd * d);
    }
    return d;
}

// Calculate normal using ray marching
vec3 calcNormal(vec3 pos, float eps) {
    vec2 e = vec2(eps, 0.0);
    vec3 n = vec3(
        map(pos + e.xyy) - map(pos - e.xyy),
        map(pos + e.yxy) - map(pos - e.yxy),
        map(pos + e.yyx) - map(pos - e.yyx)
    );
    return normalize(n);
}

// Soft shadows using ray marching
float calcSoftshadow(vec3 ro, vec3 rd, float mint, float tmax) {
    float res = 1.0;
    float t = mint;
    for(int i = 0; i < 16; i++) {
        float h = map(ro + rd * t);
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.02, 0.10);
        if(res < 0.001 || t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Ambient occlusion using ray marching
float calcAO(vec3 pos, vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    for(int i = 0; i < 5; i++) {
        float h = 0.01 + 0.12 * float(i) / 4.0;
        float d = map(pos + h * nor);
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

// Complete ray marching lighting calculation
vec3 raymarchLighting(vec3 ro, vec3 rd) {
    float d = rayMarch(ro, rd, 20.0, 0.01);
    
    if(d < 20.0) {
        vec3 pos = ro + rd * d;
        vec3 nor = calcNormal(pos, 0.01);
        
        // Lighting calculations
        vec3 lightPos = vec3(5.0, 5.0, 5.0);
        vec3 lightDir = normalize(lightPos - pos);
        
        float occ = calcAO(pos, nor);
        float sha = calcSoftshadow(pos, lightDir, 0.02, 25.0);
        
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0);
        float bac = clamp(dot(nor, normalize(vec3(-lightDir.x, 0.0, -lightDir.z))), 0.0, 1.0) * clamp(1.0 - d / 20.0, 0.0, 1.0);
        
        vec3 col = vec3(0.05, 0.10, 0.20); // Ambient
        col += 1.50 * dif * vec3(1.00, 0.90, 0.70); // Diffuse
        col += 0.50 * occ * vec3(0.40, 0.60, 1.00); // Ambient occlusion
        col += 0.25 * bac * vec3(0.25, 0.20, 0.15); // Back lighting
        col += 2.00 * sha * vec3(1.00, 0.90, 0.70); // Shadow
        
        // Apply distance fading
        col *= exp(-0.1 * d);
        
        return col;
    } else {
        // Background
        return vec3(0.05, 0.10, 0.20);
    }
}
"""

def get_pseudocode():
    """Return the pseudocode for this lighting module"""
    return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'raymarching_lighting',
        'type': 'lighting',
        'patterns': ['Ray Marching', 'Normal Mapping'],
        'frequency': 282,
        'dependencies': ['normal_mapping', 'diffuse_lighting'],
        'conflicts': [],
        'description': 'Ray marching lighting calculations with soft shadows and ambient occlusion'
    }