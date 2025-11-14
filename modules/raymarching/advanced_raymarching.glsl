// Advanced Raymarching Utilities
// Based on analysis of complex raymarching shaders

#ifndef ADVANCED_RAYMARCHING_GL
#define ADVANCED_RAYMARCHING_GL

#include "raymarching_primitives.glsl"

// ---------------------------------------------------------------------------
// ADVANCED LIGHTING MODELS
// ---------------------------------------------------------------------------

// Ambient occlusion approximation
float ambientOcclusion(vec3 p, vec3 n, float maxDist, int steps) {
    float occ = 0.0;
    float sca = 1.0;
    
    for (int i = 0; i < steps; i++) {
        float h = sdScene(p + n * pow(2.0, float(i)) * 0.1);
        occ += (pow(2.0, float(-i)) * max(0.0, h));
    }
    
    return 1.0 - occ;
}

// Soft shadows
float softShadow(vec3 ro, vec3 rd, float mint, float tmax, float k) {
    float res = 1.0;
    float t = mint;
    
    for (int i = 0; i < 32; i++) {
        if (t >= tmax) break;
        float h = sdScene(ro + rd * t);
        res = min(res, k * h / t);
        t += clamp(h, 0.01, 0.10);
    }
    
    return clamp(res, 0.0, 1.0);
}

// Reflection
vec3 reflectColor(vec3 ro, vec3 rd, vec3 pos, vec3 normal, float roughness) {
    vec3 reflected = reflect(rd, normal);
    vec3 refOrigin = pos + normal * 0.01;
    float refDistance = raymarch(refOrigin, reflected, 20.0, 0.01);
    
    if (refDistance < 20.0) {
        vec3 refPos = refOrigin + reflected * refDistance;
        vec3 refNormal = calcNormal(refPos, 0.01);
        return phongLighting(refPos, refNormal, roughness);
    }
    
    return vec3(0.0); // Or environment color
}

// Refraction
vec3 refractColor(vec3 ro, vec3 rd, vec3 pos, vec3 normal, float ior, float roughness) {
    vec3 direction = refract(rd, normal, ior);
    if (dot(direction, direction) < 0.0) return reflectColor(ro, rd, pos, normal, roughness); // Total internal reflection
    
    vec3 refOrigin = pos - normal * 0.01;
    float refDistance = raymarch(refOrigin, direction, 20.0, 0.01);
    
    if (refDistance < 20.0) {
        vec3 refractedPos = refOrigin + direction * refDistance;
        vec3 refractNormal = calcNormal(refractedPos, 0.01);
        return phongLighting(refractedPos, refractNormal, roughness);
    }
    
    return vec3(0.0);
}

// Fresnel term for realistic reflections
float fresnel(vec3 I, vec3 N, float ior) {
    float cosi = clamp(dot(I, N), -1.0, 1.0);
    float etai = 1.0, etat = ior;
    if (cosi > 0.0) { swap(etai, etat); }
    
    // Compute sini using Snell's law
    float sint = etai / etat * sqrt(max(0.0, 1.0 - cosi * cosi));
    if (sint >= 1.0) {
        // Total internal reflection
        return 1.0;
    } else {
        float cost = sqrt(max(0.0, 1.0 - sint * sint));
        cosi = abs(cosi);
        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        return (Rs * Rs + Rp * Rp) / 2.0;
    }
}

// ---------------------------------------------------------------------------
// POST-PROCESSING EFFECTS FOR RAYMARCHED SCENES
// ---------------------------------------------------------------------------

// Depth of field simulation
vec3 depthOfField(vec3 color, vec3 ro, vec3 rd, vec2 uv, float focusDistance, float aperture) {
    // Create a randomized ray based on DOF parameters
    vec2 disk = vec2(
        (hash(uv + iTime) * 2.0 - 1.0) * aperture,
        (hash(uv + iTime + 1.0) * 2.0 - 1.0) * aperture
    );
    
    vec3 focusPoint = ro + rd * focusDistance;
    vec3 newOrigin = ro + disk.x * vec3(1.0, 0.0, 0.0) + disk.y * vec3(0.0, 1.0, 0.0);
    vec3 newDir = normalize(focusPoint - newOrigin);
    
    // Perform raymarch with the new ray
    float dist = raymarch(newOrigin, newDir, 30.0, 0.01);
    vec3 newColor = vec3(0.0);
    
    if (dist < 30.0) {
        vec3 hitPos = newOrigin + newDir * dist;
        vec3 normal = calcNormal(hitPos, 0.01);
        newColor = phongLighting(hitPos, normal, 0.1);
    }
    
    return mix(color, newColor, smoothstep(0.0, 0.5, abs(dist - focusDistance) / 10.0));
}

// Motion blur simulation
vec3 motionBlur(vec3 color, vec3 ro, vec3 rd, float velocity, float time) {
    const int samples = 8;
    vec3 accumulated = vec3(0.0);
    
    for (int i = 0; i < samples; i++) {
        float t_sample = time + (float(i) / float(samples) - 0.5) * velocity;
        vec3 rd_sample = rd + vec3(sin(t_sample), cos(t_sample), sin(t_sample * 0.7)) * 0.01;
        
        float dist = raymarch(ro, rd_sample, 30.0, 0.01);
        if (dist < 30.0) {
            vec3 hitPos = ro + rd_sample * dist;
            vec3 normal = calcNormal(hitPos, 0.01);
            accumulated += phongLighting(hitPos, normal, 0.1);
        }
    }
    
    return accumulated / float(samples);
}

// Volumetric effects (fog, atmosphere)
vec3 volumetricFog(vec3 ro, vec3 rd, float tmin, float tmax, vec3 color, vec3 fogColor, float density) {
    float integral = 0.0;
    float steps = 16.0;
    float dt = (tmax - tmin) / steps;
    
    for (float i = 0.0; i < steps; i++) {
        float t = tmin + (i + 0.5) * dt;
        vec3 pos = ro + rd * t;
        float d = sdScene(pos);
        
        // Approximate density based on distance from surface
        float fog = exp(-density * max(0.0, -d));
        integral += (1.0 - fog) * dt;
    }
    
    return mix(color, fogColor, 1.0 - exp(-density * integral));
}

// ---------------------------------------------------------------------------
// CAMERA SYSTEMS
// ---------------------------------------------------------------------------

// Standard camera setup
mat3 setCamera(vec3 ro, vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    return mat3(cu, cv, cw);
}

// Orthographic camera
mat3 orthographicCamera(vec3 forward, vec3 up) {
    vec3 right = normalize(cross(forward, up));
    up = normalize(cross(right, forward));
    return mat3(right, up, forward);
}

// Perspective camera with variable FOV
mat3 perspectiveCamera(vec3 ro, vec3 ta, float fov) {
    vec3 forward = normalize(ta - ro);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));
    
    // Apply FOV scaling
    right *= tan(fov * 0.5);
    up *= tan(fov * 0.5);
    
    return mat3(right, up, forward);
}

// ---------------------------------------------------------------------------
// ADVANCED SCENE DEFINITIONS
// ---------------------------------------------------------------------------

// Multi-material scene with material IDs
struct MaterialInfo {
    float distance;
    float material_id;
    vec3 color;
};

MaterialInfo multiMaterialScene(vec3 p) {
    float d1 = sdSphere(p - vec3(0.0, 0.0, 0.0), 1.0);
    float d2 = sdBox(p - vec3(2.0, 0.0, 0.0), vec3(0.8));
    float d3 = sdTorus(p - vec3(-2.0, 0.0, 0.0), vec2(0.8, 0.2));
    
    // Determine which primitive is closest
    if (d1 <= d2 && d1 <= d3) {
        return MaterialInfo(d1, 1.0, vec3(1.0, 0.0, 0.0)); // Red sphere
    } else if (d2 <= d3) {
        return MaterialInfo(d2, 2.0, vec3(0.0, 1.0, 0.0)); // Green box
    } else {
        return MaterialInfo(d3, 3.0, vec3(0.0, 0.0, 1.0)); // Blue torus
    }
}

// Phong lighting model for raymarched scenes
vec3 phongLighting(vec3 pos, vec3 normal, float roughness) {
    // Light setup
    vec3 lightPos = vec3(5.0, 5.0, 5.0);
    vec3 lightColor = vec3(1.0, 1.0, 0.9);
    vec3 eyePos = pos + vec3(0.0, 0.0, -5.0); // Approximate eye position
    
    // Phong lighting components
    vec3 lightDir = normalize(lightPos - pos);
    vec3 viewDir = normalize(eyePos - pos);
    vec3 reflectDir = reflect(-lightDir, normal);
    
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    
    // Material color based on position or material ID
    vec3 materialColor = vec3(0.5); // Default
    // Could use material ID to determine color
    
    return (ambient + diffuse + specular) * materialColor;
}

// Physically Based Rendering (simplified)
vec3 pbrLighting(vec3 pos, vec3 normal, vec3 viewDir, vec3 albedo, float metallic, float roughness, vec3 lightPos, vec3 lightColor) {
    vec3 lightDir = normalize(lightPos - pos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    
    // Roughness clamping to avoid numerical issues
    roughness = max(roughness, 0.05);
    
    // Fresnel term (Schlick approximation)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 fresnel = F0 + (1.0 - F0) * pow(1.0 - dot(halfwayDir, viewDir), 5.0);
    
    // Normal distribution function (Blinn-Phong approximation)
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = max(dot(normal, halfwayDir) * dot(normal, halfwayDir) * (alpha2 - 1.0) + 1.0, 0.001);
    float NDF = alpha2 / (3.14159 * denom * denom);
    
    // Geometry function
    float geoA = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float geoB = dot(normal, viewDir) * (1.0 - geoA) + geoA;
    float geoC = dot(normal, lightDir) * (1.0 - geoA) + geoA;
    float G = geoB * geoC;
    
    // Cook-Torrance specular
    vec3 specular = (NDF * G * fresnel) / max(4.0 * dot(normal, viewDir) * dot(normal, lightDir), 0.001);
    
    // Lambertian diffuse
    vec3 kS = fresnel;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;
    
    float irradiance = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = irradiance * albedo / 3.14159;
    
    return (kD * diffuse + specular) * lightColor;
}

// ---------------------------------------------------------------------------
// ANTI-ALIASING TECHNIQUES
// ---------------------------------------------------------------------------

// Supersample anti-aliasing for raymarching
vec3 ssaaRaymarch(vec2 fragCoord, vec2 resolution, int samples) {
    vec3 color = vec3(0.0);
    vec2 invRes = 1.0 / resolution;
    
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < samples; j++) {
            vec2 offset = vec2(float(i) / float(samples), float(j) / float(samples)) * invRes;
            vec2 uv = (fragCoord + offset) / resolution;
            color += raymarchColor(uv);
        }
    }
    
    return color / float(samples * samples);
}

// Adaptive anti-aliasing - more samples at edges
vec3 adaptiveAA(vec2 fragCoord, vec2 resolution) {
    // Get the base color
    vec2 uv = fragCoord / resolution;
    vec3 color = raymarchColor(uv);
    
    // Check for high contrast (indicating an edge)
    vec2 px = vec2(1.0) / resolution;
    vec3 colorL = raymarchColor((fragCoord - vec2(px.x, 0.0)) / resolution);
    vec3 colorR = raymarchColor((fragCoord + vec2(px.x, 0.0)) / resolution);
    vec3 colorT = raymarchColor((fragCoord - vec2(0.0, px.y)) / resolution);
    vec3 colorB = raymarchColor((fragCoord + vec2(0.0, px.y)) / resolution);
    
    float contrast = length(color - colorL) + length(color - colorR) + 
                     length(color - colorT) + length(color - colorB);
    
    // If contrast is high, do more sampling
    if (contrast > 0.5) {
        color = ssaaRaymarch(fragCoord, resolution, 3); // 3x3 SSAA
    }
    
    return color;
}

// ---------------------------------------------------------------------------
// PERFORMANCE OPTIMIZATIONS
// ---------------------------------------------------------------------------

// Early ray termination if distance is very large
float earlyRayTermination(vec3 ro, vec3 rd, float maxDist, float precision, float earlyTermThresh) {
    float d = 0.0;
    int steps = 0;
    
    for (int i = 0; i < 128; i++) {
        vec3 p = ro + rd * d;
        float res = sdScene(p);
        
        if (res < precision || d > maxDist) {
            break;
        }
        
        d += res * 0.8;
        
        // Early termination if we're far from anything
        if (res > earlyTermThresh) {
            d = maxDist; // Jump to max distance
            break;
        }
        steps++;
    }
    
    return d;
}

// Adaptive step sizing based on distance from surface
float adaptiveStepRaymarch(vec3 ro, vec3 rd, float maxDist, float minStep, float maxStep) {
    float d = 0.0;
    int steps = 0;
    
    for (int i = 0; i < 100; i++) {
        vec3 p = ro + rd * d;
        float res = sdScene(p);
        
        if (res < 0.001 || d > maxDist) {
            break;
        }
        
        // Adaptive step size based on distance to surface
        float stepSize = mix(minStep, maxStep, smoothstep(0.0, 5.0, d));
        d += max(res * 0.8, stepSize);
        steps++;
    }
    
    return d;
}

// Optimized binary search refinement
float optimizedRefine(vec3 ro, vec3 rd, float tmin, float tmax, float precision) {
    // First ensure we have a valid range with a hit
    vec3 pmin = ro + rd * tmin;
    vec3 pmax = ro + rd * tmax;
    
    float dmin = sdScene(pmin);
    float dmax = sdScene(pmax);
    
    if (dmin > 0.0 && dmax > 0.0) return tmax; // No intersection in range
    if (dmin < 0.0 && dmax < 0.0) return tmin; // Both sides behind surface
    
    // Binary search
    for (int i = 0; i < 16; i++) {
        float tmid = (tmin + tmax) * 0.5;
        vec3 pmid = ro + rd * tmid;
        float dmid = sdScene(pmid);
        
        if (dmid < precision) {
            tmax = tmid;
        } else {
            tmin = tmid;
        }
    }
    
    return (tmin + tmax) * 0.5;
}

// ---------------------------------------------------------------------------
// FRACTAL AND COMPLEX GEOMETRY UTILITIES
// ---------------------------------------------------------------------------

// Smooth min operation for blending fractal elements
float smoothMin(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * (1.0 / 6.0);
}

// Smooth max operation
float smoothMax(float a, float b, float k) {
    return -smoothMin(-a, -b, k);
}

// Repeat pattern in 3D space with blending
vec3 opRepeatBlend(vec3 p, vec3 spacing, float blendDistance) {
    vec3 q = mod(p, spacing) - 0.5 * spacing;
    return q + blendDistance * sin(p / blendDistance);
}

// Twist deformation
vec3 opTwist(vec3 p, float twistAmount) {
    float c = cos(twistAmount * p.y);
    float s = sin(twistAmount * p.y);
    mat2 m = mat2(c, -s, s, c);
    return vec3(m * p.xz, p.y);
}

// Bend deformation
vec3 opBend(vec3 p, float bendAmount) {
    float c = cos(bendAmount * p.x);
    float s = sin(bendAmount * p.x);
    mat2 m = mat2(c, -s, s, c);
    return vec3(p.x, m * p.yz);
}

// Scale invariant (for infinite fractals)
vec3 opScaleInvariant(vec3 p, float scale) {
    return p / scale;
}

// Endless road with repeating pattern
float endlessRoad(vec3 p, float frequency) {
    // Create a road with repeating patterns, trees, etc.
    float road = sdBox(vec3(mod(p.x, 10.0) - 5.0, p.y, p.z), vec3(4.0, 0.1, 0.1));
    float laneMarkings = sdBox(vec3(mod(p.x, 2.0) - 1.0, p.y + 0.01, p.z), vec3(1.8, 0.02, 0.05));
    
    return min(road, -laneMarkings); // Lane markings are carved out (negative distance)
}

#endif // ADVANCED_RAYMARCHING_GL