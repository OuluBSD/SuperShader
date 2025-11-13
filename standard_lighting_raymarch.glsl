#version 330 core

// Common lighting definitions
#define PI 3.14159265359

// Input variables (to be provided by vertex shader or calculated)
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

// Uniforms
uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform sampler2D normalMap;
uniform sampler2D shadowMap;

// Output
out vec4 FragColor;

// Normal Mapping Implementation
vec3 getNormalFromMap(sampler2D normalMap, vec2 uv, vec3 pos, vec3 normal, vec3 tangent) {
    vec3 tangentNormal = texture(normalMap, uv).xyz * 2.0 - 1.0;
    vec3 T = normalize(tangent);
    vec3 N = normalize(normal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    vec3 finalNormal = normalize(TBN * tangentNormal);
    return finalNormal;
}
vec3 sampleNormalMap(sampler2D normalMap, vec2 uv) {
    vec3 normal = texture(normalMap, uv).xyz * 2.0 - 1.0;
    normal.xy *= -1.0;  // Flip X and Y for correct orientation
    return normalize(normal);
}
vec3 calculateDiffuseLambert(vec3 lightDir, vec3 normal, vec3 diffuseColor) {
    float diff = max(dot(normal, lightDir), 0.0);
    return diff * diffuseColor;
}
vec3 calculateDiffuseOrenNayar(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 diffuseColor, float roughness) {
    float NdotL = dot(normal, lightDir);
    float NdotV = dot(normal, viewDir);
    float sigma2 = roughness * roughness;
    float A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
    float B = 0.45 * sigma2 / (sigma2 + 0.09);
    float angleLV = acos(dot(lightDir, viewDir));
    float alpha = max(NdotL, NdotV);
    float beta = min(NdotL, NdotV);
    float C = sin(angleLV) * tan(angleLV);
    float orenNayar = A + B * C * max(0.0, alpha) * beta;
    return max(0.0, NdotL) * diffuseColor * orenNayar;
}
vec3 calculateAmbient(vec3 ambientColor, float ambientStrength) {
    return ambientStrength * ambientColor;
}
vec3 calculateDiffuseAndAmbient(vec3 position, vec3 normal, vec3 lightPos, vec3 lightColor, 
                               vec3 ambientColor, float ambientStrength) {
    vec3 lightDir = normalize(lightPos - position);
    vec3 diffuse = calculateDiffuseLambert(lightDir, normal, lightColor);
    vec3 ambient = calculateAmbient(ambientColor, ambientStrength);
    return diffuse + ambient;
}
// Ray Marching Lighting Implementation
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
vec3 raymarchLighting(vec3 ro, vec3 rd) {
    float d = rayMarch(ro, rd, 20.0, 0.01);
    if(d < 20.0) {
        vec3 pos = ro + rd * d;
        vec3 nor = calcNormal(pos, 0.01);
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
        col *= exp(-0.1 * d);
        return col;
    } else {
        return vec3(0.05, 0.10, 0.20);
    }
}

void main() {
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Initialize color
    vec3 result = vec3(0.0);
    
    // Apply lighting calculations based on selected modules
    
    // Diffuse lighting
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 diffuse = calculateDiffuseLambert(lightDir, norm, lightColor);
    result += diffuse;
    
    // Normal mapping
    vec3 tangentNormal = sampleNormalMap(normalMap, TexCoords);
    vec3 diffuseNormalMapped = calculateDiffuseLambert(tangentNormal, tangentNormal, lightColor);
    result += diffuseNormalMapped;

    // Final color
    FragColor = vec4(result, 1.0);
}
