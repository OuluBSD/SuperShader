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
vec3 calculateSpecularPhong(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 specularColor, float shininess) {
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    return specularColor * spec;
}
vec3 calculateSpecularBlinnPhong(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 specularColor, float shininess) {
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    return specularColor * spec;
}
float fresnelSchlick(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
vec3 calculateSpecularFresnel(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 F0) {
    vec3 H = normalize(lightDir + viewDir);
    float cosTheta = max(dot(viewDir, H), 0.0);
    return fresnelSchlick(cosTheta, F0);
}
// PBR Lighting Implementation
// Normal Distribution Function (GGX/Trowbridge-Reitz)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / denom;
}
// Geometry function (Smith's method)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, roughness);
    float ggx2 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
vec4 calculatePBRLighting(vec3 position, vec3 normal, vec3 viewDir, 
                         vec3 lightPos, vec3 lightColor, 
                         vec3 albedo, float metallic, float roughness) {
    vec3 N = normalize(normal);
    vec3 L = normalize(lightPos - position);
    vec3 V = normalize(viewDir);
    vec3 H = normalize(L + V);
    // Calculate distances and attenuation
    float distance = length(lightPos - position);
    float attenuation = 1.0 / (distance * distance);
    // Calculate light contribution
    vec3 radiance = lightColor * attenuation;
    float NDF = DistributionGGX(N, H, roughness);   
    float G = GeometrySmith(N, V, L, roughness);      
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), vec3(0.04));
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;	  
    vec3 nominator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = nominator / denominator;
    float NdotL = max(dot(N, L), 0.0);        
    return vec4((kD * albedo / PI + specular) * radiance * NdotL, 1.0);
}
// Shadow Mapping Implementation
float calculateShadow(sampler2D shadowMap, vec4 fragPosLightSpace, float bias) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    return shadow;
}
float calculatePCFShadow(sampler2D shadowMap, vec4 fragPosLightSpace, float bias) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }
    }
    return shadow / 9.0;
}
float calculateExponentialShadow(sampler2D shadowMap, vec4 fragPosLightSpace, float k) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float depth = texture(shadowMap, projCoords.xy).r;
    float w = texture(shadowMap, projCoords.xy).g;  // Weight or moment
    float currentDepth = projCoords.z;
    // Calculate probability of being lit
    float p = (currentDepth <= depth) ? 1.0 : w / (w + currentDepth - depth);
    return 1.0 - p;
}
// Basic Point Light Implementation
vec3 calculatePointLight(vec3 position, vec3 normal, vec3 lightPos, vec3 lightColor) {
    // Calculate light direction
    vec3 lightDir = normalize(lightPos - position);
    // Calculate distance and attenuation
    float distance = length(lightPos - position);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    diffuse *= attenuation;
    return diffuse;
}

void main() {
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Initialize color
    vec3 result = vec3(0.0);
    
    // Apply lighting calculations based on selected modules
    
    // Basic point light calculation
    vec3 pointLight = calculatePointLight(FragPos, norm, lightPos, lightColor);
    result += pointLight;
    
    // Specular lighting
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 specular = calculateSpecularPhong(lightDir, viewDir, norm, lightColor, 32.0);
    result += specular;
    
    // PBR lighting (overrides other lighting if present)
    // For simplicity, we'll use a basic PBR calculation
    vec3 albedo = vec3(0.5); 
    float metallic = 0.0;
    float roughness = 0.5;
    vec4 pbrResult = calculatePBRLighting(FragPos, norm, viewDir, lightPos, lightColor, albedo, metallic, roughness);
    result = pbrResult.rgb;  // PBR typically replaces other lighting

    // Final color
    FragColor = vec4(result, 1.0);
}
