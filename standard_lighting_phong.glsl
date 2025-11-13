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
    
    // Specular lighting
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 specular = calculateSpecularPhong(lightDir, viewDir, norm, lightColor, 32.0);
    result += specular;
    
    // Normal mapping
    vec3 tangentNormal = sampleNormalMap(normalMap, TexCoords);
    vec3 diffuseNormalMapped = calculateDiffuseLambert(tangentNormal, tangentNormal, lightColor);
    result += diffuseNormalMapped;

    // Final color
    FragColor = vec4(result, 1.0);
}
