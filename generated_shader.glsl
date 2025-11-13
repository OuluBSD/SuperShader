#version 330 core

// Uniforms
uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform sampler2D normalMap;
uniform sampler2D shadowMap;

// Input variables
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

// Output
out vec4 FragColor;

// Basic Point Light Implementation
vec3 calculatePointLight(vec3 position, vec3 normal, vec3 lightPos, vec3 lightColor) {
    // Calculate light direction
    vec3 lightDir = normalize(lightPos - position);
    // Calculate distance and attenuation
    float distance = length(lightPos - position);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    // Diffuse lighting
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    // Apply attenuation
    diffuse *= attenuation;
    return diffuse;
}
// Lambert Diffuse Lighting
vec3 calculateDiffuseLambert(vec3 lightDir, vec3 normal, vec3 diffuseColor) {
    float diff = max(dot(normal, lightDir), 0.0);
    return diff * diffuseColor;
}
// Oren-Nayar Diffuse (for rough surfaces)
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
// Ambient lighting
vec3 calculateAmbient(vec3 ambientColor, float ambientStrength) {
    return ambientStrength * ambientColor;
}
// Combined diffuse and ambient
vec3 calculateDiffuseAndAmbient(vec3 position, vec3 normal, vec3 lightPos, vec3 lightColor, 
                               vec3 ambientColor, float ambientStrength) {
    vec3 lightDir = normalize(lightPos - position);
    vec3 diffuse = calculateDiffuseLambert(lightDir, normal, lightColor);
    vec3 ambient = calculateAmbient(ambientColor, ambientStrength);
    return diffuse + ambient;
}
// Phong Specular Lighting
vec3 calculateSpecularPhong(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 specularColor, float shininess) {
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    return specularColor * spec;
}
// Blinn-Phong Specular Lighting
vec3 calculateSpecularBlinnPhong(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 specularColor, float shininess) {
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    return specularColor * spec;
}
// Fresnel Effect
float fresnelSchlick(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
// Complete specular calculation with Fresnel
vec3 calculateSpecularFresnel(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 F0) {
    vec3 H = normalize(lightDir + viewDir);
    float cosTheta = max(dot(viewDir, H), 0.0);
    return fresnelSchlick(cosTheta, F0);
}
// Normal Mapping Implementation
vec3 getNormalFromMap(sampler2D normalMap, vec2 uv, vec3 pos, vec3 normal, vec3 tangent) {
    // Sample the normal map
    vec3 tangentNormal = texture(normalMap, uv).xyz * 2.0 - 1.0;
    // Create TBN matrix
    vec3 T = normalize(tangent);
    vec3 N = normalize(normal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    // Transform normal from tangent space to world space
    vec3 finalNormal = normalize(TBN * tangentNormal);
    return finalNormal;
}
// Alternative: Simple normal mapping with normal map sampling
vec3 sampleNormalMap(sampler2D normalMap, vec2 uv) {
    vec3 normal = texture(normalMap, uv).xyz * 2.0 - 1.0;
    normal.xy *= -1.0;  // Flip X and Y for correct orientation
    return normalize(normal);
}

void main() {
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Initialize color
    vec3 result = vec3(0.0);
    
    // Basic point light calculation
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 pointLight = diff * lightColor;
    
    // Apply distance attenuation
    float distance = length(lightPos - FragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    pointLight *= attenuation;
    
    result += pointLight;
    
    // Specular lighting
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    result += specular;
    
    // Normal mapping if available
    vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;
    tangentNormal = normalize(tangentNormal);
    // Use tangentNormal instead of norm for lighting calculations
    float diff = max(dot(tangentNormal, lightDir), 0.0);
    vec3 normalMappedDiffuse = diff * lightColor;
    result = mix(result, normalMappedDiffuse, 0.5); // Blend with original
    
    // Final color
    FragColor = vec4(result, 1.0);
}
