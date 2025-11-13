#version 330 core
// Common uniforms
uniform vec2 resolution;
uniform float time;
uniform vec2 mouse;
uniform int frame;
// Inputs from vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
// Output color
out vec4 FragColor;
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
// Fresnel equation (Schlick approximation)
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
// Complete PBR lighting calculation
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
    // Cook-Torrance BRDF
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
// Shadow Mapping Implementation
// Basic shadow calculation
float calculateShadow(sampler2D shadowMap, vec4 fragPosLightSpace, float bias) {
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // Get closest depth value from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // Check whether current frag pos is in shadow
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    return shadow;
}
// Percentage-Closer Filtering for smoother shadows
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
// Exponential Shadow Maps
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
void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    // Default color based on UV position
    vec3 color = 0.5 + 0.5 * cos(time + uv.xyx + vec3(0, 2, 4));
    // Add some animation
    color *= abs(sin(time * 0.5)) * 0.5 + 0.5;
    FragColor = vec4(color, 1.0);
}