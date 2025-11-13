// Physically Based Rendering (PBR) lighting module
// Complete PBR lighting calculations using Cook-Torrance BRDF

const float PI = 3.14159265359;

// Normal Distribution Function (GGX/Trowbridge-Reitz)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
    
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return nom / denom;
}

// Geometry Function (Smith's method)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
    
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// Fresnel Function (Schlick approximation)
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

vec3 F_SchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// Complete PBR lighting function
vec3 calculatePBRLighting(vec3 N, vec3 V, vec3 L, vec3 albedo, float metallic, float roughness, vec3 lightColor) {
    vec3 H = normalize(V + L); // Half vector
    
    // Calculate lighting parameters
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = F_Schlick(max(dot(H, V), 0.0), vec3(0.04));
    
    vec3 nominator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // Prevent division by zero
    vec3 specular = nominator / denominator;
    
    // kS is equal to Fresnel
    vec3 kS = F;
    // For energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve this relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3(1.0) - kS;
    // Multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal (pure metals have no diffuse light).
    kD *= 1.0 - metallic;
    
    // Scale light by NdotL
    float NdotL = max(dot(N, L), 0.0);
    
    // Combine results
    vec3 diffuse = kD * albedo / PI;
    vec3 spec = (specular * NdotL) * lightColor;
    vec3 diff = (diffuse * NdotL) * lightColor;
    
    return diff + spec;
}

// PBR lighting with direct and indirect components
struct PBRMaterial {
    vec3 albedo;
    float metallic;
    float roughness;
    vec3 emissive;
};

vec3 computePBR(vec3 position, vec3 normal, vec3 viewDir, PBRMaterial material, vec3 lightPos, vec3 lightColor) {
    vec3 lightDir = normalize(lightPos - position);
    
    return calculatePBRLighting(normal, viewDir, lightDir, material.albedo, material.metallic, material.roughness, lightColor);
}

// Complete PBR function with image-based lighting approximation
vec3 computePBRWithIBL(vec3 N, vec3 V, vec3 L, PBRMaterial material, vec3 F0, vec3 directLighting, vec3 indirectDiffuse, vec3 indirectSpecular, float ao) {
    vec3 H = normalize(V + L);
    
    float NDF = DistributionGGX(N, H, material.roughness);
    float G = GeometrySmith(N, V, L, material.roughness);
    vec3 F = F_Schlick(max(dot(H, V), 0.0), F0);
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - material.metallic;
    
    vec3 nominator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = nominator / denominator;
    
    vec3 numerator = kD * material.albedo;
    vec3 diffuse = numerator / PI;
    
    float NdotL = max(dot(N, L), 0.0);
    
    return (diffuse + specular) * directLighting * NdotL + (indirectDiffuse * material.albedo + indirectSpecular * F) * ao;
}