// Physically Based Rendering lighting calculations
// Standardized lighting module

// Input parameters
struct Material {
    vec3 albedo;
    float metallic;
    float roughness;
    float ao;
};

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

// Constants
const float PI = 3.14159265359;

// Normal Distribution Function - GGX
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

// Geometry Function - Smith
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// Geometry Function - Smith
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Fresnel Function - Schlick
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// Cook-Torrance BRDF
vec3 CookTorranceBRDF(vec3 L, vec3 V, vec3 N, vec3 F0, float roughness, vec3 diffuseColor) {
    vec3 H = normalize(L + V);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, L), 0.0) * max(dot(N, V), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 diffuse = (kD * diffuseColor) / PI;

    return (diffuse + specular);
}

// Main PBR lighting function
vec3 CalculatePBRLighting(Material material, Light light, vec3 worldPos, vec3 normal, vec3 viewDir) {
    vec3 L = normalize(light.position - worldPos);
    vec3 V = normalize(viewDir - worldPos);
    vec3 N = normalize(normal);
    
    vec3 F0 = mix(vec3(0.04), material.albedo, material.metallic);
    
    vec3 Lo = CookTorranceBRDF(L, V, N, F0, material.roughness, material.albedo);
    
    float NdotL = max(dot(N, L), 0.0);
    return Lo * light.color * light.intensity * NdotL;
}
