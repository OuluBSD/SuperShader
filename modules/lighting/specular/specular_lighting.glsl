// Specular lighting module - Phong, Blinn-Phong, Cook-Torrance and other specular models

// Phong specular reflection
float phongSpecular(vec3 normal, vec3 lightDir, vec3 viewDir, float shininess) {
    vec3 reflectDir = reflect(-lightDir, normal);
    return pow(max(dot(viewDir, reflectDir), 0.0), shininess);
}

// Blinn-Phong specular reflection
float blinnPhongSpecular(vec3 normal, vec3 lightDir, vec3 viewDir, float shininess) {
    vec3 halfDir = normalize(lightDir + viewDir);
    return pow(max(dot(normal, halfDir), 0.0), shininess);
}

// Blinn function (reusable Blinn-Phong function)
float Blinn(in vec3 hn, in vec3 rd, in vec3 lv, in float roughness) {
    vec3 H = normalize(rd + lv);
    float dotNH = clamp(dot(hn, H), 0., 1.);
    return (roughness + 2.) / (8. * pi) * pow(dotNH, roughness);
}

// Blinn-Phong reference implementation
float BlinnPhongRef(float shininess, vec3 n, vec3 vd, vec3 ld) {
    vec3 h  = normalize(-vd+ld);
    return 1.-pow(max(0., dot(h, n)), shininess);
}

// Normalized Blinn-Phong
float normalizedBlinnPhong(float shininess, vec3 n, vec3 vd, vec3 ld) {
    float norm_factor = (shininess+1.) / (2.*PI);
    vec3 h  = normalize(-vd+ld);
    return pow(max(0., dot(h, n)), shininess) * norm_factor;
}

// Cook-Torrance specular BRDF components
float distributionGGX(vec3 normal, vec3 halfway, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(normal, halfway), 0.0);
    float NdotH2 = NdotH*NdotH;
    
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return num / denom;
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
    
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return num / denom;
}

float geometrySmith(vec3 normal, vec3 viewDir, vec3 lightDir, float roughness) {
    float NdotV = max(dot(normal, viewDir), 0.0);
    float NdotL = max(dot(normal, lightDir), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// Complete Cook-Torrance specular function
vec3 cookTorranceSpecular(vec3 normal, vec3 viewDir, vec3 lightDir, float roughness, float F0) {
    vec3 halfway = normalize(lightDir + viewDir);
    
    float NDF = distributionGGX(normal, halfway, roughness);
    float G = geometrySmith(normal, viewDir, lightDir, roughness);
    vec3 F = fresnelSchlick(max(dot(halfway, viewDir), 0.0), vec3(F0));
    
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
    vec3 specular = numerator / max(denominator, 0.001);
    
    return specular;
}