// Diffuse lighting module - Lambert, Oren-Nayar, and other diffuse models
// Standard diffuse lighting calculations

// Lambert diffuse lighting
float lambertDiffuse(vec3 normal, vec3 lightDir) {
    return max(dot(normal, lightDir), 0.0);
}

// Oren-Nayar diffuse lighting model
vec3 orenNayarDiffuse(vec3 rd, vec3 ld, vec3 n, float albedo) {
    vec3 col = vec3(0.);
    float RDdotN = dot(-rd, n);
    float NdotLD = dot(n, ld);
    float aRDN = acos(RDdotN);
    float aNLD = acos(NdotLD);
    float mu = 5.; // roughness
    float A = 1.-.5*mu*mu/(mu*mu+0.57);
    float B = .45*mu*mu/(mu*mu+0.09);
    float alpha = max(aRDN, aNLD);
    float beta = min(aRDN, aNLD);
    float e0 = 4.8;
    col = vec3(albedo / mPi) * cos(aNLD) * (A + ( B * max(0.,cos(alpha - beta)) * sin(alpha) * tan(beta)))*e0;
    return col;
}

// Standard diffuse lighting with wrap-around
vec3 wrappedDiffuse(vec3 normal, vec3 lightDir, float wrapFactor) {
    return max((dot(normal, lightDir) + wrapFactor) / ((1.0 + wrapFactor) * (1.0 + wrapFactor)), 0.0);
}

// Basic diffuse lighting function with color
vec3 computeDiffuse(vec3 normal, vec3 lightDir, vec3 lightColor, float intensity) {
    float diff = max(dot(normal, lightDir), 0.0);
    return diff * lightColor * intensity;
}