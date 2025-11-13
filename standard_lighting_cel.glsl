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
// Cel Shading Implementation
vec3 calculateCelShading(vec3 normal, vec3 lightDir, vec3 diffuseColor, vec3 specularColor) {
    float NdotL = dot(normal, lightDir);
    float intensity = smoothstep(0.0, 0.01, NdotL);
    intensity += step(0.5, NdotL);
    intensity += step(0.8, NdotL);
    intensity = min(intensity, 1.0);
    vec3 toonDiffuse = diffuseColor * vec3(intensity);
    float spec = step(0.9, NdotL) * step(0.2, dot(reflect(-lightDir, normal), viewDir));
    vec3 toonSpecular = spec * specularColor;
    return toonDiffuse + toonSpecular;
}
vec3 calculateMultiToneCelShading(vec3 normal, vec3 lightDir, vec3 lightColor, 
                                 vec3 darkColor, vec3 midColor, vec3 lightColorOut) {
    float NdotL = dot(normal, lightDir);
    float darkThreshold = 0.3;
    float midThreshold = 0.6;
    vec3 finalColor;
    if (NdotL < darkThreshold) {
        finalColor = darkColor;
    } else if (NdotL < midThreshold) {
        finalColor = midColor;
    } else {
        finalColor = lightColorOut;
    }
    return finalColor * lightColor;
}
float calculateOutline(vec3 normal, float edgeThreshold) {
    float edge = 1.0 - abs(dot(normal, vec3(0.0, 0.0, 1.0)));
    edge = smoothstep(edgeThreshold, 1.0, edge);
    return edge;
}
vec4 calculateCompleteCelShading(vec3 position, vec3 normal, vec3 viewDir, 
                                vec3 lightPos, vec3 diffuseColor, vec3 specularColor) {
    vec3 lightDir = normalize(lightPos - position);
    // Calculate toon shading
    vec3 toonColor = calculateCelShading(normal, lightDir, diffuseColor, specularColor);
    // Calculate outline
    float outline = calculateOutline(normal, 0.8);
    vec3 finalColor = mix(toonColor, vec3(0.0), outline);
    return vec4(finalColor, 1.0);
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
    
    // Apply cel shading to the result
    vec3 celResult = calculateCelShading(norm, normalize(lightPos - FragPos), result, vec3(1.0));
    result = celResult;

    // Final color
    FragColor = vec4(result, 1.0);
}
