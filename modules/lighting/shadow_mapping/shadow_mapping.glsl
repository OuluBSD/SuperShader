// Shadow mapping module
// Implements shadow calculations for various light types

// Basic shadow calculation function
float calculateShadow(vec4 fragPosLightSpace, sampler2D shadowMap, vec3 normal, vec3 lightDir) {
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get closest depth value from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    
    // Check whether current frag pos is in shadow
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    
    return shadow;
}

// Percentage-closer filtering for smoother shadows
float PCFShadow(vec4 fragPosLightSpace, sampler2D shadowMap, vec3 normal, vec3 lightDir) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    
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

// Soft shadow with distance-based filtering
float softShadow(vec3 fragPos, vec3 lightPos, float biasMultiplier) {
    vec3 lightDir = normalize(lightPos - fragPos);
    
    // Ray march toward light to check for occlusion
    float minDist = 0.1;
    float maxDist = length(lightPos - fragPos);
    
    float shadow = 1.0;
    float currentDepth = 0.0;
    
    for(int i = 0; i < 20; i++) {
        currentDepth += 0.1;
        if(currentDepth > maxDist) break;
        
        vec3 samplePos = fragPos + lightDir * currentDepth;
        float sceneDepth = getSceneDepth(samplePos);
        
        if(sceneDepth + biasMultiplier * currentDepth < currentDepth - 0.01) {
            shadow -= 0.05;
        }
    }
    
    return max(shadow, 0.0);
}

// Directional light shadow calculation
float directionalShadow(vec4 fragPosLightSpace, sampler2D shadowMap) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    if(projCoords.z > 1.0) return 0.0;
    
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    
    float shadow = currentDepth > closestDepth + 0.0005 ? 1.0 : 0.0;
    
    return shadow;
}

// Point light shadow calculation
float pointLightShadow(vec3 fragPos, vec3 lightPos, samplerCube shadowMap, float farPlane) {
    vec3 lightToFragment = fragPos - lightPos;
    float closestDepth = texture(shadowMap, lightToFragment).r;
    closestDepth *= farPlane;   // Undo mapping [0;1]
    float currentDepth = length(lightToFragment);
    
    return currentDepth > closestDepth + 0.025 ? 1.0 : 0.0;
}

// General shadow calculation function
float calculateLightAttenuationByShadow(vec3 fragPos, vec3 lightPos, vec3 normal) {
    vec3 lightDir = normalize(lightPos - fragPos);
    float bias = 0.025;
    
    // Simple shadow test using ray marching
    float lightDistance = length(lightPos - fragPos);
    float shadowFactor = 1.0;
    
    for(float i = 0.1; i < lightDistance; i += 0.1) {
        vec3 shadowPos = fragPos + lightDir * i;
        float sceneDepth = getSceneDepth(shadowPos);
        
        if(sceneDepth < i - bias) {
            shadowFactor -= 0.2;
        }
    }
    
    return max(shadowFactor, 0.0);
}