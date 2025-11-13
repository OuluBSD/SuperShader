# blur

**Category:** effects
**Type:** standardized

## Dependencies
texture_sampling

## Tags
texturing, effects

## Code
```glsl
// Blur effect module
// Standardized blur functions

// Gaussian blur implementation
vec4 GaussianBlur(sampler2D texture, vec2 texCoord, vec2 texSize, float radius) {
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    
    // Sample in a 5x5 area around the center
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 offset = vec2(float(x), float(y)) * radius / texSize;
            float weight = exp(-(x*x + y*y) / (2.0 * radius * radius));
            result += texture2D(texture, texCoord + offset) * weight;
            totalWeight += weight;
        }
    }
    
    return result / totalWeight;
}

// Simple box blur implementation
vec4 BoxBlur(sampler2D texture, vec2 texCoord, vec2 texSize, float radius) {
    vec4 result = vec4(0.0);
    int count = 0;
    
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 offset = vec2(float(x), float(y)) * radius / texSize;
            result += texture2D(texture, texCoord + offset);
            count++;
        }
    }
    
    return result / float(count);
}

// Motion blur in a specific direction
vec4 MotionBlur(sampler2D texture, vec2 texCoord, vec2 motionVector, int samples) {
    vec4 result = vec4(0.0);
    vec2 sampleStep = motionVector / float(max(samples, 1));
    
    for(int i = 0; i < samples; i++) {
        vec2 offset = texCoord + sampleStep * float(i) - motionVector * 0.5;
        result += texture2D(texture, offset);
    }
    
    return result / float(max(samples, 1));
}

```