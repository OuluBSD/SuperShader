# noise_functions

**Category:** procedural
**Type:** standardized

## Tags
procedural

## Code
```glsl
// Noise functions module
// Standardized noise function implementations

// Classic Perlin noise by Ken Perlin
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// 2D Perlin noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Four corners in 2D of a tile
    vec2 I = vec2(0.0, 1.0);
    float a = rand(i + I.yy);
    float b = rand(i + I.xy);
    float c = rand(i + I.yx);
    float d = rand(i + I.xx);
    
    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    // Mix 4 coorners percentages
    return mix(a, b, u.x) + 
          (c - a)* u.y * (1.0 - u.x) + 
          (d - b) * u.x * u.y;
}

// Fractional Brownian Motion (fBm)
float fbm(vec2 p, int octaves, float amp, float freq) {
    float total = 0.0;
    float amplitude = amp;
    float frequency = freq;
    
    for(int i = 0; i < octaves; i++) {
        total += noise(p * frequency) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return total;
}

// Turbulence function (absolute value of fBm)
float turbulence(vec2 p, int octaves, float amp, float freq) {
    float total = 0.0;
    float amplitude = amp;
    float frequency = freq;
    
    for(int i = 0; i < octaves; i++) {
        total += abs(noise(p * frequency) * amplitude);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return total;
}

```