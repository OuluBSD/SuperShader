# vignette_effect

**Category:** effects
**Type:** standardized

## Tags
effects, color

## Code
```glsl
// Vignette effect module
// Standardized vignette implementation

// Create a vignette effect based on distance from center
vec3 ApplyVignette(vec3 color, vec2 uv, float strength, float radius) {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    float vignette = 1.0 - pow(dist / radius, strength);
    return color * vignette;
}

// Radial vignette with smooth falloff
vec3 RadialVignette(vec3 color, vec2 uv, vec2 center, float softness, float intensity) {
    float dist = distance(uv, center);
    dist = smoothstep(0.0, softness, dist * intensity);
    return color * (1.0 - dist);
}

```