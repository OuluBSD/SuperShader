# procedural_textures

**Category:** texturing
**Type:** standardized

## Dependencies
texture_sampling

## Tags
texturing

## Code
```glsl
// Procedural textures module
// Standardized procedural texture implementations

// Checkerboard pattern
float checkerPattern(vec2 uv, float scale) {
    uv *= scale;
    return mod(floor(uv.x) + floor(uv.y), 2.0);
}

// Stripes pattern
float stripesPattern(vec2 uv, float scale, float width) {
    return mod(uv.x * scale, 1.0) < width ? 1.0 : 0.0;
}

// Radial pattern
float radialPattern(vec2 uv, float rings) {
    uv -= 0.5;
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    return mod(angle * rings / (2.0 * 3.14159), 1.0);
}

// Diamond pattern
float diamondPattern(vec2 uv, float scale) {
    uv = fract(uv * scale) - 0.5;
    return 1.0 - abs(uv.x) - abs(uv.y);
}

// Hexagonal pattern
vec2 hexagonalPattern(vec2 uv, float scale) {
    uv *= scale;
    float hexSize = 1.0;
    
    float q = uv.x * 0.57735 * 2.0;
    float r = (uv.y - uv.x * 0.5) / 0.57735;
    
    float x = q;
    float z = r;
    float y = -x - z;
    
    // Round to nearest hex
    float rx = floor(x + 0.5);
    float ry = floor(y + 0.5);
    float rz = floor(z + 0.5);
    
    float x_diff = abs(rx - x);
    float y_diff = abs(ry - y);
    float z_diff = abs(rz - z);
    
    if (x_diff > y_diff && x_diff > z_diff) {
        rx = -ry - rz;
    } else if (y_diff > z_diff) {
        ry = -rx - rz;
    } else {
        rz = -rx - ry;
    }
    
    return vec2(rx, ry);
}

// Generate procedural texture based on pattern
vec3 proceduralTexture(vec2 uv, float patternType) {
    uv *= 10.0; // Scale for visibility
    
    if (patternType < 1.0) {
        // Checkerboard
        float checker = checkerPattern(uv, 4.0);
        return vec3(checker);
    } else if (patternType < 2.0) {
        // Stripes
        float stripes = stripesPattern(uv, 8.0, 0.3);
        return vec3(stripes);
    } else if (patternType < 3.0) {
        // Radial
        float radial = radialPattern(uv, 6.0);
        return vec3(radial);
    } else if (patternType < 4.0) {
        // Diamond
        float diamond = diamondPattern(uv, 4.0);
        diamond = max(diamond, 0.0);
        return vec3(diamond);
    } else {
        // Hexagonal
        vec2 hex = hexagonalPattern(uv, 4.0);
        float hexPattern = fract(length(hex));
        return vec3(hexPattern);
    }
}

```