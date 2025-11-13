# edge_detection

**Category:** effects
**Type:** standardized

## Dependencies
texture_sampling

## Tags
texturing, effects

## Code
```glsl
// Edge detection module
// Standardized edge detection functions

// Sobel edge detection
float SobelEdge(sampler2D texture, vec2 texCoord, vec2 texSize) {
    vec2 texelSize = 1.0 / texSize;
    
    // Sample 3x3 neighborhood
    float tl = texture2D(texture, texCoord + vec2(-texelSize.x, -texelSize.y)).r;
    float tm = texture2D(texture, texCoord + vec2(0.0, -texelSize.y)).r;
    float tr = texture2D(texture, texCoord + vec2(texelSize.x, -texelSize.y)).r;
    float ml = texture2D(texture, texCoord + vec2(-texelSize.x, 0.0)).r;
    float mm = texture2D(texture, texCoord).r;
    float mr = texture2D(texture, texCoord + vec2(texelSize.x, 0.0)).r;
    float bl = texture2D(texture, texCoord + vec2(-texelSize.x, texelSize.y)).r;
    float bm = texture2D(texture, texCoord + vec2(0.0, texelSize.y)).r;
    float br = texture2D(texture, texCoord + vec2(texelSize.x, texelSize.y)).r;
    
    // Sobel X kernel
    float x = (-1.0 * tl) + (1.0 * tr) +
              (-2.0 * ml) + (2.0 * mr) +
              (-1.0 * bl) + (1.0 * br);
    
    // Sobel Y kernel
    float y = (-1.0 * tl) + (-2.0 * tm) + (-1.0 * tr) +
              ( 1.0 * bl) + ( 2.0 * bm) + ( 1.0 * br);
    
    return sqrt(x * x + y * y);
}

// Prewitt edge detection
float PrewittEdge(sampler2D texture, vec2 texCoord, vec2 texSize) {
    vec2 texelSize = 1.0 / texSize;
    
    // Sample 3x3 neighborhood
    float tl = texture2D(texture, texCoord + vec2(-texelSize.x, -texelSize.y)).r;
    float tm = texture2D(texture, texCoord + vec2(0.0, -texelSize.y)).r;
    float tr = texture2D(texture, texCoord + vec2(texelSize.x, -texelSize.y)).r;
    float ml = texture2D(texture, texCoord + vec2(-texelSize.x, 0.0)).r;
    float mm = texture2D(texture, texCoord).r;
    float mr = texture2D(texture, texCoord + vec2(texelSize.x, 0.0)).r;
    float bl = texture2D(texture, texCoord + vec2(-texelSize.x, texelSize.y)).r;
    float bm = texture2D(texture, texCoord + vec2(0.0, texelSize.y)).r;
    float br = texture2D(texture, texCoord + vec2(texelSize.x, texelSize.y)).r;
    
    // Prewitt X kernel
    float x = (-1.0 * tl) + (0.0 * tm) + (1.0 * tr) +
              (-1.0 * ml) + (0.0 * mm) + (1.0 * mr) +
              (-1.0 * bl) + (0.0 * bm) + (1.0 * br);
    
    // Prewitt Y kernel
    float y = (-1.0 * tl) + (-1.0 * tm) + (-1.0 * tr) +
              ( 0.0 * ml) + ( 0.0 * mm) + ( 0.0 * mr) +
              ( 1.0 * bl) + ( 1.0 * bm) + ( 1.0 * br);
    
    return sqrt(x * x + y * y);
}

```