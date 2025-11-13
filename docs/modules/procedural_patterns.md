# procedural_patterns

**Category:** procedural
**Type:** standardized

## Tags
procedural

## Code
```glsl
// Procedural patterns module
// Standardized procedural pattern implementations

// Checkerboard pattern
float checker(vec2 p) {
    vec2 i = floor(p);
    return mod(i.x + i.y, 2.0);
}

// Stripes pattern
float stripes(vec2 p) {
    return abs(fract(p.x) - 0.5);
}

// Radial pattern
float radial(vec2 p) {
    return length(p);
}

// Angle pattern
float angle(vec2 p) {
    return atan(p.y, p.x);
}

// Hexagonal tiling pattern
float hexTile(vec2 p) {
    vec2 q = vec2(p.x * 1.1547, p.y + p.x * 0.57735);
    vec2 i = floor(q);
    vec2 f = fract(q);
    
    if(f.y < 1.0 - f.x) {
        return dot(i, vec2(1.0, 0.0));
    } else {
        return dot(i + vec2(1.0, 1.0), vec2(1.0, 0.0));
    }
}

// Voronoi pattern (simplified)
float voronoi(vec2 p) {
    vec2 g = floor(p);
    vec2 f = fract(p);
    
    float distance = 1.0;
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash2(g + neighbor) * 0.5 + 0.25; // Assume hash2 function exists
            vec2 diff = neighbor + point - f;
            float dist = length(diff);
            distance = min(distance, dist);
        }
    }
    
    return distance;
}

```