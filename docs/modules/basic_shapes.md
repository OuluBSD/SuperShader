# basic_shapes

**Category:** ui
**Type:** standardized

## Tags
ui

## Code
```glsl
// Basic shapes module
// Standardized 2D shape implementations

// Draw a rectangle
float rect(vec2 st, vec2 pos, vec2 size) {
    vec2 adjustedST = st - pos;
    float horizontal = step(0.0, adjustedST.x) * step(0.0, size.x - adjustedST.x);
    float vertical = step(0.0, adjustedST.y) * step(0.0, size.y - adjustedST.y);
    return horizontal * vertical;
}

// Draw a rectangle with rounded corners
float roundedRect(vec2 st, vec2 pos, vec2 size, float radius) {
    vec2 adjustedST = st - pos - size * 0.5;
    size -= vec2(radius * 2.0);
    vec2 corner = vec2(radius);
    
    return rect(st, pos + vec2(radius), size) +
           rect(st, pos + vec2(0, size.y * 0.5), vec2(radius * 2.0, size.y)) +
           rect(st, pos + vec2(size.x, size.y * 0.5), vec2(radius * 2.0, size.y)) +
           rect(st, pos + vec2(size.x * 0.5, 0), vec2(size.x, radius * 2.0)) +
           rect(st, pos + vec2(size.x * 0.5, size.y), vec2(size.x, radius * 2.0)) +
           circle(st, pos + vec2(radius, radius), radius) +
           circle(st, pos + vec2(size.x + radius, radius), radius) +
           circle(st, pos + vec2(radius, size.y + radius), radius) +
           circle(st, pos + size + vec2(radius, radius), radius);
}

// Draw a circle/ellipse
float circle(vec2 st, vec2 center, float radius) {
    float d = distance(st, center);
    d -= radius;
    return 1.0 - smoothstep(0.0, 1.0, d);
}

// Draw an ellipse
float ellipse(vec2 st, vec2 center, vec2 axes) {
    vec2 d = (st - center) / axes;
    return 1.0 - smoothstep(0.9, 1.0, dot(d, d));
}

// Draw a line
float line(vec2 st, vec2 a, vec2 b, float thickness) {
    vec2 ba = b - a;
    vec2 pa = st - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    float d = length(pa - ba * h);
    return 1.0 - smoothstep(thickness * 0.5, thickness * 0.5 + 0.01, d);
}

// Draw a triangle
float triangle(vec2 st, vec2 a, vec2 b, vec2 c) {
    vec3 bary = vec3(
        (b.y - c.y) * (st.x - c.x) + (c.x - b.x) * (st.y - c.y),
        (c.y - a.y) * (st.x - c.x) + (a.x - c.x) * (st.y - c.y),
        (a.y - b.y) * (st.x - c.x) + (b.x - a.x) * (st.y - c.y)
    );
    
    // If p is on the same side of all edges, return 1
    return step(0.0, bary.x) * step(0.0, bary.y) * step(0.0, bary.z);
}

// Draw a polygon (simplified for quadrilateral)
float quad(vec2 st, vec2 a, vec2 b, vec2 c, vec2 d) {
    float result = 0.0;
    
    // Check if point is inside the quad by testing against each edge
    vec2 edges[4];
    edges[0] = b - a;
    edges[1] = c - b;
    edges[2] = d - c;
    edges[3] = a - d;
    
    vec2 points[4];
    points[0] = a;
    points[1] = b;
    points[2] = c;
    points[3] = d;
    
    for (int i = 0; i < 4; i++) {
        vec2 edge = edges[i];
        vec2 point = points[i];
        vec2 perp = vec2(-edge.y, edge.x);
        vec2 toPoint = st - point;
        
        float side = dot(toPoint, perp);
        result = i == 0 ? step(0.0, side) : min(result, step(0.0, side));
    }
    
    return result;
}

```