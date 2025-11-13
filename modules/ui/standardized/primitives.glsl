// Primitives module
// Additional 2D primitive implementations

// Draw a rounded rectangle with adjustable corner radius
float roundedRectCustom(vec2 st, vec2 pos, vec2 size, vec4 radii) {
    // radii: x=top-left, y=top-right, z=bottom-right, w=bottom-left
    vec2 adjustedST = st - pos;
    vec2 halfSize = size * 0.5;
    vec2 center = pos + halfSize;
    
    // Top-left corner
    float tl = circle(adjustedST, vec2(radii.x, radii.x), radii.x);
    if (adjustedST.x < radii.x && adjustedST.y < radii.x) return tl;
    
    // Top-right corner  
    float tr = circle(adjustedST, vec2(size.x - radii.y, radii.y), radii.y);
    if (adjustedST.x > size.x - radii.y && adjustedST.y < radii.y) return tr;
    
    // Bottom-right corner
    float br = circle(adjustedST, vec2(size.x - radii.z, size.y - radii.z), radii.z);
    if (adjustedST.x > size.x - radii.z && adjustedST.y > size.y - radii.z) return br;
    
    // Bottom-left corner
    float bl = circle(adjustedST, vec2(radii.w, size.y - radii.w), radii.w);
    if (adjustedST.x < radii.w && adjustedST.y > size.y - radii.w) return bl;
    
    // Center rectangle
    float centerRect = rect(adjustedST, vec2(radii.x, 0.0), vec2(size.x - radii.x - radii.y, radii.x)) + 
                       rect(adjustedST, vec2(0.0, radii.w), vec2(radii.w, size.y - radii.w - radii.z)) + 
                       rect(adjustedST, vec2(radii.y, radii.y), vec2(size.x - radii.x - radii.y, size.y - radii.w - radii.z));
    
    return max(max(max(tl, tr), max(br, bl)), centerRect);
}

// Draw a polygon using ray casting method (simplified for triangle and quad)
float polygon(vec2 st, vec2[] points, int numPoints) {
    if (numPoints < 3) return 0.0;
    
    // Simplified code for triangle
    if (numPoints == 3) {
        return triangle(st, points[0], points[1], points[2]);
    }
    
    // For quads we can break into triangles
    if (numPoints == 4) {
        float tri1 = triangle(st, points[0], points[1], points[2]);
        float tri2 = triangle(st, points[0], points[2], points[3]);
        return max(tri1, tri2);
    }
    
    // For other polygons, implement ray casting (simplified version)
    int crossings = 0;
    for (int i = 0; i < numPoints; i++) {
        int next = (i + 1) % numPoints;
        vec2 p1 = points[i];
        vec2 p2 = points[next];
        
        if (((p1.y > st.y) != (p2.y > st.y)) &&
            (st.x < (p2.x - p1.x) * (st.y - p1.y) / (p2.y - p1.y) + p1.x)) {
            crossings++;
        }
    }
    
    return float(crossings % 2);
}

// Draw a star
float star(vec2 st, vec2 center, float outerRadius, float innerRadius, int numPoints) {
    st -= center;
    
    float angle = atan(st.y, st.x);
    float radius = length(st);
    float angleStep = 3.14159 * 2.0 / float(numPoints);
    
    // Determine which section of the star we're in
    float section = mod(angle, angleStep);
    section = min(section, angleStep - section);
    
    // Calculate the distance to the star edge in this section
    float maxRadius = mix(innerRadius, outerRadius, 
                         abs(2.0 * section / angleStep - 1.0));
    
    return 1.0 - smoothstep(maxRadius - 0.01, maxRadius + 0.01, radius);
}

// Draw an arc
float arc(vec2 st, vec2 center, float radius, float startAngle, float endAngle, float thickness) {
    st -= center;
    float angle = atan(st.y, st.x);
    float dist = length(st);
    
    // Normalize angles to [0, 2*PI]
    startAngle = mod(startAngle, 3.14159 * 2.0);
    endAngle = mod(endAngle, 3.14159 * 2.0);
    
    if (startAngle > endAngle) endAngle += 3.14159 * 2.0;
    
    float inArc = 0.0;
    if (endAngle > 3.14159 * 2.0) {
        // Handle case where arc crosses 0 angle
        float normalizedAngle = mod(angle + 3.14159 * 2.0, 3.14159 * 2.0);
        inArc = step(startAngle, normalizedAngle) * step(normalizedAngle, mod(endAngle, 3.14159 * 2.0));
    } else {
        float normalizedAngle = mod(angle + 3.14159 * 2.0, 3.14159 * 2.0);
        float start = mod(startAngle + 3.14159 * 2.0, 3.14159 * 2.0);
        float end = mod(endAngle + 3.14159 * 2.0, 3.14159 * 2.0);
        if (start <= end) {
            inArc = step(start, normalizedAngle) * step(normalizedAngle, end);
        } else {
            inArc = (step(start, normalizedAngle) + step(0.0, normalizedAngle) * step(normalizedAngle, end));
        }
    }
    
    float distToEdge = abs(dist - radius);
    float inThickness = 1.0 - smoothstep(thickness * 0.5 - 0.01, thickness * 0.5 + 0.01, distToEdge);
    
    return inArc * inThickness;
}

// Draw a pie slice
float pie(vec2 st, vec2 center, float radius, float startAngle, float endAngle) {
    st -= center;
    float angle = atan(st.y, st.x);
    float dist = length(st);
    
    // Normalize angles to [0, 2*PI]
    float normStart = mod(startAngle + 3.14159 * 2.0, 3.14159 * 2.0);
    float normEnd = mod(endAngle + 3.14159 * 2.0, 3.14159 * 2.0);
    
    float inArc = 0.0;
    if (normStart <= normEnd) {
        float normalizedAngle = mod(angle + 3.14159 * 2.0, 3.14159 * 2.0);
        inArc = step(normStart, normalizedAngle) * step(normalizedAngle, normEnd);
    } else {
        // Handle wrap-around case
        float normalizedAngle = mod(angle + 3.14159 * 2.0, 3.14159 * 2.0);
        inArc = step(normStart, normalizedAngle) + step(0.0, normalizedAngle) * step(normalizedAngle, normEnd);
        inArc = clamp(inArc, 0.0, 1.0);
    }
    
    float inCircle = 1.0 - smoothstep(radius - 0.01, radius + 0.01, dist);
    
    return inArc * inCircle;
}
