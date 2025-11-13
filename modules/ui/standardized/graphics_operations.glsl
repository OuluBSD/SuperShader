// Graphics operations module
// Standardized graphics operation implementations

// Apply linear gradient
vec3 linearGradient(vec2 st, vec2 startPoint, vec2 endPoint, vec3 startColor, vec3 endColor) {
    vec2 direction = endPoint - startPoint;
    float lengthSquared = dot(direction, direction);
    
    if (lengthSquared == 0.0) {
        return startColor;
    }
    
    vec2 fromStart = st - startPoint;
    float projection = dot(fromStart, direction) / lengthSquared;
    projection = clamp(projection, 0.0, 1.0);
    
    return mix(startColor, endColor, projection);
}

// Apply radial gradient
vec3 radialGradient(vec2 st, vec2 center, float radius, vec3 innerColor, vec3 outerColor) {
    float dist = distance(st, center);
    float t = clamp(dist / radius, 0.0, 1.0);
    return mix(innerColor, outerColor, t);
}

// Apply box blur
vec3 boxBlur(sampler2D texture, vec2 uv, vec2 resolution, float radius) {
    vec3 color = vec3(0.0);
    float total = 0.0;
    
    float radiusInUV = radius / min(resolution.x, resolution.y);
    
    for (float x = -radius; x <= radius; x++) {
        for (float y = -radius; y <= radius; y++) {
            vec2 offset = vec2(x, y) * radiusInUV;
            color += texture2D(texture, uv + offset).rgb;
            total += 1.0;
        }
    }
    
    return color / total;
}

// Apply anti-aliased line
float antiAliasedLine(vec2 st, vec2 a, vec2 b, float thickness) {
    vec2 ba = b - a;
    vec2 pa = st - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    float d = length(pa - ba * h);
    
    // Anti-aliasing using smoothstep over a small range
    return 1.0 - smoothstep(thickness * 0.5 - 0.01, thickness * 0.5 + 0.01, d);
}

// Apply fill with optional stroke
vec3 fillAndStroke(vec2 shape, vec2 st, vec2 center, float strokeWidth, vec3 fillColor, vec3 strokeColor) {
    float filled = shape;
    float outline = shape - circle(st, center, strokeWidth);  // Simplified outline
    
    return mix(fillColor, strokeColor, outline);
}

// Apply alpha blending
vec4 alphaBlend(vec4 src, vec4 dst) {
    float alpha = src.a + dst.a * (1.0 - src.a);
    if (alpha == 0.0) {
        return vec4(0.0);
    }
    
    vec3 color = (src.rgb * src.a + dst.rgb * dst.a * (1.0 - src.a)) / alpha;
    return vec4(color, alpha);
}

// Apply color tint
vec3 applyTint(vec3 color, vec3 tint, float intensity) {
    return mix(color, color * tint, intensity);
}

// Apply color adjustment (brightness, contrast, saturation)
vec3 applyColorAdjustment(vec3 color, float brightness, float contrast, float saturation) {
    // Apply brightness
    color += brightness;
    
    // Apply contrast
    color = (color - 0.5) * contrast + 0.5;
    
    // Apply saturation
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(gray), color, saturation);
    
    return clamp(color, 0.0, 1.0);
}
