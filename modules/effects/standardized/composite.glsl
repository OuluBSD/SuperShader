// Compositing module
// Standardized compositing functions

// Standard alpha blending
vec4 AlphaBlend(vec4 src, vec4 dst) {
    return src + dst * (1.0 - src.a);
}

// Multiplicative blending
vec3 MultiplyBlend(vec3 src, vec3 dst) {
    return src * dst;
}

// Additive blending
vec3 AddBlend(vec3 src, vec3 dst) {
    return min(src + dst, 1.0);
}

// Screen blending
vec3 ScreenBlend(vec3 src, vec3 dst) {
    return 1.0 - (1.0 - src) * (1.0 - dst);
}

// Overlay blending
vec3 OverlayBlend(vec3 src, vec3 dst) {
    vec3 result;
    result.r = dst.r < 0.5 ? 2.0 * dst.r * src.r : 1.0 - 2.0 * (1.0 - dst.r) * (1.0 - src.r);
    result.g = dst.g < 0.5 ? 2.0 * dst.g * src.g : 1.0 - 2.0 * (1.0 - dst.g) * (1.0 - src.g);
    result.b = dst.b < 0.5 ? 2.0 * dst.b * src.b : 1.0 - 2.0 * (1.0 - dst.b) * (1.0 - src.b);
    return result;
}

// Linear interpolation blend
vec3 LerpBlend(vec3 src, vec3 dst, float factor) {
    return mix(dst, src, factor);
}
