// Standard Effects Library
// Common post-processing and filtering effects for shaders

#ifndef STANDARD_EFFECTS_GL
#define STANDARD_EFFECTS_GL

// ---------------------------------------------------------------------------
// COLOR EFFECTS
// ---------------------------------------------------------------------------

// Brightness/Contrast adjustment
vec3 brightnessContrast(vec3 color, float brightness, float contrast) {
    return (color - 0.5) * (1.0 + contrast) + 0.5 + brightness;
}

// Saturation adjustment
vec3 saturation(vec3 color, float sat) {
    float grey = dot(color, vec3(0.2125, 0.7154, 0.0721));
    return mix(vec3(grey), color, 1.0 + sat);
}

// Hue rotation
vec3 hueShift(vec3 color, float hueAdjust) {
    const vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(color.bg, K.wz), vec4(color.gb, K.xy), step(color.b, color.g));
    vec4 q = mix(vec4(p.xyw, color.r), vec4(color.r, p.yzx), step(p.x, color.r));
    
    float d = q.x;
    float e = 1.0e-10;
    return vec3(d * (K.w + clamp((q.y - d) * (hueAdjust + K.z) / (d * e + abs(q.z - d)), 0.0, 1.0)));
}

// Vibrance - increases saturation of less saturated colors more
vec3 vibrance(vec3 color, float vibrance) {
    float avg = (color.r + color.g + color.b) / 3.0;
    float mx = max(color.r, max(color.g, color.b));
    float amt = (mx - avg) * (-vibrance * 0.5);
    return mix(color, vec3(mx), amt);
}

// ---------------------------------------------------------------------------
// FILTER EFFECTS
// ---------------------------------------------------------------------------

// Sepia tone effect
vec3 sepia(vec3 color) {
    vec3 sepiaTransform = vec3(0.0);
    sepiaTransform.r = dot(color, vec3(0.393, 0.769, 0.189));
    sepiaTransform.g = dot(color, vec3(0.349, 0.686, 0.168));
    sepiaTransform.b = dot(color, vec3(0.272, 0.534, 0.131));
    return sepiaTransform;
}

// Black and white conversion with optional luminance weights
vec3 grayscale(vec3 color, vec3 weights) {
    float lum = dot(color, weights);
    return vec3(lum);
}

vec3 grayscale(vec3 color) {
    return grayscale(color, vec3(0.2126, 0.7152, 0.0722));
}

// Invert colors
vec3 invert(vec3 color) {
    return 1.0 - color;
}

// Threshold effect
vec3 threshold(vec3 color, float threshold) {
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float t = step(threshold, lum);
    return vec3(t);
}

// ---------------------------------------------------------------------------
// BLOOM EFFECTS
// ---------------------------------------------------------------------------

// Extract bright areas for bloom
vec3 bloomExtract(vec3 color, float threshold, float strength) {
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float bright = max(0.0, lum - threshold) / (1.0 - threshold);
    bright = pow(bright, strength);
    return color * bright;
}

// ---------------------------------------------------------------------------
// BLUR EFFECTS
// ---------------------------------------------------------------------------

// Simple box blur (should be used with multiple samples for better quality)
vec3 boxBlur(sampler2D tex, vec2 uv, vec2 resolution, vec2 direction) {
    vec3 color = vec3(0.0);
    vec2 off1 = vec2(1.3333333333333333) * direction / resolution;
    color += texture(tex, uv).rgb * 0.29411764705882354;
    color += texture(tex, uv + off1).rgb * 0.35294117647058826;
    color += texture(tex, uv - off1).rgb * 0.35294117647058826;
    return color;
}

// Gaussian blur approximation
vec3 gaussianBlur(sampler2D tex, vec2 uv, vec2 resolution, vec2 direction, float radius) {
    vec3 color = vec3(0.0);
    vec2 invResolution = 1.0 / resolution;
    
    // Pre-computed Gaussian weights for 5-tap
    float gWeights[5];
    gWeights[0] = 0.2270270270;  // center
    gWeights[1] = 0.1945945946;  // 1st neighbor
    gWeights[2] = 0.1216216216;  // 2nd neighbor
    gWeights[3] = 0.0540540541;  // 3rd neighbor
    gWeights[4] = 0.0162162162;  // 4th neighbor
    
    color += texture(tex, uv).rgb * gWeights[0];
    for (int i = 1; i < 5; i++) {
        vec2 offset = float(i) * radius * direction * invResolution;
        color += texture(tex, uv + offset).rgb * gWeights[i];
        color += texture(tex, uv - offset).rgb * gWeights[i];
    }
    return color;
}

// ---------------------------------------------------------------------------
// EDGE DETECTION
// ---------------------------------------------------------------------------

// Sobel edge detection
float sobelEdge(sampler2D tex, vec2 uv, vec2 resolution) {
    vec2 invResolution = 1.0 / resolution;
    
    vec3 tl = texture(tex, uv + vec2(-1.0, -1.0) * invResolution).rgb;
    vec3 tm = texture(tex, uv + vec2( 0.0, -1.0) * invResolution).rgb;
    vec3 tr = texture(tex, uv + vec2( 1.0, -1.0) * invResolution).rgb;
    vec3 ml = texture(tex, uv + vec2(-1.0,  0.0) * invResolution).rgb;
    vec3 mr = texture(tex, uv + vec2( 1.0,  0.0) * invResolution).rgb;
    vec3 bl = texture(tex, uv + vec2(-1.0,  1.0) * invResolution).rgb;
    vec3 bm = texture(tex, uv + vec2( 0.0,  1.0) * invResolution).rgb;
    vec3 br = texture(tex, uv + vec2( 1.0,  1.0) * invResolution).rgb;
    
    vec3 avg = (tl + tm + tr + ml + mr + bl + bm + br) * 0.125;
    
    // Convert to luminance
    float l_tl = dot(tl, vec3(0.2126, 0.7152, 0.0722));
    float l_tm = dot(tm, vec3(0.2126, 0.7152, 0.0722));
    float l_tr = dot(tr, vec3(0.2126, 0.7152, 0.0722));
    float l_ml = dot(ml, vec3(0.2126, 0.7152, 0.0722));
    float l_mr = dot(mr, vec3(0.2126, 0.7152, 0.0722));
    float l_bl = dot(bl, vec3(0.2126, 0.7152, 0.0722));
    float l_bm = dot(bm, vec3(0.2126, 0.7152, 0.0722));
    float l_br = dot(br, vec3(0.2126, 0.7152, 0.0722));
    float l_avg = dot(avg, vec3(0.2126, 0.7152, 0.0722));
    
    // Sobel kernels
    float edgeX = -l_tl - 2.0 * l_ml - l_bl + l_tr + 2.0 * l_mr + l_br;
    float edgeY = -l_tl - 2.0 * l_tm - l_tr + l_bl + 2.0 * l_bm + l_br;
    
    return sqrt(edgeX * edgeX + edgeY * edgeY);
}

// ---------------------------------------------------------------------------
// NOISE EFFECTS
// ---------------------------------------------------------------------------

// Hash function for noise generation
float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

float hash(vec2 p) {
    return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x))));
}

// Value noise
float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Simplex noise implementation (simplified)
float simplexNoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

vec3 permute(vec3 x) {
    return mod(((x * 34.0) + 1.0) * x, 289.0);
}

// ---------------------------------------------------------------------------
// DISTORTION EFFECTS
// ---------------------------------------------------------------------------

// Barrel distortion
vec2 barrelDistortion(vec2 uv, float amount) {
    vec2 center = uv - 0.5;
    float dist = dot(center, center);
    return uv + center * dist * amount;
}

// Chromatic aberration
vec3 chromaticAberration(sampler2D tex, vec2 uv, vec2 offset) {
    float r = texture(tex, uv - offset).r;
    float g = texture(tex, uv).g;
    float b = texture(tex, uv + offset).b;
    return vec3(r, g, b);
}

// ---------------------------------------------------------------------------
// COMPOSITE EFFECTS
// ---------------------------------------------------------------------------

// Blend modes
vec3 blendMultiply(vec3 base, vec3 blend) {
    return base * blend;
}

vec3 blendScreen(vec3 base, vec3 blend) {
    return 1.0 - (1.0 - base) * (1.0 - blend);
}

vec3 blendOverlay(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
        step(base, vec3(0.5))
    );
}

vec3 blendSoftLight(vec3 base, vec3 blend) {
    return (1.0 - 2.0 * blend) * base * base + 2.0 * blend * base;
}

vec3 blendHardLight(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
        step(blend, vec3(0.5))
    );
}

// Combine multiple effects
vec3 applyPostProcessing(
    sampler2D tex, 
    vec2 uv, 
    vec2 resolution,
    float brightness, 
    float contrast, 
    float saturation,
    bool doSepia,
    bool doInvert
) {
    vec3 color = texture(tex, uv).rgb;
    
    // Apply color adjustments
    color = brightnessContrast(color, brightness, contrast);
    color = saturation(color, saturation);
    
    // Apply filters
    if (doSepia) color = sepia(color);
    if (doInvert) color = invert(color);
    
    return color;
}

#endif // STANDARD_EFFECTS_GL