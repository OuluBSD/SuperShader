// Advanced Post-Processing Effects Library
// Based on analysis of real shader implementations

#ifndef ADVANCED_POST_EFFECTS_GL
#define ADVANCED_POST_EFFECTS_GL

// ---------------------------------------------------------------------------
// COLOR GRADING AND TONE MAPPING
// ---------------------------------------------------------------------------

//ACES Filmic Tone Mapping (approximation)
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

// Reinhard Tone Mapping
vec3 reinhard(vec3 color) {
    return color / (1.0 + color);
}

// Uncharted 2 Tone Mapping
vec3 uncharted2Tonemap(vec3 color) {
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

// Vibrance with luminance preservation
vec3 vibranceWithLuminance(vec3 color, float vibrance) {
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float maxColor = max(color.r, max(color.g, color.b));
    float minColor = min(color.r, min(color.g, color.b));
    float sat = (maxColor - minColor) / maxColor;
    float amt = (1.0 - sat) * vibrance * lum;
    float gray = dot(color, vec3(0.3333));
    return mix(vec3(gray), color, 1.0 + amt);
}

// High Pass Filter for sharpening
vec3 highPassFilter(sampler2D tex, vec2 uv, vec2 resolution, float strength) {
    vec2 invResolution = 1.0 / resolution;
    
    float center = dot(texture(tex, uv).rgb, vec3(0.2126, 0.7152, 0.0722));
    float left = dot(texture(tex, uv + vec2(-invResolution.x, 0.0)).rgb, vec3(0.2126, 0.7152, 0.0722));
    float right = dot(texture(tex, uv + vec2(invResolution.x, 0.0)).rgb, vec3(0.2126, 0.7152, 0.0722));
    float top = dot(texture(tex, uv + vec2(0.0, invResolution.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
    float bottom = dot(texture(tex, uv + vec2(0.0, -invResolution.y)).rgb, vec3(0.2126, 0.7152, 0.0722));
    
    float laplacian = 4.0 * center - left - right - top - bottom;
    return texture(tex, uv).rgb + vec3(laplacian * strength);
}

// ---------------------------------------------------------------------------
// MOTION BLUR
// ---------------------------------------------------------------------------

// Simulated motion blur using temporal accumulation
vec3 motionBlur(vec3 currentColor, vec3 previousColor, float blendFactor) {
    return mix(previousColor, currentColor, blendFactor);
}

// ---------------------------------------------------------------------------
// BLOOM IMPLEMENTATION
// ---------------------------------------------------------------------------

// Separable Gaussian Blur for bloom
vec3 separableGaussianBlur(sampler2D tex, vec2 uv, vec2 resolution, vec2 direction, float intensity) {
    vec2 invResolution = 1.0 / resolution;
    vec3 result = vec3(0.0);
    float totalWeight = 0.0;
    
    // 5-tap Gaussian kernel
    float offsets[3] = float[](0.0, 1.0, 2.0);
    float weights[3] = float[](0.383103, 0.241971, 0.053991);
    
    for (int i = 0; i < 3; i++) {
        float weight = weights[i];
        vec2 offset = direction * offsets[i] * invResolution;
        
        if (i == 0) {
            result += texture(tex, uv).rgb * weight;
        } else {
            result += texture(tex, uv + offset).rgb * weight;
            result += texture(tex, uv - offset).rgb * weight;
        }
        
        totalWeight += i == 0 ? weight : weight * 2.0;
    }
    
    return result / totalWeight;
}

// Bloom composition
vec3 composeBloom(vec3 original, vec3 bloom, float intensity) {
    return original + bloom * intensity;
}

// Extract bright pixels for bloom
vec3 extractBright(vec3 color, float threshold) {
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float factor = max(0.0, lum - threshold) / (1.0 - threshold);
    factor = pow(factor, 2.0);  // Square to enhance bright areas
    return color * factor;
}

// ---------------------------------------------------------------------------
// DEPTH OF FIELD SIMULATION
// ---------------------------------------------------------------------------

// Simple depth of field (assuming depth texture available)
vec3 depthOfField(
    sampler2D colorTex, 
    sampler2D depthTex,
    vec2 uv, 
    vec2 resolution,
    float focusDistance,
    float focusRange
) {
    vec3 color = texture(colorTex, uv).rgb;
    float depth = texture(depthTex, uv).r;
    
    float coc = abs(depth - focusDistance) / focusRange;
    coc = clamp(coc, 0.0, 1.0);
    
    if (coc > 0.01) {
        // Apply simple blur based on CoC (Circle of Confusion)
        vec2 invResolution = 1.0 / resolution;
        vec3 blurred = vec3(0.0);
        float total = 0.0;
        
        for (float x = -2.0; x <= 2.0; x += 1.0) {
            for (float y = -2.0; y <= 2.0; y += 1.0) {
                vec2 offset = vec2(x, y) * invResolution * coc * 2.0;
                vec3 sampleColor = texture(colorTex, uv + offset).rgb;
                float weight = 1.0 - (abs(x) + abs(y)) * 0.2;
                blurred += sampleColor * weight;
                total += weight;
            }
        }
        
        color = blurred / total;
    }
    
    return color;
}

// ---------------------------------------------------------------------------
// FXAA (Fast Approximate Anti-Aliasing)
// ---------------------------------------------------------------------------

vec3 fxaa(sampler2D tex, vec2 uv, vec2 resolution) {
    vec2 invResolution = 1.0 / resolution;
    
    // Calculate texture samples
    vec3 rgbNW = texture(tex, uv + (vec2(-1.0, -1.0) * invResolution)).xyz;
    vec3 rgbNE = texture(tex, uv + (vec2(1.0, -1.0) * invResolution)).xyz;
    vec3 rgbSW = texture(tex, uv + (vec2(-1.0, 1.0) * invResolution)).xyz;
    vec3 rgbSE = texture(tex, uv + (vec2(1.0, 1.0) * invResolution)).xyz;
    vec3 rgbM = texture(tex, uv).xyz;

    // Luminance calculations
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM = dot(rgbM, luma);

    // Find minimum and maximum luminance in neighborhood
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    // Early termination if contrast is too low
    float lumaRange = lumaMax - lumaMin;
    if (lumaRange < max(0.05, lumaMax * 0.08)) {
        return rgbM;
    }

    // Gradient calculation
    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    // Normalize direction
    float dirReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * lumaRange),
        0.0078125
    );
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(8.0, 8.0), max(vec2(-8.0, -8.0), dir * rcpDirMin)) * invResolution;

    // Sampling locations
    vec3 rgbL = texture(tex, uv + dir * (1.0/3.0 - 0.5)).xyz;
    vec3 rgbR = texture(tex, uv + dir * (2.0/3.0 - 0.5)).xyz;
    
    vec3 rgbA = 0.5 * (rgbL + rgbR);
    vec3 rgbB = texture(tex, uv - dir * 0.5).xyz + texture(tex, uv + dir * 0.5).xyz;
    rgbB *= 0.25;
    
    float lumaB = dot(rgbB, luma);
    if (lumaB < lumaMin || lumaB > lumaMax) {
        return rgbA;
    } else {
        return rgbB;
    }
}

// ---------------------------------------------------------------------------
// CHROMATIC ABERRATION
// ---------------------------------------------------------------------------

vec3 chromaticAberrationComplex(sampler2D tex, vec2 uv, vec2 resolution, vec2 offset) {
    vec2 invResolution = 1.0 / resolution;
    vec2 redOffset = offset * invResolution;
    vec2 blueOffset = -offset * invResolution;
    
    float r = texture(tex, uv + redOffset).r;
    float g = texture(tex, uv).g;
    float b = texture(tex, uv + blueOffset).b;
    
    return vec3(r, g, b);
}

// ---------------------------------------------------------------------------
// HEAT SHIMMER EFFECT
// ---------------------------------------------------------------------------

vec3 heatShimmer(sampler2D tex, vec2 uv, vec2 resolution, float time) {
    vec2 invResolution = 1.0 / resolution;
    
    float noise1 = valueNoise(vec2(uv.x * 3.0, time * 0.5));
    float noise2 = valueNoise(vec2(uv.y * 3.0, time * 0.3));
    
    vec2 distortion = vec2(noise1, noise2) * 0.01;
    vec2 distortedUV = uv + distortion;
    
    return texture(tex, distortedUV).rgb;
}

// ---------------------------------------------------------------------------
// FILM GRAIN
// ---------------------------------------------------------------------------

vec3 addFilmGrain(vec3 color, vec2 uv, float intensity) {
    float noise = hash(uv + iTime);
    return color + (noise - 0.5) * intensity;
}

// ---------------------------------------------------------------------------
// SCANLINE EFFECT
// ---------------------------------------------------------------------------

vec3 scanlines(vec3 color, vec2 uv, float time, float intensity) {
    float scanLine = sin(uv.y * 600.0 - time * 5.0) * 0.5 + 0.5;
    scanLine = pow(scanLine, 10.0);
    return mix(color, color * vec3(0.3), scanLine * intensity);
}

// ---------------------------------------------------------------------------
// VIGNETTE EFFECT
// ---------------------------------------------------------------------------

vec3 vignette(vec3 color, vec2 uv, float strength, float darkness) {
    vec2 coord = uv - 0.5;
    float dist = length(coord);
    float vignette = 1.0 - dist * strength;
    return color * mix(1.0, vignette, darkness);
}

#endif // ADVANCED_POST_EFFECTS_GL