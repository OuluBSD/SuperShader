#version 330 core
// Common uniforms
uniform vec2 resolution;
uniform float time;
uniform vec2 mouse;
uniform int frame;
// Inputs from vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
// Output color
out vec4 FragColor;
// Bloom Effect Implementation
// Extract bright areas from the scene
vec3 extractBrightness(vec3 color, float threshold) {
    vec3 bright = max(color - threshold, 0.0);
    return bright;
}
// Simple Gaussian blur in one direction
vec3 blurH(sampler2D image, vec2 uv, vec2 resolution, float radius) {
    vec3 color = vec3(0.0);
    float total = 0.0;
    for (float i = -2.0; i <= 2.0; i += 1.0) {
        float weight = exp(-i * i / (2.0 * radius * radius));
        color += texture(image, uv + vec2(i * resolution.x, 0.0)).rgb * weight;
        total += weight;
    }
    return color / total;
}
// Simple Gaussian blur in the other direction
vec3 blurV(sampler2D image, vec2 uv, vec2 resolution, float radius) {
    vec3 color = vec3(0.0);
    float total = 0.0;
    for (float i = -2.0; i <= 2.0; i += 1.0) {
        float weight = exp(-i * i / (2.0 * radius * radius));
        color += texture(image, uv + vec2(0.0, i * resolution.y)).rgb * weight;
        total += weight;
    }
    return color / total;
}
// Perform a full Gaussian blur
vec3 gaussianBlur(sampler2D image, vec2 uv, vec2 resolution, float radius) {
    // First pass: horizontal blur
    vec3 h_blur = blurH(image, uv, vec2(1.0/resolution.x, 0.0), radius);
    // Second pass: vertical blur on the horizontally blurred result
    return blurV(image, uv, vec2(0.0, 1.0/resolution.y), radius);
}
// Apply bloom effect by combining base image with blurred bright areas
vec4 applyBloom(vec3 baseColor, vec3 brightColor, float intensity) {
    // Apply bloom by adding the blurred bright areas to the base image
    vec3 bloom = brightColor * intensity;
    vec3 finalColor = baseColor + bloom;
    return vec4(finalColor, 1.0);
}
// Full bloom pass with multiple blur iterations for better quality
vec4 bloomPass(sampler2D scene, vec2 uv, vec2 resolution, float threshold, float intensity, float blurRadius) {
    // Extract bright areas
    vec3 sceneColor = texture(scene, uv).rgb;
    vec3 bright = extractBrightness(sceneColor, threshold);
    // Apply blur to the bright areas
    vec3 blurredBright = gaussianBlur(scene, uv, resolution, blurRadius);
    // Apply bloom effect
    return applyBloom(sceneColor, blurredBright, intensity);
}
// Fast approximate bloom using a simple blur
vec4 fastBloom(sampler2D scene, vec2 uv, float threshold, float intensity) {
    // Sample the center and surrounding pixels
    vec3 center = texture(scene, uv).rgb;
    vec3 sample1 = texture(scene, uv + vec2(0.01, 0.0)).rgb;
    vec3 sample2 = texture(scene, uv + vec2(-0.01, 0.0)).rgb;
    vec3 sample3 = texture(scene, uv + vec2(0.0, 0.01)).rgb;
    vec3 sample4 = texture(scene, uv + vec2(0.0, -0.01)).rgb;
    // Extract bright areas
    vec3 brightCenter = extractBrightness(center, threshold);
    vec3 brightAvg = (extractBrightness(sample1, threshold) + 
                      extractBrightness(sample2, threshold) + 
                      extractBrightness(sample3, threshold) + 
                      extractBrightness(sample4, threshold)) * 0.25;
    // Combine and apply bloom
    vec3 bloom = (brightCenter + brightAvg) * 0.5 * intensity;
    return vec4(center + bloom, 1.0);
}
// Post-Processing Effects Implementation
// Basic tone mapping using Reinhard operator
vec3 reinhardToneMapping(vec3 color, float exposure) {
    color *= exposure;
    return color / (color + 1.0);
}
// ACES tone mapping (approximation)
vec3 acesToneMapping(vec3 color) {
    color *= 0.6;  // Scale down to prevent clipping
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}
// Gamma correction
vec3 gammaCorrection(vec3 color, float gamma) {
    return pow(color, vec3(1.0 / gamma));
}
// Color grading with RGB scale factors
vec3 colorGrade(vec3 color, vec3 redScale, vec3 greenScale, vec3 blueScale) {
    // Apply individual channel scaling
    vec3 graded = color;
    graded.r = dot(graded, redScale);
    graded.g = dot(graded, greenScale);
    graded.b = dot(graded, blueScale);
    return graded;
}
// Simple color grading with lift, gamma, gain
vec3 liftGammaGain(vec3 color, vec3 lift, vec3 gamma, vec3 gain) {
    color = color * gain + lift;
    color = pow(color, 1.0 / gamma);
    return clamp(color, 0.0, 1.0);
}
// Vignette effect
vec3 vignette(vec2 uv, vec3 color, float strength, float falloff) {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    float vig = 1.0 - dist * strength;
    vig = pow(vig, falloff);
    return color * vig;
}
// Chromatic aberration
vec3 chromaticAberration(sampler2D source, vec2 uv, vec2 offset) {
    float r = texture(source, uv + offset).r;
    float g = texture(source, uv).g;  // No offset for green (center)
    float b = texture(source, uv - offset).b;
    return vec3(r, g, b);
}
// Film grain effect
vec3 filmGrain(vec3 color, vec2 uv, float time, float intensity) {
    // Simple noise function
    float noise = sin(uv.x * 100.0) * sin(uv.y * 77.0) * sin(time) * 0.1;
    noise = fract(noise);
    return color + noise * intensity;
}
// Depth of field effect (simplified)
vec3 depthOfField(sampler2D source, vec2 uv, sampler2D depthTexture, 
                  float focusPoint, float focusRange) {
    float depth = texture(depthTexture, uv).r;
    float coc = abs(depth - focusPoint) / focusRange;  // Circle of confusion
    coc = clamp(coc, 0.0, 1.0);
    // Simplified CoC-based blur
    vec3 color = texture(source, uv).rgb;
    if (coc > 0.1) {
        // Apply more blur for out-of-focus areas
        vec2 blurOffset = vec2(0.01, 0.0);
        color = (texture(source, uv).rgb + 
                 texture(source, uv + blurOffset).rgb + 
                 texture(source, uv - blurOffset).rgb) / 3.0;
    }
    return color;
}
// Complete post-processing pass combining multiple effects
vec4 postProcess(vec3 baseColor, vec2 uv, float time, float exposure, 
                 float gamma, float vignetteStrength) {
    vec3 color = baseColor;
    // Apply tone mapping
    color = reinhardToneMapping(color, exposure);
    // Apply gamma correction
    color = gammaCorrection(color, gamma);
    // Apply vignette
    color = vignette(uv, color, vignetteStrength, 2.0);
    // Apply film grain
    color = filmGrain(color, uv, time, 0.02);
    return vec4(color, 1.0);
}
// Full screen effect with multiple post-processing operations
vec4 fullScreenEffect(sampler2D source, vec2 uv, vec2 resolution, float time) {
    // Sample the original color
    vec3 color = texture(source, uv).rgb;
    // Apply post-processing effects
    color = reinhardToneMapping(color, 1.2);  // Tone mapping
    color = gammaCorrection(color, 2.2);      // Gamma correction
    color = vignette(uv, color, 0.3, 2.0);    // Vignette
    color = filmGrain(color, uv, time, 0.01); // Film grain
    return vec4(color, 1.0);
}
void main() {
    // Sample from texture (typically a rendered scene)
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    vec3 color = texture(gSceneTexture, uv).rgb;
    // Apply bloom effect
    vec3 brightColor = max(color - 0.5, 0.0);
    vec3 bloomColor = brightColor;  // In a real implementation, this would be the blurred version
    color += bloomColor * 0.5;
    // Apply vignette
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    float vig = 1.0 - dist * 0.5;
    color *= pow(vig, 2.0);
    // Apply tone mapping if available
    color = color / (color + vec3(1.0));
    // Apply gamma correction
    color = pow(color, vec3(0.4545));
    FragColor = vec4(color, 1.0);
}