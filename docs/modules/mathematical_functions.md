# mathematical_functions

**Category:** procedural
**Type:** standardized

## Tags
procedural

## Code
```glsl
// Mathematical functions module
// Standardized mathematical function implementations

// Smooth minimum function
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
    return mix(a, b, h) - k * h * (1.0 - h);
}

// Smooth maximum function
float smax(float a, float b, float k) {
    return -smin(-a, -b, k);
}

// Power function with smooth transition
float smoothPow(float base, float exponent, float smoothness) {
    return pow(base, exponent + smoothness * sin(exponent));
}

// Smooth absolute value
float smoothAbs(float x, float k) {
    return k * log(exp(x / k) + exp(-x / k));
}

// Sigmoid function
float sigmoid(float x, float sharpness) {
    return 1.0 / (1.0 + exp(-x * sharpness));
}

// Smooth pulse function
float smoothPulse(float edge0, float edge1, float x) {
    return smoothstep(edge0, edge0 + (edge1 - edge0) * 0.1, x) * 
           (1.0 - smoothstep(edge1 - (edge1 - edge0) * 0.1, edge1, x));
}

```