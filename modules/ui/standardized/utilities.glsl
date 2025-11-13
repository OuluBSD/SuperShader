// Utilities module
// Standardized UI/2D graphics utility functions

// Remap a value from one range to another
float remap(float value, float inputMin, float inputMax, float outputMin, float outputMax) {
    return outputMin + (outputMax - outputMin) * (value - inputMin) / (inputMax - inputMin);
}

// Clamp with float3
vec3 clamp3(vec3 value, vec3 minVal, vec3 maxVal) {
    return vec3(clamp(value.x, minVal.x, maxVal.x),
                clamp(value.y, minVal.y, maxVal.y),
                clamp(value.z, minVal.z, maxVal.z));
}

// Linear interpolation for vec3
vec3 lerp3(vec3 a, vec3 b, float t) {
    return a + (b - a) * t;
}

// Calculate UV for a specific element in a layout
vec2 calculateElementUV(vec2 globalUV, vec2 elementPos, vec2 elementSize) {
    return (globalUV - elementPos) / elementSize;
}

// Convert screen coordinates to normalized coordinates [0, 1]
vec2 screenToNormalized(vec2 screenPos, vec2 resolution) {
    return screenPos / resolution;
}

// Convert normalized coordinates [0, 1] to screen coordinates
vec2 normalizedToScreen(vec2 normalizedPos, vec2 resolution) {
    return normalizedPos * resolution;
}

// Calculate aspect ratio
float calculateAspectRatio(vec2 resolution) {
    return resolution.x / resolution.y;
}

// Apply aspect ratio correction
vec2 applyAspectRatio(vec2 uv, vec2 resolution) {
    float aspectRatio = resolution.x / resolution.y;
    if (aspectRatio > 1.0) {
        uv.x *= aspectRatio;
        uv.x -= (aspectRatio - 1.0) * 0.5;
    } else {
        uv.y /= aspectRatio;
        uv.y -= (1.0 / aspectRatio - 1.0) * 0.5;
    }
    return uv;
}

// Create a checkerboard pattern
vec3 checkerboard(vec2 uv, float scale) {
    vec2 c = floor(uv * scale);
    return mod(c.x + c.y, 2.0) > 0.0 ? vec3(1.0) : vec3(0.8);
}

// Apply transformation matrix to UV
vec2 applyTransform(vec2 uv, mat2 transformMatrix, vec2 translation) {
    return (transformMatrix * uv) + translation;
}

// Get the distance field for multiple shapes combined
float combineShapes(float shape1, float shape2, int operation) {
    // operation: 0 = union, 1 = intersection, 2 = difference
    if (operation == 0) { // Union
        return max(shape1, shape2);
    } else if (operation == 1) { // Intersection
        return min(shape1, shape2);
    } else if (operation == 2) { // Difference
        return min(shape1, 1.0 - shape2);
    }
    return shape1;
}

// Smoothly combine shapes
float smoothCombine(float shape1, float shape2, float k, int operation) {
    // Using smooth minimum/maximum functions
    if (operation == 0) { // Union
        return -k * log(exp(-shape1/k) + exp(-shape2/k));
    } else if (operation == 1) { // Intersection
        return k * log(exp(shape1/k) + exp(shape2/k));
    }
    return shape1; // Default to first shape
}
