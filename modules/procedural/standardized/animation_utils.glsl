// Animation utilities module
// Standardized animation function implementations

// Linear interpolation with time
float linear(float start, float end, float time, float duration) {
    float t = clamp(time / duration, 0.0, 1.0);
    return mix(start, end, t);
}

// Smoothstep interpolation
float smoothStep(float start, float end, float time, float duration) {
    float t = clamp(time / duration, 0.0, 1.0);
    return mix(start, end, smoothstep(0.0, 1.0, t));
}

// Sine wave oscillation
float sineWave(float frequency, float amplitude, float time, float offset) {
    return sin(time * frequency + offset) * amplitude;
}

// Square wave oscillation
float squareWave(float frequency, float amplitude, float time) {
    return sign(sin(time * frequency)) * amplitude;
}

// Triangle wave oscillation
float triangleWave(float frequency, float amplitude, float time) {
    return abs(fract(time * frequency + 0.25) * 2.0 - 1.0) * 2.0 * amplitude - amplitude;
}

// Sawtooth wave oscillation
float sawtoothWave(float frequency, float amplitude, float time) {
    return (fract(time * frequency) * 2.0 - 1.0) * amplitude;
}

// Ping-pong oscillation (smooth back and forth)
float pingPong(float minVal, float maxVal, float time, float duration) {
    float t = (time / duration);
    float range = maxVal - minVal;
    float cycle = 2.0 * range;
    float position = fract(t) * cycle;
    return position > range ? (2.0 * range - position) + minVal : position + minVal;
}
