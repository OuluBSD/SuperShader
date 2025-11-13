// Frequency analysis module
// Standardized frequency analysis implementations

// Get frequency data at position
float getFrequency(sampler2D fftChannel, float position) {
    return texture2D(fftChannel, vec2(position, 0.0)).x;
}

// Get frequency band at index
float getFrequencyBand(sampler2D fftChannel, float bandIndex, float numBands) {
    return texture2D(fftChannel, vec2(bandIndex / numBands, 0.0)).x;
}

// Get average frequency in a range
float getAverageFrequency(sampler2D fftChannel, float start, float end) {
    float sum = 0.0;
    float count = 0.0;
    for(float i = start; i < end; i += 0.01) {
        sum += getFrequency(fftChannel, i);
        count += 1.0;
    }
    return sum / max(count, 1.0);
}

// Get frequency intensity with scaling
float getFrequencyIntensity(sampler2D fftChannel, float position, float scale) {
    return pow(texture2D(fftChannel, vec2(position, 0.0)).x, scale);
}

// Get multiple frequency bands
vec3 getFrequencyBands(sampler2D fftChannel, float time) {
    float bass = getFrequencyBand(fftChannel, 0.0, 3.0);
    float mid = getFrequencyBand(fftChannel, 1.0, 3.0);
    float treble = getFrequencyBand(fftChannel, 2.0, 3.0);
    return vec3(bass, mid, treble);
}

// Calculate frequency gradient
vec2 getFrequencyGradient(sampler2D fftChannel, float position) {
    float left = getFrequency(fftChannel, max(position - 0.01, 0.0));
    float right = getFrequency(fftChannel, min(position + 0.01, 1.0));
    return vec2(left, right);
}
