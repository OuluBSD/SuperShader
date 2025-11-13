// Audio sampling module
// Standardized audio sampling implementations

// Sample audio data from channel
vec2 sampleAudio(sampler2D iChannel) {
    return texture2D(iChannel, vec2(0.0, 0.0)).xy; // x: left channel, y: right channel
}

// Sample audio at specific time
vec2 sampleAudioAtTime(sampler2D audioChannel, float time) {
    return texture2D(audioChannel, vec2(time, 0.0)).xy;
}

// Sample audio with position-based offset
vec2 sampleAudioWithOffset(sampler2D audioChannel, vec2 offset) {
    return texture2D(audioChannel, offset).xy;
}

// Get audio level at current frame
float getAudioLevel(sampler2D audioChannel) {
    return length(texture2D(audioChannel, vec2(0.0)).xy);
}

// Sample audio data for waveform visualization
vec2 sampleWaveform(sampler2D audioChannel, float position) {
    vec2 sample = texture2D(audioChannel, vec2(position, 0.0)).xy;
    return sample;
}

// Sample audio with smoothing
vec2 sampleAudioSmooth(sampler2D audioChannel, float position, float smoothing) {
    vec2 current = texture2D(audioChannel, vec2(position, 0.0)).xy;
    vec2 previous = texture2D(audioChannel, vec2(position - smoothing, 0.0)).xy;
    return mix(previous, current, 0.5);
}
