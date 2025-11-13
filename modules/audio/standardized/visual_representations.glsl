// Visual representations module
// Standardized audio visualization implementations

// Create frequency bar visualization
float createFrequencyBar(float barIndex, float totalBars, sampler2D fftChannel, float barWidth, float barHeight) {
    float bandValue = getFrequencyBand(fftChannel, barIndex, totalBars);
    float barPos = (barIndex / totalBars) - (barWidth / 2.0);
    
    // Calculate bar dimensions
    float bar = smoothstep(barPos, barPos + barWidth, uv.x);
    bar *= smoothstep(0.0, bandValue * barHeight, 1.0 - uv.y);
    
    return bar;
}

// Create waveform visualization
float createWaveform(float posX, sampler2D audioChannel, float amplitudeScale) {
    float waveformValue = texture2D(audioChannel, vec2(posX, 0.0)).x;
    float centerY = 0.5;
    float scaledValue = waveformValue * amplitudeScale * 0.5;
    
    // Draw line at the waveform position
    return 1.0 - smoothstep(0.01, 0.0, abs(uv.y - (centerY + scaledValue)));
}

// Create radial spectrum visualization
vec3 createRadialSpectrum(sampler2D fftChannel, vec2 center, float radius, float rotation) {
    vec2 dir = uv - center;
    float dist = length(dir);
    float angle = atan(dir.y, dir.x) + rotation;
    
    // Map angle to frequency band
    float bandIndex = mod(angle * 0.5 / 3.14159 * 64.0, 64.0); // Assuming 64 bands
    float freqValue = getFrequencyBand(fftChannel, bandIndex, 64.0);
    
    // Return color based on frequency value
    float intensity = step(dist, radius) * freqValue;
    return vec3(intensity, intensity * 0.7, intensity * 0.3);
}

// Create particle system based on audio
vec2 getAudioParticlePosition(float particleIndex, sampler2D audioChannel, float time) {
    float amplitude = getAmplitude(audioChannel);
    float angle = particleIndex * 0.1 + time;
    float radius = amplitude * 0.3;
    
    return vec2(cos(angle) * radius, sin(angle) * radius);
}

// Create pulse effect based on audio
float createPulseEffect(float centerX, float centerY, float time, sampler2D audioChannel, float speed) {
    float amplitude = getAmplitude(audioChannel);
    float dist = distance(uv, vec2(centerX, centerY));
    
    // Create concentric circles based on audio amplitude
    float wave = sin(dist * 50.0 - time * speed + amplitude * 10.0);
    return smoothstep(0.8, 1.0, wave) * amplitude;
}

// Create spectrum analyzer bars
vec3 createSpectrumBars(float y, sampler2D fftChannel, float numBars) {
    float barWidth = 1.0 / numBars;
    float barIndex = floor(uv.x / barWidth);
    float bandValue = getFrequencyBand(fftChannel, barIndex, numBars);
    
    // Draw bars
    float bar = step(mod(uv.x, barWidth), barWidth * 0.8);
    bar *= step(y, bandValue);
    
    // Color based on frequency
    return vec3(bar * (barIndex / numBars), bar * 0.5, bar * (1.0 - barIndex / numBars));
}
