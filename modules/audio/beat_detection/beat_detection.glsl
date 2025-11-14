// Beat Detection Module
// Implements beat detection and audio analysis for visualization

// Simple energy-based beat detection
float detectBeatEnergy(sampler2D audioBuffer, float time, float sensitivity, float decay, float threshold) {
    float energy = 0.0;

    // Calculate average energy across frequency bands
    for (int i = 0; i < 16; i++) {
        float freq = float(i) / 16.0;
        energy += texture(audioBuffer, vec2(freq, 0.0)).x;
    }

    energy /= 16.0;

    // Apply decay to reduce false positives
    float decayedEnergy = energy * decay;

    // Detect beat if energy exceeds threshold
    float beat = step(threshold, decayedEnergy * sensitivity);

    return beat;
}

// Frequency-domain beat detection
float detectBeatFrequency(sampler2D audioBuffer, float time, float sensitivity, float decay) {
    float lowFreq = 0.0;
    float midFreq = 0.0;

    // Analyze low frequencies (bass)
    for (int i = 0; i < 4; i++) {
        float freq = float(i) / 64.0;
        lowFreq += texture(audioBuffer, vec2(freq, 0.0)).x;
    }
    lowFreq /= 4.0;

    // Analyze mid frequencies
    for (int i = 4; i < 12; i++) {
        float freq = float(i) / 64.0;
        midFreq += texture(audioBuffer, vec2(freq, 0.0)).x;
    }
    midFreq /= 8.0;

    // Beat is detected when low frequencies are significantly higher than mid frequencies
    float beat = step(midFreq * 1.5, lowFreq * sensitivity);

    return beat;
}

// Advanced beat detection combining multiple methods
float detectBeatAdvanced(sampler2D audioBuffer, float time, float sensitivity, float decay, float threshold) {
    float energyBeat = detectBeatEnergy(audioBuffer, time, sensitivity, decay, threshold);
    float frequencyBeat = detectBeatFrequency(audioBuffer, time, sensitivity, decay);

    // Combine both detection methods with weights
    float combinedBeat = mix(energyBeat, frequencyBeat, 0.3);
    float finalBeat = step(0.5, combinedBeat);

    return finalBeat;
}

// Frequency analysis for spectral visualization
vec4 analyzeFrequencies(sampler2D audioBuffer) {
    vec4 bands = vec4(0.0);

    // Low frequencies (bass)
    for (int i = 0; i < 4; i++) {
        float freq = float(i) / 64.0;
        bands.x += texture(audioBuffer, vec2(freq, 0.0)).x;
    }
    bands.x /= 4.0;

    // Mid-low frequencies
    for (int i = 4; i < 8; i++) {
        float freq = float(i) / 64.0;
        bands.y += texture(audioBuffer, vec2(freq, 0.0)).x;
    }
    bands.y /= 4.0;

    // Mid-high frequencies
    for (int i = 8; i < 16; i++) {
        float freq = float(i) / 64.0;
        bands.z += texture(audioBuffer, vec2(freq, 0.0)).x;
    }
    bands.z /= 8.0;

    // High frequencies (treble)
    for (int i = 16; i < 32; i++) {
        float freq = float(i) / 64.0;
        bands.w += texture(audioBuffer, vec2(freq, 0.0)).x;
    }
    bands.w /= 16.0;

    return bands;
}

// Spectrum visualization
vec3 createSpectrum(sampler2D audioBuffer, float position, float time) {
    float intensity = texture(audioBuffer, vec2(position, 0.0)).x;
    float frequency = position * 2000.0;  // Convert to Hz

    // Map frequency to color (blue to red)
    vec3 color = vec3(0.0);
    color.r = smoothstep(0.0, 0.5, position);
    color.b = 1.0 - smoothstep(0.0, 0.7, position);
    color.g = 0.5 - abs(0.5 - position);

    return color * intensity;
}

// Oscilloscope visualization
vec3 createOscilloscope(sampler2D audioBuffer, vec2 uv, float time) {
    float waveform = texture(audioBuffer, vec2(uv.x, 0.5)).x;
    float threshold = 0.01;
    
    // Draw waveform as a line
    float line = 1.0 - smoothstep(threshold, threshold + 0.01, abs(uv.y - 0.5 - waveform * 0.3));
    
    vec3 color = vec3(line);
    color *= 1.0 + 2.0 * abs(uv.y - 0.5); // Brighten center
    
    return color;
}

// Bass pulse effect
float bassPulse(sampler2D audioBuffer, float time) {
    float bassLevel = 0.0;
    
    // Average low frequency content
    for (int i = 0; i < 4; i++) {
        float freq = float(i) / 64.0;
        bassLevel += texture(audioBuffer, vec2(freq, 0.0)).x;
    }
    bassLevel /= 4.0;
    
    // Smooth the bass level to create a pulsing effect
    float pulse = smoothstep(0.0, 0.5, bassLevel);
    
    return pulse;
}

// Audio reactive color shift
vec3 audioReactiveColor(sampler2D audioBuffer, vec3 baseColor, float time) {
    float energy = 0.0;
    
    // Calculate total audio energy
    for (int i = 0; i < 16; i++) {
        float freq = float(i) / 64.0;
        energy += texture(audioBuffer, vec2(freq, 0.0)).x;
    }
    energy /= 16.0;
    
    // Apply audio reactivity to color
    vec3 reactiveColor = baseColor;
    reactiveColor.r += energy * 0.5;
    reactiveColor.g += energy * 0.3 * sin(time);
    reactiveColor.b += energy * 0.4 * cos(time);
    
    return clamp(reactiveColor, 0.0, 1.0);
}

// Fourier Transform visualization (simplified)
vec2 fourierTransform(sampler2D audioBuffer, float freq, float time) {
    float real = 0.0;
    float imag = 0.0;
    
    // Simple discrete Fourier transform
    for (int i = 0; i < 64; i++) {
        float sample = texture(audioBuffer, vec2(float(i) / 64.0, 0.0)).x;
        float angle = 6.28318 * freq * float(i) / 64.0; // 2PI * f * n/N
        real += sample * cos(angle);
        imag -= sample * sin(angle);
    }
    
    return vec2(real, imag);
}

// Waveform visualization
float drawWaveform(sampler2D audioBuffer, vec2 uv, float time) {
    // Get amplitude at this horizontal position
    float amplitude = texture(audioBuffer, vec2(uv.x, 0.0)).x;
    
    // Draw waveform centered vertically
    float waveform = 0.5 + amplitude * 0.3;
    
    // Create a line at the waveform position
    float thickness = 0.02;
    float line = 1.0 - smoothstep(0.0, thickness, abs(uv.y - waveform));
    
    return line;
}