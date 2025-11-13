// Channel processing module
// Standardized audio channel processing implementations

// Get left channel value
float getLeftChannel(sampler2D audioChannel) {
    return texture2D(audioChannel, vec2(0.0, 0.0)).x;
}

// Get right channel value
float getRightChannel(sampler2D audioChannel) {
    return texture2D(audioChannel, vec2(0.0, 0.0)).y;
}

// Get stereo balance (left-right difference)
float getStereoBalance(sampler2D audioChannel) {
    float left = getLeftChannel(audioChannel);
    float right = getRightChannel(audioChannel);
    return left - right;
}

// Process stereo to mono
float stereoToMono(sampler2D audioChannel) {
    float left = getLeftChannel(audioChannel);
    float right = getRightChannel(audioChannel);
    return (left + right) * 0.5;
}

// Create left/right channel visualization
vec3 visualizeStereo(sampler2D audioChannel, float position) {
    float left = texture2D(audioChannel, vec2(position, 0.0)).x;
    float right = texture2D(audioChannel, vec2(position, 0.0)).y;
    
    // Visualize left on red, right on green
    return vec3(abs(left), abs(right), abs(left - right) * 0.5);
}

// Apply stereo separation effect
vec2 applyStereoSeparation(vec2 position, sampler2D audioChannel) {
    float left = getLeftChannel(audioChannel);
    float right = getRightChannel(audioChannel);
    
    // Apply different effects based on channel
    vec2 offset = vec2(left * 0.05, right * 0.05);
    return position + offset;
}

// Get channel difference for visualization
float getChannelDifference(sampler2D audioChannel) {
    float left = getLeftChannel(audioChannel);
    float right = getRightChannel(audioChannel);
    return abs(left - right);
}

// Create stereo field visualization
vec3 visualizeStereoField(sampler2D audioChannel, vec2 uv) {
    float left = texture2D(audioChannel, vec2(uv.x, 0.0)).x;
    float right = texture2D(audioChannel, vec2(uv.x, 0.0)).y;
    
    // Map left channel to left side, right to right side
    float leftVal = (uv.x < 0.5) ? abs(left) : 0.0;
    float rightVal = (uv.x >= 0.5) ? abs(right) : 0.0;
    
    return vec3(leftVal, rightVal, (leftVal + rightVal) * 0.5);
}
