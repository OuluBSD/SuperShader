#!/usr/bin/env python3
"""
Advanced Audio Module with Branching for Conflicting Features
This module demonstrates different audio visualization approaches with branching for conflicting features
"""

# Interface definition with branching options
INTERFACE = {
    'inputs': [
        {'name': 'audioBuffer', 'type': 'sampler2D', 'direction': 'in', 'semantic': 'audio_spectrum'},
        {'name': 'uv', 'type': 'vec2', 'direction': 'in', 'semantic': 'texture_coordinates'},
        {'name': 'time', 'type': 'float', 'direction': 'uniform', 'semantic': 'audio_time'},
        {'name': 'sensitivity', 'type': 'float', 'direction': 'uniform', 'semantic': 'audio_sensitivity'},
        {'name': 'decay', 'type': 'float', 'direction': 'uniform', 'semantic': 'decay_rate'},
        {'name': 'threshold', 'type': 'float', 'direction': 'uniform', 'semantic': 'detection_threshold'}
    ],
    'outputs': [
        {'name': 'visualOutput', 'type': 'vec4', 'direction': 'out', 'semantic': 'audio_visualization'},
        {'name': 'beatSignal', 'type': 'float', 'direction': 'out', 'semantic': 'beat_detected'},
        {'name': 'frequencyBands', 'type': 'vec4', 'direction': 'out', 'semantic': 'frequency_analysis'}
    ],
    'uniforms': [
        {'name': 'time', 'type': 'float', 'semantic': 'audio_time'},
        {'name': 'sensitivity', 'type': 'float', 'semantic': 'audio_sensitivity'},
        {'name': 'decay', 'type': 'float', 'semantic': 'decay_rate'},
        {'name': 'threshold', 'type': 'float', 'semantic': 'detection_threshold'}
    ],
    'branches': {
        'visual_representation': {
            'bars': {
                'name': 'Frequency Bars',
                'description': 'Vertical bars representing frequency spectrum',
                'requires': [],
                'conflicts': ['waveform', 'circle', 'particles']
            },
            'waveform': {
                'name': 'Waveform Display',
                'description': 'Time-domain waveform visualization',
                'requires': [],
                'conflicts': ['bars', 'circle', 'particles']
            },
            'circle': {
                'name': 'Circular Spectrum',
                'description': 'Polar coordinate frequency representation',
                'requires': [],
                'conflicts': ['bars', 'waveform', 'particles']
            },
            'particles': {
                'name': 'Particle System',
                'description': 'Audio-reactive particle visualization',
                'requires': ['particle_support'],
                'conflicts': ['bars', 'waveform', 'circle']
            }
        },
        'beat_detection': {
            'energy': {
                'name': 'Energy-based Detection',
                'description': 'Detect beats based on overall energy',
                'requires': [],
                'conflicts': ['frequency', 'amplitude']
            },
            'frequency': {
                'name': 'Frequency-based Detection',
                'description': 'Detect beats by analyzing frequency bands',
                'requires': [],
                'conflicts': ['energy', 'amplitude']
            },
            'amplitude': {
                'name': 'Amplitude-based Detection',
                'description': 'Detect beats by analyzing amplitude changes',
                'requires': [],
                'conflicts': ['energy', 'frequency']
            }
        },
        'color_mapping': {
            'spectral': {
                'name': 'Spectral Color Mapping',
                'description': 'Map frequencies to colors of the spectrum',
                'requires': [],
                'conflicts': ['intensity', 'temporal']
            },
            'intensity': {
                'name': 'Intensity-based Colors',
                'description': 'Colors based on audio signal intensity',
                'requires': [],
                'conflicts': ['spectral', 'temporal']
            },
            'temporal': {
                'name': 'Temporal Color Shifts',
                'description': 'Colors that shift over time with audio',
                'requires': [],
                'conflicts': ['spectral', 'intensity']
            }
        }
    }
}

# Pseudocode for different audio algorithms
pseudocode = {
    'bars_visualization': '''
// Frequency bars visualization
vec3 visualizeBars(sampler2D audioBuffer, vec2 uv, float time) {
    float barWidth = 0.01;
    float barCount = 32.0;
    float barIndex = floor(uv.x * barCount);
    float freq = barIndex / barCount;
    
    // Get audio level for this frequency band
    float level = texture(audioBuffer, vec2(freq, 0.0)).x;
    
    // Create bar effect
    float barX = barIndex / barCount;
    float barMask = smoothstep(barWidth * 0.5, 0.0, abs(uv.x - barX - barWidth * 0.5));
    
    // Scale the bar height based on audio level
    float barHeight = level * 0.8;
    float barY = smoothstep(barHeight, barHeight - 0.02, uv.y);
    
    // Color based on frequency
    vec3 barColor = vec3(0.2, 0.6, 1.0) * (0.5 + 0.5 * level);
    
    return barMask * barY * barColor;
}
    ''',
    
    'waveform_visualization': '''
// Waveform visualization
vec3 visualizeWaveform(sampler2D audioBuffer, vec2 uv, float time) {
    // Sample the waveform data
    float waveform = texture(audioBuffer, vec2(uv.x, 0.5)).x;
    
    // Create waveform line
    float waveY = 0.5 + waveform * 0.3;
    float waveThickness = 0.01;
    float waveLine = 1.0 - smoothstep(waveThickness, 0.0, abs(uv.y - waveY));
    
    // Color the waveform
    vec3 waveColor = vec3(0.0, 0.8, 1.0) * (0.5 + 0.5 * abs(waveform));
    
    return waveLine * waveColor;
}
    ''',
    
    'circle_visualization': '''
// Circular spectrum visualization
vec3 visualizeCircle(sampler2D audioBuffer, vec2 uv, float time) {
    vec2 center = vec2(0.5, 0.5);
    vec2 dir = uv - center;
    float dist = length(dir);
    float angle = atan(dir.y, dir.x);
    
    // Normalize angle to [0, 1]
    float normAngle = (angle + 3.14159) / (2.0 * 3.14159);
    
    // Get audio level for this angle (frequency band)
    float level = texture(audioBuffer, vec2(normAngle, 0.0)).x;
    
    // Create radial bars
    float radialMask = smoothstep(0.1, 0.15, dist) * 
                      (1.0 - smoothstep(0.8, 0.85, dist)) *
                      step(0.3, level);
    
    // Color based on distance and level
    vec3 color = vec3(1.0, 0.5, 0.2) * (level * 0.8 + 0.2);
    
    return radialMask * color;
}
    ''',
    
    'particles_visualization': '''
// Particle system visualization
vec3 visualizeParticles(sampler2D audioBuffer, vec2 uv, float time) {
    vec3 particles = vec3(0.0);
    
    // Calculate number of particles based on audio energy
    float energy = 0.0;
    for (int i = 0; i < 16; i++) {
        float freq = float(i) / 16.0;
        energy += texture(audioBuffer, vec2(freq, 0.0)).x;
    }
    energy /= 16.0;
    
    int particleCount = int(10.0 + energy * 50.0);
    
    for (int i = 0; i < particleCount; i++) {
        float freq = float(i) / float(particleCount);
        float level = texture(audioBuffer, vec2(freq, 0.0)).x;
        
        // Calculate particle position
        vec2 particlePos = vec2(freq, 0.5 + level * 0.4);
        float particleSize = 0.005 + level * 0.01;
        float particleDist = distance(uv, particlePos);
        float particleMask = 1.0 - smoothstep(0.0, particleSize, particleDist);
        
        // Color particles based on frequency
        vec3 particleColor = vec3(freq, 1.0 - freq, 0.5) * (0.5 + level * 0.5);
        particles += particleMask * particleColor;
    }
    
    return particles;
}
    ''',
    
    'energy_beat_detection': '''
// Energy-based beat detection
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
    ''',
    
    'frequency_beat_detection': '''
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
    ''',
    
    'amplitude_beat_detection': '''
// Amplitude-based beat detection
float detectBeatAmplitude(sampler2D audioBuffer, float time, float sensitivity, float decay, float threshold) {
    float currentAmp = texture(audioBuffer, vec2(0.0, 0.0)).x; // Assume amplitude at position 0
    float prevAmp = texture(audioBuffer, vec2(0.001, 0.0)).x;  // Stored in buffer or previous frame
    
    // Calculate amplitude change
    float delta = abs(currentAmp - prevAmp);
    
    // Detect beat if amplitude change exceeds threshold
    float beat = step(threshold, delta * sensitivity);
    
    return beat;
}
    ''',
    
    'spectral_color_mapping': '''
// Spectral color mapping
vec3 mapSpectralColor(float frequency, float intensity) {
    // Map frequency to color (blue -> green -> yellow -> red)
    vec3 color = vec3(0.0);
    color.r = smoothstep(0.2, 0.8, frequency);
    color.g = 1.0 - abs(0.5 - frequency) * 2.0;
    color.b = 1.0 - smoothstep(0.0, 0.6, frequency);
    
    // Apply intensity
    return color * intensity;
}
    ''',
    
    'intensity_color_mapping': '''
// Intensity-based color mapping
vec3 mapIntensityColor(float intensity) {
    // Colors change based on intensity level
    vec3 color = vec3(0.2, 0.4, 1.0); // Blue for low intensity
    
    if (intensity > 0.3) {
        color = vec3(0.2, 1.0, 0.4);  // Green for medium
    }
    if (intensity > 0.6) {
        color = vec3(1.0, 1.0, 0.2);  // Yellow for high
    }
    if (intensity > 0.8) {
        color = vec3(1.0, 0.2, 0.2);  // Red for very high
    }
    
    return color * intensity;
}
    ''',
    
    'temporal_color_mapping': '''
// Temporal color shifting
vec3 mapTemporalColor(float time, float intensity) {
    // Colors shift over time based on audio reactivity
    vec3 color = vec3(
        sin(time * 0.5) * 0.5 + 0.5,
        sin(time * 0.7 + 2.0) * 0.5 + 0.5,
        sin(time * 0.9 + 4.0) * 0.5 + 0.5
    );
    
    return color * intensity;
}
    '''
}

def get_interface():
    """Return the interface definition for this module"""
    return INTERFACE

def get_pseudocode(branch_name=None):
    """Return the pseudocode for this audio module or specific branch"""
    if branch_name and branch_name in pseudocode:
        return pseudocode[branch_name]
    else:
        # Return all pseudocodes
        return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'audio_advanced_branching',
        'type': 'audio',
        'patterns': ['Frequency Bars Visualization', 'Waveform Visualization', 'Circular Spectrum Visualization',
                     'Particle System Visualization', 'Energy-based Beat Detection', 'Frequency-based Beat Detection',
                     'Amplitude-based Beat Detection', 'Spectral Color Mapping', 'Intensity-based Color Mapping',
                     'Temporal Color Mapping'],
        'frequency': 180,
        'dependencies': [],
        'conflicts': [],
        'description': 'Advanced audio visualization algorithms with branching for different approaches',
        'interface': INTERFACE,
        'branches': INTERFACE['branches']
    }

def validate_branches(selected_branches):
    """Validate that the selected branches don't have conflicts"""
    # Check for conflicts between different branch categories
    if 'visual_representation' in selected_branches:
        vis_method = selected_branches['visual_representation']
        
        # Visual representation methods conflict with each other
        valid_vis = True
        return valid_vis
        
    return True