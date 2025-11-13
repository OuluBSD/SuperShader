#!/usr/bin/env python3
"""
Process audio visualization shaders from JSON files to identify common patterns
and extract reusable modules.
"""

import json
import os
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def find_audio_visualization_shaders(json_dir='json'):
    """
    Find all JSON files that contain audio visualization related tags.

    Args:
        json_dir (str): Directory containing JSON shader files

    Returns:
        list: List of tuples (filepath, shader_info) for audio visualization shaders
    """
    print("Finding audio visualization related shaders...")
    
    keywords = [
        'audio', 'sound', 'music', 'frequency', 'spectrum', 'visualization', 'visualizer',
        'beat', 'rhythm', 'waveform', 'oscilloscope', 'spectrogram', 'fourier', 'fft',
        'amplitude', 'volume', 'intensity', 'bass', 'treble', 'mid', 'frequency',
        'bands', 'analyzer', 'analyzer', 'channel', 'left', 'right', 'stereo',
        'mono', 'samples', 'sample', 'time domain', 'frequency domain', 'vumeter',
        'bars', 'frequency bars', 'wave', 'audio reactive', 'frequency analysis'
    ]
    
    audio_shaders = []
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} JSON files for audio visualization tags...")
    
    for i, filepath in enumerate(json_files):
        if i % 1000 == 0:
            print(f"Scanned {i}/{len(json_files)} files...")
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict) and 'info' in data:
                info = data['info']
                tags = [tag.lower() for tag in info.get('tags', [])]
                name = info.get('name', '').lower()
                description = info.get('description', '').lower()
                
                # Check if this shader is audio visualization related 
                is_audio_related = False
                
                # Check tags
                for tag in tags:
                    if any(keyword in tag for keyword in keywords):
                        is_audio_related = True
                        break
                
                # Check name
                if not is_audio_related:
                    for keyword in keywords:
                        if keyword in name:
                            is_audio_related = True
                            break
                
                # Check description
                if not is_audio_related:
                    for keyword in keywords:
                        if keyword in description:
                            is_audio_related = True
                            break
                
                if is_audio_related:
                    shader_info = {
                        'id': info.get('id', os.path.basename(filepath).replace('.json', '')),
                        'name': info.get('name', ''),
                        'tags': tags,
                        'username': info.get('username', ''),
                        'description': info.get('description', ''),
                        'filepath': filepath
                    }
                    audio_shaders.append((filepath, shader_info))
                    
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue

    print(f"Found {len(audio_shaders)} audio visualization related shaders")
    return audio_shaders


def extract_shader_code(filepath):
    """
    Extract GLSL code from a JSON shader file.

    Args:
        filepath (str): Path to the JSON shader file

    Returns:
        str: GLSL code extracted from the file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        shader_data = json.load(f)

    glsl_code = []

    # Extract GLSL code based on render passes
    if 'renderpass' in shader_data:
        for i, pass_data in enumerate(shader_data['renderpass']):
            if 'code' in pass_data:
                code = pass_data['code']
                pass_type = pass_data.get('type', 'fragment')
                name = pass_data.get('name', f'Pass {i}')
                
                glsl_code.append(f"// {name} ({pass_type})")
                glsl_code.append(code)
                glsl_code.append("")  # Empty line separator
    else:
        # Try to find shader code in other possible fields
        possible_fields = ['fragment_shader', 'vertex_shader', 'shader', 'code', 'main']
        for field in possible_fields:
            if field in shader_data and isinstance(shader_data[field], str):
                code = shader_data[field]
                glsl_code.append(f"// From field: {field}")
                glsl_code.append(code)
                glsl_code.append("")
    
    return "\n".join(glsl_code)


def identify_audio_patterns(shader_code):
    """
    Identify common audio visualization patterns in shader code.

    Args:
        shader_code (str): GLSL code to analyze

    Returns:
        dict: Dictionary of identified audio patterns
    """
    patterns = {
        # Audio sampling and processing
        'audio_sampling': 'ichannel' in shader_code.lower() and ('audio' in shader_code.lower() or 'sound' in shader_code.lower()),
        'frequency_spectrum': 'fft' in shader_code.lower() or 'frequency' in shader_code.lower() and 'spectrum' in shader_code.lower(),
        'amplitude_analysis': 'amplitude' in shader_code.lower() or 'volume' in shader_code.lower() or 'intensity' in shader_code.lower(),
        'time_domain': 'time' in shader_code.lower() and 'domain' in shader_code.lower(),
        'frequency_domain': 'frequency' in shader_code.lower() and 'domain' in shader_code.lower(),
        
        # Visual representations
        'frequency_bars': 'bar' in shader_code.lower() and ('frequency' in shader_code.lower() or 'spectrum' in shader_code.lower()),
        'waveform_visualization': 'wave' in shader_code.lower() and ('visualize' in shader_code.lower() or 'form' in shader_code.lower()),
        'oscilloscope': 'oscillo' in shader_code.lower() or 'scope' in shader_code.lower(),
        'spectrogram': 'spectro' in shader_code.lower() and 'gram' in shader_code.lower(),
        
        # Audio reactive elements
        'audio_reactive': 'audio' in shader_code.lower() and 'reactive' in shader_code.lower(),
        'beat_detection': 'beat' in shader_code.lower() and ('detect' in shader_code.lower() or 'pulse' in shader_code.lower()),
        'rhythm_visualization': 'rhythm' in shader_code.lower() and 'visualize' in shader_code.lower(),
        
        # Channel processing
        'stereo_processing': 'left' in shader_code.lower() and 'right' in shader_code.lower(),
        'channel_analysis': 'channel' in shader_code.lower() and ('ichannel' in shader_code.lower() or 'analyze' in shader_code.lower()),
        
        # Frequency bands
        'bass_band': 'bass' in shader_code.lower() and ('freq' in shader_code.lower() or 'band' in shader_code.lower()),
        'treble_band': 'treble' in shader_code.lower() and ('freq' in shader_code.lower() or 'band' in shader_code.lower()),
        'mid_band': 'mid' in shader_code.lower() and ('freq' in shader_code.lower() or 'band' in shader_code.lower()),
        
        # Visual effects
        'vumeter': 'vumeter' in shader_code.lower() or ('vu' in shader_code.lower() and 'meter' in shader_code.lower()),
        'frequency_analysis': 'analyze' in shader_code.lower() and ('freq' in shader_code.lower() or 'frequency' in shader_code.lower()),
        'fourier_transform': 'fourier' in shader_code.lower() and 'transform' in shader_code.lower(),
    }
    
    # Filter only the patterns that were found
    active_patterns = {k: v for k, v in patterns.items() if v}
    return active_patterns


def find_matching_brace(code, start_pos):
    """
    Find the matching closing brace for an opening brace at start_pos.
    
    Args:
        code (str): The code string
        start_pos (int): Position of the opening brace
    
    Returns:
        int: Position of the matching closing brace, or -1 if not found
    """
    brace_count = 1
    pos = start_pos + 1
    
    while pos < len(code) and brace_count > 0:
        if code[pos] == '{':
            brace_count += 1
        elif code[pos] == '}':
            brace_count -= 1
        pos += 1
    
    return pos - 1 if brace_count == 0 else -1


def extract_complete_functions(code, pattern):
    """
    Extract complete functions that match a given pattern.
    
    Args:
        code (str): GLSL code to search in
        pattern (str): Base pattern to match (like 'audio', 'frequency', 'beat', etc.)
    
    Returns:
        list: List of complete function definitions
    """
    functions = []
    
    # Updated regex to find function declarations with name containing the pattern
    func_patterns = [
        rf'(\w+)\s+[\w\d_]*{pattern}[\w\d_]*\s*\([^)]*\)\s*\{{',
        rf'(\w+)\s+[\w\d_]*{pattern}[\w\d_]*\s*\([^)]*\)\s+[\w\d_]*\s*\{{',  # With qualifier like 'const'
        rf'(\w+)\s+[\w\d_]*{pattern}[\w\d_]*\s*\([^)]*\)\s*\n\s*\{{',  # With newline
    ]
    
    for func_pattern in func_patterns:
        matches = re.finditer(func_pattern, code, re.IGNORECASE)
        
        for match in matches:
            # Find the opening brace position
            open_brace_pos = match.end() - 1  # Position of the opening brace
            
            # Find the matching closing brace
            close_brace_pos = find_matching_brace(code, open_brace_pos)
            
            if close_brace_pos != -1:
                # Extract the complete function
                func_start = match.start()
                func_end = close_brace_pos + 1
                function_code = code[func_start:func_end]
                
                # Clean up and add to results if it's not already there
                function_code = function_code.strip()
                if function_code and function_code not in functions:
                    functions.append(function_code)
    
    return functions


def analyze_audio_shaders():
    """
    Main function to analyze audio visualization shaders.
    """
    print("Analyzing audio visualization shaders...")
    
    audio_shaders = find_audio_visualization_shaders()
    
    # Store shader codes and identified patterns
    shader_codes = []
    all_patterns = []
    pattern_counts = Counter()
    
    print("\nAnalyzing audio patterns in shaders...")
    for i, (filepath, shader_info) in enumerate(audio_shaders):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(audio_shaders)} audio shaders...")
        
        shader_code = extract_shader_code(filepath)
        patterns = identify_audio_patterns(shader_code)
        
        shader_codes.append({
            'info': shader_info,
            'code': shader_code,
            'patterns': patterns
        })
        
        all_patterns.append(patterns)
        
        # Update pattern counts
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    print(f"\nAnalysis complete! Found {len(audio_shaders)} audio visualization shaders.")
    
    # Print pattern distribution
    print(f"\nCommon audio patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count} shaders")
    
    # Save analysis results
    save_audio_analysis(shader_codes, pattern_counts)
    
    return shader_codes, pattern_counts


def save_audio_analysis(shader_codes, pattern_counts):
    """
    Save the audio analysis results to files.
    """
    os.makedirs('analysis/audio', exist_ok=True)
    
    # Save pattern statistics
    with open('analysis/audio/pattern_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Audio Pattern Statistics\n")
        f.write("=" * 50 + "\n")
        for pattern, count in pattern_counts.most_common():
            f.write(f"{pattern.replace('_', ' ').title()}: {count}\n")
    
    # Save detailed shader analysis
    with open('analysis/audio/shader_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Audio Visualization Shader Analysis\n")
        f.write("=" * 50 + "\n")
        for shader_data in shader_codes[:50]:  # Only first 50 for file size
            info = shader_data['info']
            patterns = shader_data['patterns']
            
            f.write(f"\nShader ID: {info['id']}\n")
            f.write(f"Name: {info['name']}\n")
            f.write(f"Author: {info['username']}\n")
            f.write(f"Tags: {', '.join(info['tags'])}\n")
            f.write(f"Patterns: {', '.join([p.replace('_', ' ').title() for p in patterns])}\n")
            f.write("-" * 30 + "\n")
    
    print("Audio visualization shader analysis saved to analysis/audio/ directory")


def extract_audio_modules(shader_codes):
    """
    Extract reusable audio visualization modules from analyzed shaders.

    Args:
        shader_codes (list): List of shader data from analysis

    Returns:
        dict: Dictionary of extracted audio visualization modules
    """
    print("\nExtracting reusable audio visualization modules...")
    
    modules = {
        'audio_sampling': set(),
        'frequency_analysis': set(),
        'amplitude_processing': set(),
        'visual_representations': set(),
        'beat_detection': set(),
        'channel_processing': set()
    }
    
    total_processed = 0
    
    for shader_data in shader_codes:
        code = shader_data['code']
        
        # Extract audio sampling functions
        audio_funcs = extract_complete_functions(code, 'audio')
        audio_funcs += extract_complete_functions(code, 'sample')
        audio_funcs += extract_complete_functions(code, 'channel')
        modules['audio_sampling'].update(audio_funcs)
        
        # Extract frequency analysis functions
        freq_funcs = extract_complete_functions(code, 'frequency')
        freq_funcs += extract_complete_functions(code, 'fft')
        freq_funcs += extract_complete_functions(code, 'spectrum')
        modules['frequency_analysis'].update(freq_funcs)
        
        # Extract amplitude processing functions
        amp_funcs = extract_complete_functions(code, 'amplitude')
        amp_funcs += extract_complete_functions(code, 'volume')
        amp_funcs += extract_complete_functions(code, 'intensity')
        modules['amplitude_processing'].update(amp_funcs)
        
        # Extract visual representation functions
        visual_funcs = extract_complete_functions(code, 'visual')
        visual_funcs += extract_complete_functions(code, 'bar')
        visual_funcs += extract_complete_functions(code, 'wave')
        modules['visual_representations'].update(visual_funcs)
        
        # Extract beat detection functions
        beat_funcs = extract_complete_functions(code, 'beat')
        beat_funcs += extract_complete_functions(code, 'pulse')
        modules['beat_detection'].update(beat_funcs)
        
        # Extract channel processing functions
        channel_funcs = extract_complete_functions(code, 'left')
        channel_funcs += extract_complete_functions(code, 'right')
        channel_funcs += extract_complete_functions(code, 'stereo')
        modules['channel_processing'].update(channel_funcs)
        
        total_processed += 1
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(shader_codes)} shaders...")
    
    print(f"Extraction complete! Found:")
    for module_type, funcs in modules.items():
        print(f"  {module_type}: {len(funcs)} functions")
    
    # Save modules
    save_audio_modules(modules)
    
    return modules


def save_audio_modules(modules):
    """
    Save extracted audio visualization modules to files.
    """
    os.makedirs('modules/audio', exist_ok=True)
    
    for module_type, func_list in modules.items():
        if func_list:  # Only save if there are modules of this type
            with open(f'modules/audio/{module_type}_functions.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} Audio Functions\n")
                f.write("// Automatically extracted from audio visualization-related shaders\n\n")
                
                for i, func in enumerate(func_list, 1):
                    f.write(f"// Function {i}\n")
                    f.write(func)
                    f.write("\n\n")
    
    print("Audio visualization modules saved to modules/audio/ directory")


def create_standardized_audio_modules():
    """
    Create standardized audio visualization modules based on patterns found.
    """
    print("Creating standardized audio visualization modules...")
    
    # Define standardized module templates with actual GLSL implementations
    standardized_modules = {
        'audio_sampling.glsl': generate_audio_sampling_glsl(),
        'frequency_analysis.glsl': generate_frequency_analysis_glsl(),
        'amplitude_processing.glsl': generate_amplitude_processing_glsl(),
        'visual_representations.glsl': generate_visual_representations_glsl(),
        'beat_detection.glsl': generate_beat_detection_glsl(),
        'channel_processing.glsl': generate_channel_processing_glsl()
    }
    
    os.makedirs('modules/audio/standardized', exist_ok=True)
    
    # Create standardized modules
    for filename, code in standardized_modules.items():
        filepath = f'modules/audio/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Created {len(standardized_modules)} standardized audio visualization modules")


def generate_audio_sampling_glsl():
    """Generate GLSL implementation for audio sampling."""
    return """// Audio sampling module
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
"""


def generate_frequency_analysis_glsl():
    """Generate GLSL implementation for frequency analysis."""
    return """// Frequency analysis module
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
"""


def generate_amplitude_processing_glsl():
    """Generate GLSL implementation for amplitude processing."""
    return """// Amplitude processing module
// Standardized amplitude processing implementations

// Get current amplitude level
float getAmplitude(sampler2D audioChannel) {
    return length(texture2D(audioChannel, vec2(0.0, 0.0)).xy);
}

// Get amplitude with smoothing
float getSmoothedAmplitude(sampler2D audioChannel, float smoothness) {
    float current = getAmplitude(audioChannel);
    float prev = texture2D(audioChannel, vec2(0.001, 0.0)).x; // Previous frame value if available
    return mix(prev, current, smoothness);
}

// Apply amplitude-based scaling
vec2 scaleByAmplitude(vec2 position, float amplitude, float maxScale) {
    return position * (1.0 + amplitude * maxScale);
}

// Generate amplitude-based color
vec3 amplitudeToColor(float amplitude) {
    // Map amplitude to color: low = blue/green, high = red/yellow
    return vec3(amplitude, min(1.0, amplitude * 0.5), min(1.0, 1.0 - amplitude));
}

// Calculate amplitude envelope
float getAmplitudeEnvelope(sampler2D audioChannel, float attack, float release) {
    float currentAmp = getAmplitude(audioChannel);
    float previousAmp = texture2D(audioChannel, vec2(0.001, 0.0)).x; // Assuming prev frame data
    
    if(currentAmp > previousAmp) {
        // Attack phase
        return currentAmp * attack;
    } else {
        // Release phase
        return currentAmp * release;
    }
}

// Get root mean square (RMS) of amplitude
float getRMSAmplitude(sampler2D audioChannel, float numSamples) {
    float sum = 0.0;
    float sampleInc = 1.0 / numSamples;
    
    for(float i = 0.0; i < 1.0; i += sampleInc) {
        float sample = texture2D(audioChannel, vec2(i, 0.0)).x;
        sum += sample * sample;
    }
    
    return sqrt(sum / numSamples);
}
"""


def generate_visual_representations_glsl():
    """Generate GLSL implementation for visual representations."""
    return """// Visual representations module
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
"""


def generate_beat_detection_glsl():
    """Generate GLSL implementation for beat detection."""
    return """// Beat detection module
// Standardized beat detection implementations

// Detect beats based on amplitude threshold
bool detectBeat(sampler2D audioChannel, float threshold) {
    float currentAmp = getAmplitude(audioChannel);
    float prevAmp = texture2D(audioChannel, vec2(0.001, 0.0)).x; // Previous frame if stored there
    
    return currentAmp > threshold && currentAmp > prevAmp;
}

// Simple peak detection
float detectPeak(sampler2D audioChannel, float sensitivity) {
    float current = getAmplitude(audioChannel);
    float prev = texture2D(audioChannel, vec2(0.001, 0.0)).x;
    float next = texture2D(audioChannel, vec2(-0.001, 0.0)).x;
    
    // Check if current is higher than neighbors
    return (current > prev * sensitivity && current > next * sensitivity) ? current : 0.0;
}

// Beat-based flashing effect
float beatFlash(sampler2D audioChannel, float flashThreshold, float time, float speed) {
    float amplitude = getAmplitude(audioChannel);
    bool beat = amplitude > flashThreshold;
    
    // Create flashing effect
    return beat ? sin(time * speed) : 0.0;
}

// Calculate beat intensity
float calculateBeatIntensity(sampler2D audioChannel, float prevBeatIntensity) {
    float currentAmp = getAmplitude(audioChannel);
    float beat = max(0.0, currentAmp - 0.5) * 2.0; // Amplify strong beats
    
    // Apply decay to previous intensity
    return max(beat, prevBeatIntensity * 0.9); // Beat decays over time
}

// Detect rhythm pattern
float detectRhythm(sampler2D audioChannel, float time, float patternInterval) {
    float amplitude = getAmplitude(audioChannel);
    
    // Check for rhythmic pattern based on regular intervals
    float patternBeat = step(0.7, amplitude) * sin(time / patternInterval * 6.283);
    return abs(patternBeat);
}

// Create beat-reactive scaling factor
float getBeatScale(sampler2D audioChannel, float baseScale, float maxScale, float threshold) {
    float amplitude = getAmplitude(audioChannel);
    if(amplitude > threshold) {
        return baseScale + (amplitude - threshold) * maxScale;
    }
    return baseScale;
}
"""


def generate_channel_processing_glsl():
    """Generate GLSL implementation for channel processing."""
    return """// Channel processing module
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
"""


def main():
    # Find audio visualization shaders
    audio_shaders = find_audio_visualization_shaders()
    
    # Extract shader codes for a subset (first 500) for efficiency
    shader_codes = []
    for filepath, shader_info in audio_shaders[:500]:  # Limit to first 500 for efficiency
        shader_code = extract_shader_code(filepath)
        shader_codes.append({
            'info': shader_info,
            'code': shader_code
        })
    
    # Analyze shader patterns
    analyzed_shaders, pattern_counts = analyze_audio_shaders()
    
    # Extract specific audio functions
    modules = extract_audio_modules(analyzed_shaders)
    
    # Create standardized modules
    create_standardized_audio_modules()
    
    print("Audio visualization shader analysis and module extraction completed!")


if __name__ == "__main__":
    main()