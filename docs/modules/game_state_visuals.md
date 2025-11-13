# game_state_visuals

**Category:** game
**Type:** standardized

## Tags
color, game

## Code
```glsl
// Game state visuals module
// Standardized game state visual implementations

// Draw a menu background
vec3 drawMenuBackground(vec2 uv) {
    // Create a subtle animated background
    float pattern = sin(uv.x * 10.0 + iTime) * cos(uv.y * 8.0 + iTime * 0.5);
    pattern = abs(pattern) * 0.1 + 0.1;
    
    return vec3(pattern, pattern * 0.8, pattern * 1.2);
}

// Draw a pause screen overlay
vec3 drawPauseOverlay(vec2 uv) {
    // Semi-transparent dark overlay
    float overlay = 0.7;
    
    // Draw pause symbol (two vertical bars)
    float pauseSymbol = 0.0;
    float barWidth = 0.02;
    float barHeight = 0.1;
    vec2 center = vec2(0.5, 0.5);
    
    // Left bar
    pauseSymbol += (step(center.x - 0.03, uv.x) - step(center.x - 0.03 + barWidth, uv.x)) * 
                   (step(center.y - barHeight * 0.5, uv.y) - step(center.y + barHeight * 0.5, uv.y));
    
    // Right bar
    pauseSymbol += (step(center.x + 0.01, uv.x) - step(center.x + 0.01 + barWidth, uv.x)) * 
                   (step(center.y - barHeight * 0.5, uv.y) - step(center.y + barHeight * 0.5, uv.y));
    
    return vec3(overlay * 0.7 + pauseSymbol * 0.3);
}

// Draw a level transition effect
vec3 drawLevelTransition(vec2 uv, float progress) {
    // Radial transition - starts from center and expands
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    
    // Transition from 0 to 1 based on progress
    float transition = smoothstep(progress - 0.1, progress, dist);
    
    // Add a bright edge at the transition line
    float edge = 1.0 - smoothstep(progress - 0.02, progress + 0.02, abs(dist - progress));
    vec3 edgeColor = vec3(1.0, 1.0, 0.8) * edge;
    
    return vec3(transition) + edgeColor;
}

// Draw a game over screen
vec3 drawGameOver(vec2 uv, float time) {
    // Dark background
    vec3 bg = vec3(0.1, 0.05, 0.05);
    
    // "GAME OVER" text (simplified)
    vec2 textSize = vec2(0.1, 0.05);
    vec2 textPos = vec2(0.5 - 0.3, 0.5);
    
    // Simple character drawing for "GAME OVER"
    float text = 0.0;
    for(int i = 0; i < 4; i++) {
        vec2 charPos = textPos + vec2(float(i) * textSize.x * 1.2, 0.0);
        text += drawChar(uv, charPos, float(i));
    }
    for(int i = 0; i < 4; i++) {
        vec2 charPos = textPos + vec2(float(i) * textSize.x * 1.2, textSize.y * 1.5);
        text += drawChar(uv, charPos, float(i + 4));
    }
    
    // Pulsing effect
    float pulse = 0.8 + 0.2 * sin(time * 5.0);
    
    return bg * (1.0 - text) + vec3(1.0, pulse, pulse) * text;
}

// Draw a loading screen
vec3 drawLoadingScreen(vec2 uv, float progress) {
    // Dark background
    vec3 bg = vec3(0.1);
    
    // Progress bar
    vec2 barPos = vec2(0.25, 0.7);
    vec2 barSize = vec2(0.5, 0.03);
    
    float loading = drawProgressBar(uv, barPos, barSize, progress);
    
    // Add spinning indicator
    vec2 center = vec2(0.5, 0.5);
    float angle = atan(uv.y - center.y, uv.x - center.x);
    float dist = distance(uv, center);
    
    float spinner = (1.0 - smoothstep(0.05, 0.07, dist)) * 
                    step(0.7, sin((angle + iTime * 5.0) * 8.0));
    
    return bg * (1.0 - loading - spinner) + 
           vec3(0.2, 0.6, 1.0) * loading + 
           vec3(1.0, 0.8, 0.2) * spinner;
}

// Draw a win screen
vec3 drawWinScreen(vec2 uv, float time) {
    // Colorful animated background
    vec3 bg = vec3(
        sin(uv.x * 5.0 + time * 2.0) * 0.2 + 0.3,
        sin(uv.y * 4.0 + time * 1.5) * 0.2 + 0.4,
        sin((uv.x + uv.y) * 3.0 + time * 1.0) * 0.2 + 0.5
    );
    
    // "YOU WIN" text (simplified)
    vec2 textSize = vec2(0.1, 0.05);
    vec2 textPos = vec2(0.5 - 0.25, 0.5);
    
    float text = 0.0;
    for(int i = 0; i < 3; i++) { // "YOU"
        vec2 charPos = textPos + vec2(float(i) * textSize.x * 1.2, 0.0);
        text += drawChar(uv, charPos, float(i));
    }
    for(int i = 0; i < 3; i++) { // "WIN"
        vec2 charPos = textPos + vec2(float(i) * textSize.x * 1.2, textSize.y * 1.5);
        text += drawChar(uv, charPos, float(i + 3));
    }
    
    // Add celebration effects
    float confetti = sin(uv.x * 20.0 + time * 10.0) * sin(uv.y * 15.0 + time * 8.0);
    confetti = abs(confetti) * 0.2;
    
    return (bg + vec3(confetti)) * (1.0 - text) + vec3(1.0) * text;
}

```