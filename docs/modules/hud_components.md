# hud_components

**Category:** game
**Type:** standardized

## Dependencies
texture_sampling, raymarching

## Tags
texturing, color, game

## Code
```glsl
// HUD components module
// Standardized HUD component implementations

// Draw a health bar
vec3 drawHealthBar(vec2 uv, vec2 pos, vec2 size, float health, vec3 fillColor, vec3 emptyColor) {
    // Background
    vec2 bgUV = (uv - pos) / size;
    float background = step(0.0, bgUV.x) * step(0.0, bgUV.y) * 
                       (1.0 - step(1.0, bgUV.x)) * (1.0 - step(1.0, bgUV.y));
    
    // Health fill
    vec2 fillUV = (uv - pos) / size;
    float fill = step(0.0, fillUV.x) * step(0.0, fillUV.y) * 
                 step(fillUV.x, health) * (1.0 - step(1.0, fillUV.y));
    
    return mix(emptyColor, fillColor, fill) * background;
}

// Draw an ammo counter
vec3 drawAmmoCounter(vec2 uv, vec2 pos, vec2 size, int currentAmmo, int maxAmmo) {
    // Draw background
    vec2 bgUV = (uv - pos) / size;
    float background = step(0.0, bgUV.x) * step(0.0, bgUV.y) * 
                       (1.0 - step(1.0, bgUV.x)) * (1.0 - step(1.0, bgUV.y));
    
    // Draw ammo count (simplified as bars)
    float ammoPercent = float(currentAmmo) / float(maxAmmo);
    vec2 fillUV = (uv - pos) / size;
    float ammo = step(0.0, fillUV.x) * step(0.0, fillUV.y) * 
                 step(fillUV.x, ammoPercent) * (1.0 - step(1.0, fillUV.y));
    
    // Draw ammo boxes based on count
    vec3 color = vec3(0.0);
    if(background > 0.5) {
        color = vec3(0.2, 0.2, 0.2); // Background color
        if(ammo > 0.5) {
            color = vec3(1.0, 1.0, 0.0); // Ammo color
        }
    }
    
    return color;
}

// Draw a score display
vec3 drawScore(vec2 uv, vec2 pos, vec2 charSize, int score) {
    // Simple digit drawing
    vec3 color = vec3(0.0);
    int remainingScore = score;
    int digitCount = 0;
    
    // Count digits
    int tempScore = max(1, score);  // Ensure at least 1 digit
    while(tempScore > 0) {
        tempScore /= 10;
        digitCount++;
    }
    
    // Draw each digit
    vec2 digitPos = pos;
    for(int i = 0; i < 6; i++) { // Max 6 digits
        if(i < digitCount) {
            int digit = int(mod(float(remainingScore), 10.0));
            remainingScore /= 10;
            
            // Draw digit (simplified)
            vec2 localUV = (uv - digitPos) / charSize;
            localUV -= vec2(0.5);
            
            // Simple box for digit
            float digitShape = (step(-0.4, localUV.x) - step(0.4, localUV.x)) * 
                               (step(-0.4, localUV.y) - step(0.4, localUV.y));
            
            color += vec3(digitShape * float(digit + 1) / 10.0);
            digitPos.x += charSize.x * 1.2;  // Space between digits
        }
    }
    
    return color;
}

// Draw a minimap
vec3 drawMinimap(vec2 uv, vec2 pos, vec2 size, sampler2D worldTexture, vec2 playerPos) {
    // Background
    vec2 bgUV = (uv - pos) / size;
    float background = step(0.0, bgUV.x) * step(0.0, bgUV.y) * 
                       (1.0 - step(1.0, bgUV.x)) * (1.0 - step(1.0, bgUV.y));
    
    if(background < 0.5) return vec3(0.0);
    
    // Sample the world at the relative position
    vec2 worldUV = bgUV; // Simple mapping from minimap UV to world UV
    vec3 worldSample = texture2D(worldTexture, worldUV).rgb;
    
    // Draw player indicator
    vec2 playerUV = (playerPos - pos) / size;
    float playerIndicator = drawCircle(uv, pos + playerUV * size, 0.01);
    
    return worldSample * (1.0 - playerIndicator) + vec3(1.0, 0.0, 0.0) * playerIndicator;
}

// Draw a crosshair
vec3 drawCrosshair(vec2 uv, vec2 center, float size, vec3 color, float thickness) {
    float horizontal = (abs(uv.y - center.y) < thickness) * 
                       (abs(uv.x - center.x) < size ? 1.0 : 0.0);
    float vertical = (abs(uv.x - center.x) < thickness) * 
                     (abs(uv.y - center.y) < size ? 1.0 : 0.0);
    
    // Don't double count the center
    float centerPixel = (abs(uv.x - center.x) < thickness) * 
                        (abs(uv.y - center.y) < thickness);
    
    return color * (horizontal + vertical - centerPixel);
}

// Draw a compass
vec3 drawCompass(vec2 uv, vec2 center, float radius, float rotation) {
    float dist = distance(uv, center);
    float angle = atan(uv.y - center.y, uv.x - center.x);
    
    // Only draw within the circle
    float circle = step(dist, radius) - step(dist, radius * 0.9);
    
    // Calculate angle relative to player direction
    float adjustedAngle = angle - rotation;
    
    // Draw N, E, S, W markers
    float north = step(0.95, cos(adjustedAngle)) * step(dist, radius * 0.95);
    float east = step(0.95, sin(adjustedAngle)) * step(dist, radius * 0.95);
    float south = step(0.95, cos(adjustedAngle + 3.14159)) * step(dist, radius * 0.95);
    float west = step(0.95, sin(adjustedAngle + 3.14159)) * step(dist, radius * 0.95);
    
    vec3 compassColor = vec3(0.0);
    compassColor += vec3(1.0, 0.0, 0.0) * north;  // N = Red
    compassColor += vec3(0.0, 1.0, 0.0) * east;   // E = Green
    compassColor += vec3(0.0, 0.0, 1.0) * south;  // S = Blue
    compassColor += vec3(1.0, 1.0, 0.0) * west;   // W = Yellow
    
    return compassColor * circle;
}

```