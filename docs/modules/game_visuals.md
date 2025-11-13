# game_visuals

**Category:** game
**Type:** standardized

## Tags
game

## Code
```glsl
// Game visuals module
// Standardized game visual implementations

// Draw a player character indicator
vec3 drawPlayer(vec2 uv, vec2 pos, float size, float rotation) {
    vec2 localUV = (uv - pos) / size;
    localUV = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation)) * localUV;
    
    // Draw a simple arrow or triangle for the player
    float playerShape = 1.0 - step(0.0, -localUV.y) * 
                        step(abs(localUV.x), -localUV.y * 0.5);
    
    return vec3(playerShape);
}

// Draw an enemy indicator
vec3 drawEnemy(vec2 uv, vec2 pos, float size) {
    vec2 localUV = (uv - pos) / size;
    localUV -= vec2(0.5);
    
    // Draw a simple diamond or cross shape for enemies
    float distToCenter = length(localUV);
    float enemyShape = step(distToCenter, 0.3) * (1.0 - step(distToCenter, 0.1));
    
    // Add cross shape
    float cross = max(abs(localUV.x), abs(localUV.y)) < 0.2 ? 1.0 : 0.0;
    enemyShape = max(enemyShape, cross);
    
    return vec3(enemyShape * 0.8, enemyShape * 0.2, enemyShape * 0.2);  // Red enemy
}

// Draw an item/pickup indicator
vec3 drawItem(vec2 uv, vec2 pos, float size) {
    vec2 localUV = (uv - pos) / size;
    localUV -= vec2(0.5);
    
    // Draw a simple circle with pulsing effect
    float dist = length(localUV);
    float pulse = 0.8 + 0.2 * sin(iTime * 5.0); // Pulsing effect
    float itemShape = (1.0 - smoothstep(0.3 * pulse - 0.02, 0.3 * pulse, dist)) * 
                      smoothstep(0.2 * pulse - 0.02, 0.2 * pulse, dist);
    
    // Add shine effect
    vec2 shinePos = vec2(-0.1, 0.1);
    float shine = (1.0 - smoothstep(0.05 - 0.01, 0.05, distance(localUV, shinePos))) * 0.5;
    
    return vec3(itemShape * 0.8 + shine, itemShape * 0.9 + shine, itemShape * 0.2 + shine);  // Yellow/golden item
}

// Draw a weapon indicator
vec3 drawWeapon(vec2 uv, vec2 pos, float size, float rotation) {
    vec2 localUV = (uv - pos) / size;
    localUV = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation)) * localUV;
    
    // Draw a simple gun shape
    float barrel = (abs(localUV.x - 0.3) < 0.05) * (abs(localUV.y) < 0.15) * step(0.0, localUV.x);
    float body = (abs(localUV.x) < 0.25) * (abs(localUV.y) < 0.2);
    float grip = (abs(localUV.x + 0.15) < 0.05) * (abs(localUV.y - 0.15) < 0.15) * step(localUV.y - 0.15, 0.0);
    
    float weaponShape = max(max(barrel, body), grip);
    return vec3(weaponShape * 0.5, weaponShape * 0.5, weaponShape * 0.6);  // Gray weapon
}

// Draw a health pickup
vec3 drawHealthPickup(vec2 uv, vec2 pos, float size) {
    vec2 localUV = (uv - pos) / size;
    localUV -= vec2(0.5);
    
    // Draw a simple cross shape for health
    float horizontal = (abs(localUV.y) < 0.2) * (abs(localUV.x) < 0.05);
    float vertical = (abs(localUV.x) < 0.2) * (abs(localUV.y) < 0.05);
    float cross = max(horizontal, vertical);
    
    return vec3(cross * 0.8, cross * 0.2, cross * 0.2);  // Red health
}

// Create a damage indicator effect
vec3 drawDamageIndicator(vec2 uv, float damageTime, vec2 damagePos) {
    float elapsed = iTime - damageTime;
    if(elapsed > 0.5) return vec3(0.0); // Effect only lasts 0.5 seconds
    
    // Radial red flash
    float dist = distance(uv, damagePos);
    float flash = (1.0 - smoothstep(0.0, 0.2, dist)) * (1.0 - elapsed / 0.5);
    
    return vec3(flash, flash * 0.2, flash * 0.2);  // Red damage indicator
}

```