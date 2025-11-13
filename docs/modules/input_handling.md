# input_handling

**Category:** game
**Type:** standardized

## Dependencies
texture_sampling, normal_mapping

## Tags
texturing, game

## Code
```glsl
// Input handling module
// Standardized input handling implementations

// Get mouse position in UV coordinates
vec2 getMousePosition(sampler2D mouseChannel) {
    return texture2D(mouseChannel, vec2(0.0, 0.0)).xy;
}

// Get mouse click state
bool isMouseClicked(sampler2D mouseChannel) {
    return texture2D(mouseChannel, vec2(0.0, 0.0)).z > 0.0;
}

// Get mouse button state
bool isMouseButtonDown(sampler2D mouseChannel, int buttonIndex) {
    // Assuming button states are stored in different channels
    float buttonState = texture2D(mouseChannel, vec2(float(buttonIndex) / 4.0, 0.0)).x;
    return buttonState > 0.5;
}

// Get keyboard key state
bool isKeyDown(sampler2D keyboardChannel, int keyCode) {
    // Assuming each key state is stored at a specific position in the texture
    float keyState = texture2D(keyboardChannel, vec2(float(keyCode) / 256.0, 0.0)).x;
    return keyState > 0.5;
}

// Check if mouse is over an area
bool isMouseOver(vec2 mousePos, vec2 elementPos, vec2 elementSize) {
    return (mousePos.x > elementPos.x && mousePos.x < elementPos.x + elementSize.x &&
            mousePos.y > elementPos.y && mousePos.y < elementPos.y + elementSize.y);
}

// Get input-based selection
int getSelection(vec2 mousePos, vec2[] elementPositions, vec2[] elementSizes, int numElements) {
    for(int i = 0; i < numElements; i++) {
        if(isMouseOver(mousePos, elementPositions[i], elementSizes[i])) {
            return i;
        }
    }
    return -1; // No selection
}

// Handle joystick input (simplified)
vec2 getJoystickInput(sampler2D joystickChannel, int stickIndex) {
    vec2 offset = vec2(float(stickIndex) / 4.0, 0.0);
    return texture2D(joystickChannel, offset).xy;
}

// Get input direction from D-pad or WASD
vec2 getDirectionInput(sampler2D inputChannel) {
    vec4 keys = texture2D(inputChannel, vec2(0.0, 0.0)); // Assuming W, A, S, D in rgba
    vec2 direction = vec2(0.0);
    
    direction.x = (keys.y > 0.5 ? -1.0 : 0.0) + (keys.w > 0.5 ? 1.0 : 0.0); // A/D
    direction.y = (keys.x > 0.5 ? 1.0 : 0.0) + (keys.z > 0.5 ? -1.0 : 0.0); // W/S
    
    return normalize(direction);
}

```