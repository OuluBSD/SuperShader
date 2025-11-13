# integration_methods

**Category:** physics
**Type:** standardized

## Tags
physics

## Code
```glsl
// Integration methods module
// Standardized physics integration implementations

// Euler integration (simple but less stable)
vec3 eulerIntegration(vec3 position, vec3 velocity, vec3 acceleration, float deltaTime) {
    velocity += acceleration * deltaTime;
    position += velocity * deltaTime;
    return position;
}

// Velocity-based Euler integration
vec3 velocityEulerIntegration(vec3 position, vec3 velocity, vec3 acceleration, float deltaTime) {
    vec3 newVelocity = velocity + acceleration * deltaTime;
    vec3 newPosition = position + newVelocity * deltaTime;
    return newPosition;
}

// Verlet integration (more stable for physics simulation)
vec3 verletIntegration(vec3 currentPosition, vec3 previousPosition, vec3 acceleration, float deltaTime) {
    vec3 temp = currentPosition;
    vec3 newPosition = 2.0 * currentPosition - previousPosition + acceleration * deltaTime * deltaTime;
    // Store current position for next frame as previous position
    previousPosition = temp;
    return newPosition;
}

// Verlet integration with damping
vec3 verletIntegrationDamped(vec3 currentPosition, vec3 previousPosition, vec3 acceleration, float deltaTime, float damping) {
    vec3 temp = currentPosition;
    vec3 newPosition = 2.0 * currentPosition - previousPosition + acceleration * deltaTime * deltaTime;
    newPosition = mix(newPosition, currentPosition, damping); // Apply damping
    previousPosition = temp;
    return newPosition;
}

// Calculate velocity from positions (for Verlet integration)
vec3 calculateVelocity(vec3 currentPosition, vec3 previousPosition, float deltaTime) {
    return (currentPosition - previousPosition) / deltaTime;
}

```