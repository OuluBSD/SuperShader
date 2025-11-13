// Simulation kernels module
// Standardized simulation function implementations

// Basic particle update
vec2 particleUpdate(vec2 position, vec2 velocity, float deltaTime) {
    return position + velocity * deltaTime;
}

// Simple Euler integration
vec2 eulerIntegration(vec2 position, vec2 velocity, vec2 acceleration, float deltaTime) {
    velocity += acceleration * deltaTime;
    position += velocity * deltaTime;
    return position;
}

// Velocity damping
vec2 applyDamping(vec2 velocity, float dampingFactor) {
    return velocity * (1.0 - dampingFactor);
}

// Constrain to bounds
vec2 constrainToBounds(vec2 position, vec2 minBounds, vec2 maxBounds) {
    return clamp(position, minBounds, maxBounds);
}

// Distance constraint for springs
vec2 springConstraint(vec2 posA, vec2 posB, float restLength) {
    vec2 delta = posB - posA;
    float distance = length(delta);
    if(distance > 0.0) {
        float diff = (distance - restLength) / distance;
        vec2 offset = delta * diff * 0.5;
        return offset;
    }
    return vec2(0.0);
}

// Simple collision detection with response
vec2 collisionResponse(vec2 position, vec2 size, vec2 minBounds, vec2 maxBounds, vec2 velocity) {
    vec2 newPos = position;
    
    if(position.x - size.x < minBounds.x) {
        newPos.x = minBounds.x + size.x;
        velocity.x = -velocity.x * 0.8; // 80% bounce
    } else if(position.x + size.x > maxBounds.x) {
        newPos.x = maxBounds.x - size.x;
        velocity.x = -velocity.x * 0.8;
    }
    
    if(position.y - size.y < minBounds.y) {
        newPos.y = minBounds.y + size.y;
        velocity.y = -velocity.y * 0.8;
    } else if(position.y + size.y > maxBounds.y) {
        newPos.y = maxBounds.y - size.y;
        velocity.y = -velocity.y * 0.8;
    }
    
    return newPos;
}
