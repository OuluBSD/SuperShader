# particle_system

**Category:** physics
**Type:** standardized

## Tags
particles, color, physics

## Code
```glsl
// Particle system module
// Standardized particle system implementations

// Particle structure
struct Particle {
    vec3 position;
    vec3 velocity;
    vec3 acceleration;
    vec3 color;
    float size;
    float mass;
    float lifetime;
    float age;
};

// Initialize a particle
Particle initParticle(vec3 pos, vec3 vel, vec3 col, float sz, float m) {
    Particle p;
    p.position = pos;
    p.velocity = vel;
    p.acceleration = vec3(0.0);
    p.color = col;
    p.size = sz;
    p.mass = m;
    p.lifetime = 5.0; // Default lifetime
    p.age = 0.0;
    return p;
}

// Update particle position with velocity
Particle updateParticle(Particle p, float deltaTime) {
    p.velocity += p.acceleration * deltaTime;
    p.position += p.velocity * deltaTime;
    p.age += deltaTime;
    p.acceleration = vec3(0.0); // Reset acceleration
    return p;
}

// Apply force to particle (F = ma => a = F/m)
Particle applyForce(Particle p, vec3 force) {
    p.acceleration += force / p.mass;
    return p;
}

// Check if particle is still alive
bool isAlive(Particle p) {
    return p.age < p.lifetime;
}

// Create a particle emitter
vec3 emitParticle(vec2 uv, float time, vec3 emitterPos) {
    // Use time and UV to create unique particle properties
    float angle = time * 2.0 + uv.x * 10.0;
    float speed = 1.0 + sin(time + uv.y * 5.0);
    vec3 direction = vec3(cos(angle), sin(angle), 0.0);
    return emitterPos + direction * speed * 0.1;
}

```