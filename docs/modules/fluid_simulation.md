# fluid_simulation

**Category:** physics
**Type:** standardized

## Dependencies
texture_sampling, normal_mapping

## Tags
texturing, particles, physics

## Code
```glsl
// Fluid simulation module
// Standardized fluid simulation implementations

// Simple grid-based fluid simulation - density advection
vec4 advectDensity(vec2 uv, vec2 velocity, float deltaTime, sampler2D density) {
    vec2 prevPos = uv - velocity * deltaTime;
    return texture(density, prevPos);
}

// Velocity advection step
vec2 advectVelocity(vec2 uv, vec2 velocity, float deltaTime, sampler2D velocityField) {
    vec2 prevPos = uv - velocity * deltaTime;
    return texture(velocityField, prevPos).xy;
}

// Apply simple buoyancy force
vec2 applyBuoyancy(vec3 velocity, float density, float temperature, vec3 ambientTemp, vec3 gravity) {
    // Simple buoyancy model: warmer and less dense fluid rises
    vec3 buoyancy = (temperature - ambientTemp) * vec3(0.0, 1.0, 0.0) * 0.1;
    return velocity + buoyancy + gravity;
}

// Apply vorticity confinement to preserve small-scale details
vec3 vorticityConfinement(vec2 uv, sampler2D velocityField, float gridScale) {
    // Calculate vorticity (curl of velocity field)
    vec2 dx = vec2(gridScale, 0.0);
    vec2 dy = vec2(0.0, gridScale);
    
    float vorticity = (texture(velocityField, uv + dx).y - texture(velocityField, uv - dx).y) / (2.0 * gridScale) 
                    - (texture(velocityField, uv + dy).x - texture(velocityField, uv - dy).x) / (2.0 * gridScale);
    
    // Calculate vorticity force
    vec2 force = vec2(-dFdy(vorticity), dFdx(vorticity));
    float curl = length(vec2(dFdx(vorticity), dFdy(vorticity)));
    force *= 0.01 / max(curl, 0.01); // Normalize and scale
    
    return vec3(force, 0.0);
}

// Simple particle-based fluid simulation using SPH (Smoothed Particle Hydrodynamics)
float sphKernel(vec3 r, float h) {
    float r2 = dot(r, r);
    float h2 = h * h;
    
    if (r2 > h2) return 0.0;
    
    float x = 1.0 - r2 / h2;
    return 8.0 / (3.14159 * h2) * x * x * x;
}

// SPH gradient for pressure calculation
vec3 sphPressureGradient(vec3 r, float h) {
    float r_len = length(r);
    float h2 = h * h;
    
    if (r_len > h || r_len < 0.0001) return vec3(0.0);
    
    float x = 1.0 - r_len / h;
    vec3 grad = -45.0 / (3.14159 * h2 * h) * x * x * normalize(r);
    return grad;
}

```