// Advanced Procedural and Animation Library
// Based on analysis of real shader implementations

#ifndef ADVANCED_PROCEDURAL_GL
#define ADVANCED_PROCEDURAL_GL

// ---------------------------------------------------------------------------
// FRACTAL GENERATION
// ---------------------------------------------------------------------------

// Mandelbulb distance estimation (simplified version)
float mandelbulbDE(vec3 pos, float power, float bailout) {
    vec3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    
    for (int i = 0; i < 8; i++) {
        r = length(z);
        if (r > bailout) break;
        
        // Convert to polar coordinates
        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        
        // Scale and rotate the point
        float zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;
        
        // Convert back to Cartesian coordinates
        z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
        z += pos;
    }
    
    return 0.5 * log(r) * r / dr;
}

// Multi-fractal (FBM with variable lacunarity and gain)
float multifractal(vec2 p, int octaves, float initialAmp, float freqMult, float ampMult) {
    float total = 0.0;
    float frequency = 1.0;
    float amplitude = initialAmp;
    
    for (int i = 0; i < octaves; i++) {
        total += abs(perlinNoise(p * frequency) - 0.5) * 2.0 * amplitude;
        frequency *= freqMult;
        amplitude *= ampMult;
    }
    
    return total;
}

// Ridged multifractal
float ridgedMultifractal(vec2 p, int octaves, float initialAmp, float freqMult, float ampMult) {
    float total = 0.0;
    float frequency = 1.0;
    float amplitude = initialAmp;
    float prev = 1.0;
    
    for (int i = 0; i < octaves; i++) {
        float n = perlinNoise(p * frequency);
        n = abs(n);  // Create ridges
        n = n * n;   // Sharpen ridges
        n = 1.0 - n; // Invert
        n *= amplitude; // Scale
        n *= prev;   // Self-masking
        total += n;
        prev = n;
        frequency *= freqMult;
        amplitude *= ampMult;
    }
    
    return total;
}

// ---------------------------------------------------------------------------
// FLUID SIMULATION UTILITIES
// ---------------------------------------------------------------------------

// Simple 2D fluid advection simulation
vec2 fluidAdvection(in vec2 velocity, in vec2 position, float dt) {
    // Simple backward tracing for advection
    vec2 pos = position - velocity * dt;
    return pos;
}

// Vorticity confinement (simplified)
vec2 vorticityConfinement(vec2 velocity, float curl) {
    // Estimate vorticity and confine it
    float vort = curl * 0.1;  // Simplified
    return vec2(-vort, vort);
}

// Simple particle interaction
vec2 particleRepulsion(vec2 pos1, vec2 pos2, float radius, float strength) {
    vec2 diff = pos1 - pos2;
    float dist = length(diff);
    
    if (dist < radius && dist > 0.001) {
        float force = strength * (1.0 - dist / radius);
        return normalize(diff) * force;
    }
    
    return vec2(0.0);
}

// ---------------------------------------------------------------------------
// PARTICLE PHYSICS
// ---------------------------------------------------------------------------

// Particle collision with boundary
vec2 particleBoundaryCollision(vec2 position, vec2 velocity, vec2 boundsMin, vec2 boundsMax) {
    vec2 newVel = velocity;
    
    if (position.x <= boundsMin.x || position.x >= boundsMax.x) {
        newVel.x = -newVel.x;
    }
    
    if (position.y <= boundsMin.y || position.y >= boundsMax.y) {
        newVel.y = -newVel.y;
    }
    
    return newVel;
}

// Particle attraction to center
vec2 particleAttraction(vec2 position, vec2 center, float strength) {
    vec2 direction = center - position;
    float distance = length(direction);
    direction = normalize(direction);
    
    // Inverse square law
    float force = min(strength / (distance * distance + 0.1), 10.0);
    return direction * force;
}

// Particle spring connection
vec2 springConnection(vec2 pos1, vec2 pos2, float restLength, float stiffness) {
    vec2 diff = pos1 - pos2;
    float distance = length(diff);
    vec2 direction = normalize(diff);
    
    float force = (distance - restLength) * stiffness;
    return direction * force;
}

// ---------------------------------------------------------------------------
// CUSTOM ANIMATION CURVES
// ---------------------------------------------------------------------------

// Bezier curve animation
vec2 bezierCurve(vec2 p0, vec2 p1, vec2 p2, float t) {
    vec2 a = mix(p0, p1, t);
    vec2 b = mix(p1, p2, t);
    return mix(a, b, t);
}

// Cubic Bezier curve (more control points)
vec2 cubicBezier(vec2 p0, vec2 p1, vec2 p2, vec2 p3, float t) {
    vec2 a = mix(p0, p1, t);
    vec2 b = mix(p1, p2, t);
    vec2 c = mix(p2, p3, t);
    vec2 d = mix(a, b, t);
    vec2 e = mix(b, c, t);
    return mix(d, e, t);
}

// Catmull-Rom spline (interpolating)
vec2 catmullRom(vec2 p0, vec2 p1, vec2 p2, vec2 p3, float t) {
    float tt = t * t;
    float ttt = tt * t;
    
    float x = 0.5 * (
        (2.0 * p1.x) +
        (-p0.x + p2.x) * t +
        (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * tt +
        (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * ttt
    );
    
    float y = 0.5 * (
        (2.0 * p1.y) +
        (-p0.y + p2.y) * t +
        (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * tt +
        (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * ttt
    );
    
    return vec2(x, y);
}

// ---------------------------------------------------------------------------
// COMPLEX ANIMATION SEQUENCES
// ---------------------------------------------------------------------------

// Morphing animation between multiple states
vec2 morphingAnimation(vec2 state1, vec2 state2, vec2 state3, float t) {
    float cycle = mod(t, 3.0);
    if (cycle < 1.0) {
        return mix(state1, state2, cycle);
    } else if (cycle < 2.0) {
        return mix(state2, state3, cycle - 1.0);
    } else {
        return mix(state3, state1, cycle - 2.0);
    }
}

// Animation with easing functions
float animatedEasing(float start, float end, float duration, float currentTime, float (*easeFunc)(float)) {
    float t = clamp(currentTime / duration, 0.0, 1.0);
    t = easeFunc(t);
    return mix(start, end, t);
}

// Sequenced particle spawning
float particleSpawnSequence(float spawnRate, float particleLifetime) {
    float spawnTime = floor(iTime * spawnRate);
    float particleAge = iTime - spawnTime / spawnRate;
    return particleAge < particleLifetime ? 1.0 - particleAge / particleLifetime : 0.0;
}

// ---------------------------------------------------------------------------
// ADVANCED NOISE VARIATIONS
// ---------------------------------------------------------------------------

// Worley noise (cellular noise)
float worleyNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float minDist = 1.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(x, y);
            vec2 point = i + neighbor;
            vec2 randomPoint = hash22(point) * 0.5 + 0.25;
            vec2 offset = neighbor + randomPoint - f;
            float dist = length(offset);
            minDist = min(minDist, dist);
        }
    }
    
    return minDist;
}

// Flow noise (using gradients)
vec2 flowNoise(vec2 p, float timeScale) {
    float angle1 = perlinNoise(p);
    float angle2 = perlinNoise(p + vec2(100.0, 100.0));
    vec2 flow = vec2(cos(angle1), sin(angle1));
    flow += vec2(cos(angle2 + iTime * timeScale), sin(angle2 + iTime * timeScale));
    return flow;
}

// Turbulent flow
vec2 turbulentFlow(vec2 p, float timeScale, float intensity) {
    vec2 baseFlow = flowNoise(p, timeScale);
    vec2 turbulence = flowNoise(p * 2.0, timeScale * 0.5) * intensity;
    return baseFlow + turbulence * 0.5;
}

// ---------------------------------------------------------------------------
// COMPLEX SHAPE ANIMATION
// ---------------------------------------------------------------------------

// Heart-shaped path for particles
vec2 heartPath(float t) {
    float x = 16.0 * pow(sin(t), 3.0);
    float y = 13.0 * cos(t) - 5.0 * cos(2.0*t) - 2.0 * cos(3.0*t) - cos(4.0*t);
    return vec2(x, y) * 0.05;  // Scale down to fit screen
}

// Flower petal animation
vec2 flowerPetal(vec2 center, float time, float petalWidth, float petalLength, float petalCount) {
    float angle = 2.0 * 3.14159 * floor(petalCount * time) / petalCount;
    float rad = radians(angle);
    
    // Create a petal shape
    float dist = length(center);
    float shape = sin(dist * 5.0) * 0.5 + 0.5;
    vec2 offset = vec2(cos(rad) * petalWidth * shape, sin(rad) * petalLength * shape);
    
    return center + offset;
}

// Spiral animation
vec2 spiral(float radius, float turns, float time) {
    float angle = time * 6.28318 * turns;
    return vec2(cos(angle) * radius * time, sin(angle) * radius * time);
}

// Rotating and scaling square animation
vec2 rotatingSquare(float size, float rotation, float scale) {
    float halfSize = size * 0.5 * scale;
    float rad = radians(rotation);
    float cs = cos(rad);
    float sn = sin(rad);
    
    // Return center of square after rotation/scaling
    return vec2(cs, sn) * halfSize;
}

// ---------------------------------------------------------------------------
// PHYSICS-BASED ANIMATIONS
// ---------------------------------------------------------------------------

// Damped harmonic oscillator
float dampedOscillation(float rest, float current, float velocity, float spring, float damping, float dt) {
    float force = -spring * (current - rest);
    velocity += force * dt;
    velocity *= (1.0 - damping);
    return current + velocity * dt;
}

// Verlet integration for physics
vec2 verletIntegration(vec2 currentPosition, vec2 previousPosition, vec2 acceleration, float dt) {
    vec2 velocity = currentPosition - previousPosition;
    vec2 newPosition = currentPosition + velocity + acceleration * dt * dt;
    return newPosition;
}

// Spring-mass system
vec2 springMassSystem(vec2 currentPos, vec2 anchorPos, vec2 velocity, float springConstant, float damping, float mass) {
    vec2 displacement = currentPos - anchorPos;
    vec2 springForce = -springConstant * displacement;
    vec2 dampingForce = -damping * velocity;
    vec2 totalForce = springForce + dampingForce;
    vec2 acceleration = totalForce / mass;
    
    return velocity + acceleration * 0.016;  // Assuming 60fps
}

// Simple cloth simulation (mass-spring system)
vec2 clothSimulation(vec2 currentPos, vec2 neighbors[4], float restLength, float stiffness, vec2 gravity) {
    vec2 force = vec2(0.0);
    
    // Connect to neighbors with springs
    for (int i = 0; i < 4; i++) {
        vec2 diff = neighbors[i] - currentPos;
        float dist = length(diff);
        vec2 direction = normalize(diff);
        float stretch = dist - restLength;
        force += direction * stretch * stiffness;
    }
    
    force += gravity;  // Apply gravity
    return force;
}

// Ripple effect (water simulation)
float rippleEffect(vec2 center, vec2 position, float time, float wavelength, float decay) {
    float dist = distance(position, center);
    float ripple = sin(dist * wavelength - time * 10.0) / (1.0 + dist * decay);
    return ripple;
}

#endif // ADVANCED_PROCEDURAL_GL