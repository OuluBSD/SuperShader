// Standard Animation and Procedural Generation Library
// Common functions for animations, physics simulations, and procedural generation

#ifndef STANDARD_ANIMATION_GL
#define STANDARD_ANIMATION_GL

// ---------------------------------------------------------------------------
// ANIMATION UTILITIES
// ---------------------------------------------------------------------------

// Smooth oscillation between two values
float oscillate(float min_val, float max_val, float speed) {
    return min_val + (max_val - min_val) * (sin(iTime * speed) * 0.5 + 0.5);
}

// Ping-pong oscillation (back and forth)
float pingPong(float min_val, float max_val, float speed) {
    return min_val + (max_val - min_val) * abs(sin(iTime * speed));
}

// Step-based animation (like a clock)
float stepAnimation(float steps, float speed) {
    return mod(floor(iTime * speed), steps);
}

// Smooth pulse between 0 and 1
float pulse(float frequency, float duration) {
    return smoothstep(0.0, duration, sin(iTime * frequency));
}

// Sawtooth wave (0 to 1 repeating)
float sawWave(float frequency) {
    return mod(iTime * frequency, 1.0);
}

// Triangle wave (0 to 1 to 0 repeating)  
float triangleWave(float frequency) {
    float s = sin(iTime * frequency);
    return abs(acos(cos(s * 3.14159)) / 3.14159);
}

// ---------------------------------------------------------------------------
// NOISE FUNCTIONS
// ---------------------------------------------------------------------------

// Classic Perlin noise (simplified implementation)
// Based on Hugo Elias implementation
float perlinNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Four corners in 2D of a tile
    float a = hash22(i);
    float b = hash22(i + vec2(1.0, 0.0));
    float c = hash22(i + vec2(0.0, 1.0));
    float d = hash22(i + vec2(1.0, 1.0));
    
    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) + 
           (c - a) * u.y * (1.0 - u.x) + 
           (d - b) * u.x * u.y;
}

// Fractional Brownian Motion (Fractal noise)
float fbm(vec2 p, int octaves, float persistence, float scale) {
    float total = 0.0;
    float frequency = scale;
    float amplitude = 1.0;
    float maxValue = 0.0;
    
    for (int i = 0; i < octaves; i++) {
        total += perlinNoise(p * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    
    return total / maxValue;
}

// Turbulence (absolute value of noise)
float turbulence(vec2 p, int octaves, float persistence, float scale) {
    float total = 0.0;
    float frequency = scale;
    float amplitude = 1.0;
    
    for (int i = 0; i < octaves; i++) {
        total += abs(perlinNoise(p * frequency)) * amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    
    return total;
}

// Domain warping using noise
vec2 warpDomain(vec2 p, float amount) {
    float noise1 = perlinNoise(p + vec2(0.0, 0.0));
    float noise2 = perlinNoise(p + vec2(10.0, 10.0));
    return p + vec2(noise1, noise2) * amount;
}

// ---------------------------------------------------------------------------
// PARTICLE SYSTEMS
// ---------------------------------------------------------------------------

// Particle motion with gravity
vec2 particleMotion(vec2 initialVelocity, vec2 gravity, vec2 initialPosition, float time, float angle) {
    float rad = radians(angle);
    vec2 rotatedVel = vec2(initialVelocity.x * cos(rad) - initialVelocity.y * sin(rad),
                           initialVelocity.x * sin(rad) + initialVelocity.y * cos(rad));
    return initialPosition + rotatedVel * time + 0.5 * gravity * time * time;
}

// Particle velocity at time
vec2 particleVelocity(vec2 initialVelocity, vec2 acceleration, float time, float angle) {
    float rad = radians(angle);
    vec2 rotatedVel = vec2(initialVelocity.x * cos(rad) - initialVelocity.y * sin(rad),
                           initialVelocity.x * sin(rad) + initialVelocity.y * cos(rad));
    return rotatedVel + acceleration * time;
}

// Simple particle with lifetime
float particleLife(float birthTime, float lifetime) {
    float age = iTime - birthTime;
    return 1.0 - smoothstep(0.0, lifetime, age);
}

// ---------------------------------------------------------------------------
// PHYSICS SIMULATIONS
// ---------------------------------------------------------------------------

// Simple spring oscillation
float spring(float restPos, float currentPos, float velocity, float stiffness, float damping) {
    float force = -stiffness * (currentPos - restPos);
    velocity += force * 0.016; // Assuming 60fps timestep
    velocity *= damping;
    return currentPos + velocity * 0.016;
}

// Pendulum motion
float pendulum(float amplitude, float frequency, float phase) {
    return amplitude * sin(frequency * iTime + phase);
}

// Wave propagation
vec2 waveMotion(vec2 position, float amplitude, float frequency, float speed, vec2 direction) {
    float wave = amplitude * sin(dot(position, direction) * frequency - iTime * speed);
    return position + normalize(direction) * wave;
}

// Orbital motion
vec2 orbitalMotion(vec2 center, float radius, float angularVelocity, float initialAngle) {
    float angle = initialAngle + angularVelocity * iTime;
    return center + vec2(radius * cos(angle), radius * sin(angle));
}

// Random walk
vec2 randomWalk(vec2 seed, float stepSize) {
    float angle = hash22(seed + vec2(iTime)) * 6.2831853; // 2PI
    return vec2(cos(angle), sin(angle)) * stepSize;
}

// ---------------------------------------------------------------------------
// PROCEDURAL PATTERNS
// ---------------------------------------------------------------------------

// Grid pattern
float gridPattern(vec2 uv, float cellSize, float lineWidth) {
    vec2 grid = abs(fract(uv / cellSize) - 0.5);
    float dist = min(grid.x, grid.y);
    return 1.0 - smoothstep(0.0, lineWidth, dist);
}

// Hexagonal grid
float hexGrid(vec2 uv, float scale) {
    vec2 q = vec2(uv.x * 1.1547, uv.y + uv.x * 0.57735);
    vec2 id = floor(q);
    vec2 odd = step(1.0 - abs(fract(q.y) * 2.0 - 1.0), fract(q.x));
    id += odd * vec2(1.0, -1.0);
    q = q - id - odd * vec2(0.0, 1.0);
    vec2 o = vec2(0.0, -1.0);
    if (abs(dot(q, vec2(0.5, 0.866))) > abs(dot(q, vec2(0.5, -0.866))))
        o = vec2(-1.0, 0.0);
    id += o;
    return length(fract(id) - 0.5);
}

// Checkerboard pattern
float checkerboard(vec2 uv, float scale) {
    vec2 s = floor(uv * scale);
    return mod(s.x + s.y, 2.0);
}

// Stripes pattern
float stripes(vec2 uv, float scale, float width) {
    return mod(uv.y * scale, 1.0) < width ? 1.0 : 0.0;
}

// Radial stripes
float radialStripes(vec2 uv, float count, float width) {
    float angle = atan(uv.y, uv.x) / 6.2831853; // 2PI
    float sector = mod(angle * count, 1.0);
    return mod(sector, width) < width * 0.5 ? 1.0 : 0.0;
}

// ---------------------------------------------------------------------------
// RANDOM FUNCTIONS
// ---------------------------------------------------------------------------

// 2D hash function
float hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Random with seed
float random(float seed) {
    return fract(sin(seed) * 43758.5453);
}

// Random in range [min, max]
float randomRange(float min, float max, float seed) {
    return min + random(seed) * (max - min);
}

// Pseudo-random number generator for 3D positions
float random3(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

// ---------------------------------------------------------------------------
// UTILITY ANIMATION FUNCTIONS
// ---------------------------------------------------------------------------

// Smooth interpolation with easing
float easeIn(float t) {
    return t * t;
}

float easeOut(float t) {
    return t * (2.0 - t);
}

float easeInOut(float t) {
    return t < 0.5 ? 2.0 * t * t : -1.0 + (4.0 - 2.0 * t) * t;
}

// Bounce easing
float easeBounce(float t) {
    if (t < 1.0 / 2.75) {
        return 7.5625 * t * t;
    } else if (t < 2.0 / 2.75) {
        return 7.5625 * (t -= 1.5 / 2.75) * t + 0.75;
    } else if (t < 2.5 / 2.75) {
        return 7.5625 * (t -= 2.25 / 2.75) * t + 0.9375;
    } else {
        return 7.5625 * (t -= 2.625 / 2.75) * t + 0.984375;
    }
}

// Exponential easing
float easeExpo(float t) {
    return t == 0.0 ? 0.0 : pow(2.0, 10.0 * (t - 1.0));
}

// Circular easing
float easeCircular(float t) {
    return 1.0 - sqrt(1.0 - t * t);
}

// Elastic easing
float easeElastic(float t) {
    return sin(-13.0 * (t + 1.0) * 3.14159 / 2.0) * pow(2.0, -10.0 * t) + 1.0;
}

// Rotation matrix
mat2 rotate2D(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

// Rotation in 3D space
mat3 rotateX(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat3(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
}

mat3 rotateY(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat3(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
}

mat3 rotateZ(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat3(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
}

// Smooth blend between two values over time
float smoothBlend(float start, float end, float duration) {
    float t = mod(iTime, duration) / duration;
    return mix(start, end, smoothstep(0.0, 1.0, t));
}

// Sequenced animation with multiple states
float sequencedAnimation(float state1, float state2, float state3, float duration) {
    float cycle = mod(iTime, duration * 3.0);
    if (cycle < duration) {
        return state1;
    } else if (cycle < 2.0 * duration) {
        return state2;
    } else {
        return state3;
    }
}

#endif // STANDARD_ANIMATION_GL