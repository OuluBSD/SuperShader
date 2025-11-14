// Perlin Noise Module
// Implements standard Perlin noise generation with FBM

// Random function for noise generation
float random(vec2 coord) {
    return fract(sin(dot(coord, vec2(12.9898, 78.233))) * 43758.5453);
}

// Interpolation function
float interpolate(float a, float b, float t) {
    // Smooth interpolation using cubic curve
    float ft = t * t * (3.0 - 2.0 * t);
    return mix(a, b, ft);
}

// Perlin noise function
float perlinNoise(vec2 coord, float scale) {
    // Scale the coordinates
    vec2 scaledCoord = coord * scale;

    // Calculate integer and fractional parts
    vec2 i = floor(scaledCoord);
    vec2 f = fract(scaledCoord);

    // Generate random values at the four corners
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Interpolate between the values
    float top = interpolate(a, b, f.x);
    float bottom = interpolate(c, d, f.x);
    float value = interpolate(top, bottom, f.y);

    return value;
}

// Fractal Brownian Motion combining multiple octaves
float fbm(vec2 coord, float scale) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < 4; i++) {
        value += amplitude * perlinNoise(coord * frequency, scale);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// Improved Perlin noise with gradients
float improvedPerlin(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    // Four corners of a unit square
    float a = random(i + vec2(0.0, 0.0));
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);

    // Mix the results
    return mix(a, b, u.x) +
           (c - a) * u.y * (1.0 - u.x) +
           (d - b) * u.x * u.y;
}

// Turbulence function using absolute value of noise
float turbulence(vec2 coord, float scale) {
    float value = 0.0;
    float amplitude = 1.0;

    for (int i = 0; i < 5; i++) {
        value += amplitude * abs(perlinNoise(coord, scale));
        amplitude *= 0.5;
        coord *= 2.0;
        scale *= 2.0;
    }

    return value;
}