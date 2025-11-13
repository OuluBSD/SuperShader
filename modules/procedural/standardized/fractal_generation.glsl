// Fractal generation module
// Standardized fractal function implementations

// Mandelbrot set calculation
float mandelbrot(vec2 c, int maxIterations) {
    vec2 z = vec2(0.0);
    float iterations = 0.0;
    
    for(int i = 0; i < maxIterations; i++) {
        if(dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }
    
    // Smooth coloring
    if(iterations < float(maxIterations)) {
        float log_zn = log(dot(z, z)) / 2.0;
        float nu = log(log_zn / log(2.0)) / log(2.0);
        iterations = iterations + 1.0 - nu;
    }
    
    return iterations / float(maxIterations);
}

// Julia set calculation
float julia(vec2 z, vec2 c, int maxIterations) {
    float iterations = 0.0;
    
    for(int i = 0; i < maxIterations; i++) {
        if(dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }
    
    return iterations / float(maxIterations);
}

// Burning Ship fractal
float burningShip(vec2 c, int maxIterations) {
    vec2 z = vec2(0.0);
    float iterations = 0.0;
    
    for(int i = 0; i < maxIterations; i++) {
        if(dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, abs(2.0 * z.x * z.y)) + abs(c);
        iterations++;
    }
    
    return iterations / float(maxIterations);
}
