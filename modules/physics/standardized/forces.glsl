// Forces module
// Standardized force calculation implementations

// Apply gravity force
vec3 applyGravity(vec3 position, vec3 gravityVector, float mass) {
    return gravityVector * mass; // F = mg
}

// Apply spring force (Hooke's law: F = -kx)
vec3 applySpringForce(vec3 particlePos, vec3 anchorPos, float stiffness, float restLength) {
    vec3 displacement = particlePos - anchorPos;
    float currentLength = length(displacement);
    vec3 normalizedDisp = normalize(displacement);
    
    // F = -k * (currentLength - restLength)
    float forceMagnitude = -stiffness * (currentLength - restLength);
    return forceMagnitude * normalizedDisp;
}

// Apply damping force (proportional to velocity)
vec3 applyDamping(vec3 velocity, float dampingCoefficient) {
    return -velocity * dampingCoefficient;
}

// Apply drag force (proportional to velocity squared)
vec3 applyDrag(vec3 velocity, float dragCoefficient) {
    float speed = length(velocity);
    float dragMagnitude = dragCoefficient * speed * speed;
    return -normalize(velocity) * dragMagnitude;
}

// Apply repulsion force between particles
vec3 applyRepulsion(vec3 pos1, vec3 pos2, float strength) {
    vec3 diff = pos1 - pos2;
    float distance = max(length(diff), 0.1); // Avoid division by zero
    return normalize(diff) * strength / (distance * distance);
}

// Apply attraction force between particles
vec3 applyAttraction(vec3 pos1, vec3 pos2, float strength) {
    vec3 diff = pos2 - pos1;
    float distance = max(length(diff), 0.1); // Avoid division by zero
    return normalize(diff) * strength / (distance * distance);
}

// Apply wind force
vec3 applyWindForce(vec3 velocity, vec3 windDirection, float windStrength) {
    vec3 windEffect = windDirection * windStrength;
    return windEffect;
}
