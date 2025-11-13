// Constraints module
// Standardized constraint solving implementations

// Distance constraint between two points
vec3 applyDistanceConstraint(vec3 posA, vec3 posB, float restDistance) {
    vec3 delta = posB - posA;
    float currentDistance = length(delta);
    
    if (currentDistance > 0.0) {
        float diff = (currentDistance - restDistance) / currentDistance;
        vec3 offset = delta * diff * 0.5;
        return offset;
    }
    
    return vec3(0.0);
}

// Solve distance constraint for two particles
void solveDistanceConstraint(inout vec3 posA, inout vec3 posB, float restDistance) {
    vec3 delta = posB - posA;
    float currentDistance = length(delta);
    float diff = (currentDistance - restDistance) / currentDistance;
    
    // Move both particles proportionally
    vec3 offset = delta * diff * 0.5;
    posA += offset;
    posB -= offset;
}

// Angle constraint between three points (A-B-C where B is the vertex)
void solveAngleConstraint(inout vec3 posA, inout vec3 posB, inout vec3 posC, float targetAngle) {
    vec3 ab = normalize(posA - posB);
    vec3 cb = normalize(posC - posB);
    
    float currentAngle = acos(dot(ab, cb));
    float angleDiff = targetAngle - currentAngle;
    
    // Rotate points to achieve target angle
    // This is a simplified implementation
    if (abs(angleDiff) > 0.01) {
        // Calculate rotation axis
        vec3 axis = normalize(cross(ab, cb));
        
        // Apply small rotation toward target angle
        float rotation = angleDiff * 0.1;
        mat3 rotationMatrix = rotationMatrix(axis, rotation);
        
        vec3 newA = posB + rotationMatrix * (posA - posB);
        vec3 newC = posB + rotationMatrix * (posC - posB);
        
        posA = newA;
        posC = newC;
    }
}

// Create rotation matrix around axis by angle
mat3 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat3(
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c
    );
}

// Position constraint (keep point within bounds)
vec3 applyPositionConstraint(vec3 position, vec3 minBounds, vec3 maxBounds) {
    return clamp(position, minBounds, maxBounds);
}

// Keep particles within a spherical volume
vec3 applySphericalConstraint(vec3 position, vec3 center, float radius) {
    vec3 offset = position - center;
    float distance = length(offset);
    
    if (distance > radius) {
        return center + normalize(offset) * radius;
    }
    
    return position;
}
