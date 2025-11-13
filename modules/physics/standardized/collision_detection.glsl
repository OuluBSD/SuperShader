// Collision detection module
// Standardized collision detection implementations

// Check collision between sphere and point
bool spherePointCollision(vec3 spherePos, float sphereRadius, vec3 point) {
    float distance = length(point - spherePos);
    return distance < sphereRadius;
}

// Check collision between two spheres
bool sphereSphereCollision(vec3 pos1, float radius1, vec3 pos2, float radius2) {
    float minDistance = radius1 + radius2;
    float actualDistance = length(pos2 - pos1);
    return actualDistance < minDistance;
}

// Check collision between sphere and plane
bool spherePlaneCollision(vec3 spherePos, float sphereRadius, vec3 planeNormal, float planeDistance) {
    float distanceToPlane = dot(spherePos, planeNormal) - planeDistance;
    return abs(distanceToPlane) < sphereRadius;
}

// Get collision response vector for sphere-plane collision
vec3 spherePlaneCollisionResponse(vec3 spherePos, vec3 sphereVel, vec3 planeNormal, float sphereRadius, float planeDistance) {
    float distance = dot(spherePos, planeNormal) - planeDistance;
    
    if (distance < sphereRadius) {
        // Calculate reflection vector
        vec3 reflection = reflect(sphereVel, planeNormal);
        // Add some damping to the reflection
        return reflection * 0.8;
    }
    
    return sphereVel; // No collision, return original velocity
}

// Check collision with rectangular bounds
bool boundsCollision(vec3 position, vec3 minBounds, vec3 maxBounds) {
    return position.x < minBounds.x || position.x > maxBounds.x ||
           position.y < minBounds.y || position.y > maxBounds.y ||
           position.z < minBounds.z || position.z > maxBounds.z;
}

// Constrain position to rectangular bounds with bounce
vec3 bounceBounds(vec3 position, vec3 velocity, vec3 minBounds, vec3 maxBounds) {
    vec3 newPos = position;
    vec3 newVel = velocity;
    
    if (position.x <= minBounds.x || position.x >= maxBounds.x) {
        newVel.x *= -0.8; // Reverse and dampen velocity
        newPos.x = position.x <= minBounds.x ? minBounds.x : maxBounds.x;
    }
    
    if (position.y <= minBounds.y || position.y >= maxBounds.y) {
        newVel.y *= -0.8;
        newPos.y = position.y <= minBounds.y ? minBounds.y : maxBounds.y;
    }
    
    if (position.z <= minBounds.z || position.z >= maxBounds.z) {
        newVel.z *= -0.8;
        newPos.z = position.z <= minBounds.z ? minBounds.z : maxBounds.z;
    }
    
    return newVel;
}
