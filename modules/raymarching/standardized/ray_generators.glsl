// Ray generation module
// Standardized ray generation functions for raymarching

// Generate a ray from camera position through a screen pixel
vec3 getRayDirection(vec2 uv, vec3 cameraPos, vec3 cameraTarget) {
    vec3 forward = normalize(cameraTarget - cameraPos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));
    
    vec3 rayDir = normalize(forward + uv.x * right + uv.y * up);
    return rayDir;
}

// Generate a ray with FOV consideration
vec3 getRayDirectionWithFOV(vec2 uv, vec3 rd, float fov) {
    rd = normalize(rd + fov * uv.x * vec3(1, 0, 0) + fov * uv.y * vec3(0, 1, 0));
    return rd;
}

// Generate primary ray with aspect ratio correction
vec3 generatePrimaryRay(vec2 fragCoord, vec2 resolution, vec3 cameraPos, vec3 cameraTarget) {
    vec2 uv = (fragCoord - 0.5 * resolution.xy) / resolution.y;
    return getRayDirection(uv, cameraPos, cameraTarget);
}
