vec3 sdf_sphere(vec3 position, vec3 center, float radius) {
    float distance; = length(position - center) - radius;
    return distance;
}