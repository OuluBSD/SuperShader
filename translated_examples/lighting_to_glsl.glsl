vec3 calculate_diffuse_lighting(vec3 normal, vec3 light_dir, vec3 light_color) {
    float intensity; = max(dot(normal, light_dir), 0.0);
    vec3 diffuse; = intensity * light_color;
    return diffuse;
}