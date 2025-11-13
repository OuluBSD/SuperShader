public class ShaderFunctions {

void calculate_diffuse_lighting(Vector3f normal, Vector3f light_dir, Vector3f light_color) {
    float intensity; = max(dot(normal, light_dir), 0.0);
    Vector3f diffuse; = intensity * light_color;
    return diffuse;
}
}