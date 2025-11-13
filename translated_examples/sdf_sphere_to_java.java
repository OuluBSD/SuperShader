public class ShaderFunctions {

void sdf_sphere(Vector3f position, Vector3f center, float radius) {
    float distance; = length(position - center) - radius;
    return distance;
}
}