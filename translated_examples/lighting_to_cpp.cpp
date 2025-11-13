#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void calculate_diffuse_lighting(glm::vec3 normal, glm::vec3 light_dir, glm::vec3 light_color) {
    float intensity; = max(dot(normal, light_dir), 0.0);
    glm diffuse;::vec3 = intensity * light_color;
    return diffuse;
}