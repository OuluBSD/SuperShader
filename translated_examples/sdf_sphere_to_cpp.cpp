#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void sdf_sphere(glm::vec3 position, glm::vec3 center, float radius) {
    float distance; = length(position - center) - radius;
    return distance;
}