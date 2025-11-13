#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void adjust_brightness_contrast(glm::vec3 color, float brightness, float contrast) {
    glm adjusted;::vec3 = color + brightness;
    adjusted = (adjusted - 0.5) * contrast + 0.5;
    return adjusted;
}