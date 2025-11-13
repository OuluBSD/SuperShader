#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <vector>

// Software shader utilities
struct Texture2D {
    std::vector<float> data;
    int width, height, channels;
    
    glm::vec4 sample(glm::vec2 uv) const {
        int x = static_cast<int>(uv.x * width) % width;
        int y = static_cast<int>(uv.y * height) % height;
        int idx = (y * width + x) * channels;
        return glm::vec4(data[idx], data[idx+1], data[idx+2], 
                        channels > 3 ? data[idx+3] : 1.0f);
    }
};

inline glm::vec4 texture_cpu(const Texture2D& tex, glm::vec2 uv) {
    return tex.sample(uv);
}

inline glm::vec4 texture2D_cpu(const Texture2D& tex, glm::vec2 uv) {
    return tex.sample(uv);
}

// Basic lighting calculation
glm::vec3 calculateLighting(glm::vec3 position, glm::vec3 normal, glm::vec3 lightPos, glm::vec3 lightColor) {
    glm::vec3 lightDir = normalize(lightPos - position);
    float diff = std::max(dot(normal, lightDir), 0.0);
    return diff * lightColor;
}

// Main shader function
glm::vec4 mainShader(glm::vec2 uv, glm::vec3 normal, glm::vec3 position) {
    glm::vec3 lightPos = glm::vec3(5.0, 5.0, 5.0);
    glm::vec3 lightColor = glm::vec3(1.0, 1.0, 1.0);
    
    glm::vec3 lighting = calculateLighting(position, normal, lightPos, lightColor);
    return glm::vec4(lighting, 1.0);
}
