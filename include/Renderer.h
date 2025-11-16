#pragma once

#include <string>
#include <vector>
#include <memory>

// Forward declarations
class Shader;
class Scene;

class Renderer {
public:
    Renderer(int width = 800, int height = 600);
    ~Renderer();

    bool initialize();
    void render();
    void cleanup();

    void setViewport(int width, int height);
    void swapBuffers();

    // Scene management
    void setScene(std::shared_ptr<Scene> scene);
    
    // Shader management
    void addShader(std::shared_ptr<Shader> shader);
    void removeShader(const std::string& name);

private:
    int width_, height_;
    bool initialized_;

    std::shared_ptr<Scene> current_scene_;
    std::vector<std::shared_ptr<Shader>> shaders_;
    
    // OpenGL context handles (platform-specific)
    void* window_context_;
};