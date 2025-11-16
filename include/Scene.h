#pragma once

#include <string>
#include <vector>
#include <memory>

struct Vertex {
    float x, y, z;          // Position
    float nx, ny, nz;       // Normal
    float u, v;             // Texture coordinates
    float r, g, b, a;       // Color
};

struct Entity {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::string shader_name;
};

struct Light {
    std::string name;
    float position[3];
    float color[3];
    float intensity;
    std::string type;       // "directional", "point", "spot"
};

struct Camera {
    float position[3];
    float target[3];
    float up[3];
    float fov;
    float near_plane;
    float far_plane;
};

class Scene {
public:
    Scene();
    ~Scene();

    void addEntity(const Entity& entity);
    void removeEntity(const std::string& name);
    const std::vector<Entity>& getEntities() const { return entities_; }

    void addLight(const Light& light);
    void removeLight(const std::string& name);
    const std::vector<Light>& getLights() const { return lights_; }

    void setCamera(const Camera& camera);
    const Camera& getCamera() const { return camera_; }

    void update(float deltaTime);

private:
    std::vector<Entity> entities_;
    std::vector<Light> lights_;
    Camera camera_;
};