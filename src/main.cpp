#include "Renderer.h"
#include "Scene.h"
#include "Shader.h"
#include "Communication.h"

#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

int main(int argc, char* argv[]) {
    std::cout << "SuperShader Rendering Process Starting..." << std::endl;
    
    // Parse command line arguments
    std::string project_path = (argc > 1) ? argv[1] : "";
    if (!project_path.empty()) {
        std::cout << "Loading project: " << project_path << std::endl;
    } else {
        std::cout << "No project specified, using default scene" << std::endl;
    }
    
    // Initialize communication with IDE
    Communication comm;
    if (!comm.connectToIDE()) {
        std::cerr << "Failed to connect to IDE" << std::endl;
        return -1;
    }
    
    // Send connection confirmation
    Json::Value connect_msg;
    connect_msg["type"] = "connection";
    connect_msg["status"] = "connected";
    connect_msg["process"] = "renderer";
    comm.sendMessage(connect_msg);
    
    // Initialize renderer
    auto renderer = std::make_shared<Renderer>(1024, 768);
    if (!renderer->initialize()) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return -1;
    }
    
    // Create a basic scene
    auto scene = std::make_shared<Scene>();
    renderer->setScene(scene);
    
    // Main rendering loop
    std::cout << "Entering rendering loop..." << std::endl;
    
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (comm.isConnected()) {
        // Calculate delta time
        auto current_time = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float>(current_time - start_time).count();
        start_time = current_time;
        
        // Update scene
        scene->update(delta_time);
        
        // Render frame
        renderer->render();
        
        // Send frame update to IDE periodically
        if (frame_count % 60 == 0) { // Every 60 frames
            Json::Value frame_msg;
            frame_msg["type"] = "frame_update";
            frame_msg["frame"] = frame_count;
            frame_msg["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            comm.sendMessage(frame_msg);
        }
        
        frame_count++;
        
        // Simple frame rate control (simulate 60 FPS)
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    
    std::cout << "Renderer shutting down..." << std::endl;
    
    renderer->cleanup();
    comm.disconnectFromIDE();
    
    return 0;
}