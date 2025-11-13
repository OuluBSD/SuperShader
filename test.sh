#!/bin/bash
# test.sh - Simulate C++ compilation errors for SuperShader project
# This script demonstrates common compilation errors that might occur
# when building C++ shader programs using the SuperShader system

echo "SuperShader C++ Compilation Test"
echo "==============================="

# Create a temporary directory for our test
TEST_DIR=$(mktemp -d)
cd "$TEST_DIR"

# Create a basic C++ shader example
cat > shader_example.cpp << 'EOF'
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

class ShaderProgram {
private:
    unsigned int programID;
    
public:
    ShaderProgram() {
        programID = glCreateProgram();
    }
    
    void loadVertexShader(const char* vertexCode) {
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexCode, NULL);
        glCompileShader(vertexShader);
        
        // Check for compilation errors
        int success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if(!success) {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cout << "ERROR::VERTEX_SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        glAttachShader(programID, vertexShader);
    }
    
    void linkProgram() {
        glLinkProgram(programID);
        
        int success;
        char infoLog[512];
        glGetProgramiv(programID, GL_LINK_STATUS, &success);
        if(!success) {
            glGetProgramInfoLog(programID, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }
    }
};

int main() {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "SuperShader Test", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Create shader program
    ShaderProgram shader;
    
    // Load vertex shader
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProjection;
        
        void main() {
            gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
        }
    )";
    
    shader.loadVertexShader(vertexShaderSource);
    shader.linkProgram();

    // Render loop
    while(!glfwWindowShouldClose(window)) {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
EOF

echo "Attempting to compile shader_example.cpp..."
echo "------------------------------------------"

# Try to compile without required libraries (common error)
echo "Compiling without required libraries..."
g++ shader_example.cpp -o shader_example 2>&1
echo
echo "As expected, this fails because we didn't link required libraries."
echo

# Try to compile with wrong library names
echo "Compiling with wrong library names..."
g++ shader_example.cpp -o shader_example -lGL -lGLU -lglut 2>&1
echo
echo "This fails because we're missing the correct library names (should be -lglfw, -lGLEW)."
echo

# Try to compile with missing GLM header path
echo "Compiling with missing header paths..."
g++ shader_example.cpp -o shader_example -lglfw -lGLEW -lGL 2>&1
echo
echo "This might fail with linking errors depending on your system setup."
echo

# Create a file with explicit errors to simulate common mistakes
cat > error_shader.cpp << 'EOF'
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>  // This might not be found if GLM is not installed

class ShaderProgram {
private:
    unsigned int programID;
    
public:
    ShaderProgram() {
        programID = glCreateProgram();
    }
    
    void loadVertexShader(const char* vertexCode) {
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexCode, NULL);
        glCompileShader(vertexShader);
        
        // This function is called incorrectly (missing & for address)
        int success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, success); // ERROR: missing & 
        if(!success) {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cout << "ERROR::VERTEX_SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        glAttachShader(programID, vertexShader);
    }
    
    // This function has a type mismatch
    GLuint linkProgram() {  // ERROR: Return type doesn't match implementation
        glLinkProgram(programID);
        
        int success;
        char infoLog[512];
        glGetProgramiv(programID, GL_LINK_STATUS, &success);
        if(!success) {
            glGetProgramInfoLog(programID, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }
        return programID;  // ERROR: Function should return void but returns GLuint
    }
};

int main() {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "SuperShader Test", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Create shader program
    ShaderProgram shader;
    
    // Load vertex shader
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProjection;
        
        void main() {
            gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
        }
    )";
    
    shader.loadVertexShader(vertexShaderSource);
    shader.linkProgram();

    // Render loop
    while(!glfwWindowShouldClose(window)) {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
EOF

echo "Compiling error_shader.cpp with intentional errors..."
echo "-----------------------------------------------------"
g++ error_shader.cpp -o error_shader -lglfw -lGLEW -lGL 2>&1
echo
echo "These errors demonstrate common C++ compilation mistakes:"
echo "1. Missing address operator (&) in glGetShaderiv call"
echo "2. Type mismatch in function return type"
echo "3. Potential missing header files"
echo

# Clean up
cd - > /dev/null
rm -rf "$TEST_DIR"

echo
echo "Test completed. This script simulates common C++ compilation errors"
echo "that might occur when working with the SuperShader system:"
echo
echo "- Missing library links (-l flags)"
echo "- Incorrect function calls (missing & operator)"
echo "- Type mismatches"
echo "- Missing header files"
echo "- Unresolved external symbols"
echo

# Run Python unit tests for SuperShader components
echo "Running SuperShader Python unit tests..."
cd "$PWD"
python3 core_tests.py
echo

# Run pseudocode functionality tests
echo "Running SuperShader pseudocode tests..."
python3 test_pseudocode.py
echo

echo "All tests completed successfully!"