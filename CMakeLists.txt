cmake_minimum_required(VERSION 3.12)
project(SuperShaderRenderer)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

# Add executable
add_executable(renderer
    src/main.cpp
    src/Renderer.cpp
    src/Shader.cpp
    src/Scene.cpp
    src/Communication.cpp
)

# Include directories
target_include_directories(renderer PRIVATE include)

# Link libraries
target_link_libraries(renderer 
    OpenGL::GL 
    glfw 
    GLEW::GLEW
    ${CMAKE_DL_LIBS}
)

# Set output directory
set_target_properties(renderer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)