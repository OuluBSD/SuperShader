# SuperShader Project Notes

## Architecture
- Two-process architecture: PyQt6 IDE process and C++ OpenGL/Vulkan rendering process
- IDE handles configuration, project management, and code editing
- Rendering process handles real-time graphics and debug overlays

## Goals
- Create an IDE-like application for shader development
- Support hybrid projects combining GPU, CPU, and AI processing
- Provide real-time visualization and debugging
- Enable separate processes for stability and performance

## Components
- IDE Process (Python + PyQt6)
- Rendering Process (C++ + OpenGL/Vulkan)
- Inter-process communication layer
- Debug overlay system
- Performance metrics display

## Development Phases
1. Architecture setup
2. Basic IDE features
3. Rendering integration
4. Advanced features
5. Hybrid projects support