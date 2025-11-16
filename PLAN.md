# SuperShader Project Plan

## Architecture Overview

The SuperShader project will be restructured into a two-process architecture:

1. **IDE Process (PyQt6)**:
   - Project management and overall coordination
   - Configuration panels and settings UI
   - Code editing for shaders and configuration files
   - Process management for rendering engine
   - Settings and project file management

2. **Rendering Process (C++ with OpenGL/Vulkan)**:
   - Real-time rendering engine
   - Debug overlays and visualization
   - Performance metrics display
   - Actual GPU/CPU shader execution
   - Lean and optimized for real-time performance

## Core Components

### IDE Process Components
- Project management system
- Configuration editor
- Shader code editor with syntax highlighting
- Process launcher and monitor
- Communication layer with rendering process

### Rendering Process Components
- Graphics rendering engine
- Shader compilation and execution
- CPU shader engine integration
- Debug information overlay
- Performance metrics display
- Real-time visualization

## Development Roadmap

### Phase 1: Architecture Setup
1. Create IDE Process with basic PyQt6 structure
2. Create Rendering Process with basic OpenGL context
3. Establish inter-process communication
4. Set up build system for both processes

### Phase 2: Basic IDE Features
1. Project creation and loading
2. Configuration editor
3. Basic shader code editor
4. Process management

### Phase 3: Rendering Integration
1. Shader compilation and deployment
2. Basic rendering pipeline
3. Communication of parameters between processes
4. Real-time updates

### Phase 4: Advanced Features
1. Debug overlays and visualization
2. Performance metrics
3. CPU shader engine integration
4. AI graphics features integration

### Phase 5: Hybrid Projects
1. Support for multiple processing types (GPU/CPU/AI)
2. Module interconnectivity visualization
3. Advanced project templates
4. Performance optimization

## Technology Stack

### IDE Process
- Language: Python
- GUI Framework: PyQt6
- Inter-Process Communication: TCP Sockets or named pipes

### Rendering Process
- Language: C++
- Graphics API: OpenGL/Vulkan
- Build System: CMake
- Math Library: GLM (OpenGL Mathematics)

## Communication Protocol

The two processes will communicate through:
- JSON messages over TCP/IP
- Shared memory for large data transfers (frames, textures)
- Process signals for lifecycle management