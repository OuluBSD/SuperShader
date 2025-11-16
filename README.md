# SuperShader

An advanced IDE-like shader development environment with hybrid GPU/CPU/AI processing capabilities.

## Overview

SuperShader is a professional-grade development environment for creating complex graphics applications. It features a two-process architecture:

- **IDE Process**: PyQt6-based interface for project management, configuration, and code editing
- **Rendering Process**: High-performance C++/OpenGL rendering engine with real-time visualization

## Features

### Project Management
- Create, load, and manage shader projects
- Support for hybrid projects combining GPU, CPU, and AI processing
- Configuration templates for common use cases

### Shader Development
- Multi-language shader generation (GLSL, HLSL, Metal, WGSL)
- Syntax-highlighted code editor
- Real-time compilation and preview
- CPU and GPU shader execution

### Real-time Visualization
- Live preview of rendered scenes
- Performance metrics display
- Debug overlay system
- Frame analysis tools

### Advanced Processing
- CPU-based shader execution engine
- AI-enhanced graphics features (denoising, upscaling, anti-aliasing)
- Physics simulation integration
- Post-processing effects

## Architecture

The application uses a two-process architecture for stability and performance:

```
[PyQt6 IDE Process] ←→ [C++ OpenGL Rendering Process]
     │                           │
     ├─ Project Management       ├─ Real-time Rendering
     ├─ Configuration UI         ├─ Shader Execution
     ├─ Code Editor              ├─ Debug Visualization
     └─ Process Management       └─ Performance Metrics
```

## Installation

### Prerequisites
- Python 3.8+
- C++ compiler (GCC, Clang, or MSVC)
- OpenGL 4.1+ compatible graphics driver
- CMake 3.12+
- Make (Linux/macOS) or Ninja (Windows)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SuperShader
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build the C++ rendering engine:
   ```bash
   # Using the provided build script:
   ./build_renderer.sh
   
   # Or manually:
   mkdir build
   cd build
   cmake ..
   make
   ```

4. Run the IDE:
   ```bash
   python supershader_ide.py
   ```

## Usage

1. Launch the IDE:
   ```bash
   python supershader_ide.py
   ```

2. Create a new project or open an existing one

3. Configure your scene, lighting, and materials

4. Write and edit shaders using the integrated editor

5. View real-time results in the preview window

## Project Structure

- `ide/` - PyQt6-based IDE components
- `renderer/` - C++ OpenGL rendering engine
- `core/` - Shared components and utilities
- `modules/` - Shader module definitions
- `examples/` - Sample projects and configurations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository.