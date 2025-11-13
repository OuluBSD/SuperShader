# SuperShader

Tongue-in-cheek project that combines all shaders into one unified system using generic modules.

## Goal

The SuperShader project aims to:
- Break down all shaders into generic, reusable modules
- Identify and eliminate duplicate code
- Find similar modules and group them logically
- Create a management system to combine modules into useful shaders
- Store code in a universal format (pseudocode) that can be translated to multiple languages and rendering APIs

## Approach

- Modules are organized by genre, with conflicting features separated into branches
- Management code (Python) combines modules into functional shaders
- Use assimp for 3D model loading, replacing built-in 3D primitives when appropriate
- Utilize platform-appropriate math libraries (GLM for C/C++, DirectX math, etc.) instead of built-in mathematical code
- Recognize code that duplicates system library functionality for potential replacement
- Universal pseudocode serves as storage format, easily translatable to:
  - Programming languages: C/C++, Java, Python, JavaScript, C#
  - Graphics APIs: OpenGL, DirectX, Vulkan, Metal, Software rendering

## Structure

- `modules/` - Contains generic shader modules organized by genre
- `management/` - Python scripts to combine modules into shaders
- `extract_glsl.py` - Utility to extract GLSL code from JSON files
- `search.py` - Search functionality for shader analysis
- `TASKS.md` - Detailed project tasks
- `PLAN.md` - Project phases and implementation plan
- `QWEN.md` - Project context and instructions for AI assistants
- `json/` - Original shader data files in JSON format
- `common/` - Common resources and assets

## Features

- Extract GLSL code from JSON shader files
- Analyze and categorize shader modules
- Identify duplicate and similar code segments
- Create generic, reusable modules from specific implementations
- Generate shaders by combining modules
- Tag analysis for organizing shaders by genre
- Feature cataloging system for identifying common shader components
- Data processing pipeline for batch analysis of shaders
- Universal pseudocode system for cross-platform shaders

## Python Scripts Overview

The SuperShader project includes a comprehensive set of Python scripts for shader analysis, module management, and application building:

### Core Management Scripts
- `create_module_engine.py` - Module combination engine to combine modules into functional shaders with validation
- `create_pseudocode_translator.py` - Translates pseudocode to multiple target languages (GLSL, HLSL, Metal, WGSL, C++)
- `create_module_registry.py` - Registry of all available modules with metadata and search functionality
- `create_performance_system.py` - Performance benchmarking for different module combinations

### Shader Generation & Optimization
- `generator/shader_generation_pipeline.py` - Complete pipeline to combine modules and produce final shaders
- `shader_optimizer.py` - Optimization passes for generated shaders with multiple optimization levels
- `advanced_shader_translator.py` - Enhanced translation system with sophisticated rules for complex shader features
- `target_specific_optimizer.py` - Target-specific optimizations tailored to different graphics APIs and platforms

### Module Creation & Processing
- `process_raymarching_shaders.py` - Extract raymarching/raytracing patterns from shaders
- `process_effects_shaders.py` - Extract effects and post-processing patterns
- `process_texturing_shaders.py` - Extract texturing and UV mapping patterns
- `extract_glsl.py` - Extract GLSL code from JSON shader files
- `pipeline.py` - Batch processing pipeline for shader analysis organized by tags

### Analysis & Cataloging Tools
- `analyze_tags.py` - Analyze JSON files to extract all available tags for shader organization
- `catalog_features.py` - Identify duplicate code patterns and compare shader structures
- `search_modules.py` - Find similar code patterns and identify duplicate functionality
- `search.py` - Enhanced search capabilities for shader analysis

### Data Flow & Validation
- `modules/data_flow_validator.py` - Validates data flow between modules with connection validation
- `modules/cross_genre_data_flow_validator.py` - Cross-genre validation system for different shader types
- `pseudocode_translator.py` - Translation tools from pseudocode to target implementations

### Application Building
- `build_app.py` - Application builder with configuration switches for scenes, shaders, cameras, lighting, post effects, neural models, and more
- `software_parallella_shaders.py` - Generator for software C++ and Epiphany Parallella multicore chip shader implementations

### Testing & Utilities
- `test_suite.py` - Comprehensive test suite for all modules and combinations
- `core_tests.py` - Core functionality tests for the module system
- `benchmark_system.py` - Performance benchmarking for different shader configurations
- `module_tester.py` - Testing environment for individual modules
- `generate_docs.py` - Documentation generation for modules and APIs
- `performance_profiler.py` - Performance profiling for generated shaders

## Application Building with build_app.py

The `build_app.py` script allows you to create complete 3D applications with various configurable features:

### Basic Usage
```bash
python build_app.py --help  # Show help (default when no arguments provided)
python build_app.py --camera following --lighting pbr --bloom --shadows --target glsl
```

### Key Options
- `--camera {static,following,free}` - Camera control modes
- `--lighting {pbr,phong,blinn_phong,cel_shading}` - Different lighting models
- `--bloom, --ssao, --shadows` - Post-processing effects
- `--target {glsl,hlsl,metal,wgsl}` - Target graphics API
- `--width, --height` - Window dimensions
- `--output` - Output directory for generated application

The generated application includes keyboard controls (WASD for movement, mouse for look-around) and is ready for compilation with appropriate graphics libraries.

## Testing and Building

- Use the `test.sh` script to simulate common C++ compilation errors
- This helps identify potential build issues in SuperShader C++ implementations

## License

The SuperShader code (excluding the `json/` directory) is licensed under the BSD 3-Clause License (see LICENSE file).
The JSON files in the `json/` directory are not covered by this license and remain the property of their respective copyright holders.