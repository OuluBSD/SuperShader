# SuperShader Project Context

## Project Overview

This is the Qwen Code context for the SuperShader repository. SuperShader is a tongue-in-cheek project that combines all shaders into one unified system using generic modules.

## Project Goals

- Break down all shaders into generic, reusable modules
- Identify and eliminate duplicate code
- Find similar modules and group them logically
- Create a management system to combine modules into useful shaders
- Store code in a universal format (pseudocode) translatable to multiple languages and rendering APIs

## Key Directories and Files

- `modules/` - Contains generic shader modules organized by genre
- `search.py` - Enhanced search functionality for shader analysis
- `extract_glsl.py` - Utility to extract GLSL code from JSON files
- `TASKS.md` - Detailed project tasks organized by phases
- `PLAN.md` - Project phases and implementation approach
- `QWEN.md` (this file) - Project context and instructions for AI assistants
- `json/` - Shader data files in JSON format (original data)
- `common/` - Common resources and assets

## Working with JSON Shader Files

The project maintains a large collection of shaders in JSON format in the `json/` directory. To analyze these files:

1. Use `search.py` for tag-based searching and filtering
2. Use the GLSL extraction utility to get shader code
3. Identify patterns across different shaders
4. Group similar functionality into modules

## GLSL Extraction Utility

The `extract_glsl.py` script allows easy extraction of GLSL code from JSON files:

```bash
# Extract GLSL code to stdout for analysis
python extract_glsl.py --json-file path/to/shader.json

# Extract specific shader type (vertex, fragment, etc.)
python extract_glsl.py --json-file path/to/shader.json --type fragment

# Process all JSON files in a directory
python extract_glsl.py --json-dir ./json/
```

## Module Organization

Modules are organized by genre in the `modules/` directory:

```
modules/
├── geometry/
│   ├── transforms/
│   │   ├── matrix_ops/
│   │   │   ├── standard/
│   │   │   └── optimized/
│   │   └── projections/
│   │       ├── perspective/
│   │       └── orthographic/
├── lighting/
│   ├── models/
│   │   ├── phong/
│   │   ├── blinn_phong/
│   │   └── pbr/
```

Branches within module directories handle conflicting features between different implementations.

## Development Workflow

1. Analyze shaders in JSON files using search and extraction tools
2. Identify similar patterns and functionality across shaders
3. Extract common functionality into reusable modules
4. Organize modules by genre and function
5. Handle conflicting features with branching directories
6. Create pseudocode modules for universal compatibility
7. Test modules using the management system
8. Generate complete shaders by combining modules

## Pseudocode Format

Modules are stored in a pseudocode format that can be translated to:
- Programming languages: C/C++, Java, Python, JavaScript, C#
- Graphics APIs: OpenGL, DirectX, Vulkan, Metal, Software rendering

The pseudocode should be clear, maintainable, and easily translatable to specific implementations while preserving the algorithmic essence of the original shader code.

## Working with Tags

Use the tag information from JSON files to group shaders by functionality:
- Process shaders by tag categories (geometry, lighting, effects, etc.)
- Identify common patterns within each tag category
- Create genre-specific modules based on tag organization

## 3D Model and Math Library Integration

- Use assimp library for 3D model loading, replacing built-in 3D primitives when appropriate
- Wrap with intuitive helpers when needed for simple cases (e.g., adding red spheres)
- Maintain primitive support but consider replacing with 3D model loaders when beneficial
- Utilize platform-specific math libraries (GLM for C/C++, DirectX math, etc.) instead of built-in mathematical operations
- Recognize code that duplicates system library functionality for potential replacement with native libraries
- Consider future collection of "library code" that duplicates native language library functionality (currently a curiosity only)

## Documentation and Analysis Tasks

The project includes extensive documentation and analysis phases:

- Create PlantUML diagrams visualizing common, average, minimal, semi-advanced, and app-in-shader architectures
- Generate inline PNG images for all documentation diagrams
- Develop hypotheses for best universal shaders for different complexity levels
- Formulate extension approaches for app-in-shader systems with filesystem access between shaders
- Explore possibilities for inter-shader communication and complex application architectures

## Important Notes

- Original JSON shader files are in the `json/` directory (over 100MB)
- The focus is on analysis and extraction, not modifying the original JSON files
- Module directories should be created to organize functionality, not stored in JSON files
- Use `search.py` to analyze shader content and requirements
- Aim to minimize duplicate code between branches in the same module