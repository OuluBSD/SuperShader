# SuperShader User Documentation

## Welcome to SuperShader

SuperShader is a modular shader generation system that breaks down complex shaders into reusable, generic modules. This allows for efficient shader composition, code reuse, and cross-platform compatibility through a universal pseudocode format.

## Table of Contents
1. [Getting Started](#getting-started)
2. [System Architecture](#system-architecture)
3. [Using SuperShader](#using-supershader)
4. [Module System](#module-system)
5. [Pseudocode Format](#pseudocode-format)
6. [Application Building](#application-building)
7. [Performance and Optimization](#performance-and-optimization)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic understanding of shader programming concepts
- Familiarity with GLSL (OpenGL Shading Language)

### Installation

#### Method 1: Direct Installation
```bash
# Clone the repository
git clone <repository-url>
cd SuperShader

# The system is ready to use - no additional installation needed
```

#### Method 2: Using the Distribution
If you have a distribution package:
```bash
# Extract the distribution
unzip supershader-1.0.0-src.zip
cd supershader-1.0.0

# The system is ready to use
```

### Quick Start
```bash
# Generate a simple shader
python create_pseudocode_translator.py

# Process a batch of shaders
python pipeline.py --analyze-all

# Build a 3D application
python build_app.py --camera following --lighting pbr --bloom --target glsl --output my_app
```

## System Architecture

### Core Components

#### 1. Module Management System
- **`create_module_registry.py`**: Maintains a registry of all available modules
- **`management/module_combiner.py`**: Combines modules into complete shaders
- **`create_pseudocode_translator.py`**: Translates pseudocode to target languages

#### 2. Shader Generation Pipeline
- **`generator/shader_generation_pipeline.py`**: Complete pipeline for shader generation
- **`shader_optimizer.py`**: Optimizes generated shaders
- **`advanced_shader_translator.py`**: Enhanced translation with complex features

#### 3. Analysis and Processing Tools
- **`analyze_tags.py`**: Extracts and categorizes shader tags
- **`catalog_features.py`**: Identifies common shader components
- **`process_*_shaders.py`**: Genre-specific shader processing scripts

### Data Flow Architecture
The system follows a validated data flow architecture to ensure correct module connections:
- Inputs and outputs are type-checked
- Semantic compatibility is validated
- Connection graphs prevent invalid combinations

## Using SuperShader

### 1. Exploring Existing Shaders

To analyze existing shaders and understand what modules are available:

```bash
# Extract GLSL code from a JSON shader file
python extract_glsl.py --json-file path/to/shader.json

# Search for shaders by tag
python search.py --tag lighting

# Analyze all tags in the collection
python analyze_tags.py
```

### 2. Creating Custom Shaders

#### Basic Module Combination
```python
from management.module_combiner import ModuleCombiner

combiner = ModuleCombiner()
# Combine modules to create a custom shader
custom_shader = combiner.combine_modules([
    'basic_point_light',
    'diffuse_lighting', 
    'normal_mapping'
])
print(custom_shader)
```

#### Advanced Shader Generation
```python
from generator.shader_generation_pipeline import ShaderGenerationPipeline

pipeline = ShaderGenerationPipeline()

# Create a complex shader with validation
result = pipeline.generate_shader_with_validation([
    'pbr_lighting',
    'bloom_effect',
    'shadow_mapping'
])

# Save the generated shader
pipeline.save_shader(result, 'my_custom_shader.glsl')
```

### 3. Pseudocode Translation

Convert pseudocode to different target languages:

```python
from create_pseudocode_translator import PseudocodeTranslator

translator = PseudocodeTranslator()

sample_pseudocode = """
float calculateLighting(vec3 normal, vec3 lightDir) {
    return max(dot(normal, lightDir), 0.0);
}
"""

# Translate to GLSL (default)
glsl_code = translator.translate_to_glsl(sample_pseudocode)

# Translate to other languages
metal_code = translator.translate(sample_pseudocode, 'metal')
hlsl_code = translator.translate(sample_pseudocode, 'hlsl')
```

## Module System

### Module Organization

Modules are organized by genre in the `modules/` directory:

```
modules/
├── lighting/
│   ├── diffuse/
│   ├── specular/
│   ├── pbr/
│   └── advanced/
├── geometry/
│   ├── transforms/
│   ├── projections/
│   └── primitives/
├── effects/
│   ├── bloom/
│   ├── motion_blur/
│   └── ssao/
├── texturing/
│   ├── uv_mapping/
│   ├── filtering/
│   └── triplanar/
├── physics/
├── audio/
├── game/
└── ui/
```

### Branching for Conflicts

When modules have conflicting features, the system uses branching:

```
modules/lighting/pbr/
├── standard/          # Standard PBR implementation
├── optimized/       # Performance-optimized PBR
└── advanced/        # Advanced PBR with more features
```

### Module Parameters

Modules can be parameterized for customization:

```python
from module_parametrization_system import AdvancedParameterizer

param_system = AdvancedParameterizer()

# Define parameter templates for modules
param_system.define_parameter_template('pbr_lighting', {
    'albedo': {
        'type': 'vec3',
        'default': [0.5, 0.5, 0.5],
        'description': 'Base color of the material',
        'min': 0.0,
        'max': 1.0
    },
    'metallic': {
        'type': 'float',
        'default': 0.0,
        'description': 'Metallic property of the material',
        'min': 0.0,
        'max': 1.0
    }
})

# Create parameterized module
parameterized_module = param_system.create_parameterized_module('pbr_lighting', {
    'albedo': [0.8, 0.1, 0.1],  # Red material
    'metallic': 0.9,             # Very metallic
    'roughness': 0.2
})
```

## Pseudocode Format

SuperShader uses a universal pseudocode format that can be translated to multiple languages:

### Basic Syntax
```pseudocode
// Comments use C-style comment syntax

// Data types are GLSL-like
vec3 calculateLighting(vec3 normal, vec3 lightDir, vec3 lightColor) {
    // Function body
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    return diffuse;
}

// Variables
float intensity = 1.0;
vec2 uv = vec2(0.0, 0.0);
```

### Advanced Features
```pseudocode
// Uniforms
uniform vec3 lightPos;
uniform sampler2D texture0;

// Vertex shader structure
in vec3 aPosition;
in vec3 aNormal;
in vec2 aTexCoord;

out vec3 vFragPos;
out vec3 vNormal;
out vec2 vTexCoord;

void main() {
    vFragPos = aPosition;
    vNormal = aNormal;
    vTexCoord = aTexCoord;
    
    gl_Position = /* transformation */;
}
```

## Application Building

### Using the Application Builder

SuperShader includes a powerful application builder that creates complete 3D applications:

```bash
# Build an application with PBR lighting and effects
python build_app.py \
    --camera following \
    --lighting pbr \
    --bloom --ssao --shadows \
    --target glsl \
    --width 1920 --height 1080 \
    --output my_3d_app
```

### Available Options

#### Camera Controls
- `--camera static`: Static camera position
- `--camera following`: Camera follows a target
- `--camera free`: Free movement camera

#### Lighting Models
- `--lighting pbr`: Physically-Based Rendering
- `--lighting phong`: Phong lighting model
- `--lighting blinn_phong`: Blinn-Phong lighting model
- `--lighting cel_shading`: Cel shading (toon rendering)

#### Effects
- `--bloom`: Add bloom effect
- `--ssao`: Screen Space Ambient Occlusion
- `--shadows`: Shadow mapping
- `--motion_blur`: Motion blur effect
- `--depth_of_field`: Depth of field effect

#### Target Platforms
- `--target glsl`: OpenGL Shading Language
- `--target hlsl`: High Level Shading Language
- `--target metal`: Apple Metal
- `--target wgsl`: WebGPU Shading Language

### Generated Application Features
- Keyboard controls (WASD for movement, mouse for look-around)
- Configurable lighting and effects
- Optimized for the selected target platform
- Ready-to-compile with appropriate graphics libraries

## Performance and Optimization

### Performance Testing
```python
from performance_tests import PerformanceTestSuite

suite = PerformanceTestSuite()
results = suite.run_performance_tests(iterations=100)
```

### Shader Optimization
```python
from shader_optimizer import optimize_shader

# Optimize a shader for performance
optimized_shader = optimize_shader(original_shader, optimization_level='high')
```

### Profiling
```python
from performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()
raw_results, analysis = profiler.run_complete_profiling()
```

## Troubleshooting

### Common Issues

#### Module Not Found
**Problem**: `Module 'some_module' not found`
**Solution**: Check that the module exists in the `modules/` directory and that you're using the correct module name.

#### Translation Issues
**Problem**: Pseudocode translation fails
**Solution**: Ensure your pseudocode follows the supported syntax. Check for unsupported operations or syntax errors.

#### Connection Validation Errors
**Problem**: Modules have incompatible interfaces
**Solution**: Verify that module inputs and outputs have compatible types and semantics.

### Getting Help
- Check the existing issue tracker
- Look at example implementations in the codebase
- Examine the `TASKS.md` and `PLAN.md` files for system design details
- Use the test scripts to verify functionality

### Debugging Tips
1. Start with simple module combinations to verify the system works
2. Use the validation tools to check your module combinations
3. Test in small increments rather than trying to build complex shaders immediately
4. Check the generated output step-by-step

## Advanced Topics

### Conditional Module Inclusion
```python
from conditional_module_includer import ModuleInclusionOptimizer

optimizer = ModuleInclusionOptimizer()

# Context for different platforms/requirements
high_end_context = {
    'platform': 'glsl',
    'gpu_capability': 'high',
    'available_texture_units': 32,
    'config': {'use_compute': True}
}

# Optimize module selection for the context
result = optimizer.optimize_module_selection(['raytracing_advanced', 'basic_reflections'], high_end_context)
```

### Version Management
```python
from module_versioning_system import ModuleVersioningSystem

versioning_system = ModuleVersioningSystem()

# Register a new version of a module
success = versioning_system.register_module_version(
    "lighting", 
    "// module content here", 
    "2.1.0", 
    "developer_name", 
    "Added new features"
)
```

### Creating Custom Modules
1. Create a new module in the appropriate genre directory
2. Implement the module in the universal pseudocode format
3. Define the module's interface (inputs, outputs, uniforms)
4. Test the module in isolation
5. Integrate with other modules to form complete shaders

## Contributing

### Adding New Modules
1. Identify common patterns in existing shaders
2. Break down functionality into reusable components
3. Implement in pseudocode format
4. Add to the appropriate genre directory with proper branching for conflicts
5. Test integration with module combination system

### Improving Documentation
- Add examples for complex features
- Update usage instructions for new functionality
- Provide performance benchmarks
- Create tutorials for common use cases

---

## Support and Community

For questions, bug reports, and feature requests:
- Check the `TASKS.md` file for planned features
- Review the `PLAN.md` for project direction
- Look at the existing codebase for examples and patterns
- Report issues through the appropriate channels

This documentation is maintained alongside the codebase. If you find errors or omissions, please update the documentation accordingly.