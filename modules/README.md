# SuperShader Initial Module Library

This is the initial module library for SuperShader, containing essential modules to get started with shader generation.

## Included Modules

### Math
- **basic_math_functions**: Essential mathematical operations including clamping, interpolation, distance calculations, and vector operations.

### Lighting
- **simple_diffuse_lighting**: Basic diffuse lighting calculations using Lambert's cosine law, with support for material properties.

### Texturing
- **uv_mapping_standard**: Standard UV coordinate transformations including tiling, offset, and wrapping operations.

### Procedural
- **simple_noise_generator**: Basic noise generation functions including pseudo-random generation and value noise.

## Using the Library

These modules can be combined to create simple shaders. For example, to create a basic lit textured surface:

```python
from management.module_combiner import ModuleCombiner

combiner = ModuleCombiner()
shader = combiner.combine_modules([
    'basic_math_functions',
    'simple_diffuse_lighting', 
    'uv_mapping_standard'
])
```

## Extending the Library

To add your own modules:
1. Create a new module file in the appropriate category directory
2. Follow the module format with pseudocode, dependencies, and metadata
3. Update the registry if needed
4. Test the module in isolation before combining with others

## License

These modules are provided under the BSD 3-Clause License (see LICENSE file in the main repository).