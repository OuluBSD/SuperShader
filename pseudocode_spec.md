# SuperShader Pseudocode Language Specification

## Overview

The SuperShader pseudocode language is designed to represent shader algorithms in a universal, translatable format. It should be clear, maintainable, and easily convertible to multiple target programming languages and graphics APIs.

## Syntax Guidelines

### Basic Structure
```
// Comments use C-style syntax
/* Multi-line comments are supported */

// Functions are defined with return type, name, and parameters
<return_type> <function_name>(<parameter_list>) {
    <function_body>
}

// Variables are declared with their type
<type> <variable_name>;
<type> <variable_name> = <initial_value>;
```

### Supported Types

#### Basic Types
- `int` - Integer values
- `float` - Floating point values
- `bool` - Boolean values (true/false)

#### Vector Types
- `vec2` - 2-component vector (x, y)
- `vec3` - 3-component vector (x, y, z)
- `vec4` - 4-component vector (x, y, z, w)

#### Matrix Types
- `mat2` - 2x2 matrix
- `mat3` - 3x3 matrix
- `mat4` - 4x4 matrix

#### Sampler Types
- `sampler2D` - 2D texture sampler
- `samplerCube` - Cube map sampler

## Core Operations

### Arithmetic Operations
```
a + b    // Addition
a - b    // Subtraction
a * b    // Multiplication
a / b    // Division
a % b    // Modulo
```

### Vector Operations
```
// Component-wise operations
vec3 a, b;
vec3 result = a + b;        // Component-wise addition
vec3 result = a - b;        // Component-wise subtraction
vec3 result = a * b;        // Component-wise multiplication
vec3 result = a / b;        // Component-wise division

// Scalar operations
vec3 a;
vec3 result = a * 2.0;      // Multiply all components by scalar
vec3 result = 2.0 * a;      // Multiply all components by scalar

// Built-in vector functions
float length = length(v);       // Vector magnitude
vec3 norm = normalize(v);       // Normalize vector
float dist = distance(v1, v2);  // Distance between vectors
float dot_product = dot(v1, v2); // Dot product
vec3 cross_product = cross(v1, v2); // Cross product (vec3 only)
```

### Matrix Operations
```
// Matrix multiplication
mat4 m1, m2;
mat4 result = m1 * m2;

// Matrix-vector multiplication
mat4 m;
vec4 v;
vec4 result = m * v;
```

### Control Flow
```
// If statements
if (condition) {
    // code block
} else if (other_condition) {
    // code block
} else {
    // code block
}

// For loops
for (int i = 0; i < count; i++) {
    // loop body
}

// While loops
while (condition) {
    // loop body
}

// Ternary operator
result = condition ? value_if_true : value_if_false;
```

## Graphics API Mapping

### OpenGL/GLSL Mapping
- `vec2`, `vec3`, `vec4` → `vec2`, `vec3`, `vec4`
- `mat2`, `mat3`, `mat4` → `mat2`, `mat3`, `mat4`
- `sampler2D` → `sampler2D`
- `length(v)` → `length(v)`
- `normalize(v)` → `normalize(v)`
- `distance(a, b)` → `distance(a, b)`
- `dot(a, b)` → `dot(a, b)`
- `cross(a, b)` → `cross(a, b)`

### DirectX/HLSL Mapping
- `vec2`, `vec3`, `vec4` → `float2`, `float3`, `float4`
- `mat2`, `mat3`, `mat4` → `float2x2`, `float3x3`, `float4x4`
- `sampler2D` → `Texture2D` with `SamplerState`
- `length(v)` → `length(v)`
- `normalize(v)` → `normalize(v)`
- `distance(a, b)` → `distance(a, b)`
- `dot(a, b)` → `dot(a, b)`
- `cross(a, b)` → `cross(a, b)`

### Vulkan/Metal Mapping
- Similar to OpenGL for mathematical operations
- Texture sampling handled through descriptor sets (Vulkan) or buffers (Metal)

## Built-in Functions

### Mathematical Functions
```
// Basic math
float abs(float x)                // Absolute value
float sign(float x)               // Sign of value (-1, 0, or 1)
float floor(float x)              // Floor function
float ceil(float x)               // Ceiling function
float fract(float x)              // Fractional part
float mod(float x, float y)       // Modulo operation
float min(float a, float b)       // Minimum of two values
float max(float a, float b)       // Maximum of two values
float clamp(float x, float minVal, float maxVal) // Clamp value
float mix(float a, float b, float t) // Linear interpolation
float step(float edge, float x)   // Step function
float smoothstep(float a, float b, float x) // Smooth Hermite interpolation

// Trigonometric functions
float sin(float x)
float cos(float x)
float tan(float x)
float asin(float x)
float acos(float x)
float atan(float x)               // Two-argument version: atan(y, x)

// Exponential functions
float pow(float x, float y)       // Power function
float exp(float x)                // Natural exponential
float log(float x)                // Natural logarithm
float exp2(float x)               // Base-2 exponential
float log2(float x)               // Base-2 logarithm
float sqrt(float x)               // Square root
float inversesqrt(float x)        // Inverse square root

// Vector functions (apply to vec2, vec3, vec4)
float length(vec3 v)              // Vector magnitude
vec3 normalize(vec3 v)            // Normalize vector
float distance(vec3 a, vec3 b)    // Distance between points
float dot(vec3 a, vec3 b)         // Dot product
vec3 cross(vec3 a, vec3 b)        // Cross product (vec3 only)
```

### Texture/Sampler Functions
```
// Texture sampling
vec4 texture(sampler2D sampler, vec2 coord)           // Standard sampling
vec4 texture(sampler2D sampler, vec2 coord, float bias) // With bias
vec4 textureLod(sampler2D sampler, vec2 coord, float lod) // With explicit LOD
```

## Translation Tools Concept

The translation system should work as follows:

1. Parse the pseudocode to create an abstract syntax tree (AST)
2. Apply transformation rules specific to the target language/API
3. Generate the appropriate code for the target platform

The pseudocode should maintain algorithmic intent while abstracting away platform-specific details like:

- Shader stage entry points (different between OpenGL, DirectX, Vulkan)
- Uniform buffer layouts and bindings
- Specific texture sampling syntax
- Built-in variables (gl_Position, gl_FragColor, etc.)

## Example Pseudocode

```
// Function to calculate lighting with Phong model
vec3 phongLighting(vec3 position, vec3 normal, vec3 viewDir, vec3 lightPos, vec3 lightColor) {
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse 
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - position);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    
    return ambient + diffuse + specular;
}

// Main image function (equivalent to fragment shader main)
vec4 mainImage(vec2 fragCoord, sampler2D texture0) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 texColor = texture(texture0, uv);
    
    // Simple lighting effect
    vec3 normal = vec3(0.0, 0.0, 1.0); // Simple normal
    vec3 lightPos = vec3(0.5, 0.5, 1.0);
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    
    vec3 lighting = phongLighting(vec3(uv, 0.0), normal, viewDir, lightPos, vec3(1.0, 1.0, 1.0));
    
    return texColor * vec4(lighting, 1.0);
}
```