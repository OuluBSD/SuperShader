# Raymarching Modules

This directory contains standardized raymarching and distance field rendering modules extracted from various shaders in the dataset.

## Raymarching Primitives Library (`raymarching_primitives.glsl`)

Contains basic and advanced signed distance functions (SDFs) and raymarching utilities:

- **Basic Primitives**: Sphere, box, torus, cylinder, cone, plane, capsule
- **Advanced Primitives**: Hexagonal prism, mandelbulb, menger sponge, octahedron
- **SDF Operations**: Union, subtraction, intersection, smooth blending operations
- **Transformations**: Repetition, rotation, mirroring, elongation
- **Fractal Functions**: Mandelbulb distance estimation, menger sponge approximation
- **Noise-based SDFs**: Terrain generation using FBM, ripple effects
- **Normal Calculation**: Gradient-based normal computation
- **Raymarching Algorithms**: Basic and adaptive raymarching with precision controls

## Advanced Raymarching Utilities (`advanced_raymarching.glsl`)

Contains sophisticated raymarching features:

- **Lighting Models**: Ambient occlusion, soft shadows, reflection/refraction
- **PBR Shading**: Physically based rendering with metallic/roughness workflow
- **Camera Systems**: Perspective, orthographic, and custom camera setups
- **Post-Processing**: Depth of field, motion blur, volumetric effects
- **Anti-Aliasing**: Supersampling and adaptive anti-aliasing techniques
- **Performance Optimizations**: Adaptive step sizing, early ray termination
- **Fractal Geometry**: Smooth operations, deformations (twist, bend), infinite patterns
- **Material Systems**: Multi-material scenes with ID tracking

## Usage

Include the desired raymarching library in your shader:

```glsl
#include "modules/raymarching/raymarching_primitives.glsl"
// or  
#include "modules/raymarching/advanced_raymarching.glsl"
```

Then use the functions in your raymarching pipeline:

```glsl
float sceneSDF(vec3 p) {
    float sphere = sdSphere(p - vec3(0.0), 1.0);
    float box = sdBox(p - vec3(2.0, 0.0, 0.0), vec3(0.5));
    return opSmoothUnion(sphere, box, 0.2);  // Smooth blend
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - iResolution.xy * 0.5) / iResolution.y;
    
    vec3 ro = vec3(0.0, 0.0, 3.0);  // Ray origin
    vec3 rd = normalize(vec3(uv, -1.0));  // Ray direction
    
    float dist = raymarch(ro, rd, 20.0, 0.01);  // Raymarch
    
    if (dist < 20.0) {
        vec3 pos = ro + rd * dist;
        vec3 normal = calcNormal(pos, 0.01);
        vec3 color = phongLighting(pos, normal);
        fragColor = vec4(color, 1.0);
    } else {
        fragColor = vec4(0.0);
    }
}
```

## Parameters

The modules include various parameters to control the behavior of distance functions and raymarching algorithms:

- `precision`: Controls the accuracy of raymarching (lower = more accurate but slower)
- `max_iterations`: Maximum steps for raymarching algorithms
- `ao_samples`: Number of samples for ambient occlusion
- `shadow_quality`: Quality setting for soft shadows

## Dependencies

The advanced module builds upon the primitives library for basic operations.

## Performance Notes

For optimal performance:
- Use adaptive step sizing in complex scenes
- Implement early ray termination where possible
- Choose appropriate precision values for your scene
- Consider using simpler normal calculations for animated scenes