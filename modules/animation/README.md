# Animation and Procedural Generation Modules

This directory contains standardized animation and procedural generation modules extracted from various shaders in the dataset.

## Standard Animation Library (`standard_animation.glsl`)

Contains basic animation and procedural functions:

- **Animation Utilities**: Oscillations, pulses, waves, and timing functions
- **Noise Functions**: Perlin noise, FBM (Fractional Brownian Motion), turbulence, and domain warping
- **Particle Systems**: Motion equations, velocity calculations, and lifetime functions
- **Physics Simulations**: Spring systems, pendulum motion, orbital mechanics
- **Procedural Patterns**: Grids, hexagons, checkerboards, stripes, and radial patterns
- **Random Functions**: Hash-based random generators and seeded randomness
- **Utility Functions**: Easing functions, rotation matrices, and animation blending

## Advanced Procedural Library (`advanced_procedural.glsl`)

Contains more sophisticated procedural generation functions:

- **Fractal Generation**: Mandelbulb distance estimation, multifractal, and ridged noise
- **Fluid Simulation**: Advection, vorticity confinement, and particle interaction
- **Particle Physics**: Collisions, attraction, and spring connections
- **Custom Curves**: Bezier curves, cubic Bezier, and Catmull-Rom splines
- **Animation Sequences**: Morphing, easing, and particle spawning
- **Advanced Noise**: Worley noise, flow noise, and turbulent flows
- **Complex Shapes**: Heart paths, flower petals, spirals, and rotating shapes
- **Physics-Based Animations**: Damped oscillation, Verlet integration, spring systems

## Usage

Include the desired animation library in your shader:

```glsl
#include "modules/animation/standard_animation.glsl"
// or
#include "modules/animation/advanced_procedural.glsl"
```

Then use the functions in your shader pipeline:

```glsl
vec2 uv = fragCoord / iResolution.xy;
float noise = perlinNoise(uv * 10.0);
vec2 animatedPos = waveMotion(uv, 0.1, 5.0, 1.0, vec2(1.0, 0.0));
```

## Parameters

Each animation function typically accepts time parameters and geometric parameters that control the behavior of the animation.

## Dependencies

Advanced procedural library depends on standard animation library for basic operations.