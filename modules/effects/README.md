# Effects Modules

This directory contains standardized effect and post-processing modules extracted from various shaders in the dataset.

## Standard Effects Library (`standard_effects.glsl`)

Contains basic post-processing functions:

- **Color Adjustments**: Brightness/contrast, saturation, hue shifting, vibrance
- **Filters**: Sepia, grayscale, invert, threshold
- **Bloom**: Bright area extraction for bloom effects
- **Blurs**: Box blur, Gaussian blur approximations
- **Edge Detection**: Sobel edge detection
- **Noise**: Value noise and simplex noise implementations
- **Distortions**: Barrel distortion, chromatic aberration
- **Compositing**: Blend modes (multiply, screen, overlay, soft light, hard light)

## Advanced Post-Processing Effects (`advanced_post_effects.glsl`)

Contains more sophisticated post-processing functions:

- **Tone Mapping**: ACES, Reinhard, Uncharted 2 tone mapping operators
- **Bloom**: Separable blur, bright extraction, composition
- **Depth of Field**: Bokeh simulation based on depth buffer
- **Anti-Aliasing**: FXAA implementation
- **Special Effects**: Heat shimmer, film grain, scanlines, vignette
- **Color Grading**: Advanced color grading with luminance preservation

## Usage

Include the desired effects library in your shader:

```glsl
#include "modules/effects/standard_effects.glsl"
// or
#include "modules/effects/advanced_post_effects.glsl"
```

Then use the functions in your shader pipeline:

```glsl
vec3 color = texture(iChannel0, uv).rgb;
color = brightnessContrast(color, 0.2, 0.1);
color = saturation(color, -0.2);
color = sepia(color);
```

## Parameters

Each effect function typically accepts parameters that control the strength or behavior of the effect.

## Dependencies

Advanced effects depend on standard effects for basic operations.