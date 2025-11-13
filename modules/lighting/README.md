# Standardized Lighting Modules

This directory contains standardized lighting modules that follow consistent interfaces for use in the SuperShader system.

## Module Structure

Each lighting module follows the same basic structure:
- Standard inputs and outputs
- Consistent naming conventions
- Compatible with the module combination engine

## Available Lighting Modules

### 1. Diffuse Lighting (`diffuse_lighting.glsl`)
- Lambert diffuse
- Oren-Nayar diffuse
- Wrapped diffuse

### 2. Specular Lighting (`specular_lighting.glsl`)
- Phong specular
- Blinn-Phong specular  
- Cook-Torrance specular
- Complete BRDF components

### 3. Physically Based Rendering (`pbr_lighting.glsl`)
- Complete PBR lighting calculation
- Cook-Torrance BRDF implementation
- Material property support

### 4. Point Light (`basic_point_light.glsl`)
- Point light calculations with attenuation
- Spot light implementations
- Radius-based lighting

### 5. Directional Light (`directional_light.glsl`)
- Directional light calculations
- Multiple light source support
- Shadow-aware lighting

### 6. Shadow Mapping (`shadow_mapping.glsl`)
- Shadow calculation functions
- Percentage-closer filtering
- Soft shadow implementations

### 7. Normal Mapping (`normal_mapping.glsl`)
- Tangent space normal mapping
- World space normal mapping
- TBN matrix calculations

### 8. Lighting Functions (`lighting_functions.glsl`)
- Unified lighting functions
- Multiple lighting models
- Shadow-aware calculations

### 9. Cel Shading (`cel_shading.glsl`)
- Toon/cel shading implementations
- Edge detection
- Stylized lighting

### 10. Standardized Interface (`lighting_interface.glsl`)
- Standard interface definitions
- Consistent data structures
- Module compatibility framework

## Usage

These modules can be combined using the module combination engine to create complex lighting effects. Each module is designed to work independently or in combination with others.

## Data Structures

The modules use standardized data structures:

```glsl
struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

struct Material {
    vec3 albedo;
    float metallic;
    float roughness;
    float ao;
};

struct SurfaceData {
    vec3 position;
    vec3 normal;
    vec3 viewDir;
    vec2 texCoords;
};
```

These structures ensure consistency across all lighting modules.