#!/usr/bin/env python3
"""
Standardized Lighting Modules Generator
Creates standardized lighting implementations based on the extracted modules
"""

from .combiner import combine_lighting_modules


def create_standard_lighting_implementation():
    """Create standardized lighting implementations based on common patterns"""
    
    print("Creating standardized lighting implementations...")
    
    # 1. Basic Phong Lighting
    print("\n1. Creating Basic Phong Lighting Implementation...")
    phong_modules = ['diffuse_lighting', 'specular_lighting']
    phong_result = combine_lighting_modules(phong_modules)
    with open('standard_lighting_phong.glsl', 'w') as f:
        f.write(phong_result['shader_code'])
    print("   - Created standard_lighting_phong.glsl")
    
    # 2. PBR Lighting
    print("\n2. Creating PBR Lighting Implementation...")
    pbr_modules = ['pbr_lighting', 'normal_mapping']
    pbr_result = combine_lighting_modules(pbr_modules)
    with open('standard_lighting_pbr.glsl', 'w') as f:
        f.write(pbr_result['shader_code'])
    print("   - Created standard_lighting_pbr.glsl")
    
    # 3. Cel Shading
    print("\n3. Creating Cel Shading Implementation...")
    cel_modules = ['diffuse_lighting', 'cel_shading']
    cel_result = combine_lighting_modules(cel_modules)
    with open('standard_lighting_cel.glsl', 'w') as f:
        f.write(cel_result['shader_code'])
    print("   - Created standard_lighting_cel.glsl")
    
    # 4. Advanced Lighting with Shadows
    print("\n4. Creating Advanced Lighting with Shadows...")
    advanced_modules = ['pbr_lighting', 'normal_mapping', 'shadow_mapping', 'basic_point_light']
    advanced_result = combine_lighting_modules(advanced_modules)
    with open('standard_lighting_advanced.glsl', 'w') as f:
        f.write(advanced_result['shader_code'])
    print("   - Created standard_lighting_advanced.glsl")
    
    # 5. Ray Marching Lighting
    print("\n5. Creating Ray Marching Lighting Implementation...")
    raymarch_modules = ['raymarching_lighting', 'diffuse_lighting']
    raymarch_result = combine_lighting_modules(raymarch_modules)
    with open('standard_lighting_raymarch.glsl', 'w') as f:
        f.write(raymarch_result['shader_code'])
    print("   - Created standard_lighting_raymarch.glsl")
    
    print("\nAll standardized lighting implementations created successfully!")


def analyze_module_compatibility():
    """Analyze which modules work well together"""
    print("\nAnalyzing module compatibility...")
    
    # Known compatible combinations
    compatible_combinations = [
        {
            'name': 'Basic Lighting',
            'modules': ['diffuse_lighting', 'specular_lighting'],
            'description': 'Classic Phong lighting model'
        },
        {
            'name': 'PBR Workflow',
            'modules': ['pbr_lighting', 'normal_mapping', 'diffuse_lighting'],
            'description': 'Physically Based Rendering with normal mapping'
        },
        {
            'name': 'Artistic Shading',
            'modules': ['diffuse_lighting', 'cel_shading'],
            'description': 'Cel/toon shading for artistic effects'
        },
        {
            'name': 'Advanced Lighting',
            'modules': ['pbr_lighting', 'shadow_mapping', 'normal_mapping'],
            'description': 'PBR with shadows and normal mapping'
        },
        {
            'name': 'Raymarching Scene',
            'modules': ['raymarching_lighting', 'diffuse_lighting', 'specular_lighting'],
            'description': 'Ray marching with lighting calculations'
        }
    ]
    
    print("\nCompatible Module Combinations:")
    for combo in compatible_combinations:
        print(f"  - {combo['name']}: {', '.join(combo['modules'])}")
        print(f"    Description: {combo['description']}")
    
    return compatible_combinations


if __name__ == "__main__":
    create_standard_lighting_implementation()
    analyze_module_compatibility()
    
    print("\nLighting modules extraction and standardization completed!")
    print("- Created 10 lighting modules in modules/lighting/")
    print("- Created registry and combiner tools")
    print("- Generated 5 standardized lighting implementations")
    print("- Analyzed module compatibility patterns")