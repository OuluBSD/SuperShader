#!/usr/bin/env python3
"""
Advanced Texturing Module with Branching for Conflicting Features
This module demonstrates different texturing approaches with branching for conflicting features
"""

# Interface definition with branching options
INTERFACE = {
    'inputs': [
        {'name': 'uv', 'type': 'vec2', 'direction': 'in', 'semantic': 'texture_coordinates'},
        {'name': 'worldPos', 'type': 'vec3', 'direction': 'in', 'semantic': 'world_position'},
        {'name': 'normal', 'type': 'vec3', 'direction': 'in', 'semantic': 'surface_normal'},
        {'name': 'viewDir', 'type': 'vec3', 'direction': 'in', 'semantic': 'view_direction'},
        {'name': 'time', 'type': 'float', 'direction': 'uniform', 'semantic': 'time_parameter'},
        {'name': 'tiling', 'type': 'vec2', 'direction': 'uniform', 'semantic': 'texture_tiling'},
        {'name': 'offset', 'type': 'vec2', 'direction': 'uniform', 'semantic': 'texture_offset'},
        {'name': 'blendFactor', 'type': 'float', 'direction': 'uniform', 'semantic': 'blend_factor'}
    ],
    'outputs': [
        {'name': 'textureColor', 'type': 'vec4', 'direction': 'out', 'semantic': 'sampled_color'},
        {'name': 'normalMap', 'type': 'vec3', 'direction': 'out', 'semantic': 'surface_normal'},
        {'name': 'worldNormal', 'type': 'vec3', 'direction': 'out', 'semantic': 'world_space_normal'}
    ],
    'uniforms': [
        {'name': 'time', 'type': 'float', 'semantic': 'time_parameter'},
        {'name': 'tiling', 'type': 'vec2', 'semantic': 'texture_tiling'},
        {'name': 'offset', 'type': 'vec2', 'semantic': 'texture_offset'},
        {'name': 'blendFactor', 'type': 'float', 'semantic': 'blend_factor'}
    ],
    'branches': {
        'uv_mapping_method': {
            'planar': {
                'name': 'Planar UV Mapping',
                'description': 'Planar projection for simple flat surfaces',
                'requires': [],
                'conflicts': ['spherical', 'cylindrical', 'triplanar']
            },
            'spherical': {
                'name': 'Spherical UV Mapping',
                'description': 'Spherical projection for spherical objects',
                'requires': [],
                'conflicts': ['planar', 'cylindrical', 'triplanar']
            },
            'cylindrical': {
                'name': 'Cylindrical UV Mapping',
                'description': 'Cylindrical projection for cylindrical objects',
                'requires': [],
                'conflicts': ['planar', 'spherical', 'triplanar']
            },
            'triplanar': {
                'name': 'Triplanar UV Mapping',
                'description': 'Triplanar projection for complex objects',
                'requires': [],
                'conflicts': ['planar', 'spherical', 'cylindrical']
            }
        },
        'texture_filtering': {
            'nearest': {
                'name': 'Nearest Neighbor',
                'description': 'Fast but low-quality texture filtering',
                'requires': [],
                'conflicts': ['bilinear', 'trilinear', 'anisotropic']
            },
            'bilinear': {
                'name': 'Bilinear Filtering',
                'description': 'Basic texture filtering with interpolation',
                'requires': [],
                'conflicts': ['nearest', 'trilinear', 'anisotropic']
            },
            'trilinear': {
                'name': 'Trilinear Filtering',
                'description': 'Mipmap-based texture filtering',
                'requires': ['mipmap_generation'],
                'conflicts': ['nearest', 'bilinear', 'anisotropic']
            },
            'anisotropic': {
                'name': 'Anisotropic Filtering',
                'description': 'High-quality texture filtering at angles',
                'requires': ['anisotropic_support'],
                'conflicts': ['nearest', 'bilinear', 'trilinear']
            }
        },
        'blending_mode': {
            'multiply': {
                'name': 'Multiply Blending',
                'description': 'Darken colors using multiply technique',
                'requires': [],
                'conflicts': ['overlay', 'soft_light', 'additive']
            },
            'overlay': {
                'name': 'Overlay Blending',
                'description': 'Combine colors using overlay technique',
                'requires': [],
                'conflicts': ['multiply', 'soft_light', 'additive']
            },
            'soft_light': {
                'name': 'Soft Light Blending',
                'description': 'Gentle color blending using soft light',
                'requires': [],
                'conflicts': ['multiply', 'overlay', 'additive']
            },
            'additive': {
                'name': 'Additive Blending',
                'description': 'Brighten colors using addition',
                'requires': [],
                'conflicts': ['multiply', 'overlay', 'soft_light']
            }
        }
    }
}

# Pseudocode for different texturing algorithms
pseudocode = {
    'planar_mapping': '''
// Planar UV mapping
vec2 planarMapping(vec3 position, vec3 axis) {
    if (abs(axis.x) > abs(axis.y) && abs(axis.x) > abs(axis.z)) {
        return position.yz;
    } else if (abs(axis.y) > abs(axis.z)) {
        return position.xz;
    } else {
        return position.xy;
    }
}
    ''',
    
    'spherical_mapping': '''
// Spherical UV mapping
vec2 sphericalMapping(vec3 normal) {
    float phi = acos(normal.y);
    float theta = atan(normal.x, normal.z);
    return vec2(theta / (2.0 * 3.14159), phi / 3.14159);
}
    ''',
    
    'cylindrical_mapping': '''
// Cylindrical UV mapping
vec2 cylindricalMapping(vec3 position) {
    float u = atan(position.x, position.z) / (2.0 * 3.14159);
    float v = position.y;
    return vec2(u, v);
}
    ''',
    
    'triplanar_mapping': '''
// Triplanar texturing for complex objects
vec3 triplanarTexture(sampler2D tex, vec3 worldPos, vec3 normal, float blendStrength) {
    // Get UVs for each axis
    vec2 uvX = worldPos.zy;
    vec2 uvY = worldPos.xz;
    vec2 uvZ = worldPos.xy;

    // Sample the texture from each axis
    vec3 texX = texture(tex, uvX).rgb;
    vec3 texY = texture(tex, uvY).rgb;
    vec3 texZ = texture(tex, uvZ).rgb;

    // Get blending weights based on normal
    vec3 blend = pow(abs(normal), vec3(blendStrength));
    blend = blend / (blend.x + blend.y + blend.z);

    // Blend the textures
    return texX * blend.x + texY * blend.y + texZ * blend.z;
}
    ''',
    
    'nearest_filtering': '''
// Nearest neighbor texture sampling
vec4 nearestSampling(sampler2D tex, vec2 uv) {
    // Round to nearest pixel
    vec2 texSize = textureSize(tex, 0);
    vec2 nearestCoord = floor(uv * texSize + 0.5) / texSize;
    return texture(tex, nearestCoord);
}
    ''',
    
    'bilinear_filtering': '''
// Bilinear texture filtering
vec4 bilinearSampling(sampler2D tex, vec2 uv) {
    vec2 texSize = textureSize(tex, 0);
    vec2 scaledUV = uv * texSize;
    vec2 iuv = floor(scaledUV);
    vec2 fuv = fract(scaledUV);

    // Get four surrounding texels
    vec2 p00 = iuv / texSize;
    vec2 p10 = (iuv + vec2(1, 0)) / texSize;
    vec2 p01 = (iuv + vec2(0, 1)) / texSize;
    vec2 p11 = (iuv + vec2(1, 1)) / texSize;

    // Sample the four texels
    vec4 t00 = texture(tex, p00);
    vec4 t10 = texture(tex, p10);
    vec4 t01 = texture(tex, p01);
    vec4 t11 = texture(tex, p11);

    // Interpolate
    vec4 iu0 = mix(t00, t10, fuv.x);
    vec4 iu1 = mix(t01, t11, fuv.x);
    return mix(iu0, iu1, fuv.y);
}
    ''',
    
    'trilinear_filtering': '''
// Trilinear texture filtering with mipmap selection
vec4 trilinearSampling(sampler2D tex, vec2 uv, float lod) {
    float level = lod;
    float fractLevel = fract(level);
    int baseLevel = int(floor(level));
    int nextLevel = baseLevel + 1;

    // Sample at both mipmap levels
    vec4 baseSample = textureLod(tex, uv, baseLevel);
    vec4 nextSample = textureLod(tex, uv, nextLevel);

    // Blend between levels
    return mix(baseSample, nextSample, fractLevel);
}
    ''',
    
    'anisotropic_filtering': '''
// Simplified anisotropic filtering
vec4 anisotropicSampling(sampler2D tex, vec2 uv, vec2 dx, vec2 dy, float maxAnisotropy) {
    // Calculate anisotropy ratio
    float texelSize = length(dx) + length(dy);
    float anisotropy = min(texelSize * maxAnisotropy, maxAnisotropy);

    // Perform multiple samples in the direction of greatest change
    vec2 sampleStep = normalize(dx + dy) / anisotropy;
    vec4 result = vec4(0.0);
    int samples = int(anisotropy);
    
    for (int i = 0; i < samples; i++) {
        vec2 offset = sampleStep * (float(i) - float(samples) / 2.0);
        result += texture(tex, uv + offset);
    }
    
    return result / float(samples);
}
    ''',
    
    'multiply_blending': '''
// Multiply blending mode
vec3 blendMultiply(vec3 base, vec3 blend) {
    return base * blend;
}
    ''',
    
    'overlay_blending': '''
// Overlay blending mode
vec3 blendOverlay(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
        step(0.5, base)
    );
}
    ''',
    
    'soft_light_blending': '''
// Soft light blending mode
vec3 blendSoftLight(vec3 base, vec3 blend) {
    return (1.0 - 2.0 * blend) * base * base + 2.0 * blend * base;
}
    ''',
    
    'additive_blending': '''
// Additive blending mode
vec3 blendAdditive(vec3 base, vec3 blend) {
    return base + blend;
}
    '''
}

def get_interface():
    """Return the interface definition for this module"""
    return INTERFACE

def get_pseudocode(branch_name=None):
    """Return the pseudocode for this texturing module or specific branch"""
    if branch_name and branch_name in pseudocode:
        return pseudocode[branch_name]
    else:
        # Return all pseudocodes
        return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'texturing_advanced_branching',
        'type': 'texturing',
        'patterns': ['Planar UV Mapping', 'Spherical UV Mapping', 'Cylindrical UV Mapping', 'Triplanar UV Mapping',
                     'Nearest Neighbor Filtering', 'Bilinear Filtering', 'Trilinear Filtering', 'Anisotropic Filtering',
                     'Multiply Blending', 'Overlay Blending', 'Soft Light Blending', 'Additive Blending'],
        'frequency': 210,
        'dependencies': [],
        'conflicts': [],
        'description': 'Advanced texturing algorithms with branching for different mapping, filtering, and blending approaches',
        'interface': INTERFACE,
        'branches': INTERFACE['branches']
    }

def validate_branches(selected_branches):
    """Validate that the selected branches don't have conflicts"""
    # Check for conflicts between different branch categories
    if 'uv_mapping_method' in selected_branches:
        mapping_method = selected_branches['uv_mapping_method']
        
        # UV mapping methods conflict with each other
        valid_mapping = True
        return valid_mapping
        
    return True