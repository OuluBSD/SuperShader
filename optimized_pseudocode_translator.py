#!/usr/bin/env python3
"""
Optimized Pseudocode to GLSL Translator
Improved version with caching, more efficient regex operations, and better performance
"""

import re
from functools import lru_cache
import threading
from typing import Dict, Callable, Any


class OptimizedPseudocodeTranslator:
    def __init__(self):
        # Create optimized regex patterns (compiling them once for reuse)
        self._compile_patterns()
        
        # Define mappings for different target languages
        self.translation_methods = {
            'glsl': {
                'data_types': self._glsl_data_types,
                'vector_types': self._glsl_vector_types,
                'math_functions': self._glsl_math_functions,
                'syntax': self._glsl_syntax
            },
            'hlsl': {
                'data_types': self._hlsl_data_types,
                'vector_types': self._hlsl_vector_types,
                'math_functions': self._hlsl_math_functions,
                'syntax': self._hlsl_syntax
            },
            'metal': {
                'data_types': self._metal_data_types,
                'vector_types': self._metal_vector_types,
                'math_functions': self._metal_math_functions,
                'syntax': self._metal_syntax
            },
            'c_cpp': {
                'data_types': self._c_cpp_data_types,
                'vector_types': self._c_cpp_vector_types,
                'math_functions': self._c_cpp_math_functions,
                'syntax': self._c_cpp_syntax
            },
            'wgsl': {
                'data_types': self._wgsl_data_types,
                'vector_types': self._wgsl_vector_types,
                'math_functions': self._wgsl_math_functions,
                'syntax': self._wgsl_syntax
            }
        }

        # Cache for translations
        self._translation_cache = {}
        self._cache_lock = threading.RLock()
        
        # Pre-computed values for performance
        self._glsl_type_mappings = {
            'vec3': 'vec3',
            'vec2': 'vec2', 
            'vec4': 'vec4',
            'float': 'float',
            'int': 'int',
            'bool': 'bool',
            'sampler2D': 'sampler2D',
            'samplerCube': 'samplerCube',
            'mat3': 'mat3',
            'mat4': 'mat4'
        }

        self._metal_type_mappings = {
            'vec3': 'float3',
            'vec2': 'float2',
            'vec4': 'float4',
            'mat3': 'float3x3',
            'mat4': 'float4x4',
            'float': 'float',
            'int': 'int',
            'bool': 'bool',
            'sampler2D': 'texture2d<float>',
            'samplerCube': 'texturecube<float>'
        }

        self._wgsl_type_mappings = {
            'vec3': 'vec3<f32>',
            'vec2': 'vec2<f32>',
            'vec4': 'vec4<f32>',
            'mat3': 'mat3x3<f32>',
            'mat4': 'mat4x4<f32>',
            'float': 'f32',
            'int': 'i32',
            'bool': 'bool',
            'sampler2D': 'texture_2d<f32>',
            'samplerCube': 'texture_cube<f32>'
        }

    def _compile_patterns(self):
        """Compile regex patterns once for reuse."""
        self._vec_pattern = re.compile(r'\\b(vec[2-4])\\(')
        self._type_pattern = re.compile(r'\\b(vec[2-4]|float|int|bool|sampler2D|samplerCube|mat[3-4])\\b')
        self._glsl_vec_pattern = re.compile(r'\\bvec([2-4])\\(')
        self._metal_vec_pattern = re.compile(r'\\bvec([2-4])\\(')
        self._wgsl_vec_pattern = re.compile(r'\\bvec([2-4])\\(')

    @lru_cache(maxsize=1000)
    def _cached_glsl_data_types(self, pseudocode: str) -> str:
        """Cached version of GLSL data type conversion."""
        result = pseudocode
        for old, new in self._glsl_type_mappings.items():
            result = result.replace(old, new)
        return result

    def _glsl_data_types(self, pseudocode: str) -> str:
        """Convert data types to GLSL with caching."""
        return self._cached_glsl_data_types(pseudocode)

    def _glsl_vector_types(self, pseudocode: str) -> str:
        """Handle GLSL vector operations."""
        # In GLSL, we don't need to change vector types, they're already correct
        return pseudocode

    def _glsl_math_functions(self, pseudocode: str) -> str:
        """Convert math functions to GLSL equivalents."""
        # GLSL has the same basic math functions as the pseudocode
        return pseudocode

    def _glsl_syntax(self, pseudocode: str) -> str:
        """Apply GLSL specific syntax rules."""
        lines = pseudocode.split('\\n')
        processed_lines = []

        for line in lines:
            processed_lines.append(line)

        return '\\n'.join(processed_lines)

    def _hlsl_data_types(self, pseudocode: str) -> str:
        """Convert data types to HLSL."""
        result = pseudocode
        # For now, HLSL is similar to GLSL - in reality, you'd have different mappings
        return result

    def _hlsl_vector_types(self, pseudocode: str) -> str:
        return pseudocode

    def _hlsl_math_functions(self, pseudocode: str) -> str:
        return pseudocode

    def _hlsl_syntax(self, pseudocode: str) -> str:
        return pseudocode

    @lru_cache(maxsize=1000)
    def _cached_metal_data_types(self, pseudocode: str) -> str:
        """Cached version of Metal data type conversion."""
        result = pseudocode
        for old, new in self._metal_type_mappings.items():
            result = result.replace(old, new)
        return result

    def _metal_data_types(self, pseudocode: str) -> str:
        """Convert data types to Metal with caching."""
        return self._cached_metal_data_types(pseudocode)

    def _metal_vector_types(self, pseudocode: str) -> str:
        """Handle Metal vector operations."""
        # Convert vector constructors without nested patterns
        result = pseudocode.replace('vec3(', 'float3(')
        result = result.replace('vec2(', 'float2(')
        result = result.replace('vec4(', 'float4(')
        return result

    def _metal_math_functions(self, pseudocode: str) -> str:
        """Convert math functions to Metal equivalents."""
        return pseudocode

    def _metal_syntax(self, pseudocode: str) -> str:
        """Apply Metal specific syntax rules."""
        return pseudocode

    @lru_cache(maxsize=1000)
    def _cached_wgsl_data_types(self, pseudocode: str) -> str:
        """Cached version of WGSL data type conversion."""
        result = pseudocode
        for old, new in self._wgsl_type_mappings.items():
            result = result.replace(old, new)
        return result

    def _wgsl_data_types(self, pseudocode: str) -> str:
        """Convert data types to WGSL (WebGPU Shading Language) with caching."""
        return self._cached_wgsl_data_types(pseudocode)

    def _wgsl_vector_types(self, pseudocode: str) -> str:
        """Handle WGSL vector operations."""
        # Convert vector constructors
        result = pseudocode.replace('vec3(', 'vec3<f32>(')
        result = result.replace('vec2(', 'vec2<f32>(')
        result = result.replace('vec4(', 'vec4<f32>(')
        return result

    def _wgsl_math_functions(self, pseudocode: str) -> str:
        """Convert math functions to WGSL equivalents."""
        return pseudocode

    def _wgsl_syntax(self, pseudocode: str) -> str:
        """Apply WGSL specific syntax rules."""
        return pseudocode

    def _c_cpp_data_types(self, pseudocode: str) -> str:
        return pseudocode

    def _c_cpp_vector_types(self, pseudocode: str) -> str:
        return pseudocode

    def _c_cpp_math_functions(self, pseudocode: str) -> str:
        return pseudocode

    def _c_cpp_syntax(self, pseudocode: str) -> str:
        return pseudocode

    def translate_to_glsl(self, pseudocode: str) -> str:
        """Translate pseudocode to GLSL with optimized operations."""
        # Use a single string for building the result to avoid multiple string concatenations
        result = pseudocode
        result = self._glsl_data_types(result)
        result = self._glsl_vector_types(result)
        result = self._glsl_math_functions(result)
        result = self._glsl_syntax(result)

        return result

    def translate(self, pseudocode: str, target_language: str = 'glsl') -> str:
        """Translate pseudocode to target language with caching."""
        cache_key = (pseudocode, target_language)
        
        with self._cache_lock:
            if cache_key in self._translation_cache:
                return self._translation_cache[cache_key]
        
        if target_language in self.translation_methods:
            translation = self.translation_methods[target_language]

            result = pseudocode
            result = translation['data_types'](result)
            result = translation['vector_types'](result)
            result = translation['math_functions'](result)
            result = translation['syntax'](result)
            
            with self._cache_lock:
                self._translation_cache[cache_key] = result

            return result
        else:
            raise ValueError(f"Unsupported target language: {target_language}")

    def create_glsl_shader_from_modules(self, module_names):
        """Create a complete GLSL shader from module pseudocodes."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
        from modules.lighting.registry import get_module_by_name

        shader_parts = {
            'header': '#version 330 core\\n\\n',
            'uniforms': '// Uniforms\\nuniform vec3 viewPos;\\nuniform vec3 lightPos;\\nuniform vec3 lightColor;\\nuniform sampler2D normalMap;\\nuniform sampler2D shadowMap;\\n\\n',
            'inputs': '// Input variables\\nin vec3 FragPos;\\nin vec3 Normal;\\nin vec2 TexCoords;\\n\\n',
            'outputs': '// Output\\nout vec4 FragColor;\\n\\n',
            'functions': [],
            'main': ''
        }

        # Collect all functions from modules
        for module_name in module_names:
            module = get_module_by_name(module_name)
            if module and 'pseudocode' in module:
                # Translate the pseudocode to GLSL
                glsl_code = self.translate_to_glsl(module['pseudocode'])
                # Add to shader
                shader_parts['functions'].append(glsl_code)

        # Create the main function
        main_content = self._create_main_from_modules(module_names)
        shader_parts['main'] = main_content

        # Combine all parts efficiently using join
        parts_list = [
            shader_parts['header'],
            shader_parts['uniforms'],
            shader_parts['inputs'], 
            shader_parts['outputs']
        ]

        # Add all the functions
        for func in shader_parts['functions']:
            if func.strip():
                parts_list.append(func)
                parts_list.append('\\n')  # Add newline between functions

        parts_list.append(shader_parts['main'])

        shader_code = ''.join(parts_list)

        return shader_code

    def _create_main_from_modules(self, module_names):
        """Create main function based on selected modules."""
        main_func = \"\"\"
void main() {
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    // Initialize color
    vec3 result = vec3(0.0);
\"\"\"

        # Add logic based on modules
        if 'basic_point_light' in module_names:
            main_func += \"\"\"
    // Basic point light calculation
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 pointLight = diff * lightColor;

    // Apply distance attenuation
    float distance = length(lightPos - FragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    pointLight *= attenuation;

    result += pointLight;
\"\"\"

        if 'diffuse_lighting' in module_names and 'basic_point_light' not in module_names:
            main_func += \"\"\"
    // Diffuse lighting if point light module is not used
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    result += diffuse;
\"\"\"

        if 'specular_lighting' in module_names:
            main_func += \"\"\"
    // Specular lighting
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    result += specular;
\"\"\"

        if 'normal_mapping' in module_names:
            main_func += \"\"\"
    // Normal mapping if available
    vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;
    tangentNormal = normalize(tangentNormal);
    // Use tangentNormal instead of norm for lighting calculations
    float diff = max(dot(tangentNormal, lightDir), 0.0);
    vec3 normalMappedDiffuse = diff * lightColor;
    result = mix(result, normalMappedDiffuse, 0.5); // Blend with original
\"\"\"

        if 'pbr_lighting' in module_names:
            main_func += \"\"\"
    // PBR lighting model (overrides other lighting if present)
    // Simplified PBR calculation
    vec3 albedo = vec3(0.5);
    float metallic = 0.0;
    float roughness = 0.5;

    // Direct calculation without full Cook-Torrance for simplicity
    float NdotL = max(dot(norm, lightDir), 0.0);
    result = albedo * lightColor * NdotL;
\"\"\"

        if 'cel_shading' in module_names:
            main_func += \"\"\"
    // Apply cel shading effect
    float NdotL = dot(norm, lightDir);
    float intensity = smoothstep(0.0, 0.01, NdotL);
    intensity += step(0.5, NdotL);
    intensity += step(0.8, NdotL);
    intensity = min(intensity, 1.0);

    result = result * vec3(intensity);
\"\"\"

        if 'shadow_mapping' in module_names:
            main_func += \"\"\"
    // Simple shadow calculation if needed
    vec3 projCoords = /* light space transformation */;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float shadow = currentDepth > closestDepth + 0.0005 ? 1.0 : 0.0;

    result *= (1.0 - shadow * 0.5); // Apply shadow with 50% intensity
\"\"\"

        main_func += \"\"\"
    // Final color
    FragColor = vec4(result, 1.0);
}
\"\"\"
        return main_func

    def create_shader_from_modules(self, module_names, target_language='glsl'):
        \"\"\"Create a complete shader from module pseudocodes in the specified language.\"\"\"
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
        from modules.lighting.registry import get_module_by_name

        if target_language == 'glsl':
            header = '#version 330 core\\n\\n'
            uniforms = '// Uniforms\\nuniform vec3 viewPos;\\nuniform vec3 lightPos;\\nuniform vec3 lightColor;\\nuniform sampler2D normalMap;\\nuniform sampler2D shadowMap;\\n\\n'
            inputs = '// Input variables\\nin vec3 FragPos;\\nin vec3 Normal;\\nin vec2 TexCoords;\\n\\n'
            outputs = '// Output\\nout vec4 FragColor;\\n\\n'
        elif target_language == 'metal':
            header = '#include <metal_stdlib>\\nusing namespace metal;\\n\\n'
            uniforms = '// Uniforms\\nconstant float3& viewPos [[buffer(0)]];\\nconstant float3& lightPos [[buffer(1)]];\\nconstant float3& lightColor [[buffer(2)]];\\ntexture2d<float> normalMap [[texture(0)]];\\ntexture2d<float> shadowMap [[texture(1)]];\\n\\n'
            inputs = '// Input variables\\nstruct VertexOutput {\\n    float3 fragPos [[user(locn0)]];\\n    float3 normal [[user(locn1)]];\\n    float2 texCoords [[user(locn2)]];\\n};\\n\\n'
            outputs = '// Output\\nstruct FragmentOutput {\\n    float4 color [[color(0)]];\\n};\\n\\n'
        else:  # Default to GLSL
            header = '#version 330 core\\n\\n'
            uniforms = '// Uniforms\\nuniform vec3 viewPos;\\nuniform vec3 lightPos;\\nuniform vec3 lightColor;\\nuniform sampler2D normalMap;\\nuniform sampler2D shadowMap;\\n\\n'
            inputs = '// Input variables\\nin vec3 FragPos;\\nin vec3 Normal;\\nin vec2 TexCoords;\\n\\n'
            outputs = '// Output\\nout vec4 FragColor;\\n\\n'

        # Build shader parts list
        shader_parts = [header, uniforms, inputs, outputs]

        # Collect all functions from modules and translate them
        for module_name in module_names:
            module = get_module_by_name(module_name)
            if module and 'pseudocode' in module:
                # Translate the pseudocode to the target language
                translated_code = self.translate(module['pseudocode'], target_language)
                shader_parts.append(translated_code)
                shader_parts.append('\\n')  # Add newline between functions

        # Create and add the main function
        main_content = self._create_main_from_modules(module_names)
        shader_parts.append(main_content)

        # Combine all parts efficiently
        shader_code = ''.join(shader_parts)

        return shader_code


def test_optimized_translator():
    \"\"\"Test the optimized pseudocode translator\"\"\"
    print(\"Testing Optimized Pseudocode Translator...\")

    # Sample pseudocode from one of our modules
    sample_pseudocode = \"\"\"
// Basic Point Light Implementation
vec3 calculatePointLight(vec3 position, vec3 normal, vec3 lightPos, vec3 lightColor) {
    // Calculate light direction
    vec3 lightDir = normalize(lightPos - position);

    // Calculate distance and attenuation
    float distance = length(lightPos - position);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

    // Diffuse lighting
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Apply attenuation
    diffuse *= attenuation;

    return diffuse;
}
\"\"\"

    translator = OptimizedPseudocodeTranslator()

    # Translate to GLSL (should be mostly the same)
    glsl_code = translator.translate_to_glsl(sample_pseudocode)
    print(\"GLSL Translation:\")
    print(glsl_code)

    # Translate to Metal
    metal_code = translator.translate(sample_pseudocode, 'metal')
    print(\"\\nMetal Translation:\")
    print(metal_code)

    # Translate to WGSL
    wgsl_code = translator.translate(sample_pseudocode, 'wgsl')
    print(\"\\nWGSL Translation:\")
    print(wgsl_code)

    print(\"\\nAll translations completed successfully with optimizations!\")


if __name__ == \"__main__\":
    test_optimized_translator()