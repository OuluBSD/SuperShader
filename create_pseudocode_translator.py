#!/usr/bin/env python3
"""
Pseudocode to GLSL Translator
Translates pseudocode from modules to GLSL and other target languages
"""

import re


class PseudocodeTranslator:
    def __init__(self):
        # Define mappings for different target languages
        self.translations = {
            'glsl': {
                'function_start': self._glsl_function_start,
                'function_end': self._glsl_function_end,
                'data_types': self._glsl_data_types,
                'vector_types': self._glsl_vector_types,
                'math_functions': self._glsl_math_functions,
                'syntax': self._glsl_syntax
            },
            'hlsl': {
                'function_start': self._hlsl_function_start,
                'function_end': self._hlsl_function_end,
                'data_types': self._hlsl_data_types,
                'vector_types': self._hlsl_vector_types,
                'math_functions': self._hlsl_math_functions,
                'syntax': self._hlsl_syntax
            },
            'metal': {
                'function_start': self._metal_function_start,
                'function_end': self._metal_function_end,
                'data_types': self._metal_data_types,
                'vector_types': self._metal_vector_types,
                'math_functions': self._metal_math_functions,
                'syntax': self._metal_syntax
            },
            'c_cpp': {
                'function_start': self._c_cpp_function_start,
                'function_end': self._c_cpp_function_end,
                'data_types': self._c_cpp_data_types,
                'vector_types': self._c_cpp_vector_types,
                'math_functions': self._c_cpp_math_functions,
                'syntax': self._c_cpp_syntax
            },
            'wgsl': {
                'function_start': self._wgsl_function_start,
                'function_end': self._wgsl_function_end,
                'data_types': self._wgsl_data_types,
                'vector_types': self._wgsl_vector_types,
                'math_functions': self._wgsl_math_functions,
                'syntax': self._wgsl_syntax
            }
        }
    
    def _glsl_data_types(self, pseudocode):
        """Convert data types to GLSL"""
        # Common type replacements
        replacements = {
            r'\bvec3\b': 'vec3',
            r'\bvec2\b': 'vec2',
            r'\bvec4\b': 'vec4',
            r'\bfloat\b': 'float',
            r'\bint\b': 'int',
            r'\bbool\b': 'bool',
            r'\bsampler2D\b': 'sampler2D',
            r'\bsamplerCube\b': 'samplerCube'
        }
        
        result = pseudocode
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _glsl_vector_types(self, pseudocode):
        """Handle GLSL vector operations"""
        # Only modify vector constructor patterns, not function calls
        # Look for patterns like vec3(0.0, 0.0, 0.0) but not function calls
        # This is a simplified version - in a full implementation, we'd be more specific
        
        return pseudocode
    
    def _glsl_math_functions(self, pseudocode):
        """Convert math functions to GLSL equivalents"""
        # GLSL has the same basic math functions as the pseudocode
        # But we might need to handle specific cases
        return pseudocode
    
    def _glsl_syntax(self, pseudocode):
        """Apply GLSL specific syntax rules"""
        # GLSL uses specific structures
        lines = pseudocode.split('\n')
        processed_lines = []
        
        for line in lines:
            # Skip pseudocode comments that aren't GLSL ready
            if line.strip().startswith('//') and '//' in line:
                processed_lines.append(line)
            elif line.strip():
                # Process the line for GLSL compatibility
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _glsl_function_start(self, pseudocode):
        """Process function start for GLSL"""
        # In GLSL, functions are defined with specific syntax
        return pseudocode
    
    def _glsl_function_end(self, pseudocode):
        """Process function end for GLSL"""
        return pseudocode
    
    # HLSL implementations (simplified)
    def _hlsl_data_types(self, pseudocode):
        return pseudocode  # Simplified for this example
    
    def _hlsl_vector_types(self, pseudocode):
        return pseudocode
    
    def _hlsl_math_functions(self, pseudocode):
        return pseudocode
    
    def _hlsl_syntax(self, pseudocode):
        return pseudocode
    
    def _hlsl_function_start(self, pseudocode):
        return pseudocode
    
    def _hlsl_function_end(self, pseudocode):
        return pseudocode
    
    # C/C++ implementations (simplified)
    def _c_cpp_data_types(self, pseudocode):
        return pseudocode
    
    def _c_cpp_vector_types(self, pseudocode):
        return pseudocode
    
    def _c_cpp_math_functions(self, pseudocode):
        return pseudocode
    
    def _c_cpp_syntax(self, pseudocode):
        return pseudocode
    
    def _c_cpp_function_start(self, pseudocode):
        return pseudocode
    
    def _c_cpp_function_end(self, pseudocode):
        return pseudocode

    # Metal implementations
    def _metal_data_types(self, pseudocode):
        """Convert data types to Metal"""
        replacements = {
            r'\bvec3\b': 'float3',
            r'\bvec2\b': 'float2',
            r'\bvec4\b': 'float4',
            r'\bmat3\b': 'float3x3',
            r'\bmat4\b': 'float4x4',
            r'\bfloat\b': 'float',
            r'\bint\b': 'int',
            r'\bbool\b': 'bool',
            r'\bsampler2D\b': 'texture2d<float>',
            r'\bsamplerCube\b': 'texturecube<float>'
        }

        result = pseudocode
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result)

        return result

    def _metal_vector_types(self, pseudocode):
        """Handle Metal vector operations"""
        # Convert vector constructors
        result = re.sub(r'vec3\(', 'float3(', pseudocode)
        result = re.sub(r'vec2\(', 'float2(', result)
        result = re.sub(r'vec4\(', 'float4(', result)
        return result

    def _metal_math_functions(self, pseudocode):
        """Convert math functions to Metal equivalents"""
        # Metal uses the same basic functions as GLSL but with metal:: prefix in some cases
        return pseudocode  # For now, same as GLSL

    def _metal_syntax(self, pseudocode):
        """Apply Metal specific syntax rules"""
        return pseudocode

    def _metal_function_start(self, pseudocode):
        return pseudocode

    def _metal_function_end(self, pseudocode):
        return pseudocode

    # WGSL implementations
    def _wgsl_data_types(self, pseudocode):
        """Convert data types to WGSL (WebGPU Shading Language)"""
        replacements = {
            r'\bvec3\b': 'vec3<f32>',
            r'\bvec2\b': 'vec2<f32>',
            r'\bvec4\b': 'vec4<f32>',
            r'\bmat3\b': 'mat3x3<f32>',
            r'\bmat4\b': 'mat4x4<f32>',
            r'\bfloat\b': 'f32',
            r'\bint\b': 'i32',
            r'\bbool\b': 'bool',
            r'\bsampler2D\b': 'texture_2d<f32>',
            r'\bsamplerCube\b': 'texture_cube<f32>'
        }

        result = pseudocode
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result)

        return result

    def _wgsl_vector_types(self, pseudocode):
        """Handle WGSL vector operations"""
        # Convert vector constructors
        result = re.sub(r'vec3\(', 'vec3<f32>(', pseudocode)
        result = re.sub(r'vec2\(', 'vec2<f32>(', result)
        result = re.sub(r'vec4\(', 'vec4<f32>(', result)
        return result

    def _wgsl_math_functions(self, pseudocode):
        """Convert math functions to WGSL equivalents"""
        # WGSL math functions are similar but in the math namespace
        return pseudocode  # For now, same as GLSL

    def _wgsl_syntax(self, pseudocode):
        """Apply WGSL specific syntax rules"""
        return pseudocode

    def _wgsl_function_start(self, pseudocode):
        return pseudocode

    def _wgsl_function_end(self, pseudocode):
        return pseudocode
    
    def translate_to_glsl(self, pseudocode):
        """Translate pseudocode to GLSL"""
        # Apply translations in order
        result = pseudocode
        result = self._glsl_data_types(result)
        result = self._glsl_vector_types(result)
        result = self._glsl_math_functions(result)
        result = self._glsl_syntax(result)
        result = self._glsl_function_start(result)
        result = self._glsl_function_end(result)
        
        return result
    
    def translate(self, pseudocode, target_language='glsl'):
        """Translate pseudocode to target language"""
        if target_language in self.translations:
            translation = self.translations[target_language]
            
            result = pseudocode
            result = translation['data_types'](result)
            result = translation['vector_types'](result)
            result = translation['math_functions'](result)
            result = translation['syntax'](result)
            result = translation['function_start'](result)
            result = translation['function_end'](result)
            
            return result
        else:
            raise ValueError(f"Unsupported target language: {target_language}")
    
    def create_glsl_shader_from_modules(self, module_names):
        """Create a complete GLSL shader from module pseudocodes"""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
        from modules.lighting.registry import get_module_by_name
        
        shader_parts = {
            'header': '#version 330 core\n\n',
            'uniforms': '// Uniforms\nuniform vec3 viewPos;\nuniform vec3 lightPos;\nuniform vec3 lightColor;\nuniform sampler2D normalMap;\nuniform sampler2D shadowMap;\n\n',
            'inputs': '// Input variables\nin vec3 FragPos;\nin vec3 Normal;\nin vec2 TexCoords;\n\n',
            'outputs': '// Output\nout vec4 FragColor;\n\n',
            'functions': [],
            'main': ''
        }
        
        # Collect all functions from modules
        for module_name in module_names:
            module = get_module_by_name(module_name)
            if module and 'pseudocode' in module:
                # Translate the pseudocode to GLSL
                glsl_code = self.translate_to_glsl(module['pseudocode'])
                # Extract functions and add to shader
                shader_parts['functions'].append(glsl_code)
        
        # Create the main function
        main_content = self._create_main_from_modules(module_names)
        shader_parts['main'] = main_content
        
        # Combine all parts
        shader_code = shader_parts['header']
        shader_code += shader_parts['uniforms']
        shader_code += shader_parts['inputs']
        shader_code += shader_parts['outputs']
        
        # Add all the functions
        for func in shader_parts['functions']:
            shader_code += func + "\n"
        
        shader_code += shader_parts['main']
        
        return shader_code
    
    def _create_main_from_modules(self, module_names):
        """Create main function based on selected modules"""
        main_func = """
void main() {
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Initialize color
    vec3 result = vec3(0.0);
"""
        
        # Add logic based on modules
        if 'basic_point_light' in module_names:
            main_func += """    
    // Basic point light calculation
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 pointLight = diff * lightColor;
    
    // Apply distance attenuation
    float distance = length(lightPos - FragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    pointLight *= attenuation;
    
    result += pointLight;
"""
        
        if 'diffuse_lighting' in module_names and 'basic_point_light' not in module_names:
            main_func += """    
    // Diffuse lighting if point light module is not used
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    result += diffuse;
"""
        
        if 'specular_lighting' in module_names:
            main_func += """    
    // Specular lighting
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    result += specular;
"""
        
        if 'normal_mapping' in module_names:
            main_func += """    
    // Normal mapping if available
    vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;
    tangentNormal = normalize(tangentNormal);
    // Use tangentNormal instead of norm for lighting calculations
    float diff = max(dot(tangentNormal, lightDir), 0.0);
    vec3 normalMappedDiffuse = diff * lightColor;
    result = mix(result, normalMappedDiffuse, 0.5); // Blend with original
"""
        
        if 'pbr_lighting' in module_names:
            main_func += """    
    // PBR lighting model (overrides other lighting if present)
    // Simplified PBR calculation
    vec3 albedo = vec3(0.5);
    float metallic = 0.0;
    float roughness = 0.5;
    
    // Direct calculation without full Cook-Torrance for simplicity
    float NdotL = max(dot(norm, lightDir), 0.0);
    result = albedo * lightColor * NdotL;
"""
        
        if 'cel_shading' in module_names:
            main_func += """    
    // Apply cel shading effect
    float NdotL = dot(norm, lightDir);
    float intensity = smoothstep(0.0, 0.01, NdotL);
    intensity += step(0.5, NdotL);
    intensity += step(0.8, NdotL);
    intensity = min(intensity, 1.0);
    
    result = result * vec3(intensity);
"""
        
        if 'shadow_mapping' in module_names:
            main_func += """    
    // Simple shadow calculation if needed
    vec3 projCoords = /* light space transformation */;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float shadow = currentDepth > closestDepth + 0.0005 ? 1.0 : 0.0;
    
    result *= (1.0 - shadow * 0.5); // Apply shadow with 50% intensity
"""
        
        main_func += """    
    // Final color
    FragColor = vec4(result, 1.0);
}
"""
        return main_func

    def _create_shader_parts(self, target_language):
        """Create basic shader parts for different target languages"""
        if target_language == 'glsl':
            return {
                'header': '#version 330 core\n\n',
                'uniforms': '// Uniforms\nuniform vec3 viewPos;\nuniform vec3 lightPos;\nuniform vec3 lightColor;\nuniform sampler2D normalMap;\nuniform sampler2D shadowMap;\n\n',
                'inputs': '// Input variables\nin vec3 FragPos;\nin vec3 Normal;\nin vec2 TexCoords;\n\n',
                'outputs': '// Output\nout vec4 FragColor;\n\n',
                'functions': [],
                'main': ''
            }
        elif target_language == 'metal':
            return {
                'header': '#include <metal_stdlib>\nusing namespace metal;\n\n',
                'uniforms': '// Uniforms\nconstant float3& viewPos [[buffer(0)]];\nconstant float3& lightPos [[buffer(1)]];\nconstant float3& lightColor [[buffer(2)]];\ntexture2d<float> normalMap [[texture(0)]];\ntexture2d<float> shadowMap [[texture(1)]];\n\n',
                'inputs': '// Input variables\nstruct VertexOutput {\n    float3 fragPos [[user(locn0)]];\n    float3 normal [[user(locn1)]];\n    float2 texCoords [[user(locn2)]];\n};\n\n',
                'outputs': '// Output\nstruct FragmentOutput {\n    float4 color [[color(0)]];\n};\n\n',
                'functions': [],
                'main': ''
            }
        elif target_language == 'hlsl':
            return {
                'header': '#include \"common.fxh\"\n\n',
                'uniforms': '// Uniforms\nfloat3 viewPos : register(b0);\nfloat3 lightPos : register(b1);\nfloat3 lightColor : register(b2);\nTexture2D normalMap : register(t0);\nTexture2D shadowMap : register(t1);\nSamplerState samp : register(s0);\n\n',
                'inputs': '// Input variables\nstruct VertexInput {\n    float3 fragPos : POSITION;\n    float3 normal : NORMAL;\n    float2 texCoords : TEXCOORD0;\n};\n\n',
                'outputs': '// Output\nstruct FragmentOutput {\n    float4 color : SV_TARGET0;\n};\n\n',
                'functions': [],
                'main': ''
            }
        elif target_language == 'wgsl':
            return {
                'header': '@group(0) @binding(0)\nvar<uniform> viewPos: vec3f;\n@group(0) @binding(1)\nvar<uniform> lightPos: vec3f;\n@group(0) @binding(2)\nvar<uniform> lightColor: vec3f;\n\n',
                'uniforms': '@group(1) @binding(0)\nvar normalMap: texture_2d<f32>;\n@group(1) @binding(1)\nvar shadowMap: texture_2d<f32>;\n\n',
                'inputs': '@builtin(position) fragPos: vec4f,\n@location(0) normal: vec3f,\n@location(1) texCoords: vec2f,\n\n',
                'outputs': '@location(0) color: vec4f,\n\n',
                'functions': [],
                'main': ''
            }
        else:  # Default to GLSL
            return {
                'header': '#version 330 core\n\n',
                'uniforms': '// Uniforms\nuniform vec3 viewPos;\nuniform vec3 lightPos;\nuniform vec3 lightColor;\nuniform sampler2D normalMap;\nuniform sampler2D shadowMap;\n\n',
                'inputs': '// Input variables\nin vec3 FragPos;\nin vec3 Normal;\nin vec2 TexCoords;\n\n',
                'outputs': '// Output\nout vec4 FragColor;\n\n',
                'functions': [],
                'main': ''
            }

    def _create_main_from_modules(self, module_names, target_language='glsl'):
        """Create main function based on selected modules and target language"""
        if target_language == 'glsl':
            main_func = """
void main() {
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    // Initialize color
    vec3 result = vec3(0.0);
"""

        elif target_language == 'metal':
            main_func = """
fragment FragmentOutput main(VertexOutput input [[stage_in]], 
                            constant device float3* viewPos [[buffer(0)]], 
                            constant device float3* lightPos [[buffer(1)]],
                            constant device float3* lightColor [[buffer(2)]],
                            texture2d<float> normalMap [[texture(0)]],
                            sampler normalSampler [[sampler(0)]]) {
    FragmentOutput output;
    
    // Normalize the normal vector
    float3 norm = normalize(input.normal);
    float3 viewDir = normalize(*viewPos - input.fragPos);

    // Initialize color
    float3 result = float3(0.0);
"""

        elif target_language == 'hlsl':
            main_func = """
FragmentOutput main(VertexInput input) : SV_TARGET {
    FragmentOutput output;
    
    // Normalize the normal vector
    float3 norm = normalize(input.normal);
    float3 viewDir = normalize(viewPos - input.fragPos);

    // Initialize color
    float3 result = float3(0.0);
"""

        elif target_language == 'wgsl':
            main_func = """
@fragment
fn main(@builtin(position) position: vec4f,
        @location(0) normal: vec3f,
        @location(1) texCoords: vec2f) -> @location(0) vec4f {
    // Normalize the normal vector
    var norm = normalize(normal);
    var viewDir = normalize(viewPos - position.xyz);

    // Initialize color
    var result = vec3f(0.0);
"""

        else:  # Default to GLSL
            main_func = """
void main() {
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    // Initialize color
    vec3 result = vec3(0.0);
"""

        # Add logic based on modules (this part stays the same across languages for now)
        if 'basic_point_light' in module_names:
            main_func += """
    // Basic point light calculation
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 pointLight = diff * lightColor;

    // Apply distance attenuation
    float distance = length(lightPos - FragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    pointLight *= attenuation;

    result += pointLight;
"""

        if 'diffuse_lighting' in module_names and 'basic_point_light' not in module_names:
            main_func += """
    // Diffuse lighting if point light module is not used
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    result += diffuse;
"""

        if 'specular_lighting' in module_names:
            main_func += """
    // Specular lighting
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    result += specular;
"""

        if 'normal_mapping' in module_names:
            main_func += """
    // Normal mapping if available
    vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;
    tangentNormal = normalize(tangentNormal);
    // Use tangentNormal instead of norm for lighting calculations
    float diff = max(dot(tangentNormal, lightDir), 0.0);
    vec3 normalMappedDiffuse = diff * lightColor;
    result = mix(result, normalMappedDiffuse, 0.5); // Blend with original
"""

        if 'pbr_lighting' in module_names:
            main_func += """
    // PBR lighting model (overrides other lighting if present)
    // Simplified PBR calculation
    vec3 albedo = vec3(0.5);
    float metallic = 0.0;
    float roughness = 0.5;

    // Direct calculation without full Cook-Torrance for simplicity
    float NdotL = max(dot(norm, lightDir), 0.0);
    result = albedo * lightColor * NdotL;
"""

        if 'cel_shading' in module_names:
            main_func += """
    // Apply cel shading effect
    float NdotL = dot(norm, lightDir);
    float intensity = smoothstep(0.0, 0.01, NdotL);
    intensity += step(0.5, NdotL);
    intensity += step(0.8, NdotL);
    intensity = min(intensity, 1.0);

    result = result * vec3(intensity);
"""

        if 'shadow_mapping' in module_names:
            main_func += """
    // Simple shadow calculation if needed
    vec3 projCoords = /* light space transformation */;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float shadow = currentDepth > closestDepth + 0.0005 ? 1.0 : 0.0;

    result *= (1.0 - shadow * 0.5); // Apply shadow with 50% intensity
"""

        if target_language == 'metal':
            main_func += """
    output.color = float4(result, 1.0);
    return output;
}
"""
        elif target_language == 'hlsl':
            main_func += """
    output.color = float4(result, 1.0);
    return output;
}
"""
        elif target_language == 'wgsl':
            main_func += """
    return vec4f(result, 1.0);
}
"""
        else:  # GLSL and default
            main_func += """
    // Final color
    FragColor = vec4(result, 1.0);
}
"""
        return main_func

    def create_shader_from_modules(self, module_names, target_language='glsl'):
        """Create a complete shader from module pseudocodes in the specified language"""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
        from modules.lighting.registry import get_module_by_name

        # Create shader parts based on the target language
        shader_parts = self._create_shader_parts(target_language)

        # Collect all functions from modules and translate them
        for module_name in module_names:
            module = get_module_by_name(module_name)
            if module and 'pseudocode' in module:
                # Translate the pseudocode to the target language
                translated_code = self.translate(module['pseudocode'], target_language)
                # Extract functions and add to shader
                shader_parts['functions'].append(translated_code)

        # Create the main function
        main_content = self._create_main_from_modules(module_names, target_language)
        shader_parts['main'] = main_content

        # Combine all parts
        shader_code = shader_parts['header']
        shader_code += shader_parts['uniforms']
        shader_code += shader_parts['inputs']
        shader_code += shader_parts['outputs']

        # Add all the functions
        for func in shader_parts['functions']:
            shader_code += func + "\n"

        shader_code += shader_parts['main']

        return shader_code

    def create_glsl_shader_from_modules(self, module_names):
        """Create a complete GLSL shader from module pseudocodes (backward compatibility)"""
        return self.create_shader_from_modules(module_names, 'glsl')


def test_translator():
    """Test the pseudocode translator"""
    print("Testing Pseudocode Translator...")
    
    # Sample pseudocode from one of our modules
    sample_pseudocode = """
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
"""
    
    translator = PseudocodeTranslator()
    
    # Translate to GLSL (should be mostly the same)
    glsl_code = translator.translate_to_glsl(sample_pseudocode)
    print("GLSL Translation:")
    print(glsl_code)
    
    # Create a complete shader from modules
    print("\nCreating shader from modules...")
    shader_code = translator.create_glsl_shader_from_modules([
        'basic_point_light', 
        'diffuse_lighting', 
        'specular_lighting', 
        'normal_mapping'
    ])
    
    print("Complete GLSL Shader:")
    print(shader_code)
    
    # Save the shader
    with open('generated_shader.glsl', 'w') as f:
        f.write(shader_code)
    print("\nSaved complete shader to generated_shader.glsl")


if __name__ == "__main__":
    test_translator()