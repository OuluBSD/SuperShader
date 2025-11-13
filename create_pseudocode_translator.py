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
            'c_cpp': {
                'function_start': self._c_cpp_function_start,
                'function_end': self._c_cpp_function_end,
                'data_types': self._c_cpp_data_types,
                'vector_types': self._c_cpp_vector_types,
                'math_functions': self._c_cpp_math_functions,
                'syntax': self._c_cpp_syntax
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