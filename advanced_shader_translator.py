#!/usr/bin/env python3
"""
Advanced Pseudocode Translation System
Implements sophisticated translation rules for complex shader features
"""

import re
from typing import Dict, List, Tuple, Pattern
from create_pseudocode_translator import PseudocodeTranslator


class AdvancedShaderTranslator(PseudocodeTranslator):
    def __init__(self):
        super().__init__()
        
        # Enhanced translation patterns for complex features
        self.enhanced_patterns = {
            'glsl': self._get_glsl_patterns(),
            'hlsl': self._get_hlsl_patterns(),
            'metal': self._get_metal_patterns(),
            'wgsl': self._get_wgsl_patterns(),
            'c_cpp': self._get_c_cpp_patterns()
        }
        
        # Semantic transformation rules
        self.semantic_transformations = {
            'glsl': self._get_glsl_semantic_transformations(),
            'hlsl': self._get_hlsl_semantic_transformations(),
            'metal': self._get_metal_semantic_transformations(),
            'wgsl': self._get_wgsl_semantic_transformations()
        }

    def _get_glsl_patterns(self) -> Dict[str, str]:
        """Get enhanced GLSL patterns"""
        return {
            # Advanced vector operations
            r'\btexture\(([^,]+),\s*([^)]+)\)': r'texture(\1, \2)',
            r'\btexture2D\(([^,]+),\s*([^)]+)\)': r'texture(\1, \2)',
            r'\btextureCube\(([^,]+),\s*([^)]+)\)': r'texture(\1, \2)',
            r'\bnormalize\(([^)]+)\)': r'normalize(\1)',
            r'\bdot\(([^,]+),\s*([^)]+)\)': r'dot(\1, \2)',
            r'\bcross\(([^,]+),\s*([^)]+)\)': r'cross(\1, \2)',
            
            # Advanced math functions
            r'\bpow\(([^,]+),\s*2\.0\s*\)': r'(\1) * (\1)',  # pow(x, 2.0) -> x*x
            r'\bpow\(([^,]+),\s*0\.5\s*\)': r'sqrt(\1)',     # pow(x, 0.5) -> sqrt(x)
            r'\bsin\(([^)]+)\)\s*\*\s*sin\(([^)]+)\)': r'pow(sin(\1), 2.0)',  # Duplicate for same var
        }

    def _get_hlsl_patterns(self) -> Dict[str, str]:
        """Get enhanced HLSL patterns"""
        return {
            # HLSL-specific functions
            r'\btexture\(([^,]+),\s*([^)]+)\)': r'\1.Sample(\1 ## Sampler, \2)',
            r'\btexture2D\(([^,]+),\s*([^)]+)\)': r'\1.Sample(\1 ## Sampler, \2)',  # Will be fixed in semantic transform
            r'\btextureCube\(([^,]+),\s*([^)]+)\)': r'\1.Sample(\1 ## Sampler, \2)',
            r'\bnormalize\(([^)]+)\)': r'normalize(\1)',
            r'\bdot\(([^,]+),\s*([^)]+)\)': r'dot(\1, \2)',
            r'\bcross\(([^,]+),\s*([^)]+)\)': r'cross(\1, \2)',
            
            # Advanced math functions with HLSL equivalents
            r'\bpow\(([^,]+),\s*2\.0\s*\)': r'(\1) * (\1)',
            r'\bpow\(([^,]+),\s*0\.5\s*\)': r'sqrt(\1)',
        }

    def _get_metal_patterns(self) -> Dict[str, str]:
        """Get enhanced Metal patterns"""
        return {
            # Metal-specific functions and syntax
            r'\btexture\(([^,]+),\s*([^)]+)\)': r'\1.sample(\1 ## sampler, \2)',
            r'\btexture2D\(([^,]+),\s*([^)]+)\)': r'\1.sample(\1 ## sampler, \2)',
            r'\btextureCube\(([^,]+),\s*([^)]+)\)': r'\1.sample(\1 ## sampler, \2)',
            r'\bnormalize\(([^)]+)\)': r'normalize(\1)',
            r'\bdot\(([^,]+),\s*([^)]+)\)': r'dot(\1, \2)',
            r'\bcross\(([^,]+),\s*([^)]+)\)': r'cross(\1, \2)',
            
            # Vector operations
            r'\bfloat3\(([^,]+),\s*([^,]+),\s*([^)]+)\)': r'float3(\1, \2, \3)',
        }

    def _get_wgsl_patterns(self) -> Dict[str, str]:
        """Get enhanced WGSL patterns"""
        return {
            # WGSL-specific syntax
            r'\btexture\(([^,]+),\s*([^)]+)\)': r'textureSample(\1, \1 ## _sampler, \2)',
            r'\btexture2D\(([^,]+),\s*([^)]+)\)': r'textureSample(\1, \1 ## _sampler, \2)',
            r'\btextureCube\(([^,]+),\s*([^)]+)\)': r'textureSample(\1, \1 ## _sampler, \2)',
            r'\bnormalize\(([^)]+)\)': r'normalize(\1)',
            r'\bdot\(([^,]+),\s*([^)]+)\)': r'dot(\1, \2)',
            r'\bcross\(([^,]+),\s*([^)]+)\)': r'cross(\1, \2)',
        }

    def _get_c_cpp_patterns(self) -> Dict[str, str]:
        """Get enhanced C/C++ patterns"""
        return {
            # C/C++ math functions
            r'\bsin\(([^)]+)\)': r'std::sin(\1)',
            r'\bcos\(([^)]+)\)': r'std::cos(\1)',
            r'\btan\(([^)]+)\)': r'std::tan(\1)',
            r'\bsqrt\(([^)]+)\)': r'std::sqrt(\1)',
            r'\bpow\(([^,]+),\s*([^)]+)\)': r'std::pow(\1, \2)',
            r'\babs\(([^)]+)\)': r'std::abs(\1)',
            r'\bmax\(([^,]+),\s*([^)]+)\)': r'std::max(\1, \2)',
            r'\bmin\(([^,]+),\s*([^)]+)\)': r'std::min(\1, \2)',
        }

    def _get_glsl_semantic_transformations(self) -> List[Tuple[str, str]]:
        """Get semantic transformations for GLSL"""
        return [
            # GLSL-specific transformations
        ]

    def _get_hlsl_semantic_transformations(self) -> List[Tuple[str, str]]:
        """Get semantic transformations for HLSL"""
        return [
            # Fix texture sampling syntax
            (r'(\w+)\.Sample\(\1 ## Sampler,\s*([^)]+)\)', r'\1.Sample(sampler, \2)'),
        ]

    def _get_metal_semantic_transformations(self) -> List[Tuple[str, str]]:
        """Get semantic transformations for Metal"""
        return [
            # Metal-specific transformations
        ]

    def _get_wgsl_semantic_transformations(self) -> List[Tuple[str, str]]:
        """Get semantic transformations for WGSL"""
        return [
            # WGSL-specific transformations
        ]

    def apply_enhanced_translation(self, pseudocode: str, target_language: str) -> str:
        """Apply enhanced translation patterns to pseudocode"""
        if target_language not in self.enhanced_patterns:
            return pseudocode

        result = pseudocode
        patterns = self.enhanced_patterns[target_language]
        
        # Apply pattern-based transformations
        for pattern, replacement in patterns.items():
            result = re.sub(pattern, replacement, result)
        
        # Apply semantic transformations
        if target_language in self.semantic_transformations:
            for pattern, replacement in self.semantic_transformations[target_language]:
                result = re.sub(pattern, replacement, result)
        
        return result

    def translate_with_enhanced_rules(self, pseudocode: str, target_language: str = 'glsl') -> str:
        """Translate pseudocode using both basic and enhanced rules"""
        # Apply basic translation first
        basic_translation = self.translate(pseudocode, target_language)
        
        # Apply enhanced translation rules
        enhanced_translation = self.apply_enhanced_translation(basic_translation, target_language)
        
        return enhanced_translation

    def translate_shader_with_features(self, module_names: List[str], target_language: str = 'glsl') -> str:
        """Create and translate a shader with complex features"""
        # Create the shader with the basic translator
        shader_code = self.create_shader_from_modules(module_names, target_language)
        
        # Apply enhanced translation rules to the entire shader
        enhanced_shader = self.apply_enhanced_translation(shader_code, target_language)
        
        return enhanced_shader


def test_advanced_translator():
    """Test the advanced translator with complex shader features"""
    print("Testing Advanced Pseudocode Translator...")
    
    advanced_translator = AdvancedShaderTranslator()
    
    # Test enhanced translation with complex pseudocode
    complex_pseudocode = """
// Advanced Lighting Function with Complex Operations
vec3 advancedLighting(vec3 position, vec3 normal, vec3 viewDir, 
                     vec3 lightPos, vec3 lightColor, 
                     sampler2D normalMap, vec2 texCoords) {
    // Sample normal map
    vec3 tangentNormal = texture(normalMap, texCoords).xyz * 2.0 - 1.0;
    tangentNormal = normalize(tangentNormal);
    
    // Calculate light direction
    vec3 lightDir = normalize(lightPos - position);
    
    // Diffuse calculation
    float NdotL = dot(tangentNormal, lightDir);
    vec3 diffuse = max(NdotL, 0.0) * lightColor;
    
    // Specular calculation using reflection
    vec3 reflectDir = reflect(-lightDir, tangentNormal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    
    // Combine results
    return diffuse + specular;
}

// Function using mathematical shortcuts
vec3 mathematicalExample(vec3 input) {
    // Use pow(x, 2.0) which should be optimized to x*x
    float squared = pow(input.x, 2.0);
    
    // Use pow(x, 0.5) which should be optimized to sqrt(x)
    float root = pow(input.y, 0.5);
    
    return vec3(squared, root, input.z);
}
"""
    
    print("Original pseudocode:")
    print(complex_pseudocode)
    
    # Test different target languages
    languages = ['glsl', 'hlsl', 'metal', 'wgsl']
    
    for lang in languages:
        print(f"\n{'='*20} {lang.upper()} Translation {'='*20}")
        try:
            translated = advanced_translator.translate_with_enhanced_rules(complex_pseudocode, lang)
            print(translated)
        except Exception as e:
            print(f"Error translating to {lang}: {e}")
    
    # Test creating a complete shader with complex features
    print(f"\n{'='*30} Complete Shader Generation {'='*30}")
    
    # Use actual module names from the project
    test_modules = ['lighting/diffuse/diffuse_lighting', 'lighting/specular/specular_lighting']
    
    for lang in ['glsl', 'metal']:
        print(f"\nGenerating {lang.upper()} shader:")
        try:
            shader = advanced_translator.translate_shader_with_features(test_modules, lang)
            print(f"Generated {len(shader)} characters of {lang} code")
            
            # Save the shader
            with open(f'advanced_shader_{lang}.glsl', 'w') as f:
                f.write(shader)
            print(f"Saved advanced {lang} shader to advanced_shader_{lang}.glsl")
        except Exception as e:
            print(f"Error generating {lang} shader: {e}")


if __name__ == "__main__":
    test_advanced_translator()