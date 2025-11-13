#!/usr/bin/env python3
"""
Target-Specific Shader Optimization System
Applies optimizations tailored to specific target languages and platforms
"""

from typing import Dict, List, Tuple
from enum import Enum
import re


class TargetPlatform(Enum):
    OPENGL = "opengl"
    DIRECTX = "directx" 
    METAL = "metal"
    WEBGPU = "webgpu"
    VULKAN = "vulkan"


class TargetSpecificOptimizer:
    def __init__(self):
        self.optimizations = {
            'glsl': self._get_glsl_optimizations(),
            'hlsl': self._get_hlsl_optimizations(), 
            'metal': self._get_metal_optimizations(),
            'wgsl': self._get_wgsl_optimizations(),
            'c_cpp': self._get_c_cpp_optimizations()
        }
        
        self.platform_optimizations = {
            TargetPlatform.OPENGL: self._get_opengl_platform_optimizations(),
            TargetPlatform.DIRECTX: self._get_directx_platform_optimizations(),
            TargetPlatform.METAL: self._get_metal_platform_optimizations(),
            TargetPlatform.WEBGPU: self._get_webgpu_platform_optimizations(),
            TargetPlatform.VULKAN: self._get_vulkan_platform_optimizations(),
        }

    def _get_glsl_optimizations(self) -> List[Tuple[str, str]]:
        """Get GLSL-specific optimizations"""
        return [
            # Reduce precision where possible for mobile/web GLSL
            (r'\bhighp\b', 'mediump'),  # For mobile optimization
            (r'\bmediump\b', 'mediump'),  # Already mediump, keep for consistency
            
            # Optimize texture lookups
            (r'\btexture\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*\.\s*rgb', r'vec3(texture(\1, \2).rgb)'),
            
            # Optimize repeated expressions
            (r'\b(\w+)\s*=\s*([^;]+);\s*\1\s*\*\s*\1', r'\1 = \2; float \1_sqr = \1 * \1;'),  # For future use
            
            # Combine operations where possible
            (r'float\s+(\w+)\s*=\s*([^;]+);\s*float\s+(\w+)\s*=\s*\1\s*\*\s*\1', 
             r'float \1 = \2; float \3 = \1 * \1; // Optimized: \3 is square of \1'),
        ]

    def _get_hlsl_optimizations(self) -> List[Tuple[str, str]]:
        """Get HLSL-specific optimizations"""
        return [
            # HLSL-specific sampler state optimizations
            (r'sampler\s+(\w+)\s*:\s*register\(s\d+\);', r'Texture2D \1_tex : register(t0); SamplerState \1_sampler : register(s0);'),
            
            # Packing optimizations for HLSL
            (r'float4\s+\w+\s*=\s*float4\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\);\s*return\s+\w+;', 
             r'return float4(\1, \2, \3, \4); // Direct return optimization'),
        ]

    def _get_metal_optimizations(self) -> List[Tuple[str, str]]:
        """Get Metal-specific optimizations"""
        return [
            # Metal-specific texture access optimizations
            (r'(\w+)\.sample\(([^,]+),\s*([^)]+)\)', r'const auto sample_result = \1.sample(\2, \3);'),
            
            # Metal function inlining hints
            (r'float\s+(\w+)\s*\(', r'float __attribute__((always_inline)) \1('),
        ]

    def _get_wgsl_optimizations(self) -> List[Tuple[str, str]]:
        """Get WGSL-specific optimizations"""
        return [
            # WGSL-specific access optimizations
            (r'textureSample\(([^,]+),\s*([^,]+),\s*([^)]+)\)', r'textureSample(\1, \2, \3)'),
            
            # WGSL variable declaration optimizations
            (r'var\s+(\w+)\s*:\s*\w+\s*=\s*([^;]+);', r'const \1 = \2;'),  # Use const when possible
        ]

    def _get_c_cpp_optimizations(self) -> List[Tuple[str, str]]:
        """Get C/C++ specific optimizations"""
        return [
            # Inline function declarations for C++
            (r'float\s+(\w+)\s*\(', r'inline float \1('),
            
            # Constant propagation in C++
            (r'const float\s+(\w+)\s*=\s*([^;]+);\s*float\s+\1', r'const float \1 = \2; float temp_\1'),
        ]

    def _get_opengl_platform_optimizations(self) -> List[Tuple[str, str]]:
        """Get OpenGL-specific platform optimizations"""
        return [
            # OpenGL ES specific optimizations
            (r'#version 330 core', '#version 300 es\nprecision mediump float;'),
            
            # Optimize for mobile GPU constraints
            (r'pow\(([^,]+),\s*2\)', r'(\1) * (\1)'),  # pow(x, 2) -> x*x for mobile
        ]

    def _get_directx_platform_optimizations(self) -> List[Tuple[str, str]]:
        """Get DirectX-specific platform optimizations"""
        return [
            # DirectX-specific optimizations
            (r'void main\(\)', 'float4 main(float4 pos : SV_POSITION) : SV_TARGET'),
        ]

    def _get_metal_platform_optimizations(self) -> List[Tuple[str, str]]:
        """Get Metal-specific platform optimizations"""
        return [
            # Metal-specific optimizations
        ]

    def _get_webgpu_platform_optimizations(self) -> List[Tuple[str, str]]:
        """Get WebGPU-specific platform optimizations"""
        return [
            # WebGPU-specific optimizations
        ]

    def _get_vulkan_platform_optimizations(self) -> List[Tuple[str, str]]:
        """Get Vulkan-specific platform optimizations"""
        return [
            # Vulkan-specific optimizations
        ]

    def optimize_for_target(self, shader_code: str, target_language: str, 
                           target_platform: TargetPlatform = None) -> str:
        """Apply optimizations specific to the target language and platform"""
        optimized_code = shader_code
        
        # Apply language-specific optimizations
        if target_language in self.optimizations:
            for pattern, replacement in self.optimizations[target_language]:
                optimized_code = re.sub(pattern, replacement, optimized_code)
        
        # Apply platform-specific optimizations if specified
        if target_platform and target_platform in self.platform_optimizations:
            for pattern, replacement in self.platform_optimizations[target_platform]:
                optimized_code = re.sub(pattern, replacement, optimized_code)
        
        return optimized_code

    def optimize_shader_structure(self, shader_code: str, target_language: str) -> str:
        """Optimize shader structure based on target language"""
        lines = shader_code.split('\n')
        optimized_lines = []
        
        # Track which optimizations were applied
        applied_optimizations = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # GLSL-specific optimizations
            if target_language == 'glsl':
                # Optimize for uniform buffer objects if present
                if 'uniform' in line and 'struct' in line:
                    applied_optimizations.append("Structured uniform optimization")
                
                # Optimize texture declarations
                texture_match = re.search(r'sampler2D\s+(\w+)', line)
                if texture_match:
                    tex_name = texture_match.group(1)
                    # In newer GLSL, we can use texture objects
                    applied_optimizations.append(f"Texture optimization for {tex_name}")
            
            # Metal-specific optimizations
            elif target_language == 'metal':
                # Optimize for Metal's function constants
                if 'constant' in line and 'device' in line:
                    applied_optimizations.append("Metal-specific memory optimization")
            
            # HLSL-specific optimizations
            elif target_language == 'hlsl':
                # Optimize for HLSL's SV semantics
                if 'SV_' in line:
                    applied_optimizations.append("HLSL semantic optimization")
            
            optimized_lines.append(line)
            i += 1
        
        optimized_shader = '\n'.join(optimized_lines)
        
        return optimized_shader

    def get_optimization_report(self, original_code: str, optimized_code: str, 
                               target_language: str, target_platform: TargetPlatform = None) -> Dict:
        """Generate a report of optimizations applied"""
        report = {
            'target_language': target_language,
            'target_platform': target_platform.value if target_platform else None,
            'original_size': len(original_code),
            'optimized_size': len(optimized_code),
            'size_reduction': len(original_code) - len(optimized_code),
            'size_reduction_percent': ((len(original_code) - len(optimized_code)) / len(original_code)) * 100 if len(original_code) > 0 else 0,
            'optimizations_applied': [],
            'complexity_analysis': self._analyze_complexity(optimized_code)
        }
        
        return report

    def _analyze_complexity(self, shader_code: str) -> Dict:
        """Analyze the complexity of the shader code"""
        complexity = {
            'function_count': len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*\{', shader_code)),
            'texture_reads': len(re.findall(r'(texture|texture2D|sample)', shader_code, re.IGNORECASE)),
            'arithmetic_operations': len(re.findall(r'[\+\-\*\/]', shader_code)),
            'branching_statements': len(re.findall(r'\b(if|else|switch|for|while)\b', shader_code)),
            'vector_operations': len(re.findall(r'\b(vec[2-4]|float[2-4]|mat[2-4])', shader_code)),
            'loops': len(re.findall(r'\b(for|while)\b', shader_code))
        }
        
        return complexity

    def optimize_complete_shader(self, shader_code: str, target_language: str, 
                                target_platform: TargetPlatform = None) -> Tuple[str, Dict]:
        """Optimize a complete shader and return both optimized code and report"""
        # Apply structure optimizations
        struct_optimized = self.optimize_shader_structure(shader_code, target_language)
        
        # Apply target-specific optimizations
        fully_optimized = self.optimize_for_target(struct_optimized, target_language, target_platform)
        
        # Generate report
        report = self.get_optimization_report(shader_code, fully_optimized, target_language, target_platform)
        
        return fully_optimized, report


def demo_target_specific_optimization():
    """Demonstrate target-specific optimization capabilities"""
    print("Target-Specific Shader Optimization Demo")
    print("=" * 50)
    
    optimizer = TargetSpecificOptimizer()
    
    # Sample shader code to optimize
    sample_shader = """#version 330 core

uniform vec3 viewPos;
uniform vec3 lightPos; 
uniform vec3 lightColor;
uniform sampler2D normalMap;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;

float complexFunction(float x) {
    float a = x * 1.0;
    float b = 2.0 * 3.0;
    float c = a + 0.0;
    float d = pow(x, 2.0);
    float e = sqrt(1.0 - x * x);
    
    return a + b + c + d + e;
}

void main() {
    vec2 uv = TexCoords;
    float value = complexFunction(uv.x);
    FragColor = vec4(value, value, value, 1.0);
    
    vec3 redundant = uv.x * 1.0;
    float unused = 42.0;
}
"""
    
    print("Original shader:")
    print(sample_shader)
    print("\n" + "="*60 + "\n")
    
    # Test optimizations for different targets
    targets = [
        ('glsl', TargetPlatform.OPENGL),
        ('metal', TargetPlatform.METAL), 
        ('hlsl', TargetPlatform.DIRECTX),
        ('wgsl', TargetPlatform.WEBGPU)
    ]
    
    for target_lang, target_platform in targets:
        print(f"Optimizing for {target_lang.upper()} on {target_platform.value.upper()}:")
        print("-" * 40)
        
        try:
            optimized_code, report = optimizer.optimize_complete_shader(
                sample_shader, target_lang, target_platform
            )
            
            print(f"Optimized code ({report['size_reduction_percent']:.2f}% size change):")
            print(optimized_code[:300] + "..." if len(optimized_code) > 300 else optimized_code)
            
            print(f"\nReport:")
            print(f"  Size: {report['original_size']} -> {report['optimized_size']} ({report['size_reduction']} chars)")
            print(f"  Complexity: {report['complexity_analysis']}")
            
        except Exception as e:
            print(f"Error optimizing for {target_lang}: {e}")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    demo_target_specific_optimization()