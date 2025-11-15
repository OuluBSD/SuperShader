#!/usr/bin/env python3
"""
Compatibility Testing System for SuperShader
Tests compatibility of generated shaders across different target platforms and graphics APIs
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tempfile
import subprocess
import platform
import re


class PlatformCompatibilityTester:
    """
    Tests compatibility of shaders across different target platforms and graphics APIs
    """

    def __init__(self):
        self.platform_configs = {
            'opengl': {
                'versions': ['2.0', '3.3', '4.0', '4.1', '4.2', '4.3', '4.5', '4.6'],
                'extensions': ['GL_ARB_shader_bit_encoding', 'GL_ARB_gpu_shader5'],
                'validation_tool': 'glslangValidator',
                'validation_args': ['-S', 'vert', '--target-env', 'opengl'],
                'status': 'supported',
                'glsl_version_prefix': '#version {}'
            },
            'vulkan': {
                'versions': ['1.0', '1.1', '1.2', '1.3'],
                'extensions': ['VK_KHR_shader_draw_parameters', 'VK_KHR_shader_float16_int8'],
                'validation_tool': 'glslangValidator',
                'validation_args': ['-S', 'vert', '--target-env', 'vulkan1.0'],
                'status': 'supported',
                'glsl_version_prefix': '#version 450\\n#extension GL_ARB_separate_shader_objects : enable'
            },
            'metal': {
                'versions': ['1.0', '1.1', '1.2', '2.0', '2.1', '2.2', '2.3', '2.4', '3.0'],
                'extensions': [],
                'validation_tool': 'metal',  # Using metal compiler
                'validation_args': [],
                'status': 'partial',  # Would need more specific tooling
                'glsl_version_prefix': '// Metal shader'
            },
            'directx': {
                'versions': ['11.0', '12.0'],
                'extensions': [],
                'validation_tool': 'dxc',  # DirectX Shader Compiler
                'validation_args': ['-T', 'vs_5_0'],  # Vertex shader model 5.0
                'status': 'partial',  # Would need more specific tooling
                'glsl_version_prefix': '// HLSL shader'
            },
            'webgl': {
                'versions': ['1.0', '2.0'],
                'extensions': ['OES_texture_float', 'OES_standard_derivatives'],
                'validation_tool': 'glslangValidator',
                'validation_args': ['-S', 'vert', '--target-env', 'web'],
                'status': 'supported',
                'glsl_version_prefix': '#version 300 es\\nprecision highp float;'
            }
        }
        
        self.compatibility_results = {}

    def validate_shader_syntax(self, shader_code: str, platform: str, version: str = None) -> Dict[str, Any]:
        """
        Validate shader syntax for a specific platform
        """
        if platform not in self.platform_configs:
            return {'valid': False, 'error': f'Platform {platform} not supported', 'details': {}}

        config = self.platform_configs[platform]
        
        if version and version not in config['versions']:
            return {'valid': False, 'error': f'Version {version} not supported for {platform}', 'details': {}}

        # For now, we'll use a simple validation approach
        # In a real implementation, we'd call an actual shader compiler/validator
        try:
            # Create a temporary shader file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.glsl', delete=False) as temp_file:
                # Add version-specific header if needed
                version_header = config['glsl_version_prefix']
                if version:
                    if platform in ['opengl', 'webgl'] and '{' in version_header:
                        version_header = version_header.format(version.replace('.', ''))
                
                full_shader = version_header + '\\n' + shader_code
                temp_file.write(full_shader)
                temp_file_path = temp_file.name

            # In a real implementation, we would run the validation tool
            # For now, we'll simulate validation by checking for common issues
            validation_result = self._simulate_shader_validation(full_shader, platform, version)

            # Clean up temp file
            os.unlink(temp_file_path)

            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}',
                'details': {}
            }

    def _simulate_shader_validation(self, shader_code: str, platform: str, version: str) -> Dict[str, Any]:
        """
        Simulate shader validation since we can't run actual compiler tools in this environment
        """
        issues = []
        
        # Check for platform-specific issues
        if platform == 'webgl':
            # WebGL doesn't support certain features
            if 'textureCube' in shader_code and version == '1.0':
                issues.append('textureCube not supported in WebGL 1.0')
            if 'inout' in shader_code:
                issues.append('inout not supported in WebGL')
        elif platform == 'opengl':
            # Check for version compatibility
            if version and float(version) < 3.3 and 'in' in shader_code.split():
                issues.append(f'Attribute "in" may not be fully supported in OpenGL {version}')
        elif platform == 'vulkan':
            # Vulkan specific checks
            if 'gl_ModelViewMatrix' in shader_code:
                issues.append('gl_ModelViewMatrix not available in Vulkan')
        elif platform == 'metal':
            # Metal uses different conventions
            if '#version' in shader_code:
                issues.append('Metal does not use #version directives')
        elif platform == 'directx':
            # DirectX uses HLSL syntax
            if 'void main()' in shader_code:
                issues.append('DirectX/HLSL uses different main function signature')

        # Check for common issues
        if 'void main()' not in shader_code and 'main()' not in shader_code:
            issues.append('Missing main function')
        
        if re.search(r'for\s*\(', shader_code):  # Loop detection
            # Check for dynamic loops which aren't allowed in some shader environments
            pass  # Would implement more detailed checks

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'details': {
                'platform': platform,
                'version': version,
                'shader_lines': len(shader_code.split('\\n'))
            }
        }

    def test_shader_compatibility(self, shader_code: str, platforms: List[str] = None) -> Dict[str, Any]:
        """
        Test shader compatibility across multiple platforms
        """
        if platforms is None:
            platforms = list(self.platform_configs.keys())

        results = {}

        for platform in platforms:
            platform_results = {}
            
            # Test against all supported versions of the platform
            for version in self.platform_configs[platform]['versions']:
                validation_result = self.validate_shader_syntax(shader_code, platform, version)
                platform_results[version] = validation_result
            
            results[platform] = platform_results

        return results

    def get_compatibility_report(self, shader_code: str, platforms: List[str] = None) -> Dict[str, Any]:
        """
        Generate a compatibility report for a shader across platforms
        """
        test_results = self.test_shader_compatibility(shader_code, platforms)

        # Generate summary statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        platform_support = {}

        for platform, versions in test_results.items():
            platform_passed = 0
            platform_failed = 0
            
            for version, result in versions.items():
                total_tests += 1
                if result['valid']:
                    passed_tests += 1
                    platform_passed += 1
                else:
                    failed_tests += 1
                    platform_failed += 1

            platform_support[platform] = {
                'total': len(versions),
                'passed': platform_passed,
                'failed': platform_failed,
                'success_rate': (platform_passed / len(versions) * 100) if len(versions) > 0 else 0
            }

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'platform_support': platform_support,
            'detailed_results': test_results
        }

    def generate_platform_specific_shader(self, shader_code: str, target_platform: str, target_version: str = None) -> str:
        """
        Generate a platform-specific version of a shader
        """
        if target_platform not in self.platform_configs:
            return shader_code  # Return as is if platform not supported

        config = self.platform_configs[target_platform]

        # Add platform-specific header
        header = config['glsl_version_prefix']
        if target_version:
            # Format header with version if supported
            if '{}' in header:
                header = header.format(target_version.replace('.', ''))

        # Make platform-specific modifications
        modified_shader = shader_code
        
        if target_platform == 'webgl':
            # WebGL specific modifications
            modified_shader = self._modify_for_webgl(modified_shader)
        elif target_platform == 'vulkan':
            # Vulkan specific modifications
            modified_shader = self._modify_for_vulkan(modified_shader)
        elif target_platform == 'metal':
            # Metal specific modifications
            modified_shader = self._modify_for_metal(modified_shader)

        return header + '\\n' + modified_shader

    def _modify_for_webgl(self, shader_code: str) -> str:
        """
        Modify shader code to be compatible with WebGL
        """
        # Replace GLSL 3.30+ features with WebGL 2.0 equivalents
        modified = shader_code.replace('in ', 'attribute ')  # For WebGL 1.0
        modified = modified.replace('out ', 'varying ')      # For WebGL 1.0
        modified = re.sub(r'uniform\\s+\\w+\\s+(\\w+);', r'uniform highp float \\1;', modified)  # Standardize precision
        return modified

    def _modify_for_vulkan(self, shader_code: str) -> str:
        """
        Modify shader code to be compatible with Vulkan
        """
        # Vulkan specific modifications
        modified = shader_code
        # Add layout qualifiers
        modified = re.sub(r'uniform\\s+(\\w+)\\s+(\\w+);', r'layout(binding = 0) uniform \\1 \\2;', modified)
        return modified

    def _modify_for_metal(self, shader_code: str) -> str:
        """
        Modify shader code to be compatible with Metal
        """
        # Metal uses different syntax, this is a simplified conversion
        modified = shader_code.replace('#version', '// Metal compatible version')
        # Replace GLSL built-ins with Metal equivalents where possible
        return modified


class CompatibilityTestingSystem:
    """
    Main system for compatibility testing
    """

    def __init__(self):
        self.tester = PlatformCompatibilityTester()
        self.test_results = {}
        self.platform_configs = self.tester.platform_configs

    def run_platform_compatibility_tests(self, shader_code: str) -> Dict[str, Any]:
        """
        Run compatibility tests for a shader across all platforms
        """
        print("Running Platform Compatibility Tests...")
        print("=" * 60)

        report = self.tester.get_compatibility_report(shader_code)

        print(f"\\nCompatibility Results:")
        print(f"  Total tests: {report['total_tests']}")
        print(f"  Passed: {report['passed_tests']}")
        print(f"  Failed: {report['failed_tests']}")
        print(f"  Overall success rate: {report['success_rate']:.1f}%")

        print(f"\\nPlatform Support Breakdown:")
        for platform, support_data in report['platform_support'].items():
            print(f"  {platform}: {support_data['passed']}/{support_data['total']} versions "
                  f"({support_data['success_rate']:.1f}% success rate)")

        print(f"\\nDetailed Results:")
        for platform, versions in report['detailed_results'].items():
            print(f"  {platform}:")
            for version, result in versions.items():
                status = '✓' if result['valid'] else '✗'
                issue_count = len(result.get('issues', []))
                print(f"    {version}: {status} ({issue_count} issues)")

        return report

    def test_cross_platform_shader_generation(self) -> Dict[str, Any]:
        """
        Test generating shaders for different platforms and validate them
        """
        print("\\nTesting Cross-Platform Shader Generation...")

        # Sample shader code to test
        sample_vertex_shader = """in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

        results = {}

        # Test original shader compatibility
        print("\\n1. Testing original shader compatibility:")
        original_report = self.run_platform_compatibility_tests(sample_vertex_shader)
        results['original'] = original_report

        # Generate platform-specific versions
        print("\\n2. Testing platform-specific versions:")
        platforms_to_test = ['opengl', 'webgl', 'vulkan']
        
        for platform in platforms_to_test:
            if platform in self.platform_configs:
                # Generate shader for specific platform
                platform_shader = self.tester.generate_platform_specific_shader(
                    sample_vertex_shader, platform, self.platform_configs[platform]['versions'][0]
                )
                
                print(f"\\n  {platform.title()} specific shader:")
                platform_report = self.tester.get_compatibility_report(platform_shader, [platform])
                
                results[f'platform_specific_{platform}'] = platform_report
                
                # Print specific results for this platform
                support_data = platform_report['platform_support'].get(platform, {})
                print(f"    Success rate: {support_data.get('success_rate', 0):.1f}%")

        return results

    def get_compatibility_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        Get recommendations based on compatibility test results
        """
        recommendations = []

        # Check overall success rate
        if report['success_rate'] < 50:
            recommendations.append("⚠️  Low overall compatibility. Consider simplifying shader code.")
        
        # Check individual platform support
        for platform, support_data in report['platform_support'].items():
            if support_data['success_rate'] < 30:
                recommendations.append(f"⚠️  Poor {platform} support ({support_data['success_rate']:.1f}%). "
                                     f"Consider platform-specific implementations.")
            elif support_data['success_rate'] < 70:
                recommendations.append(f"📝 {platform} support could be improved ({support_data['success_rate']:.1f}%). "
                                     f"Review platform-specific requirements.")
            else:
                recommendations.append(f"✅ Good {platform} support ({support_data['success_rate']:.1f}%).")

        # Check for specific issues
        for platform, versions in report['detailed_results'].items():
            common_issues = {}
            for version, result in versions.items():
                for issue in result.get('issues', []):
                    if issue not in common_issues:
                        common_issues[issue] = 0
                    common_issues[issue] += 1
            
            # Add recommendations for common issues
            if common_issues:
                most_common = max(common_issues, key=common_issues.get)
                recommendations.append(f"🔧 Common issue on {platform}: {most_common} (affects {common_issues[most_common]} versions)")

        return recommendations


def main():
    """Main function to demonstrate the compatibility testing system"""
    print("Initializing Platform Compatibility Testing System...")

    # Initialize the compatibility testing system
    compat_system = CompatibilityTestingSystem()

    # Sample shader code to test
    sample_shader = """#version 330 core
in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

    print("\\n1. Running compatibility tests for sample shader:")
    report = compat_system.run_platform_compatibility_tests(sample_shader)

    print(f"\\n2. Generating compatibility recommendations:")
    recommendations = compat_system.get_compatibility_recommendations(report)
    for rec in recommendations:
        print(f"   {rec}")

    print(f"\\n3. Testing cross-platform shader generation:")
    cross_platform_results = compat_system.test_cross_platform_shader_generation()

    print(f"\\n4. Platform configurations supported:")
    for platform, config in compat_system.platform_configs.items():
        print(f"   {platform}: {len(config['versions'])} versions, status={config['status']}")

    print(f"\\n5. Testing specific platform validations:")
    # Test OpenGL compatibility
    opengl_result = compat_system.tester.validate_shader_syntax(sample_shader, 'opengl', '4.5')
    print(f"   OpenGL 4.5: {'✓' if opengl_result['valid'] else '✗'}")

    # Test WebGL compatibility
    webgl_result = compat_system.tester.validate_shader_syntax(sample_shader, 'webgl', '2.0')
    print(f"   WebGL 2.0: {'✓' if webgl_result['valid'] else '✗'}")

    # Test Vulkan compatibility
    vulkan_result = compat_system.tester.validate_shader_syntax(sample_shader, 'vulkan', '1.2')
    print(f"   Vulkan 1.2: {'✓' if vulkan_result['valid'] else '✗'}")

    print(f"\\n✅ Platform Compatibility Testing System initialized and tested successfully!")
    print(f"   - Tested compatibility across {len(compat_system.platform_configs)} platforms")
    print(f"   - Generated platform-specific shader variants")
    print(f"   - Provided compatibility recommendations")
    print(f"   - Simulated validation for different graphics APIs")

    return 0


if __name__ == "__main__":
    exit(main())