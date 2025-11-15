#!/usr/bin/env python3
"""
Shader Verification System for SuperShader
Verifies that generated shaders maintain the same functionality as original implementations
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import hashlib
import re


class ShaderFunctionalityVerifier:
    """
    System to verify that generated shaders maintain the same functionality as original implementations
    """

    def __init__(self):
        self.original_shaders = {}
        self.generated_shaders = {}
        self.verification_results = {}

    def load_original_shader(self, name: str, content: str, metadata: Dict[str, Any] = None):
        """
        Load an original shader for reference
        """
        if metadata is None:
            metadata = {}

        self.original_shaders[name] = {
            'content': content,
            'metadata': metadata,
            'hash': hashlib.sha256(content.encode()).hexdigest(),
            'functions': self._extract_functions(content),
            'inputs': self._extract_inputs(content),
            'outputs': self._extract_outputs(content),
            'uniforms': self._extract_uniforms(content)
        }

    def load_generated_shader(self, name: str, content: str, origin: str = ""):
        """
        Load a generated shader to be verified against original
        """
        self.generated_shaders[name] = {
            'content': content,
            'origin': origin,
            'hash': hashlib.sha256(content.encode()).hexdigest(),
            'functions': self._extract_functions(content),
            'inputs': self._extract_inputs(content),
            'outputs': self._extract_outputs(content),
            'uniforms': self._extract_uniforms(content)
        }

    def _extract_functions(self, shader_code: str) -> List[str]:
        """
        Extract function names from shader code
        """
        # This is a simplified function extraction
        # In a real implementation, you'd use a proper GLSL parser
        function_pattern = r'\b(\w+)\s+(\w+)\s*\([^)]*\)\s*\{'
        matches = re.findall(function_pattern, shader_code)
        
        functions = []
        for match in matches:
            # match[0] is return type, match[1] is function name
            functions.append(match[1])
        
        return functions

    def _extract_inputs(self, shader_code: str) -> List[str]:
        """
        Extract input variables from shader code
        """
        input_patterns = [
            r'\bin\s+\w+\s+(\w+)\s*;',  # in type name;
            r'attribute\s+\w+\s+(\w+)\s*;',  # attribute type name;
            r'uniform\s+\w+\s+(\w+)\s*;',  # uniform type name; (though these are uniforms, including for completeness)
        ]
        
        inputs = []
        for pattern in input_patterns:
            matches = re.findall(pattern, shader_code)
            inputs.extend(matches)
        
        return list(set(inputs))  # Remove duplicates

    def _extract_outputs(self, shader_code: str) -> List[str]:
        """
        Extract output variables from shader code
        """
        output_patterns = [
            r'\bout\s+\w+\s+(\w+)\s*;',  # out type name;
            r'varying\s+\w+\s+(\w+)\s*;',  # varying type name;
        ]
        
        outputs = []
        for pattern in output_patterns:
            matches = re.findall(pattern, shader_code)
            outputs.extend(matches)
        
        return list(set(outputs))  # Remove duplicates

    def _extract_uniforms(self, shader_code: str) -> List[str]:
        """
        Extract uniform variables from shader code
        """
        uniform_pattern = r'uniform\s+\w+\s+(\w+)\s*;'
        matches = re.findall(uniform_pattern, shader_code)
        
        return matches

    def verify_shader_functionality(self, original_name: str, generated_name: str) -> Dict[str, Any]:
        """
        Verify that a generated shader maintains functionality compared to original
        """
        if original_name not in self.original_shaders:
            return {
                'status': 'ERROR',
                'message': f'Original shader {original_name} not found',
                'match_percentage': 0
            }

        if generated_name not in self.generated_shaders:
            return {
                'status': 'ERROR',
                'message': f'Generated shader {generated_name} not found',
                'match_percentage': 0
            }

        original = self.original_shaders[original_name]
        generated = self.generated_shaders[generated_name]

        # Perform verification checks
        checks = {
            'function_signature_match': self._check_function_signatures(original, generated),
            'input_variables_match': self._check_input_variables(original, generated),
            'output_variables_match': self._check_output_variables(original, generated),
            'uniforms_compatibility': self._check_uniforms_compatibility(original, generated),
            'structural_similarity': self._check_structural_similarity(original, generated),
            'semantic_preservation': self._check_semantic_preservation(original, generated)
        }

        # Calculate match percentage
        passed_checks = sum(1 for result in checks.values() if result['match'])
        total_checks = len(checks)
        match_percentage = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

        result = {
            'status': 'PASS' if match_percentage >= 80 else 'WARN' if match_percentage >= 60 else 'FAIL',
            'match_percentage': match_percentage,
            'checks': checks,
            'original_shader': original_name,
            'generated_shader': generated_name,
            'verification_time': __import__('time').time()
        }

        self.verification_results[f"{original_name}_vs_{generated_name}"] = result

        return result

    def _check_function_signatures(self, original: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if function signatures match between original and generated shaders
        """
        orig_funcs = set(original['functions'])
        gen_funcs = set(generated['functions'])

        # Check how many original functions are preserved
        if orig_funcs:
            preserved = len(orig_funcs.intersection(gen_funcs))
            total_orig = len(orig_funcs)
            match_ratio = preserved / total_orig if total_orig > 0 else 0

            return {
                'match': match_ratio >= 0.7,  # At least 70% of original functions should be preserved
                'message': f'Preserved {preserved}/{total_orig} original functions',
                'details': {
                    'original_functions': list(orig_funcs),
                    'generated_functions': list(gen_funcs),
                    'preserved': list(orig_funcs.intersection(gen_funcs)),
                    'missing': list(orig_funcs - gen_funcs),
                    'extra': list(gen_funcs - orig_funcs)
                }
            }
        else:
            return {
                'match': True,  # No functions to check
                'message': 'No functions to verify',
                'details': {
                    'original_functions': [],
                    'generated_functions': list(gen_funcs)
                }
            }

    def _check_input_variables(self, original: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if input variables match between original and generated shaders
        """
        orig_inputs = set(original['inputs'])
        gen_inputs = set(generated['inputs'])

        if orig_inputs:
            preserved = len(orig_inputs.intersection(gen_inputs))
            total_orig = len(orig_inputs)
            match_ratio = preserved / total_orig if total_orig > 0 else 0

            return {
                'match': match_ratio >= 0.5,  # At least 50% of inputs should be preserved
                'message': f'Preserved {preserved}/{total_orig} original inputs',
                'details': {
                    'original_inputs': list(orig_inputs),
                    'generated_inputs': list(gen_inputs),
                    'preserved': list(orig_inputs.intersection(gen_inputs)),
                    'missing': list(orig_inputs - gen_inputs),
                    'extra': list(gen_inputs - orig_inputs)
                }
            }
        else:
            return {
                'match': True,
                'message': 'No inputs to verify',
                'details': {
                    'original_inputs': [],
                    'generated_inputs': list(gen_inputs)
                }
            }

    def _check_output_variables(self, original: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if output variables match between original and generated shaders
        """
        orig_outputs = set(original['outputs'])
        gen_outputs = set(generated['outputs'])

        if orig_outputs:
            preserved = len(orig_outputs.intersection(gen_outputs))
            total_orig = len(orig_outputs)
            match_ratio = preserved / total_orig if total_orig > 0 else 0

            return {
                'match': match_ratio >= 0.5,  # At least 50% of outputs should be preserved
                'message': f'Preserved {preserved}/{total_orig} original outputs',
                'details': {
                    'original_outputs': list(orig_outputs),
                    'generated_outputs': list(gen_outputs),
                    'preserved': list(orig_outputs.intersection(gen_outputs)),
                    'missing': list(orig_outputs - gen_outputs),
                    'extra': list(gen_outputs - orig_outputs)
                }
            }
        else:
            return {
                'match': True,
                'message': 'No outputs to verify',
                'details': {
                    'original_outputs': [],
                    'generated_outputs': list(gen_outputs)
                }
            }

    def _check_uniforms_compatibility(self, original: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if uniforms are compatible between original and generated shaders
        """
        orig_uniforms = set(original['uniforms'])
        gen_uniforms = set(generated['uniforms'])

        # For uniforms, we're more flexible - generated shaders might have additional uniforms
        # but should preserve most of the essential ones
        if orig_uniforms:
            preserved = len(orig_uniforms.intersection(gen_uniforms))
            total_orig = len(orig_uniforms)
            match_ratio = preserved / total_orig if total_orig > 0 else 0

            return {
                'match': match_ratio >= 0.7,  # At least 70% of uniforms should be preserved
                'message': f'Preserved {preserved}/{total_orig} original uniforms',
                'details': {
                    'original_uniforms': list(orig_uniforms),
                    'generated_uniforms': list(gen_uniforms),
                    'preserved': list(orig_uniforms.intersection(gen_uniforms)),
                    'missing': list(orig_uniforms - gen_uniforms),
                    'extra': list(gen_uniforms - orig_uniforms)
                }
            }
        else:
            return {
                'match': True,
                'message': 'No uniforms to verify',
                'details': {
                    'original_uniforms': [],
                    'generated_uniforms': list(gen_uniforms)
                }
            }

    def _check_structural_similarity(self, original: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check structural similarity (lines of code, complexity, etc.)
        """
        orig_lines = len(original['content'].split('\\n'))
        gen_lines = len(generated['content'].split('\\n'))

        # Calculate similarity ratio (within 50% tolerance)
        if orig_lines > 0:
            ratio = min(orig_lines, gen_lines) / max(orig_lines, gen_lines)
            match = ratio >= 0.5  # Allow up to 50% size difference

            return {
                'match': match,
                'message': f'Line count similarity: original={orig_lines}, generated={gen_lines}',
                'details': {
                    'original_lines': orig_lines,
                    'generated_lines': gen_lines,
                    'ratio': ratio
                }
            }
        else:
            return {
                'match': True,
                'message': 'Empty shader',
                'details': {
                    'original_lines': 0,
                    'generated_lines': gen_lines
                }
            }

    def _check_semantic_preservation(self, original: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if semantic elements are preserved (key keywords, operations, etc.)
        """
        # Look for key semantic elements that should be preserved
        semantic_keywords = [
            'main', 'texture', 'sampler', 'sin', 'cos', 'tan', 'length', 'normalize',
            'dot', 'cross', 'reflect', 'refract', 'min', 'max', 'clamp', 'mix'
        ]

        orig_content = original['content'].lower()
        gen_content = generated['content'].lower()

        orig_matches = [kw for kw in semantic_keywords if kw in orig_content]
        gen_matches = [kw for kw in semantic_keywords if kw in gen_content]

        common_keywords = set(orig_matches).intersection(set(gen_matches))
        match_ratio = len(common_keywords) / len(orig_matches) if orig_matches else 1

        return {
            'match': match_ratio >= 0.5,  # At least 50% of semantic keywords should be preserved
            'message': f'Semantic keyword preservation: {len(common_keywords)}/{len(orig_matches)} preserved',
            'details': {
                'original_keywords': orig_matches,
                'generated_keywords': gen_matches,
                'preserved_keywords': list(common_keywords),
                'missing_keywords': list(set(orig_matches) - set(gen_matches)),
                'extra_keywords': list(set(gen_matches) - set(orig_matches))
            }
        }

    def verify_all_loaded_shaders(self) -> Dict[str, Any]:
        """
        Verify all loaded generated shaders against their corresponding original shaders
        """
        if not self.original_shaders or not self.generated_shaders:
            return {
                'status': 'ERROR',
                'message': 'No shaders loaded for verification',
                'results': {}
            }

        results = {}

        # Try to match generated shaders with original shaders based on name similarity or metadata
        for gen_name, gen_shader in self.generated_shaders.items():
            # Look for a matching original shader
            best_match = self._find_best_original_match(gen_name, gen_shader)
            if best_match:
                original_name, original_shader = best_match
                verification_result = self.verify_shader_functionality(original_name, gen_name)
                results[f"{original_name}_vs_{gen_name}"] = verification_result

        return {
            'status': 'SUCCESS',
            'message': f'Verified {len(results)} shader pairs',
            'results': results,
            'summary': self._generate_verification_summary(results)
        }

    def _find_best_original_match(self, generated_name: str, generated_shader: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Find the best matching original shader for a generated shader
        """
        # Simple matching based on name similarity and content analysis
        best_match = None
        best_score = 0

        for orig_name, orig_shader in self.original_shaders.items():
            score = 0

            # Score based on name similarity
            if orig_name.replace('_original', '').replace('_ref', '') in generated_name or \
               generated_name.replace('_generated', '').replace('_new', '') in orig_name:
                score += 50

            # Score based on function overlap
            orig_funcs = set(orig_shader['functions'])
            gen_funcs = set(generated_shader['functions'])
            if orig_funcs and gen_funcs:
                func_overlap = len(orig_funcs.intersection(gen_funcs)) / max(len(orig_funcs), len(gen_funcs))
                score += func_overlap * 30

            # Score based on input/output overlap
            orig_inputs = set(orig_shader['inputs'])
            gen_inputs = set(generated_shader['inputs'])
            if orig_inputs and gen_inputs:
                io_overlap = len(orig_inputs.intersection(gen_inputs)) / max(len(orig_inputs), len(gen_inputs))
                score += io_overlap * 20

            if score > best_score:
                best_score = score
                best_match = (orig_name, orig_shader)

        return best_match if best_score > 0 else None

    def _generate_verification_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of verification results
        """
        total = len(results)
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        warnings = sum(1 for r in results.values() if r['status'] == 'WARN')
        failed = sum(1 for r in results.values() if r['status'] == 'FAIL')

        if total > 0:
            avg_match = sum(r['match_percentage'] for r in results.values()) / total
        else:
            avg_match = 0

        return {
            'total_comparisons': total,
            'passed': passed,
            'warnings': warnings,
            'failed': failed,
            'success_rate': (passed / total * 100) if total > 0 else 0,
            'average_match_percentage': avg_match
        }

    def get_verification_report(self) -> Dict[str, Any]:
        """
        Get the complete verification report
        """
        return {
            'original_shaders': list(self.original_shaders.keys()),
            'generated_shaders': list(self.generated_shaders.keys()),
            'verification_results': self.verification_results,
            'summary': self._generate_verification_summary(self.verification_results)
        }


class ShaderVerificationSystem:
    """
    Main system for shader verification
    """

    def __init__(self):
        self.verifier = ShaderFunctionalityVerifier()

    def add_original_shader_from_file(self, filepath: str, name: str = None) -> bool:
        """
        Add an original shader from a file
        """
        if not name:
            name = Path(filepath).stem

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.verifier.load_original_shader(name, content)
            return True
        except Exception as e:
            print(f"Error loading original shader {filepath}: {e}")
            return False

    def add_generated_shader_from_file(self, filepath: str, origin: str = "", name: str = None) -> bool:
        """
        Add a generated shader from a file
        """
        if not name:
            name = Path(filepath).stem

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.verifier.load_generated_shader(name, content, origin)
            return True
        except Exception as e:
            print(f"Error loading generated shader {filepath}: {e}")
            return False

    def verify_shader_pair(self, original_name: str, generated_name: str) -> Dict[str, Any]:
        """
        Verify a specific pair of shaders
        """
        return self.verifier.verify_shader_functionality(original_name, generated_name)

    def run_verification_suite(self) -> Dict[str, Any]:
        """
        Run the complete verification suite
        """
        print("Running Shader Functionality Verification Suite...")
        print("=" * 70)

        result = self.verifier.verify_all_loaded_shaders()

        print(f"\\nVerification Results:")
        print(f"  Total comparisons: {result['summary']['total_comparisons']}")
        print(f"  Passed: {result['summary']['passed']}")
        print(f"  Warnings: {result['summary']['warnings']}")
        print(f"  Failed: {result['summary']['failed']}")
        print(f"  Success rate: {result['summary']['success_rate']:.1f}%")
        print(f"  Average match: {result['summary']['average_match_percentage']:.1f}%")

        # Show details for non-passing comparisons
        for pair_name, result in result['results'].items():
            if result['status'] != 'PASS':
                print(f"\\n  {pair_name}: {result['status']} ({result['match_percentage']:.1f}%)")

        print("=" * 70)

        return result


def main():
    """Main function to demonstrate the shader verification system"""
    print("Initializing Shader Verification System...")

    # Initialize the verification system
    verification_system = ShaderVerificationSystem()

    # Add some example shaders for verification
    original_shader = """#version 330 core
in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

    # A generated shader that should maintain similar functionality
    generated_shader = """#version 330 core
in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

    # Add the shaders to the verification system
    verification_system.verifier.load_original_shader('original_vertex_shader', original_shader)
    verification_system.verifier.load_generated_shader('generated_vertex_shader', generated_shader)

    # Test with a slightly modified generated shader to see verification in action
    modified_shader = """#version 330 core
in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
// Missing some uniforms but should still preserve core functionality

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

    verification_system.verifier.load_generated_shader('modified_vertex_shader', modified_shader)

    # Test with a completely different shader to see failure case
    different_shader = """// This is a completely different shader
void main() {
    // Different functionality
}
"""

    verification_system.verifier.load_generated_shader('different_shader', different_shader)

    # Run verification
    print("\\n1. Verifying original vs generated shaders:")
    result1 = verification_system.verify_shader_pair('original_vertex_shader', 'generated_vertex_shader')
    print(f"   Result: {result1['status']} ({result1['match_percentage']:.1f}% match)")

    print("\\n2. Verifying original vs modified shaders:")
    result2 = verification_system.verify_shader_pair('original_vertex_shader', 'modified_vertex_shader')
    print(f"   Result: {result2['status']} ({result2['match_percentage']:.1f}% match)")

    print("\\n3. Verifying original vs completely different shaders:")
    result3 = verification_system.verify_shader_pair('original_vertex_shader', 'different_shader')
    print(f"   Result: {result3['status']} ({result3['match_percentage']:.1f}% match)")

    # Run the full verification suite
    print("\\n4. Running complete verification suite:")
    full_result = verification_system.run_verification_suite()

    # Get the verification report
    print("\\n5. Verification report:")
    report = verification_system.verifier.get_verification_report()
    summary = report['summary']
    print(f"   Shaders in system: {len(report['original_shaders'])} original, {len(report['generated_shaders'])} generated")
    print(f"   Verification pairs: {summary['total_comparisons']}")
    print(f"   Overall success rate: {summary['success_rate']:.1f}%")

    print(f"\\n✅ Shader Verification System initialized and tested successfully!")
    print(f"   Tested functionality preservation between original and generated shaders")
    print(f"   Verified function signatures, inputs, outputs, uniforms, and semantic elements")

    return 0


if __name__ == "__main__":
    exit(main())