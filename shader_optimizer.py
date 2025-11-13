#!/usr/bin/env python3
"""
Shader Optimization System
Applies various optimization passes to generated shaders
"""

import re
from typing import Dict, List, Tuple
from enum import Enum


class OptimizationLevel(Enum):
    BASIC = 1
    MODERATE = 2
    AGGRESSIVE = 3


class ShaderOptimizer:
    def __init__(self):
        self.optimization_rules = self._define_optimization_rules()
        self.passes = [
            self.remove_unused_variables,
            self.simplify_expressions,
            self.merge_constants,
            self.optimize_branches,
            self.remove_redundant_operations,
            self.inline_functions,
        ]

    def _define_optimization_rules(self) -> Dict[str, str]:
        """Define common optimization rules and patterns"""
        return {
            # Common simplifications
            r'\b(\d+\.?\d*)\s*\*\s*1\.0\b': r'\1',  # multiply by 1
            r'\b(\d+\.?\d*)\s*\+\s*0\.0\b': r'\1',  # add 0
            r'\b(\d+\.?\d*)\s*-\s*0\.0\b': r'\1',  # subtract 0
            r'\b(\d+\.?\d*)\s*/\s*1\.0\b': r'\1',  # divide by 1
            r'\bsin\(acos\(([^)]+)\)\)': r'sqrt(1.0 - (\1) * (\1))',  # sin(acos(x)) = sqrt(1-x^2)
            r'\bcos\(asin\(([^)]+)\)\)': r'sqrt(1.0 - (\1) * (\1))',  # cos(asin(x)) = sqrt(1-x^2)
        }

    def optimize_shader(self, shader_code: str, level: OptimizationLevel = OptimizationLevel.MODERATE) -> str:
        """Apply optimization passes to shader code"""
        optimized_code = shader_code

        # Apply basic optimizations
        for optimization_pass in self.passes:
            optimized_code = optimization_pass(optimized_code)

        # Apply level-specific optimizations
        if level == OptimizationLevel.MODERATE:
            optimized_code = self.apply_moderate_optimizations(optimized_code)
        elif level == OptimizationLevel.AGGRESSIVE:
            optimized_code = self.apply_aggressive_optimizations(optimized_code)

        return optimized_code

    def remove_unused_variables(self, shader_code: str) -> str:
        """Remove variables that are declared but never used"""
        lines = shader_code.split('\n')
        optimized_lines = []
        
        # Find all variable declarations and their usage
        declared_vars = []
        used_vars = set()
        
        # Simple pattern to catch variable declarations and usage
        var_pattern = r'\b(?:float|int|vec[2-4]|mat[2-4]|bool)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        for line in lines:
            # Find declarations
            declarations = re.findall(var_pattern, line)
            for decl in declarations:
                if '=' in line or (';' in line and line.index(decl) < line.index(';')) or ('{' in line and line.index(decl) < line.index('{')):
                    declared_vars.append(decl)
            
            # Find usages (simplified - looks for any occurrence of variable names)
            for var in declared_vars:
                if re.search(r'\b' + re.escape(var) + r'\b', line) and line.strip() != f"float {var};" and line.strip() != f"int {var};":
                    used_vars.add(var)
        
        # Filter out unused variables
        for line in lines:
            is_unused_declaration = False
            for var in declared_vars:
                if re.search(r'\b(?:float|int|vec[2-4]|mat[2-4]|bool)\s+' + re.escape(var) + r'\b', line):
                    if var not in used_vars:
                        is_unused_declaration = True
                        break
            
            if not is_unused_declaration:
                optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)

    def simplify_expressions(self, shader_code: str) -> str:
        """Simplify mathematical expressions"""
        for pattern, replacement in self.optimization_rules.items():
            shader_code = re.sub(pattern, replacement, shader_code)
        return shader_code

    def merge_constants(self, shader_code: str) -> str:
        """Merge constant expressions into single values"""
        # Example: Convert (2.0 * 3.0) to 6.0
        # This is a simplified implementation
        lines = shader_code.split('\n')
        optimized_lines = []
        
        for line in lines:
            # Look for simple constant expressions
            # This could be expanded significantly
            line = re.sub(r'\b(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)\b', 
                         lambda m: str(float(m.group(1)) * float(m.group(2))), line)
            line = re.sub(r'\b(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\b', 
                         lambda m: str(float(m.group(1)) + float(m.group(2))), line)
            line = re.sub(r'\b(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\b', 
                         lambda m: str(float(m.group(1)) - float(m.group(2))), line)
            line = re.sub(r'\b(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\b', 
                         lambda m: str(float(m.group(1)) / float(m.group(2))), line)
            
            optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)

    def optimize_branches(self, shader_code: str) -> str:
        """Optimize conditional branches"""
        # Remove branches with constant conditions
        lines = shader_code.split('\n')
        optimized_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for if statements with constant conditions
            if_match = re.match(r'\s*if\s*\(\s*(\w+)\s*\)', line)
            if if_match:
                condition = if_match.group(1).strip()
                
                # Check if the condition is a constant
                if condition in ['true', 'false', '1', '0']:
                    if condition in ['true', '1']:
                        # Condition is always true, skip the if and go to the then block
                        i += 1
                        while i < len(lines) and not lines[i].strip().startswith('}') and not re.search(r'\belse\b', lines[i]):
                            optimized_lines.append(lines[i])
                            i += 1
                    else:  # condition is false or 0
                        # Condition is always false, skip the if and the then block
                        # Find the corresponding else or closing brace
                        brace_count = 0
                        i += 1
                        while i < len(lines):
                            if '{' in lines[i]:
                                brace_count += lines[i].count('{')
                            if '}' in lines[i]:
                                brace_count -= lines[i].count('}')
                            
                            if brace_count == 0 and '}' in lines[i]:
                                break
                            i += 1
                        i += 1  # Skip the closing brace
                    continue
            
            optimized_lines.append(line)
            i += 1
        
        return '\n'.join(optimized_lines)

    def remove_redundant_operations(self, shader_code: str) -> str:
        """Remove redundant operations like A * 1.0 or A + 0.0"""
        # Remove multiplication by 1
        shader_code = re.sub(r'\b(\w+)\s*\*\s*1\.0\b', r'\1', shader_code)
        shader_code = re.sub(r'1\.0\s*\*\s*(\w+)\b', r'\1', shader_code)
        
        # Remove addition of 0
        shader_code = re.sub(r'\b(\w+)\s*\+\s*0\.0\b', r'\1', shader_code)
        shader_code = re.sub(r'0\.0\s*\+\s*(\w+)\b', r'\1', shader_code)
        
        # Remove subtraction of 0
        shader_code = re.sub(r'\b(\w+)\s*-\s*0\.0\b', r'\1', shader_code)
        
        # Remove division by 1
        shader_code = re.sub(r'\b(\w+)\s*/\s*1\.0\b', r'\1', shader_code)
        
        # Remove assignment to self
        shader_code = re.sub(r'\s*(\w+)\s*=\s*\1\s*;', r'// Removed redundant assignment to \1', shader_code)
        
        return shader_code

    def inline_functions(self, shader_code: str) -> str:
        """Inline simple functions that are called once"""
        # For now, just return the code as a placeholder
        # This would require more sophisticated analysis
        return shader_code

    def apply_moderate_optimizations(self, shader_code: str) -> str:
        """Apply moderate-level optimizations"""
        # Remove multiple empty lines
        shader_code = re.sub(r'\n\s*\n\s*\n', '\n\n', shader_code)
        
        # Remove trailing whitespace
        lines = shader_code.split('\n')
        lines = [line.rstrip() for line in lines]
        shader_code = '\n'.join(lines)
        
        return shader_code

    def apply_aggressive_optimizations(self, shader_code: str) -> str:
        """Apply aggressive-level optimizations"""
        # Remove all comments
        shader_code = re.sub(r'//.*', '', shader_code)
        shader_code = re.sub(r'/\*.*?\*/', '', shader_code, flags=re.DOTALL)
        
        # Remove extra whitespace
        shader_code = re.sub(r'\s+', ' ', shader_code)
        
        return shader_code

    def optimize_for_gpu(self, shader_code: str) -> str:
        """Apply GPU-specific optimizations"""
        optimized = shader_code
        
        # Optimize for GPU: replace expensive functions when possible
        # Replace pow(x, 2.0) with x*x
        optimized = re.sub(r'pow\(([^,]+),\s*2\.0\s*\)', r'(\1) * (\1)', optimized)
        
        # Replace pow(x, 0.5) with sqrt(x)
        optimized = re.sub(r'pow\(([^,]+),\s*0\.5\s*\)', r'sqrt(\1)', optimized)
        
        # Replace pow(x, -1.0) with 1.0/x
        optimized = re.sub(r'pow\(([^,]+),\s*-1\.0\s*\)', r'1.0 / (\1)', optimized)
        
        return optimized

    def optimize_for_performance(self, shader_code: str) -> Dict[str, str]:
        """Perform comprehensive optimization and return both original and optimized"""
        result = {
            'original': shader_code,
            'optimized': self.optimize_shader(shader_code, OptimizationLevel.MODERATE),
            'gpu_optimized': self.optimize_for_gpu(shader_code),
            'size_reduction': None,
            'statistics': {}
        }
        
        # Calculate size reduction
        original_size = len(shader_code)
        optimized_size = len(result['optimized'])
        result['size_reduction'] = (original_size - optimized_size) / original_size * 100 if original_size > 0 else 0
        
        # Gather statistics
        result['statistics'] = {
            'original_lines': len(shader_code.split('\n')),
            'optimized_lines': len(result['optimized'].split('\n')),
            'original_size': original_size,
            'optimized_size': optimized_size,
            'size_reduction_percent': result['size_reduction']
        }
        
        return result


def demo_shader_optimization():
    """Demonstrate shader optimization capabilities"""
    print("Shader Optimization Demo")
    print("=" * 30)
    
    optimizer = ShaderOptimizer()
    
    # Example shader code with suboptimal expressions
    example_shader = '''#version 330 core

uniform vec2 resolution;
uniform float time;

in vec2 v_texCoord;
out vec4 FragColor;

float complexFunction(float x) {
    float a = x * 1.0;  // Multiply by 1, redundant
    float b = 2.0 * 3.0;  // Should be calculated as 6.0
    float c = a + 0.0;  // Add 0, redundant
    float d = pow(x, 2.0);  // Should be x * x
    float e = sqrt(1.0 - x * x);  // Might be optimized from sin(acos(x))
    
    return a + b + c + d + e;
}

void main() {
    vec2 uv = v_texCoord;
    
    if (true) {  // Always true, can be optimized
        float value = complexFunction(uv.x);
        FragColor = vec4(value, value, value, 1.0);
    }
    
    float redundant = uv.x * 1.0;  // Multiply by 1
    float unused = 42.0;  // Unused variable
}
'''
    
    print("Original shader:")
    print(example_shader)
    print("\n" + "-"*50 + "\n")
    
    # Optimize the shader
    optimization_result = optimizer.optimize_for_performance(example_shader)
    
    print("Optimized shader:")
    print(optimization_result['optimized'])
    print("\n" + "-"*50 + "\n")
    
    print("GPU-optimized shader:")
    print(optimization_result['gpu_optimized'])
    print("\n" + "-"*50 + "\n")
    
    print("Optimization Statistics:")
    stats = optimization_result['statistics']
    print(f"  Original lines: {stats['original_lines']}")
    print(f"  Optimized lines: {stats['optimized_lines']}")
    print(f"  Original size: {stats['original_size']} characters")
    print(f"  Optimized size: {stats['optimized_size']} characters")
    print(f"  Size reduction: {stats['size_reduction_percent']:.2f}%")
    
    return optimization_result


def optimize_shader_file(input_file: str, output_file: str, level: OptimizationLevel = OptimizationLevel.MODERATE):
    """Optimize a shader file and save the result"""
    with open(input_file, 'r') as f:
        shader_code = f.read()
    
    optimizer = ShaderOptimizer()
    optimized_code = optimizer.optimize_shader(shader_code, level)
    
    with open(output_file, 'w') as f:
        f.write(optimized_code)
    
    print(f"Optimized shader saved from {input_file} to {output_file}")


if __name__ == "__main__":
    demo_shader_optimization()
    
    # If there are any generated shaders, optimize them
    import os
    generated_shaders = [f for f in os.listdir('.') if f.startswith('sample_') and f.endswith('.glsl')]
    
    if generated_shaders:
        optimizer = ShaderOptimizer()
        for shader_file in generated_shaders:
            optimized_file = shader_file.replace('.glsl', '_optimized.glsl')
            try:
                optimize_shader_file(shader_file, optimized_file)
            except:
                print(f"Could not optimize {shader_file}")