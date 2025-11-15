"""
Fuzz Testing System for Shader Modules
Part of SuperShader Project - Phase 8: Testing and Quality Assurance

This module implements fuzz testing with random input generation,
automated crash detection and reporting, edge case testing for
numerical stability, and validation for generated shader code correctness.
"""

import random
import string
import math
import re
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import json
import traceback


@dataclass
class FuzzTestResult:
    """Represents the result of a fuzz test"""
    test_id: str
    shader_code: str
    input_params: Dict[str, Any]
    passed: bool
    error_type: Optional[str]
    error_message: Optional[str]
    execution_time: float
    crash_detected: bool


class FuzzInputGenerator:
    """
    System for generating random inputs for shader testing
    """
    
    def __init__(self):
        self.value_generators = {
            'float': self._generate_float,
            'int': self._generate_int,
            'vec2': self._generate_vec2,
            'vec3': self._generate_vec3,
            'vec4': self._generate_vec4,
            'mat4': self._generate_mat4,
            'bool': self._generate_bool,
            'string': self._generate_string
        }
    
    def _generate_float(self) -> float:
        """Generate a random float value"""
        # Include edge cases: very small, very large, infinity, NaN
        edge_cases = [float('inf'), float('-inf'), float('nan')]
        if random.random() < 0.05:  # 5% chance of edge case
            return random.choice(edge_cases)
        
        # Normal cases: random values in various ranges
        ranges = [
            (0.0, 1.0),      # Standard normalized values
            (-1.0, 1.0),     # Standard range
            (0.0, 100.0),    # Larger values
            (-100.0, 100.0), # Larger signed values
            (0.0001, 0.001), # Very small values
        ]
        
        r = random.choice(ranges)
        return random.uniform(r[0], r[1])
    
    def _generate_int(self) -> int:
        """Generate a random int value"""
        # Include edge cases for integers
        edge_cases = [0, 1, -1, 2147483647, -2147483648, 255, 65535]
        if random.random() < 0.05:  # 5% chance of edge case
            return random.choice(edge_cases)
        
        # Normal cases
        ranges = [
            (0, 100),
            (-50, 50),
            (0, 1000),
            (0, 16),  # For array indices
        ]
        
        r = random.choice(ranges)
        return random.randint(r[0], r[1])
    
    def _generate_vec2(self) -> List[float]:
        """Generate a random 2D vector"""
        return [self._generate_float(), self._generate_float()]
    
    def _generate_vec3(self) -> List[float]:
        """Generate a random 3D vector"""
        return [self._generate_float(), self._generate_float(), self._generate_float()]
    
    def _generate_vec4(self) -> List[float]:
        """Generate a random 4D vector"""
        return [self._generate_float(), self._generate_float(), 
                self._generate_float(), self._generate_float()]
    
    def _generate_mat4(self) -> List[List[float]]:
        """Generate a random 4x4 matrix"""
        matrix = []
        for _ in range(4):
            row = [self._generate_float() for _ in range(4)]
            matrix.append(row)
        return matrix
    
    def _generate_bool(self) -> bool:
        """Generate a random boolean"""
        return random.choice([True, False])
    
    def _generate_string(self) -> str:
        """Generate a random string"""
        length = random.randint(1, 20)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def generate_input_set(self, param_types: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate a set of random inputs based on specified parameter types
        """
        inputs = {}
        for param_name, param_type in param_types.items():
            if param_type in self.value_generators:
                inputs[param_name] = self.value_generators[param_type]()
            else:
                # Default to float for unknown types
                inputs[param_name] = self._generate_float()
        
        return inputs
    
    def generate_multiple_input_sets(self, param_types: Dict[str, str], count: int) -> List[Dict[str, Any]]:
        """
        Generate multiple sets of random inputs
        """
        return [self.generate_input_set(param_types) for _ in range(count)]


class ShaderCodeValidator:
    """
    System for validating generated shader code correctness
    """
    
    def __init__(self):
        self.compilation_errors = []
        self.semantic_errors = []
    
    def validate_shader_syntax(self, shader_code: str) -> Tuple[bool, List[str]]:
        """
        Validate the syntax of a shader code (simulated)
        """
        errors = []
        
        # Check for common syntax issues
        lines = shader_code.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for unbalanced brackets
            if line.count('{') != line.count('}'):
                errors.append(f"Line {i}: Unbalanced braces")
            
            if line.count('(') != line.count(')'):
                errors.append(f"Line {i}: Unbalanced parentheses")
            
            if line.count('[') != line.count(']'):
                errors.append(f"Line {i}: Unbalanced brackets")
            
            # Check for incomplete statements (ending with semicolon)
            line_stripped = line.strip()
            if (line_stripped and 
                not line_stripped.endswith(';') and 
                not line_stripped.endswith('{') and
                not line_stripped.endswith('}') and
                not line_stripped.startswith('//') and
                not line_stripped.startswith('/*') and
                not line_stripped.endswith('*/')):
                errors.append(f"Line {i}: Statement not terminated with semicolon")
        
        # Check for undefined variables (basic check)
        if re.search(r'\b(uniform|attribute|varying|in|out)\s+\w+\s+(\w+)\s*;', shader_code):
            # Check if variables are used before being assigned
            pass  # Would need more sophisticated parsing
        
        return len(errors) == 0, errors
    
    def validate_shader_semantics(self, shader_code: str) -> Tuple[bool, List[str]]:
        """
        Validate the semantics of a shader code (simulated)
        """
        errors = []
        
        # Check for proper main function
        if 'void main()' not in shader_code and 'main()' not in shader_code:
            errors.append("No main function found")
        
        # Check for GLSL version directive
        if not re.search(r'#version\s+\d+', shader_code):
            errors.append("No version directive found")
        
        # Check for common semantic errors
        # Division by zero potential
        if re.search(r'/\s*[0\s]*[,);]', shader_code):
            errors.append("Potential division by zero")
        
        # Check for array out of bounds (simple patterns)
        # This is a very basic check - real implementation would be much more complex
        array_accesses = re.findall(r'\[\s*(\w+)\s*\]', shader_code)
        for var in array_accesses:
            # Check if the variable is defined and validated
            pass  # Would need more complex analysis
        
        return len(errors) == 0, errors
    
    def validate_shader_for_errors(self, shader_code: str) -> Dict[str, Any]:
        """
        Validate shader code for both syntax and semantic errors
        """
        syntax_valid, syntax_errors = self.validate_shader_syntax(shader_code)
        semantic_valid, semantic_errors = self.validate_shader_semantics(shader_code)
        
        return {
            'syntax_valid': syntax_valid,
            'semantic_valid': semantic_valid,
            'overall_valid': syntax_valid and semantic_valid,
            'syntax_errors': syntax_errors,
            'semantic_errors': semantic_errors,
            'total_errors': len(syntax_errors) + len(semantic_errors)
        }


class NumericalStabilityTester:
    """
    System for testing numerical stability and edge cases in shaders
    """
    
    def __init__(self):
        self.stability_results = []
    
    def test_very_large_values(self, shader_func: Callable, 
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """Test shader with very large input values"""
        test_params = params.copy()
        
        # Replace numerical values with very large ones
        for key, value in test_params.items():
            if isinstance(value, (int, float)):
                test_params[key] = value * 1e10  # Make it very large
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        value[i] = v * 1e10
        
        try:
            # In a real system, this would execute the shader
            # For simulation, we'll just process the parameters
            result = self._simulate_shader_execution(shader_func, test_params)
            return {"passed": True, "result": result}
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def test_very_small_values(self, shader_func: Callable, 
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """Test shader with very small input values"""
        test_params = params.copy()
        
        # Replace numerical values with very small ones
        for key, value in test_params.items():
            if isinstance(value, (int, float)):
                test_params[key] = value * 1e-10  # Make it very small
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        value[i] = v * 1e-10
        
        try:
            # In a real system, this would execute the shader
            result = self._simulate_shader_execution(shader_func, test_params)
            return {"passed": True, "result": result}
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def test_extreme_conditions(self, shader_func: Callable, 
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """Test shader with various extreme conditions"""
        conditions = [
            # Division edge cases
            {"divisor": 0.0},  # Potential division by zero
            {"divisor": 1e-20},  # Very small divisor
            {"divisor": 1e20},  # Very large divisor
            
            # Trigonometric edge cases
            {"angle": float('inf')},  # Infinite angle
            {"angle": float('nan')},  # NaN angle
        ]
        
        results = []
        for condition in conditions:
            test_params = params.copy()
            test_params.update(condition)
            
            try:
                # In a real system, this would execute the shader
                result = self._simulate_shader_execution(shader_func, test_params)
                results.append({"condition": condition, "passed": True, "result": result})
            except Exception as e:
                results.append({"condition": condition, "passed": False, "error": str(e)})
        
        return results
    
    def _simulate_shader_execution(self, shader_func: Callable, params: Dict[str, Any]) -> Any:
        """Simulate shader execution for testing"""
        # In a real system, this would actually run the shader code
        # For this simulation, we'll just perform some operations on the params
        import math
        
        # Simulate some mathematical operations that might reveal numerical instability
        if 'value' in params:
            val = params['value']
            if isinstance(val, (int, float)):
                # Operations that might cause overflow/underflow
                result = math.atan(val) + math.sin(val) * math.cos(val)
                if isinstance(val, float) and math.isinf(val):
                    result = float('inf') if val > 0 else float('-inf')
                elif isinstance(val, float) and math.isnan(val):
                    result = float('nan')
                return result
        return params


class AutomatedCrashDetection:
    """
    System for automated crash detection and reporting
    """
    
    def __init__(self):
        self.crash_reports = []
        self.error_patterns = self._define_error_patterns()
    
    def _define_error_patterns(self) -> List[re.Pattern]:
        """Define patterns for detecting common shader errors"""
        return [
            re.compile(r'access.*out of bounds', re.IGNORECASE),
            re.compile(r'buffer.*overflow', re.IGNORECASE),
            re.compile(r'stack.*overflow', re.IGNORECASE),
            re.compile(r'division by zero', re.IGNORECASE),
            re.compile(r'floating point exception', re.IGNORECASE),
            re.compile(r'segmentation fault', re.IGNORECASE),
            re.compile(r'null pointer', re.IGNORECASE),
            re.compile(r'access violation', re.IGNORECASE),
        ]
    
    def detect_crash(self, shader_code: str, execution_result: Dict[str, Any]) -> bool:
        """Detect if a crash occurred during shader execution"""
        # Check for crash indicators in execution result
        if execution_result.get('crashed', False):
            return True
        
        # Check for crash indicators in error message
        error_msg = execution_result.get('error_message', '')
        for pattern in self.error_patterns:
            if pattern.search(error_msg):
                return True
        
        # Check for specific crash conditions
        if execution_result.get('exit_code') in [11, 139, 134]:  # SIGSEGV, SIGKILL, SIGABRT
            return True
        
        return False
    
    def generate_crash_report(self, test_id: str, shader_code: str, 
                            execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a crash report for a failed test"""
        report = {
            'test_id': test_id,
            'timestamp': execution_result.get('timestamp', time.time()),
            'shader_code_preview': shader_code[:200] + "..." if len(shader_code) > 200 else shader_code,
            'error_message': execution_result.get('error_message', ''),
            'stack_trace': execution_result.get('stack_trace', ''),
            'input_parameters': execution_result.get('input_params', {}),
            'detected_crash': self.detect_crash(shader_code, execution_result)
        }
        
        self.crash_reports.append(report)
        return report


class FuzzTestingSystem:
    """
    Main system for fuzz testing shader modules
    """
    
    def __init__(self):
        self.input_generator = FuzzInputGenerator()
        self.validator = ShaderCodeValidator()
        self.stability_tester = NumericalStabilityTester()
        self.crash_detector = AutomatedCrashDetection()
        self.test_results: List[FuzzTestResult] = []
    
    def run_fuzz_test(self, shader_code: str, param_types: Dict[str, str], 
                     num_tests: int = 1000) -> List[FuzzTestResult]:
        """
        Run fuzz tests on a shader with random inputs
        """
        print(f"Running {num_tests} fuzz tests...")
        
        results = []
        
        # First validate the shader code
        validation_result = self.validator.validate_shader_for_errors(shader_code)
        if not validation_result['overall_valid']:
            print(f"Shader validation failed: {validation_result['total_errors']} errors found")
            # We can continue with fuzz testing, but note the validation issues
        
        for i in range(num_tests):
            # Generate random inputs
            inputs = self.input_generator.generate_input_set(param_types)
            
            # In a real system, we'd execute the shader with these inputs
            # For simulation, we'll just check for potential issues with the inputs
            test_result = self._simulate_fuzz_test(shader_code, inputs)
            results.append(test_result)
            
            if i % 200 == 0:  # Progress update
                print(f"Completed {i}/{num_tests} fuzz tests")
        
        # Store results
        self.test_results.extend(results)
        return results
    
    def _simulate_fuzz_test(self, shader_code: str, inputs: Dict[str, Any]) -> FuzzTestResult:
        """
        Simulate a fuzz test by checking for potential issues in the shader and inputs
        """
        import time
        start_time = time.time()
        
        # Simulate potential crash conditions
        crash_detected = False
        error_type = None
        error_message = None
        
        # Check for various crash conditions
        try:
            # Check for division by zero
            for key, value in inputs.items():
                if isinstance(value, (int, float)) and value == 0:
                    if f'/{key}' in shader_code or f'/{key} ' in shader_code:
                        error_type = "DivisionByZero"
                        error_message = f"Potential division by zero with parameter {key}"
                        crash_detected = True
                        break
            
            # Check for very large values causing overflow
            for key, value in inputs.items():
                if isinstance(value, (int, float)) and abs(value) > 1e100:
                    error_type = "NumericalOverflow"
                    error_message = f"Very large value in parameter {key} may cause overflow"
                    break
            
            # Check for invalid operations
            if 'sqrt(' in shader_code:
                for key, value in inputs.items():
                    if isinstance(value, (int, float)) and value < 0:
                        error_type = "InvalidOperation"
                        error_message = f"Negative value in parameter {key} used with sqrt"
                        break
        
        except Exception as e:
            crash_detected = True
            error_type = "ExecutionError"
            error_message = str(e)
        
        execution_time = time.time() - start_time
        
        return FuzzTestResult(
            test_id=f"fuzz_{len(self.test_results)}",
            shader_code=shader_code,
            input_params=inputs,
            passed=not crash_detected,
            error_type=error_type,
            error_message=error_message,
            execution_time=execution_time,
            crash_detected=crash_detected
        )
    
    def run_numerical_stability_tests(self, shader_func: Callable, 
                                    base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run specific numerical stability tests on a shader function
        """
        large_test = self.stability_tester.test_very_large_values(shader_func, base_params)
        small_test = self.stability_tester.test_very_small_values(shader_func, base_params)
        extreme_test = self.stability_tester.test_extreme_conditions(shader_func, base_params)
        
        return {
            'very_large_values': large_test,
            'very_small_values': small_test,
            'extreme_conditions': extreme_test
        }
    
    def generate_fuzz_report(self) -> Dict[str, Any]:
        """Generate a report on fuzz testing results"""
        if not self.test_results:
            return {"message": "No fuzz test results available"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        crashed_tests = sum(1 for r in self.test_results if r.crash_detected)
        failed_tests = total_tests - passed_tests
        
        # Identify common error patterns
        error_types = {}
        for result in self.test_results:
            if result.error_type:
                error_types[result.error_type] = error_types.get(result.error_type, 0) + 1
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'crashed_tests': crashed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'error_distribution': error_types,
            'crash_rate': crashed_tests / total_tests if total_tests > 0 else 0,
            'average_execution_time': sum(r.execution_time for r in self.test_results) / total_tests,
            'most_common_errors': sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def validate_generated_shaders(self, shader_codes: List[str]) -> List[Dict[str, Any]]:
        """Validate multiple generated shaders for correctness"""
        validation_results = []
        
        for shader_code in shader_codes:
            result = self.validator.validate_shader_for_errors(shader_code)
            result['shader_preview'] = shader_code[:100] + "..." if len(shader_code) > 100 else shader_code
            validation_results.append(result)
        
        return validation_results


def simulate_shader_function(params: Dict[str, Any]) -> Any:
    """Simulate a shader function for testing purposes"""
    # This simulates what a shader might compute with the given parameters
    # In a real system, this would be the actual shader execution
    import math
    
    # Simulate potential mathematical operations that could cause issues
    result = {}
    
    # Process different parameter types
    for key, value in params.items():
        if isinstance(value, (int, float)):
            # Operations that might cause numerical issues
            try:
                if value != 0:
                    result[f"{key}_inv"] = 1.0 / value
                if value >= 0:
                    result[f"{key}_sqrt"] = math.sqrt(value)
                result[f"{key}_sin"] = math.sin(value)
                result[f"{key}_cos"] = math.cos(value)
            except Exception:
                result[f"{key}_error"] = "Computation error"
        elif isinstance(value, list):
            # Process vectors
            try:
                if len(value) >= 3:
                    # Compute vector length
                    length = math.sqrt(sum(x*x for x in value))
                    if length != 0:
                        # Normalize
                        result[f"{key}_normalized"] = [x/length for x in value]
            except Exception:
                result[f"{key}_error"] = "Vector computation error"
    
    return result


def main():
    """
    Example usage of the Fuzz Testing System
    """
    print("Fuzz Testing System for Shader Modules")
    print("Part of SuperShader Project - Phase 8")
    
    # Create the fuzz testing system
    fuzz_system = FuzzTestingSystem()
    
    # Example shader code to test
    test_shader = """
#version 330 core
uniform float time;
uniform vec3 position;
uniform vec2 resolution;
out vec4 FragColor;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    uv = uv * 2.0 - 1.0;
    float r = length(uv);
    float angle = atan(uv.y, uv.x);
    vec3 color = 0.5 + 0.5 * cos(time + r * 5.0 + vec3(0, 2, 4));
    FragColor = vec4(color, 1.0);
}
    """
    
    # Define parameter types for input generation
    param_types = {
        'time': 'float',
        'position_x': 'float',
        'position_y': 'float', 
        'position_z': 'float',
        'resolution_x': 'float',
        'resolution_y': 'float',
        'multiplier': 'float'
    }
    
    # Run fuzz tests
    print("\n--- Running Fuzz Tests ---")
    fuzz_results = fuzz_system.run_fuzz_test(test_shader, param_types, num_tests=100)
    
    # Generate fuzz report
    fuzz_report = fuzz_system.generate_fuzz_report()
    print(f"Fuzz testing completed: {fuzz_report['success_rate']:.2%} success rate")
    print(f"Crash rate: {fuzz_report['crash_rate']:.2%}")
    print(f"Most common errors: {fuzz_report['most_common_errors']}")
    
    # Run numerical stability tests
    print("\n--- Running Numerical Stability Tests ---")
    base_params = {'time': 1.0, 'resolution_x': 800.0, 'resolution_y': 600.0}
    stability_results = fuzz_system.run_numerical_stability_tests(simulate_shader_function, base_params)
    
    print(f"Large values test passed: {stability_results['very_large_values']['passed']}")
    print(f"Small values test passed: {stability_results['very_small_values']['passed']}")
    
    # Validate some generated shaders
    print("\n--- Validating Generated Shaders ---")
    test_shaders = [
        """
        #version 330 core
        void main() { gl_Position = vec4(0.0); }
        """,
        """
        #version 330 core
        in vec3 position;
        void main() { gl_Position = vec4(position, 1.0); }
        """,
        """
        #version 330 core
        in vec3 position;
        out vec4 color;
        void main() {
            gl_Position = vec4(position, 1.0);
            color = vec4(1.0, 0.0, 0.0, 1.0);
        """,  # Missing closing brace to test error detection
    ]
    
    validation_results = fuzz_system.validate_generated_shaders(test_shaders)
    
    for i, result in enumerate(validation_results):
        print(f"Shader {i+1} valid: {result['overall_valid']}, errors: {result['total_errors']}")


if __name__ == "__main__":
    import time  # Need to import time for the example
    main()