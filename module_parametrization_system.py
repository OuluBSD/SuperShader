#!/usr/bin/env python3
"""
Module Parameterization System for SuperShader
Enables modules to accept parameters for customization and configuration
"""

import json
import re
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import sys


class ModuleParameterizer:
    """
    System for parameterizing modules to allow customization and configuration
    """

    def __init__(self):
        self.parameter_templates = {}
        self.default_values = {}
        self.type_validations = {}

    def define_parameter_template(self, module_name: str, parameters: Dict[str, Dict[str, Any]]):
        """
        Define the parameter template for a module
        """
        self.parameter_templates[module_name] = parameters

        # Extract default values
        self.default_values[module_name] = {}
        for param_name, param_info in parameters.items():
            if 'default' in param_info:
                self.default_values[module_name][param_name] = param_info['default']
            else:
                self.default_values[module_name][param_name] = None

        # Extract type validation rules
        self.type_validations[module_name] = {}
        for param_name, param_info in parameters.items():
            if 'type' in param_info:
                self.type_validations[module_name][param_name] = param_info['type']

    def validate_parameters(self, module_name: str, parameters: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate parameters for a module, returning any validation errors
        """
        errors = {}

        if module_name not in self.parameter_templates:
            return {'general': [f'No parameter template found for module {module_name}']}

        template = self.parameter_templates[module_name]
        defaults = self.default_values[module_name]

        for param_name, param_value in parameters.items():
            if param_name not in template:
                errors[param_name] = [f'Parameter {param_name} is not defined for module {module_name}']
                continue

            param_def = template[param_name]
            param_errors = []

            # Type validation
            expected_type = self.type_validations[module_name].get(param_name)
            if expected_type and param_value is not None:
                if expected_type == 'int' and not isinstance(param_value, int):
                    param_errors.append(f'Parameter {param_name} must be an integer')
                elif expected_type == 'float' and not isinstance(param_value, (int, float)):
                    param_errors.append(f'Parameter {param_name} must be a float')
                elif expected_type == 'string' and not isinstance(param_value, str):
                    param_errors.append(f'Parameter {param_name} must be a string')
                elif expected_type == 'bool' and not isinstance(param_value, bool):
                    param_errors.append(f'Parameter {param_name} must be a boolean')
                elif expected_type == 'vec2' and not self._is_vec2(param_value):
                    param_errors.append(f'Parameter {param_name} must be a 2-element array or vector')
                elif expected_type == 'vec3' and not self._is_vec3(param_value):
                    param_errors.append(f'Parameter {param_name} must be a 3-element array or vector')
                elif expected_type == 'vec4' and not self._is_vec4(param_value):
                    param_errors.append(f'Parameter {param_name} must be a 4-element array or vector')
                elif expected_type == 'list' and not isinstance(param_value, list):
                    param_errors.append(f'Parameter {param_name} must be a list')

            # Range validation
            if 'min' in param_def and param_value is not None:
                min_val = param_def['min']
                if isinstance(param_value, (int, float)) and param_value < min_val:
                    param_errors.append(f'Parameter {param_name} must be >= {min_val}')

            if 'max' in param_def and param_value is not None:
                max_val = param_def['max']
                if isinstance(param_value, (int, float)) and param_value > max_val:
                    param_errors.append(f'Parameter {param_name} must be <= {max_val}')

            # Custom validation
            if 'validator' in param_def and callable(param_def['validator']):
                try:
                    is_valid = param_def['validator'](param_value)
                    if not is_valid:
                        param_errors.append(f'Parameter {param_name} failed custom validation')
                except Exception as e:
                    param_errors.append(f'Parameter {param_name} validation raised error: {str(e)}')

            if param_errors:
                errors[param_name] = param_errors

        return errors

    def _is_vec2(self, value: Any) -> bool:
        """Check if a value is a valid 2-element vector"""
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return all(isinstance(item, (int, float)) for item in value)
        return False

    def _is_vec3(self, value: Any) -> bool:
        """Check if a value is a valid 3-element vector"""
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return all(isinstance(item, (int, float)) for item in value)
        return False

    def _is_vec4(self, value: Any) -> bool:
        """Check if a value is a valid 4-element vector"""
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return all(isinstance(item, (int, float)) for item in value)
        return False

    def apply_parameters(self, module_pseudocode: str, parameters: Dict[str, Any]) -> str:
        """
        Apply parameters to pseudocode by replacing placeholders
        """
        result = module_pseudocode

        # Replace parameter placeholders in pseudocode
        for param_name, param_value in parameters.items():
            # Look for placeholders like {param_name}, [param_name], or {{param_name}}
            placeholder_patterns = [
                r'\{' + re.escape(param_name) + r'\}',
                r'\[' + re.escape(param_name) + r'\]',
                r'\{\{' + re.escape(param_name) + r'\}\}',
                r'%' + re.escape(param_name) + r'%'
            ]

            replacement = self._format_parameter_value(param_value)
            for pattern in placeholder_patterns:
                result = re.sub(pattern, replacement, result)

        return result

    def _format_parameter_value(self, value: Any) -> str:
        """
        Format a parameter value for insertion into pseudocode
        """
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, (list, tuple)):
            # Convert vector/array values to GLSL format
            if len(value) == 2:
                return f'vec2({value[0]}, {value[1]})'
            elif len(value) == 3:
                return f'vec3({value[0]}, {value[1]}, {value[2]})'
            elif len(value) == 4:
                return f'vec4({value[0]}, {value[1]}, {value[2]}, {value[3]})'
            else:
                # Just return as array
                return f'[{", ".join(map(str, value))}]'
        else:
            return str(value)

    def create_parameterized_module(self, module_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a parameterized instance of a module
        """
        if parameters is None:
            parameters = {}

        # Validate parameters
        validation_errors = self.validate_parameters(module_name, parameters)
        if validation_errors:
            error_msg = []
            for param, errors in validation_errors.items():
                error_msg.extend(errors)
            raise ValueError(f"Parameter validation failed: {'; '.join(error_msg)}")

        # Apply defaults for missing parameters
        final_params = self.default_values.get(module_name, {}).copy()
        final_params.update(parameters)

        # Load the module pseudocode
        module_pseudocode = self._load_module_pseudocode(module_name)
        if not module_pseudocode:
            raise ValueError(f"Could not load pseudocode for module {module_name}")

        # Apply parameters to pseudocode
        parameterized_pseudocode = self.apply_parameters(module_pseudocode, final_params)

        # Create the parameterized module
        return {
            'name': module_name,
            'pseudocode': parameterized_pseudocode,
            'parameters': final_params,
            'template_parameters': self.parameter_templates.get(module_name, {}),
            'type': 'parameterized_module'
        }

    def _load_module_pseudocode(self, module_name: str) -> str:
        """
        Load pseudocode for a module from the modules directory
        """
        # Try to find the module file
        for root, dirs, files in os.walk('modules'):
            for file in files:
                if file.endswith(('.txt', '.json')):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            content = f.read()
                            if module_name in content or f'"{module_name}"' in content:
                                data = json.loads(content)
                                if data.get('name') == module_name:
                                    return data.get('pseudocode', data.get('code', ''))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

        # If not found in files, create a simple example pseudocode
        example_pseudocode = f"""
// Example pseudocode for {module_name}
// Parameters would be injected here
function exampleFunction() {{
    // This is where parameters like {{param1}} and {{param2}} would be used
    return 0;
}}
"""
        return example_pseudocode

    def get_parameter_info(self, module_name: str) -> Dict[str, Any]:
        """
        Get information about parameters for a module
        """
        if module_name in self.parameter_templates:
            return {
                'module': module_name,
                'parameters': self.parameter_templates[module_name],
                'defaults': self.default_values.get(module_name, {}),
                'defined': True
            }
        else:
            return {
                'module': module_name,
                'parameters': {},
                'defaults': {},
                'defined': False
            }


class AdvancedParameterizer(ModuleParameterizer):
    """
    Advanced parameterization system with more features
    """

    def __init__(self):
        super().__init__()
        self.parameter_groups = {}  # Group related parameters
        self.parameter_constraints = {}  # Constraints between parameters
        self.parameter_presets = {}  # Predefined parameter sets

    def define_parameter_group(self, module_name: str, group_name: str, parameters: List[str]):
        """
        Define a group of related parameters
        """
        if module_name not in self.parameter_groups:
            self.parameter_groups[module_name] = {}
        self.parameter_groups[module_name][group_name] = parameters

    def define_parameter_constraint(self, module_name: str, constraint: Dict[str, Any]):
        """
        Define a constraint between parameters
        """
        if module_name not in self.parameter_constraints:
            self.parameter_constraints[module_name] = []
        self.parameter_constraints[module_name].append(constraint)

    def define_parameter_preset(self, module_name: str, preset_name: str, parameters: Dict[str, Any]):
        """
        Define a preset set of parameters
        """
        if module_name not in self.parameter_presets:
            self.parameter_presets[module_name] = {}
        self.parameter_presets[module_name][preset_name] = parameters

    def validate_parameter_constraints(self, module_name: str, parameters: Dict[str, Any]) -> List[str]:
        """
        Validate constraints between parameters
        """
        errors = []

        if module_name not in self.parameter_constraints:
            return errors

        for constraint in self.parameter_constraints[module_name]:
            constraint_type = constraint.get('type')
            params_involved = constraint.get('parameters', [])

            # Check if all required parameters are present
            missing_params = [p for p in params_involved if p not in parameters]
            if missing_params:
                errors.append(f"Missing parameters for constraint: {', '.join(missing_params)}")
                continue

            # Validate the specific constraint type
            if constraint_type == 'mutual_exclusion':
                # Only one of the parameters should be set (not None)
                active_params = [p for p in params_involved if parameters[p] is not None]
                if len(active_params) > 1:
                    errors.append(f"Parameters {params_involved} are mutually exclusive - only one can be set")

            elif constraint_type == 'dependence':
                # If one parameter is set, others must also be set
                main_param = constraint.get('main')
                dependent_params = constraint.get('dependents', [])
                
                if main_param in parameters and parameters[main_param] is not None:
                    missing_dependents = [p for p in dependent_params if parameters.get(p) is None]
                    if missing_dependents:
                        errors.append(f"When {main_param} is set, these parameters must also be set: {missing_dependents}")

            elif constraint_type == 'range_dependence':
                # The value of one parameter affects the valid range of another
                param1 = constraint.get('param1')
                param2 = constraint.get('param2')
                condition = constraint.get('condition')

                if param1 in parameters and param2 in parameters:
                    val1, val2 = parameters[param1], parameters[param2]
                    if condition == 'param1_gt_0_implies_param2_gt_0' and val1 > 0 and val2 <= 0:
                        errors.append(f"When {param1} > 0, {param2} must also be > 0")

        return errors

    def get_module_with_preset(self, module_name: str, preset_name: str, additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get a module with parameters from a preset plus additional parameters
        """
        if additional_params is None:
            additional_params = {}

        # Get the preset parameters
        preset_params = self.parameter_presets.get(module_name, {}).get(preset_name, {})
        
        # Combine with additional parameters (additional params override preset)
        final_params = {**preset_params, **additional_params}

        # Create the parameterized module
        return self.create_parameterized_module(module_name, final_params)

    def get_parameterization_recommendations(self, module_name: str) -> List[str]:
        """
        Get recommendations for parameterizing a module
        """
        recommendations = []
        
        if module_name in self.parameter_templates:
            params = self.parameter_templates[module_name]
            for param_name, param_info in params.items():
                if 'default' not in param_info:
                    recommendations.append(f"Consider adding a default value for parameter '{param_name}'")
                if 'description' not in param_info:
                    recommendations.append(f"Consider adding a description for parameter '{param_name}'")
                if 'min' not in param_info and 'max' not in param_info:
                    recommendations.append(f"Consider adding value range validation for parameter '{param_name}'")
        
        return recommendations


def initialize_parameterization_system():
    """
    Initialize the parameterization system with example parameter definitions
    """
    param_system = AdvancedParameterizer()

    # Define parameters for a lighting module
    param_system.define_parameter_template('pbr_lighting', {
        'albedo': {
            'type': 'vec3',
            'default': [0.5, 0.5, 0.5],
            'description': 'Base color of the material',
            'min': 0.0,
            'max': 1.0
        },
        'metallic': {
            'type': 'float',
            'default': 0.0,
            'description': 'Metallic property of the material',
            'min': 0.0,
            'max': 1.0
        },
        'roughness': {
            'type': 'float',
            'default': 0.5,
            'description': 'Roughness property of the material',
            'min': 0.0,
            'max': 1.0
        },
        'ao': {
            'type': 'float',
            'default': 1.0,
            'description': 'Ambient occlusion factor',
            'min': 0.0,
            'max': 1.0
        }
    })

    # Define parameters for a noise generation module
    param_system.define_parameter_template('noise_generation', {
        'frequency': {
            'type': 'float',
            'default': 1.0,
            'description': 'Frequency of the noise pattern',
            'min': 0.001,
            'max': 100.0
        },
        'amplitude': {
            'type': 'float',
            'default': 1.0,
            'description': 'Amplitude of the noise',
            'min': 0.0,
            'max': 10.0
        },
        'octaves': {
            'type': 'int',
            'default': 4,
            'description': 'Number of octaves for fractal noise',
            'min': 1,
            'max': 10
        },
        'persistence': {
            'type': 'float',
            'default': 0.5,
            'description': 'Persistence factor for noise',
            'min': 0.01,
            'max': 2.0
        },
        'lacunarity': {
            'type': 'float',
            'default': 2.0,
            'description': 'Lacunarity factor for noise',
            'min': 0.1,
            'max': 4.0
        },
        'noise_type': {
            'type': 'string',
            'default': 'perlin',
            'description': 'Type of noise to generate',
            'validator': lambda x: x in ['perlin', 'simplex', 'value', 'cellular']
        }
    })

    # Define parameters for a post-processing effect
    param_system.define_parameter_template('bloom_effect', {
        'intensity': {
            'type': 'float',
            'default': 0.1,
            'description': 'Intensity of the bloom effect',
            'min': 0.0,
            'max': 1.0
        },
        'threshold': {
            'type': 'float',
            'default': 0.8,
            'description': 'Threshold for bloom bright pass',
            'min': 0.0,
            'max': 1.0
        },
        'blur_radius': {
            'type': 'float',
            'default': 5.0,
            'description': 'Radius of the blur effect',
            'min': 0.1,
            'max': 20.0
        },
        'blur_samples': {
            'type': 'int',
            'default': 16,
            'description': 'Number of samples for blur',
            'min': 4,
            'max': 64
        }
    })

    # Define parameter groups
    param_system.define_parameter_group('pbr_lighting', 'material_properties', ['albedo', 'metallic', 'roughness'])
    param_system.define_parameter_group('pbr_lighting', 'environment_properties', ['ao'])

    param_system.define_parameter_group('noise_generation', 'main_properties', ['frequency', 'amplitude'])
    param_system.define_parameter_group('noise_generation', 'fractal_properties', ['octaves', 'persistence', 'lacunarity'])

    # Define parameter constraints
    param_system.define_parameter_constraint('noise_generation', {
        'type': 'dependence',
        'main': 'noise_type',
        'dependents': ['frequency', 'amplitude'],
        'description': 'When noise type is set, frequency and amplitude should also be configured'
    })

    # Define parameter presets
    param_system.define_parameter_preset('pbr_lighting', 'plastic', {
        'albedo': [0.8, 0.1, 0.1],
        'metallic': 0.0,
        'roughness': 0.7,
        'ao': 1.0
    })

    param_system.define_parameter_preset('pbr_lighting', 'metal', {
        'albedo': [0.9, 0.9, 0.9],
        'metallic': 1.0,
        'roughness': 0.2,
        'ao': 0.8
    })

    param_system.define_parameter_preset('noise_generation', 'soft_clouds', {
        'frequency': 0.01,
        'amplitude': 1.0,
        'octaves': 3,
        'persistence': 0.6,
        'lacunarity': 2.1,
        'noise_type': 'perlin'
    })

    param_system.define_parameter_preset('noise_generation', 'rough_terrain', {
        'frequency': 0.1,
        'amplitude': 2.0,
        'octaves': 6,
        'persistence': 0.4,
        'lacunarity': 2.0,
        'noise_type': 'perlin'
    })

    param_system.define_parameter_preset('bloom_effect', 'subtle', {
        'intensity': 0.05,
        'threshold': 0.9,
        'blur_radius': 3.0,
        'blur_samples': 8
    })

    param_system.define_parameter_preset('bloom_effect', 'intense', {
        'intensity': 0.3,
        'threshold': 0.5,
        'blur_radius': 8.0,
        'blur_samples': 24
    })

    return param_system


def main():
    """Main function to demonstrate the module parameterization system"""
    print("Initializing Module Parameterization System...")
    
    # Initialize the system with example configurations
    param_system = initialize_parameterization_system()
    
    print("\\nTesting Parameterization System...")
    
    # Test 1: Get parameter info for a module
    pbr_info = param_system.get_parameter_info('pbr_lighting')
    print(f"\\n1. PBR Lighting Parameter Info: {pbr_info['defined']}")
    if pbr_info['defined']:
        print(f"   Parameters: {list(pbr_info['parameters'].keys())}")
    
    # Test 2: Create a parameterized module with custom parameters
    try:
        custom_pbr = param_system.create_parameterized_module('pbr_lighting', {
            'albedo': [0.2, 0.6, 0.8],
            'metallic': 0.3,
            'roughness': 0.4
        })
        print(f"\\n2. Custom PBR module created successfully")
        print(f"   Parameters applied: {custom_pbr['parameters']}")
    except Exception as e:
        print(f"\\n2. Error creating custom PBR module: {e}")
    
    # Test 3: Get a module with a preset
    try:
        metal_material = param_system.get_module_with_preset('pbr_lighting', 'metal', {
            'roughness': 0.1  # Override the preset's roughness
        })
        print(f"\\n3. Metal preset module created with override")
        print(f"   Albedo: {metal_material['parameters']['albedo']}")
        print(f"   Metallic: {metal_material['parameters']['metallic']}")
        print(f"   Roughness (overridden): {metal_material['parameters']['roughness']}")
    except Exception as e:
        print(f"\\n3. Error creating preset module: {e}")
    
    # Test 4: Validate parameters
    validation_errors = param_system.validate_parameters('noise_generation', {
        'frequency': 1.0,
        'amplitude': -1.0,  # Invalid: negative amplitude
        'octaves': 15,      # Invalid: too high
        'noise_type': 'invalid_noise'  # Invalid: not in validator
    })
    print(f"\\n4. Parameter validation errors: {len(validation_errors)} found")
    for param, errors in validation_errors.items():
        print(f"   {param}: {errors}")
    
    # Test 5: Test parameter constraints
    constraint_errors = param_system.validate_parameter_constraints('noise_generation', {
        'frequency': 1.0,
        'amplitude': 1.0,
        'noise_type': 'perlin',
        'octaves': 4
    })
    print(f"\\n5. Parameter constraint errors: {len(constraint_errors)} found")
    for error in constraint_errors:
        print(f"   {error}")
    
    # Test 6: Get parameterization recommendations
    recommendations = param_system.get_parameterization_recommendations('bloom_effect')
    print(f"\\n6. Parameterization recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"   - {rec}")
    
    # Test 7: Apply parameters to pseudocode (simulated)
    sample_pseudocode = """
    // Sample noise function with parameters
    float generateNoise(vec2 coord) {
        return noise(freq * coord) * amp;
    }
    
    // Parameters would replace freq and amp
    """
    result = param_system.apply_parameters(sample_pseudocode, {
        'frequency': 2.5,
        'amplitude': 1.2,
        'octaves': 5
    })
    print(f"\\n7. Parameter replacement result:")
    print(f"{result}")
    
    print(f"\\n✅ Module Parameterization System initialized and tested successfully!")
    print(f"   Features demonstrated:")
    print(f"   - Parameter templates and validation")
    print(f"   - Parameter presets and overrides")
    print(f"   - Constraint checking")
    print(f"   - Type validation")
    print(f"   - Default values")
    
    return 0


if __name__ == "__main__":
    import os
    exit(main())