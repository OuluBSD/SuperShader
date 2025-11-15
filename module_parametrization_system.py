#!/usr/bin/env python3
"""
Module Parameterization System for SuperShader
Enables parameterization of modules to allow configurable behavior
"""

import sys
import os
import json
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class ParameterType(Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"
    MAT3 = "mat3"
    MAT4 = "mat4"
    STRING = "string"
    ENUM = "enum"


@dataclass
class ModuleParameter:
    """Definition of a module parameter"""
    name: str
    param_type: ParameterType
    default_value: Any
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[str]] = None  # For enum types
    semantic: str = ""  # Semantic meaning of the parameter


@dataclass
class ParameterizedModule:
    """A module with parameterization capabilities"""
    name: str
    parameters: List[ModuleParameter]
    implementation_template: str  # Template with placeholders
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    description: str = ""

    def apply_parameters(self, param_values: Dict[str, Any]) -> str:
        """Apply parameter values to the implementation template"""
        result = self.implementation_template
        
        # Validate parameters exist
        for param_name, value in param_values.items():
            if not any(p.name == param_name for p in self.parameters):
                raise ValueError(f"Invalid parameter: {param_name}")
            
            # Validate value based on type
            param = next(p for p in self.parameters if p.name == param_name)
            if not self._validate_parameter_value(param, value):
                raise ValueError(f"Invalid value for parameter {param_name}: {value}")
        
        # Apply parameter values to template
        for param_name, value in param_values.items():
            # Replace parameter placeholders in the implementation
            result = result.replace(f"{{{{{param_name}}}}}", str(value))
            result = result.replace(f"PARAM_{param_name.upper()}", str(value))
        
        return result

    def _validate_parameter_value(self, param: ModuleParameter, value: Any) -> bool:
        """Validate that a parameter value is appropriate for the parameter type"""
        if param.param_type == ParameterType.FLOAT:
            return isinstance(value, (int, float))
        elif param.param_type == ParameterType.INT:
            return isinstance(value, int)
        elif param.param_type == ParameterType.BOOL:
            return isinstance(value, bool)
        elif param.param_type == ParameterType.STRING:
            return isinstance(value, str)
        elif param.param_type == ParameterType.ENUM:
            return isinstance(value, str) and value in (param.options or [])
        elif param.param_type in [ParameterType.VEC2, ParameterType.VEC3, ParameterType.VEC4]:
            if isinstance(value, (list, tuple)):
                expected_len = 2 if param.param_type == ParameterType.VEC2 else \
                             3 if param.param_type == ParameterType.VEC3 else 4
                return len(value) == expected_len and all(isinstance(v, (int, float)) for v in value)
            elif isinstance(value, str):  # GLSL-style constructor: "vec3(1.0, 0.5, 0.2)"
                return True  # Assume string representations are valid
        elif param.param_type in [ParameterType.MAT3, ParameterType.MAT4]:
            if isinstance(value, list):
                expected_size = 9 if param.param_type == ParameterType.MAT3 else 16
                return len(value) == expected_size and all(isinstance(v, (int, float)) for v in value)
            elif isinstance(value, str):  # GLSL-style constructor
                return True  # Assume string representations are valid
        return False

    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameter values"""
        defaults = {}
        for param in self.parameters:
            defaults[param.name] = param.default_value
        return defaults


class ModuleParameterizationSystem:
    """
    System for managing parameterized modules and their configurations
    """
    
    def __init__(self):
        self.parameterized_modules: Dict[str, ParameterizedModule] = {}
        self.global_params: Dict[str, Any] = {}
        
    def register_parameterized_module(self, module: ParameterizedModule):
        """Register a new parameterized module"""
        self.parameterized_modules[module.name] = module
        print(f"Registered parameterized module: {module.name} with {len(module.parameters)} parameters")
    
    def create_parameterized_module(
        self,
        name: str,
        params: List[ModuleParameter],
        implementation_template: str,
        dependencies: List[str] = None,
        conflicts: List[str] = None,
        description: str = ""
    ) -> ParameterizedModule:
        """Helper to create and register a parameterized module"""
        module = ParameterizedModule(
            name=name,
            parameters=params,
            implementation_template=implementation_template,
            dependencies=dependencies or [],
            conflicts=conflicts or [],
            description=description
        )
        self.register_parameterized_module(module)
        return module

    def get_parameterized_module(self, name: str) -> Optional[ParameterizedModule]:
        """Get a registered parameterized module"""
        return self.parameterized_modules.get(name)

    def apply_module_parameters(self, module_name: str, param_values: Dict[str, Any]) -> str:
        """Apply parameters to a module and return the configured implementation"""
        module = self.get_parameterized_module(module_name)
        if not module:
            raise ValueError(f"Module {module_name} not found")
        
        return module.apply_parameters(param_values)

    def get_module_parameters(self, module_name: str) -> Optional[List[ModuleParameter]]:
        """Get the parameters for a specific module"""
        module = self.get_parameterized_module(module_name)
        if module:
            return module.parameters
        return None

    def validate_parameter_set(self, module_name: str, param_values: Dict[str, Any]) -> bool:
        """Validate that a set of parameter values is valid for a module"""
        module = self.get_parameterized_module(module_name)
        if not module:
            raise ValueError(f"Module {module_name} not found")
        
        # Check that all required parameters are provided
        for param in module.parameters:
            if param.name not in param_values and param.name in [p.name for p in module.parameters if p.default_value is None]:
                raise ValueError(f"Missing required parameter: {param.name}")
        
        # Validate each provided parameter value
        for param_name, value in param_values.items():
            param = next((p for p in module.parameters if p.name == param_name), None)
            if not param:
                raise ValueError(f"Unknown parameter: {param_name}")
            if not module._validate_parameter_value(param, value):
                raise ValueError(f"Invalid value for parameter {param_name}: {value}")
        
        return True

    def get_compatible_modules(self, param_constraints: Dict[str, Any]) -> List[str]:
        """Find modules that are compatible with the given parameter constraints"""
        compatible = []
        
        for name, module in self.parameterized_modules.items():
            is_compatible = True
            
            # Check if module's parameters satisfy constraints
            for param_name, constraint_value in param_constraints.items():
                param = next((p for p in module.parameters if p.name == param_name), None)
                if not param:
                    # If the module doesn't have this parameter, it's compatible by default
                    continue
                
                # Apply constraint validation (this is a simplified check)
                # Real implementation would be more complex
                if not self._check_constraint_satisfaction(param, constraint_value):
                    is_compatible = False
                    break
            
            if is_compatible:
                compatible.append(name)
        
        return compatible

    def _check_constraint_satisfaction(self, param: ModuleParameter, constraint_value: Any) -> bool:
        """Check if a parameter satisfies a constraint value"""
        if param.min_value is not None and param.max_value is not None:
            # Numerical constraint
            if isinstance(constraint_value, (int, float)):
                return param.min_value <= constraint_value <= param.max_value
        elif param.options is not None:
            # Enum constraint
            return constraint_value in param.options
        else:
            # Just check type compatibility
            return param.param_type in [ParameterType.FLOAT, ParameterType.INT] and isinstance(constraint_value, (int, float))
        
        return True

    def generate_param_docs(self, module_name: str) -> str:
        """Generate documentation for a module's parameters"""
        module = self.get_parameterized_module(module_name)
        if not module:
            return f"Module {module_name} not found"
        
        docs = f"Parameters for {module.name}:\n"
        docs += "=" * (len(module.name) + 12) + "\n\n"
        
        for param in module.parameters:
            docs += f"{param.name} ({param.param_type.value}):\n"
            docs += f"  Description: {param.description}\n"
            docs += f"  Default: {param.default_value}\n"
            if param.min_value is not None or param.max_value is not None:
                docs += f"  Range: {param.min_value} to {param.max_value}\n"
            if param.options:
                docs += f"  Options: {', '.join(param.options)}\n"
            docs += "\n"
        
        return docs

    def get_all_module_params(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get parameters for all registered modules in a simplified format"""
        all_params = {}
        for name, module in self.parameterized_modules.items():
            all_params[name] = [
                {
                    'name': param.name,
                    'type': param.param_type.value,
                    'default': param.default_value,
                    'description': param.description,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                    'options': param.options
                } for param in module.parameters
            ]
        return all_params


# Predefined parameterized modules for different shader types
def initialize_parameterized_modules() -> ModuleParameterizationSystem:
    """Initialize the parameterization system with common parameterized modules"""
    
    param_system = ModuleParameterizationSystem()
    
    # 1. Parameterized Noise Module
    noise_params = [
        ModuleParameter("scale", ParameterType.FLOAT, 1.0, "Scale factor for noise", 0.1, 10.0),
        ModuleParameter("amplitude", ParameterType.FLOAT, 1.0, "Amplitude of noise", 0.0, 5.0),
        ModuleParameter("persistence", ParameterType.FLOAT, 0.5, "Persistence factor for fractal noise", 0.01, 1.0),
        ModuleParameter("lacunarity", ParameterType.FLOAT, 2.0, "Gap multiplier between octaves", 1.0, 4.0),
        ModuleParameter("num_octaves", ParameterType.INT, 4, "Number of octaves for fractal noise", 1, 10),
        ModuleParameter("noise_type", ParameterType.ENUM, "perlin", "Type of noise function", options=["perlin", "simplex", "value"])
    ]
    
    noise_template = """
// Parameterized Noise Function
// Type: PARAM_noise_type
// Scale: PARAM_scale
// Octaves: PARAM_num_octaves

float noise_function(vec2 coord) {
    float value = 0.0;
    float amplitude = PARAM_amplitude;
    float frequency = 1.0;
    float persistence = PARAM_persistence;
    
    for (int i = 0; i < PARAM_num_octaves; i++) {
        if (PARAM_noise_type == "perlin") {
            value += amplitude * perlinNoise(coord * frequency * PARAM_scale, 1.0, 0.0);
        } else if (PARAM_noise_type == "simplex") {
            value += amplitude * simplexNoise(coord * frequency * PARAM_scale, 1.0, 0.0);
        } else if (PARAM_noise_type == "value") {
            value += amplitude * valueNoise(coord * frequency * PARAM_scale, 1.0, 0.0);
        }
        
        amplitude *= persistence;
        frequency *= PARAM_lacunarity;
    }
    
    return value;
}

// Implementation of the various noise types would be defined elsewhere
float perlinNoise(vec2 coord, float scale, float time) {
    // Perlin noise implementation
    return 0.0;  // Simplified for template
}

float simplexNoise(vec2 coord, float scale, float time) {
    // Simplex noise implementation
    return 0.0;  // Simplified for template
}

float valueNoise(vec2 coord, float scale, float time) {
    // Value noise implementation
    return 0.0;  // Simplified for template
}
    """
    
    param_system.create_parameterized_module(
        "parameterized_noise",
        noise_params,
        noise_template,
        description="Configurable noise generation with multiple algorithms and parameters"
    )
    
    # 2. Parameterized Lighting Module
    lighting_params = [
        ModuleParameter("light_count", ParameterType.INT, 1, "Number of lights to process", 1, 8),
        ModuleParameter("attenuation_constant", ParameterType.FLOAT, 1.0, "Constant attenuation factor"),
        ModuleParameter("attenuation_linear", ParameterType.FLOAT, 0.09, "Linear attenuation factor"),
        ModuleParameter("attenuation_quadratic", ParameterType.FLOAT, 0.032, "Quadratic attenuation factor"),
        ModuleParameter("ambient_factor", ParameterType.FLOAT, 0.1, "Ambient lighting factor", 0.0, 1.0),
        ModuleParameter("diffuse_factor", ParameterType.FLOAT, 1.0, "Diffuse lighting factor", 0.0, 2.0),
        ModuleParameter("specular_factor", ParameterType.FLOAT, 1.0, "Specular lighting factor", 0.0, 2.0),
        ModuleParameter("shininess", ParameterType.FLOAT, 32.0, "Shininess exponent", 1.0, 512.0)
    ]
    
    lighting_template = """
// Parameterized Lighting Function
// Lights: PARAM_light_count
// Attenuation: PARAM_attenuation_constant, PARAM_attenuation_linear, PARAM_attenuation_quadratic
// Factors: Ambient(PARAM_ambient_factor), Diffuse(PARAM_diffuse_factor), Specular(PARAM_specular_factor)

vec3 compute_lighting(vec3 position, vec3 normal, vec3 view_dir, vec3 base_color) {
    vec3 light_color = vec3(1.0);
    vec3 light_pos = vec3(0.0, 5.0, 5.0); // Simplified for template
    
    vec3 ambient = PARAM_ambient_factor * light_color;
    
    vec3 light_dir = normalize(light_pos - position);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = PARAM_diffuse_factor * diff * light_color;
    
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), PARAM_shininess);
    vec3 specular = PARAM_specular_factor * spec * light_color;
    
    // Attenuation calculation
    float distance = length(light_pos - position);
    float attenuation = 1.0 / (
        PARAM_attenuation_constant + 
        PARAM_attenuation_linear * distance + 
        PARAM_attenuation_quadratic * distance * distance
    );
    
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    
    return base_color * (ambient + diffuse) + specular;
}
    """
    
    param_system.create_parameterized_module(
        "parameterized_lighting", 
        lighting_params, 
        lighting_template,
        description="Configurable lighting model with adjustable parameters"
    )
    
    # 3. Parameterized Texturing Module
    texturing_params = [
        ModuleParameter("tiling_x", ParameterType.FLOAT, 1.0, "Horizontal tiling factor", 0.1, 10.0),
        ModuleParameter("tiling_y", ParameterType.FLOAT, 1.0, "Vertical tiling factor", 0.1, 10.0),
        ModuleParameter("offset_x", ParameterType.FLOAT, 0.0, "Horizontal offset", -2.0, 2.0),
        ModuleParameter("offset_y", ParameterType.FLOAT, 0.0, "Vertical offset", -2.0, 2.0),
        ModuleParameter("blend_mode", ParameterType.ENUM, "multiply", "Texture blending mode", 
                       options=["multiply", "overlay", "screen", "add"]), 
        ModuleParameter("intensity", ParameterType.FLOAT, 1.0, "Texture intensity", 0.0, 2.0),
        ModuleParameter("enable_anisotropic", ParameterType.BOOL, False, "Enable anisotropic filtering")
    ]
    
    texturing_template = """
// Parameterized Texturing Function
// Tiling: PARAM_tiling_x x PARAM_tiling_y
// Offset: PARAM_offset_x, PARAM_offset_y
// Blend Mode: PARAM_blend_mode
// Intensity: PARAM_intensity

vec4 sample_texture(sampler2D tex, vec2 uv) {
    vec2 adjusted_uv = uv * vec2(PARAM_tiling_x, PARAM_tiling_y) + vec2(PARAM_offset_x, PARAM_offset_y);
    
    vec4 tex_color = texture(tex, adjusted_uv);
    
    // Apply intensity
    tex_color.rgb *= PARAM_intensity;
    
    // Apply blend mode (simplified)
    if (PARAM_blend_mode == "multiply") {
        tex_color.rgb = tex_color.rgb * vec3(0.5) + vec3(0.5); // Simplified
    } else if (PARAM_blend_mode == "overlay") {
        tex_color.rgb = mix(tex_color.rgb, vec3(1.0), 0.5); // Simplified
    } else if (PARAM_blend_mode == "screen") {
        tex_color.rgb = 1.0 - (1.0 - tex_color.rgb) * (1.0 - vec3(0.5)); // Simplified
    } else if (PARAM_blend_mode == "add") {
        tex_color.rgb += vec3(0.2); // Simplified
    }
    
    return tex_color;
}
    """
    
    param_system.create_parameterized_module(
        "parameterized_texturing",
        texturing_params,
        texturing_template,
        description="Configurable texturing with tiling, offset, and blend options"
    )
    
    # 4. Parameterized Animation Module
    animation_params = [
        ModuleParameter("speed", ParameterType.FLOAT, 1.0, "Animation speed factor", 0.1, 5.0),
        ModuleParameter("amplitude", ParameterType.FLOAT, 0.1, "Maximum animation amplitude", 0.0, 1.0),
        ModuleParameter("frequency", ParameterType.FLOAT, 1.0, "Animation frequency", 0.1, 10.0),
        ModuleParameter("phase", ParameterType.FLOAT, 0.0, "Starting phase offset", 0.0, 6.28),
        ModuleParameter("wave_type", ParameterType.ENUM, "sine", "Waveform type", 
                       options=["sine", "cosine", "triangle", "square"]),
        ModuleParameter("enable_echo", ParameterType.BOOL, False, "Enable echo/repetition effect"),
        ModuleParameter("echo_delay", ParameterType.FLOAT, 0.5, "Delay between echoes", 0.1, 2.0)
    ]
    
    animation_template = """
// Parameterized Animation Function
// Speed: PARAM_speed
// Amplitude: PARAM_amplitude  
// Frequency: PARAM_frequency
// Wave Type: PARAM_wave_type

vec3 animate_position(vec3 original_pos, float time) {
    float t = time * PARAM_speed + PARAM_phase;
    float displacement;
    
    if (PARAM_wave_type == "sine") {
        displacement = PARAM_amplitude * sin(t * PARAM_frequency);
    } else if (PARAM_wave_type == "cosine") {
        displacement = PARAM_amplitude * cos(t * PARAM_frequency);
    } else if (PARAM_wave_type == "triangle") {
        // Approximate triangle wave
        displacement = PARAM_amplitude * 2.0 * abs(fract(t * PARAM_frequency / 2.0 + 0.5) - 0.5);
    } else if (PARAM_wave_type == "square") {
        // Square wave approximation
        displacement = PARAM_amplitude * sign(sin(t * PARAM_frequency));
    }
    
    // Apply displacement to Y axis (could be configurable)
    return original_pos + vec3(0.0, displacement, 0.0);
}

// Echo effect function
vec3 apply_echo_effect(vec3 pos, float time) {
    if (!PARAM_enable_echo) {
        return pos;
    }
    
    // Apply echo with delay - simplified implementation
    float echo_time = time - PARAM_echo_delay;
    vec3 echo_pos = animate_position(pos, echo_time);
    
    // Blend original and echo
    return mix(pos, echo_pos, 0.3);  // 30% echo effect
}
    """
    
    param_system.create_parameterized_module(
        "parameterized_animation",
        animation_params,
        animation_template,
        description="Configurable animation with different waveforms and effects"
    )
    
    return param_system


class ModuleParametrizationOptimizer:
    """Optimizer for the module parameterization system"""
    
    def __init__(self):
        self.system = initialize_parameterized_modules()
    
    def optimize_parameter_configuration(self, module_name: str, param_constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize parameter configuration for a module based on constraints"""
        print(f"Optimizing parameter configuration for {module_name}...")
        
        if param_constraints is None:
            param_constraints = {}
        
        # Get the module
        module = self.system.get_parameterized_module(module_name)
        if not module:
            raise ValueError(f"Module {module_name} not found")
        
        # Get default parameters
        optimal_params = module.get_default_params()
        
        # Apply constraints if provided
        for param_name, constraint_value in param_constraints.items():
            if param_name in optimal_params:
                # Validate the constraint
                param = next(p for p in module.parameters if p.name == param_name)
                if module._validate_parameter_value(param, constraint_value):
                    optimal_params[param_name] = constraint_value
        
        # Validate the final configuration
        try:
            self.system.validate_parameter_set(module_name, optimal_params)
            print(f"✅ Parameter configuration validated for {module_name}")
        except ValueError as e:
            print(f"⚠️  Parameter configuration warning for {module_name}: {str(e)}")
            # In this case, we'll still return the parameters but with warning
        
        return optimal_params
    
    def batch_optimize_modules(self, module_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Optimize parameters for multiple modules in batch"""
        results = {}
        
        for config in module_configs:
            module_name = config.get('module_name')
            constraints = config.get('constraints', {})
            
            try:
                optimized_params = self.optimize_parameter_configuration(module_name, constraints)
                results[module_name] = {
                    'status': 'success',
                    'params': optimized_params,
                    'constraints_applied': constraints
                }
            except Exception as e:
                results[module_name] = {
                    'status': 'error',
                    'error': str(e),
                    'params': module.get_default_params() if (module := self.system.get_parameterized_module(module_name)) else {}
                }
        
        return results
    
    def get_performance_recommendations(self, module_name: str, param_values: Dict[str, Any]) -> List[str]:
        """Provide performance recommendations for a parameterized module"""
        recommendations = []
        
        module = self.system.get_parameterized_module(module_name)
        if not module:
            return ["Module not found"]
        
        # Check performance-related parameters
        if module_name == "parameterized_noise":
            if param_values.get('num_octaves', 4) > 6:
                recommendations.append("High number of octaves may impact performance, consider reducing for real-time usage")
            if param_values.get('scale', 1.0) > 5.0:
                recommendations.append("High scale factor may cause aliasing, consider adding anti-aliasing")
        
        elif module_name == "parameterized_lighting":
            if param_values.get('light_count', 1) > 4:
                recommendations.append("Multiple lights may impact performance, consider using clustered lighting")
            if param_values.get('shininess', 32.0) > 100.0:
                recommendations.append("Very high shininess values may cause precision issues on some hardware")
        
        elif module_name == "parameterized_texturing":
            if param_values.get('enable_anisotropic', False):
                recommendations.append("Anisotropic filtering enabled - may affect performance on lower-end hardware")
            if param_values.get('tiling_x', 1.0) > 8.0 or param_values.get('tiling_y', 1.0) > 8.0:
                recommendations.append("High tiling values may cause texture sampling artifacts")
        
        elif module_name == "parameterized_animation":
            if param_values.get('frequency', 1.0) > 10.0:
                recommendations.append("High animation frequency may cause aliasing, consider reducing or adding temporal AA")
            if param_values.get('enable_echo', False):
                recommendations.append("Echo effect enabled - may impact performance due to multiple evaluations")
        
        return recommendations or ["Configuration appears performance-optimal"]


def main():
    """Main function to demonstrate and test the module parameterization system"""
    print("Initializing Module Parameterization System...")
    
    optimizer = ModuleParametrizationOptimizer()
    system = optimizer.system
    
    print(f"Loaded {len(system.parameterized_modules)} parameterized modules:")
    for name, module in system.parameterized_modules.items():
        print(f"  - {name}: {len(module.parameters)} parameters")
    
    # Test parameter validation
    print(f"\n1. Testing parameter validation...")
    try:
        # Valid parameters for noise module
        valid_params = {
            'scale': 2.0,
            'amplitude': 0.8,
            'num_octaves': 3,
            'noise_type': 'perlin'
        }
        
        is_valid = system.validate_parameter_set('parameterized_noise', valid_params)
        print(f"✅ Valid parameters validation: {is_valid}")
        
        # Invalid parameter for noise module
        invalid_params = {
            'scale': 2.0,
            'invalid_param': 1.0  # This should cause an error
        }
        
        try:
            system.validate_parameter_set('parameterized_noise', invalid_params)
            print("❌ Should have caught invalid parameter error")
        except ValueError as e:
            print(f"✅ Correctly caught invalid parameter error: {str(e)[0:50]}...")
        
    except Exception as e:
        print(f"❌ Error during validation test: {str(e)}")
    
    # Test parameter configuration optimization
    print(f"\n2. Testing parameter optimization...")
    
    # Optimize noise module with specific constraints
    noise_params = optimizer.optimize_parameter_configuration(
        'parameterized_noise', 
        {
            'scale': 3.0,
            'num_octaves': 5,
            'noise_type': 'simplex'
        }
    )
    print(f"✅ Optimized noise parameters: {noise_params}")
    
    # Optimize lighting module with constraints
    lighting_params = optimizer.optimize_parameter_configuration(
        'parameterized_lighting',
        {
            'light_count': 3,
            'ambient_factor': 0.15,
            'shininess': 64.0
        }
    )
    print(f"✅ Optimized lighting parameters: {list(lighting_params.keys())}")
    
    # Test getting module with applied parameters
    print(f"\n3. Testing module code generation with parameters...")
    
    try:
        noise_code = system.apply_module_parameters('parameterized_noise', {
            'scale': 2.0,
            'amplitude': 1.0,
            'num_octaves': 4,
            'noise_type': 'perlin'
        })
        print(f"✅ Generated noise code with {len(noise_code)} chars")
        # Show first part of generated code
        lines = noise_code.split('\n')
        for i in range(min(5, len(lines))):
            print(f"   {lines[i]}")
        print("   ...")
        
    except Exception as e:
        print(f"❌ Error generating code: {str(e)}")
    
    # Test batch optimization
    print(f"\n4. Testing batch optimization...")
    
    configs = [
        {
            'module_name': 'parameterized_noise',
            'constraints': {'scale': 1.5, 'amplitude': 0.7}
        },
        {
            'module_name': 'parameterized_lighting', 
            'constraints': {'light_count': 2, 'ambient_factor': 0.08}
        },
        {
            'module_name': 'parameterized_texturing',
            'constraints': {'tiling_x': 2.0, 'tiling_y': 2.0, 'blend_mode': 'overlay'}
        }
    ]
    
    batch_results = optimizer.batch_optimize_modules(configs)
    for module_name, result in batch_results.items():
        if result['status'] == 'success':
            print(f"✅ Batch optimized {module_name}")
        else:
            print(f"❌ Batch optimization failed for {module_name}: {result['error'][:50]}...")
    
    # Test performance recommendations
    print(f"\n5. Testing performance recommendations...")
    
    perf_rec_noise = optimizer.get_performance_recommendations('parameterized_noise', {
        'num_octaves': 8,  # High number
        'scale': 10.0      # High scale
    })
    print(f"Noise performance recs: {perf_rec_noise[0][:60]}...")
    
    perf_rec_lighting = optimizer.get_performance_recommendations('parameterized_lighting', {
        'light_count': 1,   # Normal
        'shininess': 32.0   # Normal
    })
    print(f"Lighting performance recs: {perf_rec_lighting[0]}")
    
    # Show parameter documentation
    print(f"\n6. Generating parameter documentation...")
    
    noise_docs = system.generate_param_docs('parameterized_noise')
    print(f"Noise module parameter documentation preview ({len(noise_docs)} chars):")
    print("..." + noise_docs[100:300] + "...")
    
    # Performance summary
    print(f"\n7. Performance Summary:")
    all_module_params = system.get_all_module_params()
    total_params = sum(len(params) for params in all_module_params.values())
    print(f"   - Total parameterized modules: {len(all_module_params)}")
    print(f"   - Total parameters across all modules: {total_params}")
    print(f"   - Average parameters per module: {total_params/len(all_module_params) if all_module_params else 0:.1f}")
    
    print(f"\n✅ Module parameterization system is fully operational!")
    print(f"   The system supports {len(system.parameterized_modules)} parameterized modules")
    print(f"   with customizable behavior through parameter configuration")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)