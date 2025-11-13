#!/usr/bin/env python3
"""
Module Interface Extractor and Data Flow System
Extracts interfaces from modules and creates data flow validation
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class ParameterDirection(Enum):
    INPUT = "input"
    OUTPUT = "output" 
    INOUT = "inout"
    UNIFORM = "uniform"


@dataclass
class ShaderVariable:
    name: str
    type: str
    direction: ParameterDirection
    semantic: str = ""  # Semantic meaning (position, normal, texcoord, etc.)
    description: str = ""


@dataclass
class ModuleInterface:
    inputs: List[ShaderVariable]
    outputs: List[ShaderVariable]
    uniforms: List[ShaderVariable]
    samplers: List[ShaderVariable]
    functions: List[Dict[str, Any]]  # Function signatures


class ModuleInterfaceExtractor:
    """Extracts interface information from module pseudocode"""
    
    def __init__(self):
        # Common semantic mappings
        self.semantic_mappings = {
            'position': ['pos', 'position', 'fragpos', 'frag_pos', 'Position'],
            'normal': ['normal', 'norm', 'n', 'Normal'],
            'tex_coords': ['uv', 'texcoords', 'tex_coords', 'TexCoord', 'uv_coords'],
            'tangent': ['tangent', 't', 'Tangent'],
            'bitangent': ['bitangent', 'binormal', 'b', 'B'],
            'view_dir': ['viewdir', 'view_dir', 'viewDir', 'eyeDir'],
            'light_dir': ['lightdir', 'light_dir', 'lightDir'],
            'color': ['color', 'col', 'fragcolor', 'outcolor', 'Color'],
            'albedo': ['albedo', 'diffuse', 'diffuse_color'],
            'specular': ['specular', 'spec', 'Specular'],
            'world_pos': ['worldpos', 'world_pos', 'wpos']
        }
        
        # Common shader types
        self.shader_types = [
            'vec2', 'vec3', 'vec4',
            'mat2', 'mat3', 'mat4',
            'float', 'int', 'bool',
            'sampler2D', 'samplerCube', 'sampler3D',
            'ivec2', 'ivec3', 'ivec4',
            'bvec2', 'bvec3', 'bvec4'
        ]
    
    def extract_interface(self, pseudocode: str, module_name: str) -> ModuleInterface:
        """Extract interface from pseudocode"""
        inputs = []
        outputs = []
        uniforms = []
        samplers = []
        functions = []
        
        # Split into lines and process
        lines = pseudocode.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line or line.startswith('//'):
                continue
            
            # Extract uniforms
            if 'uniform' in line:
                uniform_vars = self._extract_uniform(line)
                uniforms.extend(uniform_vars)
                
                # Check if it's a sampler
                for var in uniform_vars:
                    if 'sampler' in var.type:
                        samplers.append(var)
            
            # Extract inputs (in variables, function parameters)
            elif 'in ' in line and any(t in line for t in self.shader_types):
                input_vars = self._extract_directional_variables(line, ParameterDirection.INPUT)
                inputs.extend(input_vars)
            
            # Extract outputs (out variables)
            elif 'out ' in line and any(t in line for t in self.shader_types):
                output_vars = self._extract_directional_variables(line, ParameterDirection.OUTPUT)
                outputs.extend(output_vars)
            
            # Extract function definitions and their parameters
            elif self._is_function_definition(line, lines, i):
                func_info = self._extract_function_info(line, lines, i)
                functions.append(func_info)
                
                # Add function parameters as inputs to the module interface
                for param in func_info.get('parameters', []):
                    inputs.append(ShaderVariable(
                        name=param['name'],
                        type=param['type'],
                        direction=ParameterDirection.INPUT,
                        semantic=self._infer_semantic(param['name'])
                    ))
        
        return ModuleInterface(
            inputs=inputs,
            outputs=outputs, 
            uniforms=uniforms,
            samplers=samplers,
            functions=functions
        )
    
    def _extract_uniform(self, line: str) -> List[ShaderVariable]:
        """Extract uniform variables from a line"""
        uniforms = []
        
        # Pattern: uniform type name;
        # or: uniform type name[arraysize];
        # or: uniform type name1, name2, name3;
        
        pattern = r'uniform\s+(\w+(?:\s*\*?)?)\s+([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)\s*;?'
        match = re.search(pattern, line)
        
        if match:
            type_name = match.group(1)
            var_names = [name.strip() for name in match.group(2).split(',')]
            
            for var_name in var_names:
                if var_name:  # Skip empty names
                    semantic = self._infer_semantic(var_name)
                    uniforms.append(ShaderVariable(
                        name=var_name,
                        type=type_name,
                        direction=ParameterDirection.UNIFORM,
                        semantic=semantic
                    ))
        
        return uniforms
    
    def _extract_directional_variables(self, line: str, direction: ParameterDirection) -> List[ShaderVariable]:
        """Extract input/output variables from a line"""
        variables = []
        
        # Pattern: in/out type name;
        # or: in/out type name[arraysize];
        # or: in/out type name1, name2, name3;
        
        dir_word = direction.value
        pattern = rf'{dir_word}\s+(\w+(?:\s*\*?)?)\s+([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)\s*;?'
        match = re.search(pattern, line)
        
        if match:
            type_name = match.group(1)
            var_names = [name.strip() for name in match.group(2).split(',')]
            
            for var_name in var_names:
                if var_name:  # Skip empty names
                    semantic = self._infer_semantic(var_name)
                    variables.append(ShaderVariable(
                        name=var_name,
                        type=type_name,
                        direction=direction,
                        semantic=semantic
                    ))
        
        return variables
    
    def _is_function_definition(self, line: str, all_lines: List[str], line_idx: int) -> bool:
        """Check if a line is a function definition"""
        # Simple pattern for function definition
        function_pattern = r'^\s*[a-zA-Z_]\w*\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*\{?'
        return bool(re.match(function_pattern, line)) or 'void main()' in line
    
    def _extract_function_info(self, line: str, all_lines: List[str], line_idx: int) -> Dict[str, Any]:
        """Extract function signature and parameters"""
        func_info = {
            'name': '',
            'return_type': '',
            'parameters': []
        }
        
        # Pattern: return_type function_name(parameters) {
        pattern = r'(\w+)\s+(\w+)\s*\(([^)]*)\)\s*(?:{|$)'
        match = re.search(pattern, line)
        
        if match:
            func_info['return_type'] = match.group(1)
            func_info['name'] = match.group(2)
            
            # Extract parameters
            params_str = match.group(3)
            if params_str.strip():
                # Split parameters by comma, but be careful with nested commas in complex types
                raw_params = [p.strip() for p in params_str.split(',')]
                
                for raw_param in raw_params:
                    param_parts = raw_param.strip().split()
                    if len(param_parts) >= 2:
                        param_type = param_parts[0]
                        param_name = param_parts[-1]  # Last part is usually the name
                        
                        # Handle cases like 'in vec3 normal' or 'out vec4 color'
                        if len(param_parts) >= 3:
                            direction_str = param_parts[1] if param_parts[1] in ['in', 'out', 'inout'] else None
                            if direction_str:
                                direction_enum = getattr(ParameterDirection, direction_str.upper())
                            else:
                                direction_enum = ParameterDirection.INPUT  # Default
                        else:
                            direction_enum = ParameterDirection.INPUT  # Default
                        
                        func_info['parameters'].append({
                            'type': param_type,
                            'name': param_name,
                            'direction': direction_enum.value
                        })
        
        return func_info
    
    def _infer_semantic(self, name: str) -> str:
        """Infer semantic meaning from variable name"""
        name_lower = name.lower()
        
        for semantic, patterns in self.semantic_mappings.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return semantic
        
        return ""  # No semantic inferred


def create_enhanced_modules_with_interfaces():
    """Update existing modules to have proper interface information"""
    
    # Update the basic point light module with interface info
    point_light_interface = """
#!/usr/bin/env python3
'''
Basic Point Light Module with Interface Definition
Extracted from common lighting patterns in shader analysis
Pattern frequency: 405 occurrences
'''

# Interface definition
INTERFACE = {
    'inputs': [
        {'name': 'FragPos', 'type': 'vec3', 'direction': 'in', 'semantic': 'position'},
        {'name': 'Normal', 'type': 'vec3', 'direction': 'in', 'semantic': 'normal'}, 
        {'name': 'lightPos', 'type': 'vec3', 'direction': 'uniform', 'semantic': 'light_position'},
        {'name': 'lightColor', 'type': 'vec3', 'direction': 'uniform', 'semantic': 'light_color'}
    ],
    'outputs': [
        {'name': 'lightColorOut', 'type': 'vec3', 'direction': 'out', 'semantic': 'light_contribution'}
    ],
    'uniforms': [
        {'name': 'lightPos', 'type': 'vec3', 'semantic': 'light_position'},
        {'name': 'lightColor', 'type': 'vec3', 'semantic': 'light_color'}
    ]
}

# Pseudocode for basic point light calculation
pseudocode = '''
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
'''

def get_interface():
    '''Return the interface definition for this module'''
    return INTERFACE

def get_pseudocode():
    '''Return the pseudocode for this lighting module'''
    return pseudocode

def get_metadata():
    '''Return metadata about this module'''
    return {
        'name': 'basic_point_light',
        'type': 'lighting',
        'patterns': ['Point Light', 'Light Attenuation'],
        'frequency': 405,
        'dependencies': [],
        'conflicts': [],
        'description': 'Basic point light calculation with attenuation',
        'interface': INTERFACE
    }
"""
    
    # Write updated module
    with open('modules/lighting/point_light/basic_point_light.py', 'w') as f:
        f.write(point_light_interface)
    
    # Update the normal mapping module too
    normal_mapping_interface = """
#!/usr/bin/env python3
'''
Normal Mapping Module with Interface Definition
Extracted from common lighting patterns in shader analysis
Pattern frequency: 533 occurrences
'''

# Interface definition
INTERFACE = {
    'inputs': [
        {'name': 'TexCoords', 'type': 'vec2', 'direction': 'in', 'semantic': 'tex_coords'},
        {'name': 'FragPos', 'type': 'vec3', 'direction': 'in', 'semantic': 'position'},
        {'name': 'Normal', 'type': 'vec3', 'direction': 'in', 'semantic': 'normal'},
        {'name': 'Tangent', 'type': 'vec3', 'direction': 'in', 'semantic': 'tangent'},
        {'name': 'normalMap', 'type': 'sampler2D', 'direction': 'uniform', 'semantic': 'normal_texture'}
    ],
    'outputs': [
        {'name': 'normalOut', 'type': 'vec3', 'direction': 'out', 'semantic': 'normal_world_space'}
    ],
    'uniforms': [
        {'name': 'normalMap', 'type': 'sampler2D', 'semantic': 'normal_texture'}
    ]
}

# Pseudocode for normal mapping
pseudocode = '''
// Normal Mapping Implementation
vec3 getNormalFromMap(sampler2D normalMap, vec2 uv, vec3 pos, vec3 normal, vec3 tangent) {
    // Sample the normal map
    vec3 tangentNormal = texture(normalMap, uv).xyz * 2.0 - 1.0;
    
    // Create TBN matrix
    vec3 T = normalize(tangent);
    vec3 N = normalize(normal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    
    // Transform normal from tangent space to world space
    vec3 finalNormal = normalize(TBN * tangentNormal);
    
    return finalNormal;
}

// Alternative: Simple normal mapping with normal map sampling
vec3 sampleNormalMap(sampler2D normalMap, vec2 uv) {
    vec3 normal = texture(normalMap, uv).xyz * 2.0 - 1.0;
    normal.xy *= -1.0;  // Flip X and Y for correct orientation
    return normalize(normal);
}
'''

def get_interface():
    '''Return the interface definition for this module'''
    return INTERFACE

def get_pseudocode():
    '''Return the pseudocode for this lighting module'''
    return pseudocode

def get_metadata():
    '''Return metadata about this module'''
    return {
        'name': 'normal_mapping',
        'type': 'lighting',
        'patterns': ['Normal Mapping'],
        'frequency': 533,
        'dependencies': [],
        'conflicts': [],
        'description': 'Normal mapping implementation with TBN matrix',
        'interface': INTERFACE
    }
"""
    
    # Write updated module
    with open('modules/lighting/normal_mapping/normal_mapping.py', 'w') as f:
        f.write(normal_mapping_interface)


if __name__ == "__main__":
    # Create enhanced modules with interfaces
    create_enhanced_modules_with_interfaces()
    print("✓ Updated modules with interface definitions")
    
    # Test the interface extractor
    extractor = ModuleInterfaceExtractor()
    
    # Sample pseudocode to test
    sample_pseudocode = """
#version 330 core

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform sampler2D normalMap;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;

vec3 calculatePointLight(vec3 position, vec3 normal, vec3 lightPos, vec3 lightColor) {
    vec3 lightDir = normalize(lightPos - position);
    float distance = length(lightPos - position);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    diffuse *= attenuation;
    return diffuse;
}
"""
    
    interface = extractor.extract_interface(sample_pseudocode, "test_module")
    
    print(f"✓ Extracted interface:")
    print(f"  Inputs: {len(interface.inputs)}")
    print(f"  Outputs: {len(interface.outputs)}")
    print(f"  Uniforms: {len(interface.uniforms)}")
    print(f"  Functions: {len(interface.functions)}")
    
    for inp in interface.inputs:
        print(f"    - {inp.type} {inp.name} ({inp.semantic})")
    
    for out in interface.outputs:
        print(f"    - {out.type} {out.name} ({out.semantic})")
    
    for uniform in interface.uniforms:
        print(f"    - uniform {uniform.type} {uniform.name} ({uniform.semantic})")