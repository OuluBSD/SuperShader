#!/usr/bin/env python3
"""
Data Flow Validation System for SuperShader Modules
Implements connection validation and data flow diagrams
"""

from typing import Dict, List, Set, Tuple, Any
import json
from enum import Enum
from create_module_registry import ModuleRegistry


class ConnectionType(Enum):
    UNIFORM = "uniform"
    ATTRIBUTE = "attribute" 
    VARYING = "varying"
    OUTPUT = "output"
    SAMPLER = "sampler"


class DataFlowValidator:
    """Validates data flow between modules"""
    
    def __init__(self):
        self.registry = ModuleRegistry()
        self.connections = {}  # module_name -> {input_connections, output_connections}
        self.type_registry = {}  # Maps variable names to types
    
    def get_module_interface(self, module_name: str) -> Dict[str, Any]:
        """Extract interface information from a module"""
        # Get module from registry
        for genre, modules in self.registry.modules.items():
            for mod_name, mod_info in modules.items():
                full_name = f"{genre}/{mod_name}"
                if full_name == module_name or mod_name.split('/')[-1] == module_name:
                    # Parse the pseudocode to get interface info
                    pseudocode = mod_info.get('pseudocode', {}).get('get_pseudocode', lambda: '')()
                    return self._parse_interface(pseudocode)
        
        return {'inputs': [], 'outputs': [], 'uniforms': [], 'samplers': []}
    
    def _parse_interface(self, pseudocode: str) -> Dict[str, Any]:
        """Parse pseudocode to extract interface definitions"""
        inputs = []
        outputs = []
        uniforms = []
        samplers = []
        
        lines = pseudocode.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for uniform declarations
            if 'uniform' in line:
                if 'sampler' in line:
                    # Extract sampler name
                    parts = line.split()
                    sampler_name = [p for p in parts if p.endswith(';') or p == parts[-1]][-1].rstrip(';')
                    samplers.append({
                        'name': sampler_name,
                        'type': 'sampler2D' if 'sampler2D' in line else 'samplerCube' if 'samplerCube' in line else 'sampler'
                    })
                else:
                    # Extract uniform
                    parts = line.split()
                    if len(parts) >= 3:
                        uniform_type = parts[1]
                        uniform_name = parts[2].rstrip(';')
                        uniforms.append({
                            'name': uniform_name,
                            'type': uniform_type
                        })
            
            # Look for function parameters (inputs)
            if '(' in line and ')' in line and 'vec' in line:
                # This is a simplified parser - in a real implementation, we'd have a more sophisticated parser
                pass
            
            # Look for output variable declarations
            if 'out' in line:
                parts = line.split()
                if len(parts) >= 3:
                    output_type = parts[1]
                    output_name = parts[2].rstrip(';')
                    outputs.append({
                        'name': output_name,
                        'type': output_type
                    })
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'uniforms': uniforms,
            'samplers': samplers
        }
    
    def validate_connection(self, source_module: str, source_output: str, 
                           target_module: str, target_input: str) -> Dict[str, Any]:
        """Validate a connection between modules"""
        source_interface = self.get_module_interface(source_module)
        target_interface = self.get_module_interface(target_module)
        
        # Find the specific output and input
        source_output_def = None
        for output in source_interface['outputs']:
            if output['name'] == source_output:
                source_output_def = output
                break
        
        target_input_def = None
        for input_var in target_interface.get('inputs', []):
            if input_var['name'] == target_input:
                target_input_def = input_var
                break
        
        # Validate type compatibility
        valid = False
        error = ""
        
        if not source_output_def:
            error = f"Source output '{source_output}' not found in module '{source_module}'"
        elif not target_input_def:
            error = f"Target input '{target_input}' not found in module '{target_module}'"
        elif source_output_def['type'] != target_input_def['type']:
            # Implement type compatibility rules
            valid, error = self._check_type_compatibility(
                source_output_def['type'], target_input_def['type']
            )
        else:
            valid = True
        
        return {
            'valid': valid,
            'error': error,
            'source_type': source_output_def['type'] if source_output_def else None,
            'target_type': target_input_def['type'] if target_input_def else None
        }
    
    def _check_type_compatibility(self, source_type: str, target_type: str) -> Tuple[bool, str]:
        """Check if two types are compatible for connection"""
        # Basic compatibility rules
        if source_type == target_type:
            return True, ""
        
        # Vector compatibility (vec3 can connect to vec4 if w component is handled)
        if (source_type == 'vec3' and target_type == 'vec4'):
            return True, "Warning: vec3 to vec4 connection - w component will be default (1.0)"  # w=1.0 for positions, 0.0 for vectors
        
        if (source_type == 'vec4' and target_type == 'vec3'):
            return True, "Warning: vec4 to vec3 connection - w component will be dropped"
        
        # Scalar compatibility
        if source_type in ['float', 'int'] and target_type in ['float', 'int']:
            return True, ""
        
        return False, f"Type mismatch: {source_type} cannot connect to {target_type}"
    
    def build_data_flow_graph(self, module_combination: List[str]) -> Dict[str, Any]:
        """Build a complete data flow graph for a module combination"""
        graph = {
            'nodes': [],
            'edges': [],
            'validation': {'errors': [], 'warnings': []}
        }
        
        # Add nodes for each module
        for module in module_combination:
            interface = self.get_module_interface(module)
            graph['nodes'].append({
                'id': module,
                'label': module.split('/')[-1],
                'interface': interface
            })
        
        # Determine possible connections based on interface compatibility
        # This is a simplified version - in a real system, we'd have explicit connection rules
        for i, source_module in enumerate(module_combination):
            for j, target_module in enumerate(module_combination):
                if i != j:  # Don't connect module to itself
                    # Check for possible connections between compatible interfaces
                    source_interface = self.get_module_interface(source_module)
                    target_interface = self.get_module_interface(target_module)
                    
                    # Connect compatible outputs to compatible inputs
                    for source_output in source_interface['outputs']:
                        for target_input in target_interface.get('inputs', []):
                            validation = self.validate_connection(
                                source_module, source_output['name'],
                                target_module, target_input['name']
                            )
                            
                            if validation['valid']:
                                edge = {
                                    'source': source_module,
                                    'target': target_module,
                                    'source_port': source_output['name'],
                                    'target_port': target_input['name'],
                                    'type': 'data',
                                    'validation': validation
                                }
                                graph['edges'].append(edge)
        
        return graph
    
    def generate_plantuml_diagram(self, graph: Dict[str, Any], filename: str = "data_flow.puml"):
        """Generate PlantUML diagram from data flow graph"""
        puml_content = "@startuml\n"
        puml_content += "title SuperShader Data Flow Diagram\n\n"
        
        # Add modules as components
        for node in graph['nodes']:
            puml_content += f"component [{node['label']}] as {node['id'].replace('/', '_').replace('-', '_')}\n"
        
        puml_content += "\n"
        
        # Add connections
        for edge in graph['edges']:
            source_id = edge['source'].replace('/', '_').replace('-', '_')
            target_id = edge['target'].replace('/', '_').replace('-', '_')
            puml_content += f"{source_id} --> {target_id} : {edge['source_port']} -> {edge['target_port']}\n"
        
        puml_content += "\n@enduml"
        
        with open(filename, 'w') as f:
            f.write(puml_content)
        
        return puml_content


class TechnicalDataValidator:
    """Validates technical data and connections using names and tags"""
    
    def __init__(self):
        self.connection_rules = self._load_connection_rules()
    
    def _load_connection_rules(self) -> Dict[str, Any]:
        """Load predefined connection rules"""
        return {
            'normal_mapping': {
                'outputs': {
                    'normal_out': ['vec3', 'normal', 'world_space_normal'],
                    'tangent_out': ['vec3', 'tangent', 'tangent_space']
                },
                'inputs': {
                    'normal_map': ['sampler2D', 'normal_texture'],
                    'tex_coords': ['vec2', 'uv_coordinates']
                }
            },
            'pbr_lighting': {
                'outputs': {
                    'albedo_out': ['vec3', 'albedo', 'diffuse_color'],
                    'specular_out': ['vec3', 'specular', 'specular_color'],
                    'emission_out': ['vec3', 'emission', 'emissive_color']
                },
                'inputs': {
                    'position_in': ['vec3', 'position', 'world_position'],
                    'normal_in': ['vec3', 'normal', 'surface_normal'],
                    'albedo_in': ['vec3', 'albedo', 'diffuse_color']
                }
            },
            'shadow_mapping': {
                'outputs': {
                    'shadow_factor': ['float', 'shadow', 'shadow_intensity']
                },
                'inputs': {
                    'light_space_pos': ['vec4', 'light_position', 'light_space_coords'],
                    'shadow_map': ['sampler2D', 'shadow_texture']
                }
            }
        }
    
    def match_by_names(self, source_name: str, target_name: str) -> bool:
        """Simple name matching algorithm"""
        # Direct match
        if source_name == target_name:
            return True
        
        # Semantic matches (common shader terms)
        semantic_matches = [
            ('pos', 'position'),
            ('normal', 'n'),
            ('uv', 'tex_coords'),
            ('albedo', 'diffuse'),
            ('specular', 'spec'),
            ('shadow', 'shadow_factor')
        ]
        
        for sm_source, sm_target in semantic_matches:
            if (sm_source in source_name.lower() and sm_target in target_name.lower()) or \
               (sm_target in source_name.lower() and sm_source in target_name.lower()):
                return True
        
        return False
    
    def match_by_tags(self, source_tags: List[str], target_tags: List[str]) -> bool:
        """Match based on tags/semantics"""
        if not source_tags or not target_tags:
            return True  # If no tags specified, allow connection
        
        # Find common tags
        common_tags = set(source_tags) & set(target_tags)
        return len(common_tags) > 0
    
    def validate_by_rules(self, source_module: str, source_output: str, 
                         target_module: str, target_input: str) -> Dict[str, Any]:
        """Validate connection using predefined rules"""
        source_rules = self.connection_rules.get(source_module.split('/')[-1], {})
        target_rules = self.connection_rules.get(target_module.split('/')[-1], {})
        
        source_outputs = source_rules.get('outputs', {})
        target_inputs = target_rules.get('inputs', {})
        
        # Check if source produces the expected output
        source_compatible = False
        for out_name, out_props in source_outputs.items():
            if out_name == source_output or source_output in out_props:
                source_compatible = True
                break
        
        # Check if target expects the provided input  
        target_compatible = False
        for in_name, in_props in target_inputs.items():
            if in_name == target_input or target_input in in_props:
                target_compatible = True
                break
        
        valid = source_compatible and target_compatible
        
        return {
            'valid': valid,
            'source_compatible': source_compatible,
            'target_compatible': target_compatible,
            'error': "" if valid else f"Connection not allowed by rules: {source_output} to {target_input}"
        }


def generate_technical_documentation():
    """Generate technical documentation with data flow diagrams"""
    print("Generating Technical Data Flow Documentation...")
    
    # Create validator
    validator = DataFlowValidator()
    tech_validator = TechnicalDataValidator()
    
    # Example module combination
    modules = [
        'lighting/normal_mapping/normal_mapping',
        'lighting/pbr/pbr_lighting', 
        'lighting/shadow_mapping/shadow_mapping'
    ]
    
    # Build data flow graph
    graph = validator.build_data_flow_graph(modules)
    
    # Generate PlantUML
    puml_diagram = validator.generate_plantuml_diagram(graph, 'data_flow_diagram.puml')
    print("✓ Generated PlantUML diagram: data_flow_diagram.puml")
    
    # Create technical validation report
    report = {
        'module_combination': modules,
        'connection_validation': [],
        'technical_specifications': {}
    }
    
    # Validate each connection
    for edge in graph['edges']:
        conn_validation = validator.validate_connection(
            edge['source'], edge['source_port'],
            edge['target'], edge['target_port']
        )
        
        tech_validation = tech_validator.validate_by_rules(
            edge['source'], edge['source_port'],
            edge['target'], edge['target_port']
        )
        
        report['connection_validation'].append({
            'connection': f"{edge['source']}:{edge['source_port']} -> {edge['target']}:{edge['target_port']}",
            'data_flow_validation': conn_validation,
            'technical_validation': tech_validation,
            'name_match': tech_validator.match_by_names(edge['source_port'], edge['target_port'])
        })
    
    # Save report
    with open('technical_data_flow_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("✓ Generated technical report: technical_data_flow_report.json")
    
    return report


if __name__ == "__main__":
    report = generate_technical_documentation()
    print(f"✓ Technical documentation generated with {len(report['connection_validation'])} validated connections")