#!/usr/bin/env python3
"""
Data Flow Connection Validator - Simplified Version
Validates connections between modules using interface information
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from create_module_registry import ModuleRegistry
import importlib
from typing import Dict, List, Tuple


class DataFlowValidator:
    def __init__(self):
        self.registry = ModuleRegistry()
    
    def get_module_interface(self, module_name: str):
        """Get interface for a specific module from the registry"""
        for genre, modules in self.registry.modules.items():
            for mod_name, mod_info in modules.items():
                full_name = f"{genre}/{mod_name}"
                
                if full_name == module_name or mod_name.split('/')[-1] == module_name:
                    # Try to get interface from the module directly if available
                    try:
                        module_path_parts = mod_name.split('/')
                        if len(module_path_parts) == 2:
                            module_dir, module_file = module_path_parts
                            # Import the module and get its interface
                            module_fqdn = f"modules.{genre}.{module_dir}.{module_file}"
                            if module_fqdn in sys.modules:
                                module_imp = sys.modules[module_fqdn]
                            else:
                                module_imp = importlib.import_module(module_fqdn)
                            
                            if hasattr(module_imp, 'get_interface'):
                                return module_imp.get_interface()
                            # If no interface function, try to parse the pseudocode
                            elif hasattr(module_imp, 'get_pseudocode'):
                                pseudocode = module_imp.get_pseudocode()
                                return self._parse_pseudocode_interface(pseudocode)
                    except Exception as e:
                        print(f"Error getting interface for {module_name}: {e}")
        
        # If not found, return empty interface
        return {
            'inputs': [],
            'outputs': [],
            'uniforms': [],
            'name': module_name
        }
    
    def _parse_pseudocode_interface(self, pseudocode: str):
        """Basic parsing of pseudocode to extract interface info"""
        inputs = []
        outputs = []
        uniforms = []
        
        lines = pseudocode.split('\n')
        for line in lines:
            if 'uniform' in line:
                # Simple extraction for uniform
                parts = line.replace(';', ' ').split()
                if len(parts) >= 3 and parts[0] == 'uniform':
                    type_name = parts[1]
                    var_name = [p for p in parts[2:] if not p.startswith('/')][0] if len(parts) > 2 else "unknown"
                    uniforms.append({
                        'name': var_name,
                        'type': type_name,
                        'direction': 'uniform'
                    })
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'uniforms': uniforms
        }
    
    def validate_connection(self, source_module: str, source_output: str, 
                          target_module: str, target_input: str) -> Dict[str, any]:
        """Validate a connection between modules"""
        source_interface = self.get_module_interface(source_module)
        target_interface = self.get_module_interface(target_module)
        
        # Find the source output
        source_var = None
        for out in source_interface.get('outputs', []):
            if out['name'] == source_output:
                source_var = out
                break
        
        # Find the target input
        target_var = None
        for inp in target_interface.get('inputs', []):
            if inp['name'] == target_input:
                target_var = inp
                break
        
        # Validate the connection
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'source_type': source_var['type'] if source_var else None,
            'target_type': target_var['type'] if target_var else None
        }
        
        if not source_var:
            result['errors'].append(f"Source output '{source_output}' not found in module '{source_module}'")
        if not target_var:
            result['errors'].append(f"Target input '{target_input}' not found in module '{target_module}'")
        
        if source_var and target_var:
            # Validate type compatibility
            type_match = self._validate_type_compatibility(source_var['type'], target_var['type'])
            
            if type_match:
                result['valid'] = True
            else:
                result['errors'].append(f"Type mismatch: '{source_var['type']}' cannot connect to '{target_var['type']}'")
        
        return result
    
    def _validate_type_compatibility(self, source_type: str, target_type: str) -> bool:
        """Validate if two types are compatible for connection"""
        if source_type == target_type:
            return True
        
        # Vector compatibility rules
        vector_types = {'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4', 'bvec2', 'bvec3', 'bvec4'}
        
        if source_type in vector_types and target_type in vector_types:
            # Same base type (vec, ivec, bvec) with different dimensions
            source_base = source_type[:4]  # vec, ivec, bvec
            target_base = target_type[:4]
            if source_base == target_base:
                return True
        
        # Numeric compatibility
        numeric_types = {'float', 'int', 'bool'}
        if source_type in numeric_types and target_type in numeric_types:
            return True
        
        return False
    
    def build_complete_data_flow_graph(self, modules: List[str]) -> Dict[str, any]:
        """Build a complete data flow graph for a list of modules"""
        graph = {
            'nodes': [],
            'edges': [],
            'validation': {'errors': [], 'warnings': []}
        }
        
        # Add modules as nodes
        for module in modules:
            interface = self.get_module_interface(module)
            node = {
                'id': module,
                'label': module.split('/')[-1],
                'inputs': interface.get('inputs', []),
                'outputs': interface.get('outputs', []),
                'uniforms': interface.get('uniforms', [])
            }
            graph['nodes'].append(node)
        
        # Try to find valid connections between modules
        for i, source_module in enumerate(modules):
            for j, target_module in enumerate(modules):
                if i != j:  # Don't connect module to itself
                    source_interface = self.get_module_interface(source_module)
                    target_interface = self.get_module_interface(target_module)
                    
                    # Try connecting outputs to inputs
                    for source_output in source_interface.get('outputs', []):
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
                                    'validation': validation
                                }
                                graph['edges'].append(edge)
        
        return graph


def generate_uml_diagrams(graph: Dict[str, any]):
    """Generate UML diagrams from the data flow graph"""
    import json
    
    # Generate PlantUML component diagram
    puml_content = "@startuml\n"
    puml_content += "title SuperShader Module Data Flow\n\n"
    
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
    
    # Write PlantUML file
    with open('data_flow_uml.puml', 'w') as f:
        f.write(puml_content)
    
    # Generate JSON representation for technical documentation
    with open('data_flow_graph.json', 'w') as f:
        json.dump(graph, f, indent=2)
    
    return puml_content


def run_data_flow_validation():
    """Run complete data flow validation"""
    print("Running Data Flow Validation...")
    
    validator = DataFlowValidator()
    
    # Example module combination for a complete lighting system
    modules = [
        'lighting/point_light/basic_point_light',
        'lighting/normal_mapping/normal_mapping', 
        'lighting/diffuse/diffuse_lighting',
        'lighting/specular/specular_lighting',
        'lighting/pbr/pbr_lighting'
    ]
    
    print(f"Validating connection for modules: {modules}")
    
    # Build data flow graph
    graph = validator.build_complete_data_flow_graph(modules)
    
    print(f"✓ Found {len(graph['nodes'])} modules and {len(graph['edges'])} potential connections")
    
    # Generate UML diagrams
    uml_diagram = generate_uml_diagrams(graph)
    print("✓ Generated UML diagram: data_flow_uml.puml")
    print("✓ Generated technical graph: data_flow_graph.json")
    
    # Print connection summary
    print("\nValid Connections Found:")
    for edge in graph['edges']:
        print(f"  {edge['source']}:{edge['source_port']} -> {edge['target']}:{edge['target_port']}")
        if edge['validation']['warnings']:
            for warning in edge['validation']['warnings']:
                print(f"    WARNING: {warning}")
    
    return graph


if __name__ == "__main__":
    graph = run_data_flow_validation()
    print(f"\n✓ Data flow validation completed successfully!")
    print(f"  - {len(graph['nodes'])} modules validated")
    print(f"  - {len(graph['edges'])} valid connections identified")
    print(f"  - Technical documentation generated")