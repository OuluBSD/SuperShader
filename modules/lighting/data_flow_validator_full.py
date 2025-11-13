#!/usr/bin/env python3
"""
Data Flow Connection Validator
Validates connections between modules using interface information
"""

from typing import Dict, List, Tuple, Optional
from .interface_extractor import ModuleInterfaceExtractor, ShaderVariable, ParameterDirection
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from create_module_registry import ModuleRegistry
import json


class DataFlowValidator:
    def __init__(self):
        self.registry = ModuleRegistry()
        self.extractor = ModuleInterfaceExtractor()
    
    def get_module_interface(self, module_name: str):
        """Get interface for a specific module from the registry"""
        for genre, modules in self.registry.modules.items():
            for mod_name, mod_info in modules.items():
                full_name = f"{genre}/{mod_name}"
                
                if full_name == module_name or mod_name.split('/')[-1] == module_name:
                    # Try to get interface from the module directly if available
                    try:
                        import importlib
                        module_path_parts = mod_name.split('/')
                        if len(module_path_parts) == 2:
                            module_dir, module_file = module_path_parts
                            module_imp = importlib.import_module(f"modules.{genre}.{module_dir}.{module_file}")
                            
                            if hasattr(module_imp, 'get_interface'):
                                interface_info = module_imp.get_interface()
                                return self._interface_from_dict(interface_info)
                            elif hasattr(module_imp, 'get_pseudocode'):
                                pseudocode = module_imp.get_pseudocode()
                                return self.extractor.extract_interface(pseudocode, module_name)
                    except Exception as e:
                        print(f"Error getting interface for {module_name}: {e}")
                        # Fallback to pseudocode parsing
                        pseudocode = mod_info.get('metadata', {}).get('pseudocode', '')
                        if pseudocode:
                            return self.extractor.extract_interface(pseudocode, module_name)
        
        # If not found, return empty interface
        return self.extractor.extract_interface("", module_name)
    
    def _interface_from_dict(self, interface_dict: Dict) -> 'ModuleInterface':
        """Convert interface dictionary to ModuleInterface object"""
        from .interface_extractor import ShaderVariable, ParameterDirection
        from dataclasses import dataclass
        from typing import List
        
        @dataclass
        class ModuleInterface:
            inputs: List[ShaderVariable]
            outputs: List[ShaderVariable] 
            uniforms: List[ShaderVariable]
            samplers: List[ShaderVariable]
            functions: List[Dict[str, Any]]
        
        inputs = []
        outputs = []
        uniforms = []
        samplers = []
        
        for inp in interface_dict.get('inputs', []):
            direction = ParameterDirection.INPUT if inp.get('direction') == 'in' else ParameterDirection.OUTPUT
            if inp.get('direction') == 'uniform':
                direction = ParameterDirection.UNIFORM
            
            var = ShaderVariable(
                name=inp['name'],
                type=inp['type'], 
                direction=direction,
                semantic=inp.get('semantic', ''),
                description=inp.get('description', '')
            )
            if inp.get('direction') == 'in':
                inputs.append(var)
            elif inp.get('direction') == 'out':
                outputs.append(var)
            elif inp.get('direction') == 'uniform':
                uniforms.append(var)
        
        for uniform in interface_dict.get('uniforms', []):
            var = ShaderVariable(
                name=uniform['name'],
                type=uniform['type'],
                direction=ParameterDirection.UNIFORM,
                semantic=uniform.get('semantic', '')
            )
            uniforms.append(var)
        
        # Identify samplers from uniforms
        for uniform in uniforms:
            if 'sampler' in uniform.type.lower():
                samplers.append(uniform)
        
        return self.extractor.__class__.__annotations__['ModuleInterface'](
            inputs=inputs,
            outputs=outputs,
            uniforms=uniforms,
            samplers=samplers,
            functions=[]
        )
    
    def validate_connection(self, source_module: str, source_output: str, 
                          target_module: str, target_input: str) -> Dict[str, any]:
        """Validate a connection between modules"""
        source_interface = self.get_module_interface(source_module)
        target_interface = self.get_module_interface(target_module)
        
        # Find the source output
        source_var = None
        for out in source_interface.outputs:
            if out.name == source_output:
                source_var = out
                break
        
        # If not in outputs, check if it's a uniform that can be passed
        if not source_var:
            for uniform in source_interface.uniforms:
                if uniform.name == source_output:
                    source_var = uniform
                    break
        
        # Find the target input
        target_var = None
        for inp in target_interface.inputs:
            if inp.name == target_input:
                target_var = inp
                break
        
        # If not in inputs, check if it's a uniform
        if not target_var:
            for uniform in target_interface.uniforms:
                if uniform.name == target_input:
                    target_var = uniform
                    break
        
        # Validate the connection
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'source_type': source_var.type if source_var else None,
            'target_type': target_var.type if target_var else None
        }
        
        if not source_var:
            result['errors'].append(f"Source output '{source_output}' not found in module '{source_module}'")
        if not target_var:
            result['errors'].append(f"Target input '{target_input}' not found in module '{target_module}'")
        
        if source_var and target_var:
            # Validate type compatibility
            type_match = self._validate_type_compatibility(source_var.type, target_var.type)
            semantic_match = self._validate_semantic_compatibility(source_var.semantic, target_var.semantic)
            
            if type_match:
                result['valid'] = True
                if not semantic_match:
                    result['warnings'].append(f"Semantic mismatch: '{source_var.semantic}' vs '{target_var.semantic}'")
            else:
                result['errors'].append(f"Type mismatch: '{source_var.type}' cannot connect to '{target_var.type}'")
        
        return result
    
    def _validate_type_compatibility(self, source_type: str, target_type: str) -> bool:
        """Validate if two types are compatible for connection"""
        # Exact match
        if source_type == target_type:
            return True
        
        # Vector compatibility rules
        vector_types = {'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4', 'bvec2', 'bvec3', 'bvec4'}
        
        if source_type in vector_types and target_type in vector_types:
            # vec3 can connect to vec4 (w=1.0 assumed for positions)
            # vec4 can connect to vec3 (w component dropped)
            if (source_type == 'vec3' and target_type == 'vec4'):
                return True
            if (source_type == 'vec4' and target_type == 'vec3'):
                return True
            # Same base type (vec, ivec, bvec) with different dimensions
            source_base = source_type[:4]  # vec, ivec, bvec
            target_base = target_type[:4]
            return source_base == target_base
        
        # Numeric compatibility
        numeric_types = {'float', 'int', 'bool'}
        if source_type in numeric_types and target_type in numeric_types:
            return True
        
        # Sampler compatibility
        if 'sampler' in source_type and 'sampler' in target_type:
            return True
        
        return False
    
    def _validate_semantic_compatibility(self, source_semantic: str, target_semantic: str) -> bool:
        """Validate if two semantics are compatible"""
        if not source_semantic or not target_semantic:
            return True  # No semantic means compatible
        
        # Common semantic compatibility rules
        semantic_groups = [
            {'position', 'pos', 'fragpos', 'worldpos'},
            {'normal', 'norm', 'n'},
            {'tex_coords', 'uv', 'texcoord'},
            {'color', 'col', 'albedo', 'diffuse'},
            {'light', 'light_color', 'lightdir', 'light_dir'},
            {'tangent', 't'},
            {'bitangent', 'binormal', 'b'}
        ]
        
        for group in semantic_groups:
            if source_semantic in group and target_semantic in group:
                return True
        
        return source_semantic == target_semantic
    
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
                'inputs': [{'name': v.name, 'type': v.type, 'semantic': v.semantic} for v in interface.inputs],
                'outputs': [{'name': v.name, 'type': v.type, 'semantic': v.semantic} for v in interface.outputs],
                'uniforms': [{'name': v.name, 'type': v.type, 'semantic': v.semantic} for v in interface.uniforms]
            }
            graph['nodes'].append(node)
        
        # Try to find valid connections between modules
        for i, source_module in enumerate(modules):
            for j, target_module in enumerate(modules):
                if i != j:  # Don't connect module to itself
                    source_interface = self.get_module_interface(source_module)
                    target_interface = self.get_module_interface(target_module)
                    
                    # Try connecting outputs to inputs
                    for source_output in source_interface.outputs:
                        for target_input in target_interface.inputs:
                            validation = self.validate_connection(
                                source_module, source_output.name,
                                target_module, target_input.name
                            )
                            
                            if validation['valid']:
                                edge = {
                                    'source': source_module,
                                    'target': target_module,
                                    'source_port': source_output.name,
                                    'target_port': target_input.name,
                                    'source_semantic': source_output.semantic,
                                    'target_semantic': target_input.semantic,
                                    'validation': validation
                                }
                                graph['edges'].append(edge)
        
        return graph


def generate_uml_diagrams(graph: Dict[str, any]):
    """Generate UML diagrams from the data flow graph"""
    
    # Generate PlantUML component diagram
    puml_content = "@startuml\n"
    puml_content += "title SuperShader Module Data Flow\n\n"
    
    # Add modules as components
    for node in graph['nodes']:
        puml_content += f"component [{node['label']}] as {node['id'].replace('/', '_').replace('-', '_')}\n"
    
    puml_content += "\n"
    
    # Add connections with semantic information
    for edge in graph['edges']:
        source_id = edge['source'].replace('/', '_').replace('-', '_')
        target_id = edge['target'].replace('/', '_').replace('-', '_')
        semantic_info = f" : {edge['source_semantic']} -> {edge['target_semantic']}"
        puml_content += f"{source_id} --> {target_id}{semantic_info}\n"
    
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
    
    # Validate specific connections if needed
    test_connections = [
        ('lighting/normal_mapping/normal_mapping', 'normalOut', 
         'lighting/diffuse/diffuse_lighting', 'normalIn'),
        ('lighting/diffuse/diffuse_lighting', 'diffuseOut', 
         'lighting/specular/specular_lighting', 'diffuseIn')
    ]
    
    print(f"\nTesting specific connections:")
    for source_mod, source_out, target_mod, target_in in test_connections:
        validation = validator.validate_connection(source_mod, source_out, target_mod, target_in)
        status = "✓ VALID" if validation['valid'] else "✗ INVALID"
        print(f"  {status}: {source_mod}.{source_out} -> {target_mod}.{target_in}")
        if validation['errors']:
            for error in validation['errors']:
                print(f"    ERROR: {error}")
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"    WARNING: {warning}")
    
    return graph


if __name__ == "__main__":
    graph = run_data_flow_validation()
    print(f"\n✓ Data flow validation completed successfully!")
    print(f"  - {len(graph['nodes'])} modules validated")
    print(f"  - {len(graph['edges'])} valid connections identified")
    print(f"  - Technical documentation generated")