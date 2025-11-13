#!/usr/bin/env python3
"""
Cross-Genre Data Flow Validation System for SuperShader Modules
Implements connection validation across different shader genres
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
    COMPUTED = "computed"


class CrossGenreDataFlowValidator:
    """Validates data flow between modules across different genres"""

    def __init__(self):
        self.registry = ModuleRegistry()
        self.connections = {}  # module_name -> {input_connections, output_connections}
        self.type_registry = {}  # Maps variable names to types
        self.genre_rules = self._define_genre_rules()

    def _define_genre_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define rules for how different genres connect to each other"""
        return {
            # Lighting can connect to most other genres
            'lighting': {
                'outputs': ['light_color', 'light_dir', 'diffuse', 'specular', 'normal', 'position'],
                'inputs': [],
                'compatible_with': ['geometry', 'effects', 'texturing', 'raymarching']
            },
            # Raymarching has specific connection requirements
            'raymarching': {
                'outputs': ['position', 'normal', 'distance', 'material_id'],
                'inputs': ['ro', 'rd', 'map'],  # ray origin, ray direction, distance function
                'compatible_with': ['lighting', 'geometry', 'effects']
            },
            # Geometry connects to most rendering types
            'geometry': {
                'outputs': ['position', 'normal', 'tangent', 'bitangent', 'uv', 'frag_pos'],
                'inputs': [],
                'compatible_with': ['lighting', 'texturing', 'effects', 'raymarching']
            },
            # Texturing connects to rendering types
            'texturing': {
                'outputs': ['diffuse_color', 'normal_map', 'specular_map', 'roughness_map'],
                'inputs': ['uv', 'normal'],
                'compatible_with': ['lighting', 'geometry', 'effects']
            },
            # Effects are generally compatible with most rendering genres
            'effects': {
                'outputs': ['color', 'alpha', 'distortion'],
                'inputs': ['color', 'position', 'uv'],
                'compatible_with': ['lighting', 'geometry', 'texturing', 'raymarching']
            }
        }

    def get_module_interface(self, module_name: str) -> Dict[str, Any]:
        """Extract interface information from a module"""
        # Get module from registry
        for genre, modules in self.registry.modules.items():
            for mod_name, mod_info in modules.items():
                full_name = f"{genre}/{mod_name}"
                if full_name == module_name or mod_name.split('/')[-1] == module_name or mod_name == module_name:
                    # Try to parse the pseudocode to get interface info
                    try:
                        import importlib.util
                        path = mod_info['path']
                        spec = importlib.util.spec_from_file_location(mod_name.split('/')[-1], path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        pseudocode = module.get_pseudocode()
                        return self._parse_interface(pseudocode, genre)
                    except Exception as e:
                        print(f"Error parsing interface for {module_name}: {e}")
                        # Return basic interface if parsing fails
                        return {'inputs': [], 'outputs': [], 'uniforms': [], 'samplers': [], 'genre': genre}

        return {'inputs': [], 'outputs': [], 'uniforms': [], 'samplers': [], 'genre': 'unknown'}

    def _parse_interface(self, pseudocode: str, genre: str) -> Dict[str, Any]:
        """Parse pseudocode to extract interface definitions"""
        inputs = []
        outputs = []
        uniforms = []
        samplers = []

        lines = pseudocode.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()

            # Look for uniform declarations
            if 'uniform' in line:
                if 'sampler' in line:
                    # Extract sampler name
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if 'sampler' in part:
                            if j+1 < len(parts):
                                sampler_name = parts[j+1]
                                if sampler_name.endswith(';'):
                                    sampler_name = sampler_name[:-1]
                                samplers.append({
                                    'name': sampler_name,
                                    'type': part,
                                    'line_number': i
                                })
                else:
                    # Extract uniform
                    parts = line.split()
                    if len(parts) >= 3:
                        uniform_type = parts[1]
                        uniform_name = parts[2]
                        if uniform_name.endswith(';'):
                            uniform_name = uniform_name[:-1]
                        uniforms.append({
                            'name': uniform_name,
                            'type': uniform_type,
                            'line_number': i
                        })

            # Look for function parameters (inputs) - more sophisticated parsing
            if '(' in line and ')' in line:
                # Look for function definitions that might have inputs
                func_def = line
                if 'vec' in func_def or 'float' in func_def or 'int' in func_def:
                    # Extract parameters from function definition
                    params_part = func_def[func_def.find('(')+1:func_def.find(')')]
                    if params_part.strip():
                        params = [p.strip() for p in params_part.split(',')]
                        for param in params:
                            if param.strip():
                                param_parts = param.split()
                                if len(param_parts) >= 2:
                                    param_type = param_parts[0]
                                    param_name = param_parts[1]
                                    inputs.append({
                                        'name': param_name,
                                        'type': param_type,
                                        'line_number': i
                                    })

            # Look for output variable declarations
            if 'out' in line:
                parts = line.split()
                if len(parts) >= 3:
                    output_type = parts[1]
                    output_name = parts[2]
                    if output_name.endswith(';'):
                        output_name = output_name[:-1]
                    outputs.append({
                        'name': output_name,
                        'type': output_type,
                        'line_number': i
                    })

        return {
            'inputs': inputs,
            'outputs': outputs,
            'uniforms': uniforms,
            'samplers': samplers,
            'genre': genre
        }

    def validate_cross_genre_connection(self, source_module: str, source_output: str,
                                      target_module: str, target_input: str) -> Dict[str, Any]:
        """Validate a connection between modules of different genres"""
        source_interface = self.get_module_interface(source_module)
        target_interface = self.get_module_interface(target_module)

        source_genre = source_interface.get('genre', 'unknown')
        target_genre = target_interface.get('genre', 'unknown')

        # Check genre compatibility
        genre_compatible = self._check_genre_compatibility(source_genre, target_genre)
        
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
        type_valid = False
        error = ""

        if not source_output_def:
            error = f"Source output '{source_output}' not found in module '{source_module}'"
        elif not target_input_def:
            error = f"Target input '{target_input}' not found in module '{target_module}'"
        else:
            # Check type compatibility
            type_valid, type_error = self._check_type_compatibility(
                source_output_def['type'], target_input_def['type']
            )
            if not type_valid:
                error = type_error

        # Combine genre and type validation
        valid = genre_compatible and type_valid

        return {
            'valid': valid,
            'genre_compatible': genre_compatible,
            'type_valid': type_valid,
            'error': error,
            'source_type': source_output_def['type'] if source_output_def else None,
            'target_type': target_input_def['type'] if target_input_def else None,
            'source_genre': source_genre,
            'target_genre': target_genre
        }

    def _check_genre_compatibility(self, source_genre: str, target_genre: str) -> bool:
        """Check if two genres are compatible for connection"""
        if source_genre == target_genre:
            return True

        genre_rules = self.genre_rules.get(source_genre, {})
        compatible_genres = genre_rules.get('compatible_with', [])
        
        return target_genre in compatible_genres

    def _check_type_compatibility(self, source_type: str, target_type: str) -> Tuple[bool, str]:
        """Check if two types are compatible for connection"""
        # Basic compatibility rules
        if source_type == target_type:
            return True, ""

        # Vector compatibility (vec3 can connect to vec4 if w component is handled)
        if (source_type == 'vec3' and target_type == 'vec4'):
            return True, "Warning: vec3 to vec4 connection - w component will be default (1.0)"

        if (source_type == 'vec4' and target_type == 'vec3'):
            return True, "Warning: vec4 to vec3 connection - w component will be dropped"

        # Scalar compatibility
        if source_type in ['float', 'int'] and target_type in ['float', 'int']:
            return True, ""

        # Texture compatibility
        if 'sampler' in source_type and 'sampler' in target_type:
            return True, ""

        return False, f"Type mismatch: {source_type} cannot connect to {target_type}"

    def build_cross_genre_data_flow_graph(self, module_combination: List[str]) -> Dict[str, Any]:
        """Build a complete data flow graph across different genres"""
        graph = {
            'nodes': [],
            'edges': [],
            'validation': {'errors': [], 'warnings': [], 'genre_connections': []}
        }

        # Add nodes for each module with genre information
        for module in module_combination:
            interface = self.get_module_interface(module)
            # Extract genre from module name (first part before slash)
            genre = interface.get('genre', module.split('/')[0] if '/' in module else 'unknown')
            
            graph['nodes'].append({
                'id': module,
                'label': module.split('/')[-1],
                'genre': genre,
                'interface': interface
            })

        # Determine possible connections between modules across genres
        for i, source_module in enumerate(module_combination):
            for j, target_module in enumerate(module_combination):
                if i != j:  # Don't connect module to itself
                    # Check for possible connections between compatible interfaces
                    source_interface = self.get_module_interface(source_module)
                    target_interface = self.get_module_interface(target_module)

                    # Connect compatible outputs to compatible inputs
                    for source_output in source_interface['outputs']:
                        for target_input in target_interface.get('inputs', []):
                            validation = self.validate_cross_genre_connection(
                                source_module, source_output['name'],
                                target_module, target_input['name']
                            )

                            if validation['valid']:
                                edge = {
                                    'source': source_module,
                                    'target': target_module,
                                    'source_port': source_output['name'],
                                    'target_port': target_input['name'],
                                    'source_genre': validation['source_genre'],
                                    'target_genre': validation['target_genre'],
                                    'type': 'data',
                                    'validation': validation
                                }
                                graph['edges'].append(edge)

        # Validate genre compatibility across the entire graph
        genre_connections = {}
        for edge in graph['edges']:
            key = f"{edge['source_genre']} -> {edge['target_genre']}"
            if key not in genre_connections:
                genre_connections[key] = 0
            genre_connections[key] += 1

        graph['validation']['genre_connections'] = genre_connections

        return graph

    def generate_plantuml_diagram(self, graph: Dict[str, Any], filename: str = "cross_genre_data_flow.puml"):
        """Generate PlantUML diagram from cross-genre data flow graph"""
        puml_content = "@startuml\n"
        puml_content += "title SuperShader Cross-Genre Data Flow Diagram\n\n"

        # Define different colors for different genres
        genre_colors = {
            'lighting': '#FFE6E6',  # Light red
            'raymarching': '#E6F3FF',  # Light blue
            'geometry': '#E6FFE6',  # Light green
            'texturing': '#FFF0E6',  # Light orange
            'effects': '#F0E6FF',  # Light purple
            'default': '#FFFFFF'  # White
        }

        # Add modules as components with genre-specific styling
        for node in graph['nodes']:
            genre = node['genre']
            color = genre_colors.get(genre, genre_colors['default'])
            puml_content += f"package \"{genre}\" {{\n"
            puml_content += f"  component [{node['label']}] as {node['id'].replace('/', '_').replace('-', '_')} # {color}\n"
            puml_content += "}\n"

        puml_content += "\n"

        # Add connections with validation status
        for edge in graph['edges']:
            source_id = edge['source'].replace('/', '_').replace('-', '_')
            target_id = edge['target'].replace('/', '_').replace('-', '_')
            
            # Color edge based on validation
            validation = edge['validation']
            if validation['valid']:
                puml_content += f"{source_id} --> {target_id} : {edge['source_port']} -> {edge['target_port']}\n"
            else:
                puml_content += f"{source_id} -[dotted]-> {target_id} : {edge['source_port']} -> {edge['target_port']} [[Validation Error: {validation['error']}]]\n"

        puml_content += "\n@enduml"

        with open(filename, 'w') as f:
            f.write(puml_content)

        return puml_content


def validate_all_modules():
    """Validate data flow across all registered modules"""
    print("Validating cross-genre data flow across all modules...")

    validator = CrossGenreDataFlowValidator()
    registry = ModuleRegistry()

    # Get all modules by genre
    all_modules = registry.get_all_modules()

    # Group modules by genre
    modules_by_genre = {}
    for module in all_modules:
        genre = module['genre']
        if genre not in modules_by_genre:
            modules_by_genre[genre] = []
        modules_by_genre[genre].append(f"{genre}/{module['module_name']}")

    print(f"Found modules in genres: {list(modules_by_genre.keys())}")

    # Test validation across some combinations
    test_combinations = [
        ['geometry', 'lighting'],  # Geometry + Lighting
        ['raymarching', 'lighting'],  # Raymarching + Lighting
        ['texturing', 'lighting'],  # Texturing + Lighting
        ['effects', 'lighting'],  # Effects + Lighting
        ['raymarching', 'effects']  # Raymarching + Effects
    ]

    results = {}
    for combo in test_combinations:
        if len(combo) == 2:
            source_genre, target_genre = combo
            if source_genre in modules_by_genre and target_genre in modules_by_genre:
                # Select a module from each genre for testing
                source_module = modules_by_genre[source_genre][0]
                target_module = modules_by_genre[target_genre][0]
                
                # Try to get an output from source and input from target
                source_interface = validator.get_module_interface(source_module)
                target_interface = validator.get_module_interface(target_module)
                
                # If we have valid outputs and inputs, validate connection
                if source_interface['outputs'] and target_interface['inputs']:
                    source_output = source_interface['outputs'][0]['name']
                    target_input = target_interface['inputs'][0]['name']
                    
                    validation = validator.validate_cross_genre_connection(
                        source_module, source_output,
                        target_module, target_input
                    )
                    
                    results[f"{source_genre} to {target_genre}"] = {
                        'source_module': source_module,
                        'target_module': target_module,
                        'connection': f"{source_output} -> {target_input}",
                        'validation': validation
                    }

    # Print validation results
    print("\nCross-genre validation results:")
    for combo, result in results.items():
        status = "✓ VALID" if result['validation']['valid'] else "✗ INVALID"
        print(f"  {combo}: {status}")
        if not result['validation']['valid']:
            print(f"    Error: {result['validation']['error']}")
        print(f"    Connection: {result['connection']}")

    # Generate a graph for a specific combination
    sample_modules = []
    for genre, modules in modules_by_genre.items():
        if modules:
            sample_modules.append(modules[0])
            if len(sample_modules) >= 4:  # Limit to 4 modules for the example
                break

    if sample_modules:
        print(f"\nGenerating data flow graph for modules: {sample_modules}")
        graph = validator.build_cross_genre_data_flow_graph(sample_modules)
        
        # Count valid vs invalid connections
        valid_count = sum(1 for edge in graph['edges'] if edge['validation']['valid'])
        total_count = len(graph['edges'])
        print(f"Graph has {valid_count}/{total_count} valid connections")
        
        # Generate PlantUML diagram
        puml_diagram = validator.generate_plantuml_diagram(graph, 'cross_genre_flow_diagram.puml')
        print("✓ Generated cross-genre data flow diagram: cross_genre_flow_diagram.puml")

    return results


if __name__ == "__main__":
    results = validate_all_modules()
    print("\nCross-genre validation completed!")