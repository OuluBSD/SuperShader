#!/usr/bin/env python3
"""
Module Configuration System with Branching
Handles branching for conflicting features in lighting modules
"""

import json
from enum import Enum


class BranchType(Enum):
    EXCLUSIVE = "exclusive"  # Only one can be selected (e.g., PBR vs Cel Shading)
    ADDITIVE = "additive"   # Multiple can be combined (e.g., different light types)
    CONDITIONAL = "conditional"  # Depends on other factors


class ModuleBrancher:
    def __init__(self):
        self.branches = {
            # Lighting approach branches (exclusive)
            'lighting_approach': {
                'type': BranchType.EXCLUSIVE,
                'options': {
                    'pbr': ['pbr_lighting', 'normal_mapping', 'specular_lighting', 'diffuse_lighting'],
                    'cel_shading': ['cel_shading', 'diffuse_lighting'],
                    'phong': ['diffuse_lighting', 'specular_lighting'],
                    'raymarching': ['raymarching_lighting']
                }
            },
            # Light type branches (additive)
            'light_types': {
                'type': BranchType.ADDITIVE,
                'options': {
                    'point_light': ['basic_point_light'],
                    'directional_light': ['directional_light'],
                    'spot_light': ['spot_light']
                }
            },
            # Special effects branches (additive/conditional)
            'effects': {
                'type': BranchType.CONDITIONAL,
                'options': {
                    'shadows': ['shadow_mapping'],
                    'multiple_lights': ['basic_point_light', 'directional_light', 'spot_light'],
                    'fresnel': ['specular_lighting']  # Only if specular is selected
                }
            }
        }
    
    def get_branch_options(self, branch_name):
        """Get available options for a specific branch"""
        if branch_name in self.branches:
            return self.branches[branch_name]['options']
        return {}
    
    def validate_selection(self, branch_selections):
        """Validate that branch selections are compatible"""
        errors = []
        
        # Check exclusive branches
        for branch_name, branch_config in self.branches.items():
            if branch_config['type'] == BranchType.EXCLUSIVE:
                if branch_name in branch_selections:
                    selected_count = len(branch_selections[branch_name]) if isinstance(branch_selections[branch_name], list) else 1
                    if selected_count > 1:
                        errors.append(f"Exclusive branch '{branch_name}' can only have one selection")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def resolve_conflicts(self, requested_modules):
        """Resolve conflicts between requested modules"""
        resolved_modules = []
        conflicts_handled = []
        
        # Group modules by their branch
        branch_modules = {}
        
        # Determine which branch each requested module belongs to
        for module in requested_modules:
            assigned = False
            for branch_name, branch_config in self.branches.items():
                for option_name, option_modules in branch_config['options'].items():
                    if module in option_modules:
                        if branch_name not in branch_modules:
                            branch_modules[branch_name] = []
                        branch_modules[branch_name].append(module)
                        assigned = True
                        break
            if not assigned:
                # Module doesn't belong to any special branch, add directly
                if module not in resolved_modules:
                    resolved_modules.append(module)
        
        # Handle each branch according to its type
        for branch_name, modules in branch_modules.items():
            branch_config = self.branches[branch_name]
            
            if branch_config['type'] == BranchType.EXCLUSIVE:
                # For exclusive branches, only take the first module from each option group
                # In a real implementation, we'd have to choose which option to keep
                if modules:
                    # For now, we just take the first module from an option group
                    for option_name, option_modules in branch_config['options'].items():
                        if any(mod in modules for mod in option_modules):
                            resolved_modules.extend([m for m in option_modules if m in modules])
                            break
            elif branch_config['type'] == BranchType.ADDITIVE:
                # Additive branches can have multiple selections
                resolved_modules.extend(modules)
            elif branch_config['type'] == BranchType.CONDITIONAL:
                # Conditional branches depend on other selections
                resolved_modules.extend(modules)
        
        return {
            'modules': list(set(resolved_modules)),  # Remove duplicates
            'conflicts_handled': conflicts_handled,
            'original_request': requested_modules
        }
    
    def generate_branch_config(self, lighting_approach='pbr', light_types=None, effects=None):
        """Generate a configuration based on branch selections"""
        if light_types is None:
            light_types = ['point_light']
        if effects is None:
            effects = ['shadows']
        
        config = {
            'selected_branches': {
                'lighting_approach': lighting_approach,
                'light_types': light_types,
                'effects': effects
            },
            'modules': []
        }
        
        # Add modules based on selections
        if lighting_approach in self.branches['lighting_approach']['options']:
            config['modules'].extend(self.branches['lighting_approach']['options'][lighting_approach])
        
        for light_type in light_types:
            if light_type in self.branches['light_types']['options']:
                config['modules'].extend(self.branches['light_types']['options'][light_type])
        
        for effect in effects:
            if effect in self.branches['effects']['options']:
                config['modules'].extend(self.branches['effects']['options'][effect])
        
        # Remove duplicates
        config['modules'] = list(set(config['modules']))
        
        return config


def create_branching_configurations():
    """Create several example configurations with different branches"""
    brancher = ModuleBrancher()
    
    configurations = []
    
    # Configuration 1: PBR with shadows
    config1 = brancher.generate_branch_config(
        lighting_approach='pbr',
        light_types=['point_light', 'directional_light'],
        effects=['shadows']
    )
    configurations.append(('pbr_with_shadows', config1))
    
    # Configuration 2: Cel shading
    config2 = brancher.generate_branch_config(
        lighting_approach='cel_shading',
        light_types=['point_light'],
        effects=[]
    )
    configurations.append(('cel_shading', config2))
    
    # Configuration 3: Phong lighting
    config3 = brancher.generate_branch_config(
        lighting_approach='phong',
        light_types=['point_light', 'spot_light'],
        effects=['shadows']
    )
    configurations.append(('phong_lighting', config3))
    
    # Configuration 4: Raymarching
    config4 = brancher.generate_branch_config(
        lighting_approach='raymarching',
        light_types=[],
        effects=[]
    )
    configurations.append(('raymarching', config4))
    
    # Save configurations
    for name, config in configurations:
        with open(f'config_{name}.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created configuration: config_{name}.json")
    
    return configurations


if __name__ == "__main__":
    brancher = ModuleBrancher()
    
    print("Module Branching System Initialized")
    print("Available Branches:")
    for branch_name, branch_config in brancher.branches.items():
        print(f"  - {branch_name}: {branch_config['type'].value}")
        print(f"    Options: {list(branch_config['options'].keys())}")
    
    # Create example configurations
    configs = create_branching_configurations()
    
    print(f"\nCreated {len(configs)} example configurations")
    
    # Example of conflict resolution
    print("\nTesting conflict resolution...")
    conflicting_request = ['pbr_lighting', 'cel_shading', 'basic_point_light', 'diffuse_lighting']
    resolved = brancher.resolve_conflicts(conflicting_request)
    print(f"Original request: {resolved['original_request']}")
    print(f"Resolved modules: {resolved['modules']}")
    print(f"Conflicts handled: {resolved['conflicts_handled']}")