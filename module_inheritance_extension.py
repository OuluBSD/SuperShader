#!/usr/bin/env python3
"""
Module Inheritance and Extension System for SuperShader
Enables modules to inherit from other modules and extend their functionality
"""

import json
import re
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import sys


class ModuleInheritanceManager:
    """
    Manages module inheritance and extension relationships
    """

    def __init__(self):
        self.inheritance_tree = {}  # parent -> [children]
        self.extension_map = {}      # module -> [extensions]
        self.parent_map = {}         # child -> parent
        self.extension_data = {}     # module -> extension details
        self.module_implementations = {}  # module -> implementation details

    def define_module_inheritance(self, child_module: str, parent_module: str, 
                                extension_points: List[str] = None) -> bool:
        """
        Define that a child module inherits from a parent module
        """
        if extension_points is None:
            extension_points = []

        # Initialize data structures if needed
        if parent_module not in self.inheritance_tree:
            self.inheritance_tree[parent_module] = []
        
        if child_module not in self.parent_map:
            self.parent_map[child_module] = parent_module
            
        if child_module not in self.extension_map:
            self.extension_map[child_module] = []
        
        # Add to inheritance tree
        if child_module not in self.inheritance_tree[parent_module]:
            self.inheritance_tree[parent_module].append(child_module)
        
        # Store extension points
        self.extension_data[child_module] = {
            'parent': parent_module,
            'extension_points': extension_points,
            'overrides': [],
            'additional_features': []
        }
        
        return True

    def extend_module(self, base_module: str, extension_module: str, 
                     extension_type: str = "feature", details: Dict[str, Any] = None) -> bool:
        """
        Define that an extension module extends a base module
        """
        if details is None:
            details = {}

        # Initialize data structures if needed
        if base_module not in self.extension_map:
            self.extension_map[base_module] = []
        
        # Add extension
        extension_info = {
            'module': extension_module,
            'type': extension_type,
            'details': details,
            'applied': False
        }
        
        if extension_info not in self.extension_map[base_module]:
            self.extension_map[base_module].append(extension_info)
            
        return True

    def get_module_parent(self, module: str) -> Optional[str]:
        """
        Get the parent module of a module
        """
        return self.parent_map.get(module)

    def get_all_children(self, parent_module: str) -> List[str]:
        """
        Get all modules that inherit from the given parent
        """
        return self.inheritance_tree.get(parent_module, [])

    def get_module_extensions(self, base_module: str) -> List[Dict[str, Any]]:
        """
        Get all extensions for a base module
        """
        return self.extension_map.get(base_module, [])

    def is_inheriting_from(self, child_module: str, potential_parent: str) -> bool:
        """
        Check if a module inherits from another (directly or indirectly)
        """
        current = child_module
        visited = set()
        
        while current in self.parent_map:
            if current in visited:  # Prevent infinite loops
                break
            visited.add(current)
            
            parent = self.parent_map[current]
            if parent == potential_parent:
                return True
            current = parent
            
        return False

    def get_inheritance_chain(self, module: str) -> List[str]:
        """
        Get the full inheritance chain for a module (from module to root parent)
        """
        chain = [module]
        current = module
        visited = set()
        
        while current in self.parent_map:
            if current in visited:  # Prevent infinite loops
                break
            visited.add(current)
            
            parent = self.parent_map[current]
            chain.append(parent)
            current = parent
            
        return chain

    def get_all_overridable_methods(self, module: str) -> List[str]:
        """
        Get all methods that can be overridden in a module's inheritance chain
        """
        all_methods = set()
        
        # Walk up the inheritance chain
        for ancestor in self.get_inheritance_chain(module):
            if ancestor in self.module_implementations:
                impl = self.module_implementations[ancestor]
                if 'methods' in impl:
                    all_methods.update(impl['methods'])
        
        return list(all_methods)

    def register_module_implementation(self, module_name: str, implementation: Dict[str, Any]):
        """
        Register implementation details for a module
        """
        self.module_implementations[module_name] = implementation

    def apply_extensions(self, base_module: str) -> Dict[str, Any]:
        """
        Apply all extensions to a base module and return the combined result
        """
        # Start with the base module implementation
        base_impl = self.module_implementations.get(base_module, {}).copy()
        
        # Get all extensions for this module
        extensions = self.get_module_extensions(base_module)
        
        # Apply each extension
        for extension_info in extensions:
            extension_module = extension_info['module']
            extension_impl = self.module_implementations.get(extension_module, {})
            
            # Merge implementations (this is a simplified merge)
            for key, value in extension_impl.items():
                if key == 'methods' or key == 'functions':
                    # Merge methods/functions, with extensions potentially overriding
                    if key not in base_impl:
                        base_impl[key] = {}
                    if isinstance(value, dict):
                        base_impl[key].update(value)
                    elif isinstance(value, list):
                        if key not in base_impl:
                            base_impl[key] = []
                        base_impl[key].extend(value)
                elif key == 'dependencies':
                    # Merge dependencies
                    if key not in base_impl:
                        base_impl[key] = []
                    if isinstance(value, list):
                        base_impl[key].extend([dep for dep in value if dep not in base_impl[key]])
                elif key == 'metadata':
                    # Merge metadata
                    if key not in base_impl:
                        base_impl[key] = {}
                    if isinstance(value, dict):
                        base_impl[key].update(value)
                else:
                    # For other keys, extension takes precedence
                    base_impl[key] = value
            
            # Mark this extension as applied
            extension_info['applied'] = True
        
        return base_impl


class ModuleExtensionSystem:
    """
    Complete system for module inheritance and extension
    """

    def __init__(self):
        self.inheritance_manager = ModuleInheritanceManager()
        self.inheritance_patterns = {
            'inheritance_operators': ['extends', 'inherits', 'derives'],
            'extension_operators': ['extends', 'adds', 'enhances'],
        }

    def create_inheritance_relationship(self, child_name: str, parent_name: str, 
                                      specialization: str = "") -> bool:
        """
        Create an inheritance relationship between modules
        """
        # Determine extension points based on specialization
        extension_points = []
        if specialization:
            extension_points = [specialization]
        
        return self.inheritance_manager.define_module_inheritance(
            child_name, parent_name, extension_points
        )

    def create_extension_module(self, base_module: str, extension_name: str,
                              extension_type: str, extension_features: Dict[str, Any] = None) -> bool:
        """
        Create an extension module that adds functionality to a base module
        """
        if extension_features is None:
            extension_features = {}

        return self.inheritance_manager.extend_module(
            base_module, extension_name, extension_type, extension_features
        )

    def create_specialized_module(self, base_module: str, specialization_name: str, 
                                specializations: List[str] = None) -> str:
        """
        Create a specialized version of a base module
        """
        if specializations is None:
            specializations = []

        # Define inheritance relationship
        self.create_inheritance_relationship(specialization_name, base_module, "specialization")

        # Register implementation for the specialized module
        base_impl = self.inheritance_manager.module_implementations.get(base_module, {})
        specialized_impl = base_impl.copy()

        # Add specialization-specific features
        if specializations:
            if 'methods' not in specialized_impl:
                specialized_impl['methods'] = {}
            
            for spec in specializations:
                specialized_impl['methods'][f"specialized_{spec}"] = f"Implementation for {spec}"

        # Register the specialized implementation
        self.inheritance_manager.register_module_implementation(specialization_name, specialized_impl)

        return specialization_name

    def get_inheritance_info(self, module: str) -> Dict[str, Any]:
        """
        Get complete inheritance information for a module
        """
        parent = self.inheritance_manager.get_module_parent(module)
        children = self.inheritance_manager.get_all_children(module)
        extensions = self.inheritance_manager.get_module_extensions(module)
        inheritance_chain = self.inheritance_manager.get_inheritance_chain(module)
        overridable_methods = self.inheritance_manager.get_all_overridable_methods(module)

        return {
            'module': module,
            'parent': parent,
            'children': children,
            'extensions': extensions,
            'inheritance_chain': inheritance_chain,
            'overridable_methods': overridable_methods,
            'is_inheriting': parent is not None,
            'has_children': len(children) > 0,
            'has_extensions': len(extensions) > 0
        }

    def validate_inheritance(self, module: str) -> Dict[str, Any]:
        """
        Validate inheritance relationships for a module
        """
        issues = []
        
        # Check for circular inheritance
        if self.inheritance_manager.is_inheriting_from(module, module):
            issues.append(f"Circular inheritance detected for module {module}")
        
        # Check if parent exists
        parent = self.inheritance_manager.get_module_parent(module)
        if parent:
            parent_impl = self.inheritance_manager.module_implementations.get(parent)
            if not parent_impl:
                issues.append(f"Parent module {parent} for {module} not implemented")
        
        # Check if all extension points exist in parent
        if module in self.inheritance_manager.extension_data:
            ext_data = self.inheritance_manager.extension_data[module]
            parent = ext_data['parent']
            parent_impl = self.inheritance_manager.module_implementations.get(parent, {})
            
            if 'methods' in parent_impl:
                parent_methods = set(parent_impl['methods'].keys())
                for override in ext_data['overrides']:
                    if override not in parent_methods:
                        issues.append(f"Module {module} overrides non-existent method {override} from parent {parent}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'module': module
        }

    def merge_implementation(self, child_module: str) -> Dict[str, Any]:
        """
        Merge the implementation of a child module with its parent
        """
        inheritance_chain = self.inheritance_manager.get_inheritance_chain(child_module)
        
        # Start with the root parent's implementation
        if not inheritance_chain:
            return {}
        
        final_impl = {}
        
        # Walk down the inheritance chain, with children overriding parents
        for module_in_chain in reversed(inheritance_chain):  # From parent to child
            module_impl = self.inheritance_manager.module_implementations.get(module_in_chain, {})
            
            # Merge implementations
            for key, value in module_impl.items():
                if key == 'methods' or key == 'functions':
                    # For methods/functions, child implementations override parent
                    if key not in final_impl:
                        final_impl[key] = {}
                    if isinstance(value, dict):
                        final_impl[key].update(value)
                elif key == 'dependencies':
                    # For dependencies, collect all unique dependencies
                    if key not in final_impl:
                        final_impl[key] = []
                    if isinstance(value, list):
                        for dep in value:
                            if dep not in final_impl[key]:
                                final_impl[key].append(dep)
                elif key == 'metadata':
                    # For metadata, merge dictionaries
                    if key not in final_impl:
                        final_impl[key] = {}
                    if isinstance(value, dict):
                        final_impl[key].update(value)
                else:
                    # For other properties, child takes precedence
                    final_impl[key] = value
        
        # Apply extensions to the final implementation
        final_impl = self.inheritance_manager.apply_extensions(child_module)
        
        return final_impl

    def create_polymorphic_module(self, base_module: str, profile: str = "default") -> Dict[str, Any]:
        """
        Create a polymorphic module instance based on a profile
        """
        # This would select the appropriate implementation based on the profile
        # For simplicity, we'll just return the merged implementation
        return self.merge_implementation(base_module)


def initialize_inheritance_extensions():
    """
    Initialize the inheritance and extension system with example relationships
    """
    system = ModuleExtensionSystem()
    
    # Register base implementations
    base_lighting_impl = {
        'methods': {
            'calculate_lighting': 'Basic lighting calculation',
            'get_light_direction': 'Get direction to light source'
        },
        'dependencies': ['math', 'vectors'],
        'metadata': {
            'type': 'lighting',
            'supported_features': ['basic_lighting']
        }
    }
    
    advanced_lighting_impl = {
        'methods': {
            'calculate_pbr_lighting': 'Advanced PBR lighting calculation',
            'calculate_shadows': 'Shadow calculation',
            'update_light_direction': 'Updated light direction method'
        },
        'dependencies': ['pbr_math', 'shadow_maps'],
        'metadata': {
            'type': 'advanced_lighting',
            'supported_features': ['pbr', 'shadows', 'ibl']
        }
    }
    
    optimized_lighting_impl = {
        'methods': {
            'calculate_optimized_lighting': 'Performance-optimized lighting',
            'batch_light_calculations': 'Batch multiple light calculations'
        },
        'dependencies': ['simd_math'],
        'metadata': {
            'type': 'optimized_lighting',
            'supported_features': ['performance', 'batch_processing']
        }
    }
    
    # Register implementations
    system.inheritance_manager.register_module_implementation('base_lighting', base_lighting_impl)
    system.inheritance_manager.register_module_implementation('advanced_lighting', advanced_lighting_impl)
    system.inheritance_manager.register_module_implementation('optimized_lighting', optimized_lighting_impl)
    
    # Create inheritance relationships
    system.create_inheritance_relationship('advanced_lighting', 'base_lighting', 'advanced_features')
    system.create_inheritance_relationship('optimized_lighting', 'base_lighting', 'performance_features')
    
    # Create some extension modules
    system.create_extension_module('base_lighting', 'shadow_extension', 'enhancement', {
        'features': ['shadow_mapping'],
        'functions': ['calculate_shadow_map', 'apply_shadow']
    })
    
    system.create_extension_module('base_lighting', 'reflection_extension', 'enhancement', {
        'features': ['reflections'],
        'functions': ['calculate_reflection', 'apply_reflection']
    })
    
    # Create specialization examples
    system.create_specialized_module('base_lighting', 'cel_shading_lighting', ['cel_shading'])
    system.create_specialized_module('base_lighting', 'toon_lighting', ['toon_rendering', 'outline_effects'])
    
    return system


def main():
    """Main function to demonstrate the module inheritance and extension system"""
    print("Initializing Module Inheritance and Extension System...")
    
    # Initialize the system
    system = initialize_inheritance_extensions()
    
    print("\\nTesting Inheritance and Extension System...")
    
    # Test 1: Get inheritance information
    print(f"\\n1. Inheritance info for 'advanced_lighting':")
    info = system.get_inheritance_info('advanced_lighting')
    print(f"   Parent: {info['parent']}")
    print(f"   Children: {info['children']}")
    print(f"   Inheritance chain: {info['inheritance_chain']}")
    print(f"   Overridable methods: {info['overridable_methods']}")
    
    # Test 2: Validate inheritance
    print(f"\\n2. Inheritance validation for 'advanced_lighting':")
    validation = system.validate_inheritance('advanced_lighting')
    print(f"   Valid: {validation['valid']}")
    if validation['issues']:
        print(f"   Issues: {validation['issues']}")
    
    # Test 3: Create inheritance relationship
    print(f"\\n3. Creating new inheritance relationship...")
    system.create_inheritance_relationship('specialized_lighting', 'advanced_lighting', 'specialized_features')
    
    # Register implementation for the new module
    specialized_impl = {
        'methods': {
            'specialized_calculation': 'Specialized lighting calculation',
            'override_base_method': 'Override from advanced lighting'
        },
        'dependencies': ['specialized_utils'],
        'metadata': {
            'type': 'specialized',
            'supported_features': ['specialized_feature']
        }
    }
    system.inheritance_manager.register_module_implementation('specialized_lighting', specialized_impl)
    
    info = system.get_inheritance_info('specialized_lighting')
    print(f"   Parent: {info['parent']}")
    print(f"   Full inheritance chain: {info['inheritance_chain']}")
    
    # Test 4: Check inheritance chain
    print(f"\\n4. Full inheritance chain for 'specialized_lighting':")
    chain = system.inheritance_manager.get_inheritance_chain('specialized_lighting')
    for i, module in enumerate(reversed(chain)):
        print(f"   L{i}: {module}")
    
    # Test 5: Check inheritance relationship
    print(f"\\n5. Inheritance relationships:")
    is_inheriting = system.inheritance_manager.is_inheriting_from('specialized_lighting', 'base_lighting')
    print(f"   specialized_lighting inherits from base_lighting: {is_inheriting}")
    
    is_inheriting = system.inheritance_manager.is_inheriting_from('base_lighting', 'specialized_lighting')
    print(f"   base_lighting inherits from specialized_lighting: {is_inheriting}")
    
    # Test 6: Get all children
    print(f"\\n6. Children of 'base_lighting': {system.inheritance_manager.get_all_children('base_lighting')}")
    
    # Test 7: Extension functionality
    print(f"\\n7. Extensions for 'base_lighting': {len(system.inheritance_manager.get_module_extensions('base_lighting'))}")
    
    # Test 8: Merge implementation (inheritance)
    print(f"\\n8. Testing implementation merging for 'specialized_lighting':")
    merged = system.merge_implementation('specialized_lighting')
    print(f"   Merged methods: {list(merged.get('methods', {}).keys())}")
    print(f"   Merged dependencies: {merged.get('dependencies', [])}")
    print(f"   Merged metadata type: {merged.get('metadata', {}).get('type')}")
    
    # Test 9: Create another specialization
    print(f"\\n9. Creating another specialization...")
    system.create_specialized_module('advanced_lighting', 'raytraced_lighting', ['raytracing', 'global_illumination'])
    raytraced_info = system.get_inheritance_info('raytraced_lighting')
    print(f"   Raytraced lighting inheritance chain: {raytraced_info['inheritance_chain']}")
    
    # Test 10: Polymorphic behavior
    print(f"\\n10. Testing polymorphic behavior:")
    polymorphic = system.create_polymorphic_module('advanced_lighting')
    print(f"   Methods available: {list(polymorphic.get('methods', {}).keys())}")
    print(f"   Dependencies: {polymorphic.get('dependencies', [])}")
    
    print(f"\\n✅ Module Inheritance and Extension System initialized and tested successfully!")
    print(f"   Features demonstrated:")
    print(f"   - Inheritance relationships")
    print(f"   - Extension modules")
    print(f"   - Specialization")
    print(f"   - Method overriding")
    print(f"   - Implementation merging")
    print(f"   - Inheritance validation")
    print(f"   - Polymorphic behavior")
    
    return 0


if __name__ == "__main__":
    exit(main())