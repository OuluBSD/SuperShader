#!/usr/bin/env python3
"""
Module Engine
Engine to combine modules into functional shaders with validation
"""

import json
from create_module_registry import ModuleRegistry
from create_pseudocode_translator import PseudocodeTranslator


class ModuleEngine:
    def __init__(self):
        self.registry = ModuleRegistry()
        self.translator = PseudocodeTranslator()
        self.selected_modules = []
        self.dependencies = set()
        self.conflicts = []
    
    def add_module(self, module_key):
        """Add a module to the combination, resolving dependencies and conflicts"""
        # Check if module exists in the registry
        full_module_key = None
        for genre, modules in self.registry.modules.items():
            for module_name, module_info in modules.items():
                full_key = f"{genre}/{module_name}"
                if full_key == module_key or module_name.split('/')[-1] == module_key:  # Handle just the filename part
                    full_module_key = full_key
                    break
            if full_module_key:
                break
        
        if full_module_key is None:
            raise ValueError(f"Module '{module_key}' not found in registry")
        
        # Check for conflicts with already selected modules
        new_conflicts = self.registry.get_module_conflicts(full_module_key)
        for selected_module in self.selected_modules:
            if selected_module in new_conflicts:
                self.conflicts.append(f"Conflict: {full_module_key} conflicts with {selected_module}")
                return False
        
        # Add dependencies
        deps = self.registry.get_module_dependencies(full_module_key)
        for dep in deps:
            # Find the full path for the dependency
            dep_full_key = None
            for genre, modules in self.registry.modules.items():
                for module_name, module_info in modules.items():
                    if module_name.split('/')[-1] == dep or f"{genre}/{module_name}" == dep:
                        dep_full_key = f"{genre}/{module_name}"
                        break
                if dep_full_key:
                    break
            
            if dep_full_key:
                if dep_full_key not in self.selected_modules and dep_full_key not in self.dependencies:
                    self.add_module(dep_full_key)  # Recursive addition of dependencies
                    self.dependencies.add(dep_full_key)
        
        # Add the module
        if full_module_key not in self.selected_modules:
            self.selected_modules.append(full_module_key)
        
        return True
    
    def remove_module(self, module_key):
        """Remove a module from the combination"""
        if module_key in self.selected_modules:
            self.selected_modules.remove(module_key)
        if module_key in self.dependencies:
            self.dependencies.remove(module_key)
    
    def validate_combination(self):
        """Validate that the current module combination is valid"""
        issues = []
        
        # Check for conflicts
        for i, module1 in enumerate(self.selected_modules):
            for module2 in self.selected_modules[i+1:]:
                conflicts1 = self.registry.get_module_conflicts(module1)
                conflicts2 = self.registry.get_module_conflicts(module2)
                
                if module2 in conflicts1 or module1 in conflicts2:
                    issues.append(f"Conflict: {module1} and {module2} are incompatible")
        
        # Check for missing dependencies
        for module_key in self.selected_modules:
            deps = self.registry.get_module_dependencies(module_key)
            for dep in deps:
                if dep not in self.selected_modules and dep not in self.dependencies:
                    issues.append(f"Missing dependency: {module_key} requires {dep}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'selected_modules': self.selected_modules.copy()
        }
    
    def generate_shader(self, target_language='glsl', include_validation=True):
        """Generate shader code from selected modules"""
        validation = self.validate_combination()
        
        if include_validation and not validation['valid']:
            print("Warning: Module combination has issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if not self.selected_modules:
            return "// No modules selected\n"
        
        # Collect pseudocode from all selected modules
        all_pseudocode = []
        module_metadata = []
        
        for module_key in self.selected_modules:
            # Find the module in the registry
            found = False
            for genre, modules in self.registry.modules.items():
                for module_name, module_info in modules.items():
                    full_key = f"{genre}/{module_name}"
                    if full_key == module_key or module_name == module_key:
                        if 'pseudocode' in dir(__import__(f"modules.{genre}.{module_name.split('/')[0]}.{module_name.split('/')[1]}", fromlist=['get_pseudocode'])):
                            # Import the specific module and get its pseudocode
                            import importlib
                            module_path_parts = module_key.split('/')
                            if len(module_path_parts) == 2:
                                module_dir, module_file = module_path_parts
                                module_imp = importlib.import_module(f"modules.{genre}.{module_dir}.{module_file}")
                                
                                if hasattr(module_imp, 'get_pseudocode'):
                                    pseudocode = module_imp.get_pseudocode()
                                    all_pseudocode.append(pseudocode)
                                    module_metadata.append({
                                        'name': module_key,
                                        'metadata': module_info['metadata']
                                    })
                        found = True
                        break
                if found:
                    break
        
        # Generate shader using the translator
        shader_code = self.translator.create_glsl_shader_from_modules(self.selected_modules)
        
        return shader_code
    
    def get_combination_summary(self):
        """Get a summary of the current module combination"""
        validation = self.validate_combination()
        
        summary = {
            'total_modules': len(self.selected_modules),
            'modules': self.selected_modules.copy(),
            'dependencies': list(self.dependencies),
            'conflicts': self.conflicts.copy(),
            'validation': validation
        }
        
        return summary
    
    def create_profile(self, profile_name):
        """Create a saved profile of the current module combination"""
        profile = {
            'name': profile_name,
            'modules': self.selected_modules.copy(),
            'dependencies': list(self.dependencies),
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        # Save to profiles directory
        import os
        os.makedirs('profiles', exist_ok=True)
        
        with open(f'profiles/{profile_name}.json', 'w') as f:
            json.dump(profile, f, indent=2)
        
        return profile


def demo_module_engine():
    """Demonstrate the module engine functionality"""
    print("Demonstrating Module Engine:")
    print("=" * 40)
    
    engine = ModuleEngine()
    
    # Add some lighting modules to combine
    modules_to_add = [
        'lighting/point_light/basic_point_light',
        'lighting/diffuse/diffuse_lighting', 
        'lighting/specular/specular_lighting',
        'lighting/normal_mapping/normal_mapping'
    ]
    
    print("Adding modules:")
    for module in modules_to_add:
        success = engine.add_module(module)
        print(f"  - {module}: {'Success' if success else 'Failed'}")
    
    print()
    
    # Validate the combination
    validation = engine.validate_combination()
    print(f"Combination valid: {validation['valid']}")
    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    print()
    
    # Generate shader
    shader = engine.generate_shader()
    print("Generated shader code (first 20 lines):")
    lines = shader.split('\n')
    for i, line in enumerate(lines[:20]):
        print(f"  {i+1:2d}: {line}")
    if len(lines) > 20:
        print(f"  ... and {len(lines) - 20} more lines")
    
    # Save the generated shader
    with open('engine_generated_shader.glsl', 'w') as f:
        f.write(shader)
    print("\nShader saved to engine_generated_shader.glsl")
    
    # Create a profile
    profile = engine.create_profile('demo_lighting_profile')
    print(f"Profile created: {profile['name']}")
    
    # Print combination summary
    summary = engine.get_combination_summary()
    print(f"\nCombination Summary:")
    print(f"  Modules: {summary['total_modules']}")
    print(f"  Dependencies: {len(summary['dependencies'])}")
    print(f"  Conflicts: {len(summary['conflicts'])}")
    
    return engine


if __name__ == "__main__":
    demo_module_engine()