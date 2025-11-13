#!/usr/bin/env python3
"""
Comprehensive Module Registry
Registry of all available modules with metadata and search functionality
"""

import os
import json
from pathlib import Path


class ModuleRegistry:
    def __init__(self, modules_dir="modules"):
        self.modules_dir = modules_dir
        self.modules = {}
        self.load_all_modules()
    
    def load_all_modules(self):
        """Load metadata for all modules in the modules directory"""
        modules_path = Path(self.modules_dir)
        
        for genre_dir in modules_path.iterdir():
            if genre_dir.is_dir():
                genre = genre_dir.name
                self.modules[genre] = {}
                
                for module_dir in genre_dir.iterdir():
                    if module_dir.is_dir():
                        # Look for module files within the directory
                        for py_file in module_dir.rglob("*.py"):
                            if py_file.name != "__init__.py":
                                module_name = py_file.stem
                                try:
                                    # Import the module dynamically
                                    import importlib.util
                                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                                    module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(module)
                                    
                                    # Get metadata if available
                                    if hasattr(module, 'get_metadata'):
                                        metadata = module.get_metadata()
                                        self.modules[genre][f"{module_dir.name}/{module_name}"] = {
                                            'path': str(py_file),
                                            'metadata': metadata,
                                            'genre': genre,
                                            'module_type': module_dir.name
                                        }
                                    else:
                                        # Create basic metadata if not available
                                        self.modules[genre][f"{module_dir.name}/{module_name}"] = {
                                            'path': str(py_file),
                                            'metadata': {
                                                'name': module_name,
                                                'type': 'unknown',
                                                'patterns': [],
                                                'frequency': 0,
                                                'dependencies': [],
                                                'conflicts': [],
                                                'description': 'No description available'
                                            },
                                            'genre': genre,
                                            'module_type': module_dir.name
                                        }
                                except Exception as e:
                                    print(f"Error loading module {py_file}: {e}")
    
    def get_all_modules(self):
        """Get all modules in the registry"""
        all_modules = []
        for genre, modules in self.modules.items():
            for module_name, module_info in modules.items():
                all_modules.append({
                    'genre': genre,
                    'module_name': module_name,
                    'info': module_info
                })
        return all_modules
    
    def search_modules(self, **criteria):
        """Search for modules based on various criteria"""
        results = []
        
        for genre, modules in self.modules.items():
            for module_name, module_info in modules.items():
                match = True
                
                # Check each criterion
                for key, value in criteria.items():
                    if key == 'genre' and value and value.lower() != genre.lower():
                        match = False
                        break
                    elif key == 'pattern' and value:
                        patterns = module_info['metadata'].get('patterns', [])
                        if value.lower() not in [p.lower() for p in patterns]:
                            match = False
                            break
                    elif key == 'module_type' and value:
                        if value.lower() != module_info['module_type'].lower():
                            match = False
                            break
                    elif key == 'conflicts_with' and value:
                        conflicts = module_info['metadata'].get('conflicts', [])
                        if value in conflicts:
                            match = False
                            break
                
                if match:
                    results.append({
                        'genre': genre,
                        'module_name': module_name,
                        'info': module_info
                    })
        
        return results
    
    def get_module_dependencies(self, module_key):
        """Get dependencies for a specific module"""
        for genre, modules in self.modules.items():
            for module_name, module_info in modules.items():
                if f"{genre}/{module_name}" == module_key or module_name == module_key:
                    return module_info['metadata'].get('dependencies', [])
        return []
    
    def get_module_conflicts(self, module_key):
        """Get conflicts for a specific module"""
        for genre, modules in self.modules.items():
            for module_name, module_info in modules.items():
                if f"{genre}/{module_name}" == module_key or module_name == module_key:
                    return module_info['metadata'].get('conflicts', [])
        return []
    
    def get_modules_by_genre(self, genre):
        """Get all modules for a specific genre"""
        if genre in self.modules:
            return self.modules[genre]
        return {}
    
    def get_statistics(self):
        """Get statistics about the registry"""
        stats = {
            'total_modules': 0,
            'genres': {},
            'patterns': {}
        }
        
        for genre, modules in self.modules.items():
            stats['genres'][genre] = len(modules)
            stats['total_modules'] += len(modules)
            
            # Count patterns
            for module_name, module_info in modules.items():
                patterns = module_info['metadata'].get('patterns', [])
                for pattern in patterns:
                    pattern_lower = pattern.lower()
                    if pattern_lower in stats['patterns']:
                        stats['patterns'][pattern_lower] += 1
                    else:
                        stats['patterns'][pattern_lower] = 1
        
        return stats
    
    def save_registry(self, filename="registry/modules.json"):
        """Save the registry to a JSON file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create a serializable version
        serializable_registry = {}
        for genre, modules in self.modules.items():
            serializable_registry[genre] = {}
            for module_name, module_info in modules.items():
                serializable_registry[genre][module_name] = {
                    'path': module_info['path'],
                    'metadata': module_info['metadata'],
                    'genre': module_info['genre'],
                    'module_type': module_info['module_type']
                }
        
        with open(filename, 'w') as f:
            json.dump(serializable_registry, f, indent=2)
        
        print(f"Registry saved to {filename}")
    
    def load_registry(self, filename="registry/modules.json"):
        """Load the registry from a JSON file"""
        try:
            with open(filename, 'r') as f:
                self.modules = json.load(f)
            print(f"Registry loaded from {filename}")
        except FileNotFoundError:
            print(f"Registry file {filename} not found. Starting with empty registry.")
            self.modules = {}


def create_default_registry():
    """Create and populate the default module registry"""
    registry = ModuleRegistry()
    
    # Add some default modules if none exist
    if not registry.modules:
        print("Creating default registry...")
        # The registry will auto-load any modules found in the modules/ directory
        # which we already created for lighting
    
    # Save the registry
    registry.save_registry()
    
    # Print statistics
    stats = registry.get_statistics()
    print(f"Registry contains {stats['total_modules']} modules across {len(stats['genres'])} genres")
    print(f"Top patterns: {sorted(stats['patterns'].items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    return registry


def demo_registry_features():
    """Demonstrate the registry features"""
    print("Demonstrating Module Registry Features:")
    print("=" * 50)
    
    registry = ModuleRegistry()
    
    # Show all modules
    print("All registered modules:")
    all_modules = registry.get_all_modules()
    for module in all_modules[:10]:  # Show first 10
        print(f"  - {module['genre']}/{module['module_name']}")
    if len(all_modules) > 10:
        print(f"  ... and {len(all_modules) - 10} more")
    
    print()
    
    # Search for specific patterns
    print("Searching for modules with 'light' pattern:")
    light_modules = registry.search_modules(pattern="light")
    for module in light_modules:
        print(f"  - {module['module_name']}: {module['info']['info']['metadata']['description']}")
    
    print()
    
    # Get statistics
    stats = registry.get_statistics()
    print("Registry Statistics:")
    print(f"  Total modules: {stats['total_modules']}")
    print(f"  Genres: {list(stats['genres'].keys())}")
    print(f"  Most common patterns: {sorted(stats['patterns'].items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    return registry


if __name__ == "__main__":
    registry = create_default_registry()
    demo_registry_features()