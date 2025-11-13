# Module Search API

import json

class ModuleSearcher:
    def __init__(self, registry_file="registry/modules.json"):
        with open(registry_file, 'r') as f:
            self.registry = json.load(f)
    
    def search_by_category(self, category):
        """Search modules by category."""
        return [m for m in self.registry['modules'] if m['category'] == category]
    
    def search_by_tag(self, tag):
        """Search modules by tag."""
        return [m for m in self.registry['modules'] if tag in m['tags']]
    
    def search_by_name(self, name):
        """Search modules by name (partial match)."""
        name_lower = name.lower()
        return [m for m in self.registry['modules'] if name_lower in m['name'].lower()]
    
    def search_by_dependencies(self, dependency):
        """Search modules that depend on a specific module/type."""
        return [m for m in self.registry['modules'] if dependency in m['dependencies']]
    
    def get_module_info(self, module_name):
        """Get full information about a specific module."""
        for m in self.registry['modules']:
            if m['name'] == module_name:
                return m
        return None
    
    def search_with_filters(self, category=None, tags=None, dependencies=None, name=None):
        """Search modules with multiple filters."""
        results = self.registry['modules']
        
        if category:
            results = [m for m in results if m['category'] == category]
        
        if tags:
            for tag in tags:
                results = [m for m in results if tag in m['tags']]
        
        if dependencies:
            for dep in dependencies:
                results = [m for m in results if dep in m['dependencies']]
        
        if name:
            name_lower = name.lower()
            results = [m for m in results if name_lower in m['name'].lower()]
        
        return results


def main():
    searcher = ModuleSearcher()
    
    # Examples of searching
    print("Modules in lighting category:")
    lighting_modules = searcher.search_by_category('lighting')
    for m in lighting_modules[:5]:  # Show first 5
        print(f"  - {m['name']}")
    
    print("\nModules with 'glow' tag:")
    glow_modules = searcher.search_by_tag('glow')
    for m in glow_modules[:5]:  # Show first 5
        print(f"  - {m['name']}")
    
    print("\nModules containing 'texture' in name:")
    texture_modules = searcher.search_by_name('texture')
    for m in texture_modules[:5]:  # Show first 5
        print(f"  - {m['name']}")


if __name__ == "__main__":
    main()
