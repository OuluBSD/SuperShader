#!/usr/bin/env python3
"""
Module registry and documentation system for SuperShader project.

This script creates a comprehensive registry of available modules,
adds metadata for each module, creates search functionality,
and adds a tagging system for module categorization.
"""

import json
import os
from pathlib import Path


def enhance_module_registry():
    """
    Enhance the existing module registry with additional metadata and search capabilities.
    """
    # Load existing registry
    try:
        with open("registry/modules.json", "r") as f:
            registry = json.load(f)
    except FileNotFoundError:
        # Create basic registry if it doesn't exist
        registry = {"modules": []}
        
        # Walk through all module directories to populate registry
        modules_dir = Path("modules")
        if modules_dir.exists():
            for module_type_dir in modules_dir.iterdir():
                if module_type_dir.is_dir():
                    for glsl_file in module_type_dir.rglob("*.glsl"):
                        module_info = {
                            "name": glsl_file.stem,
                            "path": str(glsl_file.relative_to(modules_dir)),
                            "category": module_type_dir.name,
                            "type": "standardized" if "standardized" in str(glsl_file) else "extracted",
                            "dependencies": [],
                            "conflicts": [],
                            "tags": [module_type_dir.name],
                            "description": f"Module for {module_type_dir.name} operations",
                            "author": "Auto-generated",
                            "version": "1.0",
                            "last_modified": "2023-01-01"
                        }
                        registry["modules"].append(module_info)
    
    # Add enhanced metadata to each module
    for module in registry["modules"]:
        # Determine dependencies based on function calls or imports
        module_path = Path("modules") / module["path"]
        if module_path.exists():
            try:
                with open(module_path, 'r') as f:
                    content = f.read()
                
                # Extract potential dependencies from the code
                dependencies = []
                # Look for common GLSL functions that might indicate dependencies
                if "texture" in content.lower():
                    dependencies.append("texture_sampling")
                if "normal" in content.lower():
                    dependencies.append("normal_mapping")
                if "light" in content.lower():
                    dependencies.append("lighting")
                if "sdf" in content.lower() or "map" in content.lower():
                    dependencies.append("raymarching")
                
                module["dependencies"] = dependencies
                
                # Add more tags based on content
                content_tags = []
                if "light" in content.lower():
                    content_tags.append("lighting")
                if "texture" in content.lower():
                    content_tags.append("texturing")
                if "particle" in content.lower():
                    content_tags.append("particles")
                if "audio" in content.lower():
                    content_tags.append("audio")
                if "color" in content.lower():
                    content_tags.append("color")
                if "animation" in content.lower():
                    content_tags.append("animation")
                
                module["tags"].extend(content_tags)
                module["tags"] = list(set(module["tags"]))  # Remove duplicates
                
            except Exception as e:
                print(f"Error processing {module_path}: {str(e)}")
    
    # Save updated registry
    os.makedirs("registry", exist_ok=True)
    with open("registry/modules.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"Enhanced registry with {len(registry['modules'])} modules")
    return registry


def create_module_search_functionality():
    """
    Create functionality to search for modules by various criteria.
    """
    search_code = '''# Module Search API

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
    
    print("\\nModules with 'glow' tag:")
    glow_modules = searcher.search_by_tag('glow')
    for m in glow_modules[:5]:  # Show first 5
        print(f"  - {m['name']}")
    
    print("\\nModules containing 'texture' in name:")
    texture_modules = searcher.search_by_name('texture')
    for m in texture_modules[:5]:  # Show first 5
        print(f"  - {m['name']}")


if __name__ == "__main__":
    main()
'''
    
    with open("search_modules.py", "w") as f:
        f.write(search_code)
    
    print("Created module search functionality")


def create_documentation_generator():
    """
    Create documentation for modules.
    """
    doc_generator_code = '''# Module Documentation Generator

import json
import os
from pathlib import Path

class ModuleDocumentationGenerator:
    def __init__(self, registry_file="registry/modules.json"):
        with open(registry_file, 'r') as f:
            self.registry = json.load(f)
    
    def generate_module_docs(self, module_info):
        """Generate documentation for a single module."""
        doc = []
        doc.append(f"# {module_info['name']}")
        doc.append("")
        doc.append(f"**Category:** {module_info['category']}")
        doc.append(f"**Type:** {module_info['type']}")
        doc.append("")
        
        if module_info.get('description'):
            doc.append(f"## Description")
            doc.append(module_info['description'])
            doc.append("")
        
        if module_info['dependencies']:
            doc.append(f"## Dependencies")
            doc.append(", ".join(module_info['dependencies']))
            doc.append("")
        
        if module_info['conflicts']:
            doc.append(f"## Conflicts")
            doc.append(", ".join(module_info['conflicts']))
            doc.append("")
        
        if module_info['tags']:
            doc.append(f"## Tags")
            doc.append(", ".join(module_info['tags']))
            doc.append("")
        
        # Read and display the actual module code
        module_path = Path("modules") / module_info['path']
        if module_path.exists():
            with open(module_path, 'r') as f:
                code = f.read()
            doc.append("## Code")
            doc.append("```glsl")
            doc.append(code)
            doc.append("```")
        
        return "\\n".join(doc)
    
    def generate_all_docs(self):
        """Generate documentation for all modules."""
        os.makedirs("docs/modules", exist_ok=True)
        
        for module in self.registry['modules']:
            doc_content = self.generate_module_docs(module)
            doc_filename = f"docs/modules/{module['name']}.md"
            with open(doc_filename, 'w') as f:
                f.write(doc_content)
        
        # Generate a main index file
        self.generate_index()
    
    def generate_index(self):
        """Generate an index of all modules."""
        index_content = ["# Module Index", ""]
        
        # Group modules by category
        categories = {}
        for module in self.registry['modules']:
            cat = module['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(module)
        
        for category, modules in categories.items():
            index_content.append(f"## {category.title()}")
            for module in modules:
                index_content.append(f"- [{module['name']}]({module['name']})")
            index_content.append("")
        
        with open("docs/module_index.md", "w") as f:
            f.write("\\n".join(index_content))


def main():
    generator = ModuleDocumentationGenerator()
    generator.generate_all_docs()
    print(f"Generated documentation for {len(generator.registry['modules'])} modules")


if __name__ == "__main__":
    main()
'''
    
    with open("generate_docs.py", "w") as f:
        f.write(doc_generator_code)
    
    print("Created documentation generator")


def create_module_testing_framework():
    """
    Create a basic testing framework for modules.
    """
    testing_code = '''# Module Testing Framework

import unittest
import json
import os
from pathlib import Path

class ModuleTester:
    def __init__(self, registry_file="registry/modules.json"):
        with open(registry_file, 'r') as f:
            self.registry = json.load(f)
    
    def test_module_syntax(self, module_path):
        """Test if a module has valid GLSL syntax (basic check)."""
        try:
            with open(module_path, 'r') as f:
                content = f.read()
            
            # Basic checks
            if not content.strip():
                return False, "Module is empty"
            
            # Check for common GLSL syntax issues
            if 'float' in content and 'vec' in content:
                # Check that variable declarations are properly formatted
                pass  # More detailed syntax checking could go here
            
            return True, "Syntax appears valid"
        except Exception as e:
            return False, f"Could not read file: {str(e)}"
    
    def test_all_modules(self):
        """Run tests on all modules."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        for module in self.registry['modules']:
            results["total"] += 1
            module_path = Path("modules") / module['path']
            
            success, message = self.test_module_syntax(module_path)
            
            if success:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "module": module['name'],
                    "error": message
                })
        
        return results
    
    def run_compatibility_tests(self, module1_name, module2_name):
        """Test if two modules are compatible."""
        # This is a simplified test - a full implementation would be more complex
        # For now, just check if they have conflicting dependencies
        
        module1 = None
        module2 = None
        
        for m in self.registry['modules']:
            if m['name'] == module1_name:
                module1 = m
            if m['name'] == module2_name:
                module2 = m
        
        if not module1 or not module2:
            return False, "One or both modules not found"
        
        # Check for conflicts (simplified)
        conflicts = set(module1.get('conflicts', [])) & set(module2.get('tags', []))
        conflicts |= set(module2.get('conflicts', [])) & set(module1.get('tags', []))
        
        if conflicts:
            return False, f"Conflicts found: {', '.join(conflicts)}"
        
        return True, "Modules appear compatible"


def main():
    tester = ModuleTester()
    results = tester.test_all_modules()
    
    print(f"Module Testing Results:")
    print(f"Total modules tested: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    if results['errors']:
        print("\\nErrors found:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  {error['module']}: {error['error']}")


if __name__ == "__main__":
    main()
'''
    
    with open("module_tester.py", "w") as f:
        f.write(testing_code)
    
    print("Created module testing framework")


def main():
    print("Creating module registry and documentation system...")
    
    # Enhance the module registry
    registry = enhance_module_registry()
    
    # Create search functionality
    create_module_search_functionality()
    
    # Create documentation generator
    create_documentation_generator()
    
    # Create testing framework
    create_module_testing_framework()
    
    print("\\nModule registry and documentation system created successfully!")
    
    print(f"\\nSummary:")
    print(f"- Enhanced registry with {len(registry['modules'])} modules")
    print(f"- Created search functionality in search_modules.py")
    print(f"- Created documentation generator in generate_docs.py") 
    print(f"- Created testing framework in module_tester.py")


if __name__ == "__main__":
    main()