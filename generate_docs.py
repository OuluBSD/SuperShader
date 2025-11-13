# Module Documentation Generator

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
        
        return "\n".join(doc)
    
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
            f.write("\n".join(index_content))


def main():
    generator = ModuleDocumentationGenerator()
    generator.generate_all_docs()
    print(f"Generated documentation for {len(generator.registry['modules'])} modules")


if __name__ == "__main__":
    main()
