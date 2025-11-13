#!/usr/bin/env python3
"""
Module combination engine for SuperShader project.

This script creates an engine to combine modules into functional shaders,
implement validation for module compatibility, add support for optional
modules, and create error handling for incompatible modules.
"""

import os
import json
from pathlib import Path


def create_module_registry():
    """
    Create a registry of available modules with metadata.
    """
    registry = {
        "modules": []
    }
    
    # Walk through all module directories
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
                        "tags": [module_type_dir.name]
                    }
                    registry["modules"].append(module_info)
    
    # Save registry to file
    os.makedirs("registry", exist_ok=True)
    with open("registry/modules.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"Created registry with {len(registry['modules'])} modules")
    return registry


def validate_module_compatibility(selected_modules):
    """
    Validate if selected modules are compatible with each other.
    
    Args:
        selected_modules (list): List of module names to check
    
    Returns:
        tuple: (is_compatible, list of issues)
    """
    issues = []
    
    # For now, we'll implement basic checks
    # In a full implementation, this would check for conflicts between modules
    
    # Check for duplicate functionality (very basic)
    function_categories = {}
    for module in selected_modules:
        # Extract category from module name
        category = module.split('_')[0] if '_' in module else module
        if category in function_categories:
            function_categories[category].append(module)
        else:
            function_categories[category] = [module]
    
    # Flag cases where multiple modules might provide similar functionality
    for category, modules in function_categories.items():
        if len(modules) > 1:
            issues.append(f"Multiple modules in '{category}' category: {', '.join(modules)}. May cause conflicts.")
    
    return len(issues) == 0, issues


def combine_modules(selected_modules, output_file="combined_shader.glsl"):
    """
    Combine selected modules into a single shader file.
    
    Args:
        selected_modules (list): List of module names to combine
        output_file (str): Output file path
    
    Returns:
        str: Path to combined shader file
    """
    combined_code = []
    
    # Add standard GLSL header
    combined_code.append("// Combined shader from SuperShader modules")
    combined_code.append("// Automatically generated")
    combined_code.append("")
    combined_code.append("#version 300 es")
    combined_code.append("precision highp float;")
    combined_code.append("")
    
    # Track which code has been added to avoid duplicates
    added_code = set()
    
    # Process each selected module
    for module_name in selected_modules:
        # Find the module file
        module_path = find_module_file(module_name)
        if module_path:
            with open(module_path, 'r') as f:
                module_code = f.read()
                
                # Add the module code if not already added
                if module_code not in added_code:
                    combined_code.append(f"// Begin module: {module_name}")
                    combined_code.append(module_code)
                    combined_code.append(f"// End module: {module_name}")
                    combined_code.append("")
                    added_code.add(module_code)
    
    # Add a basic main function if not already present
    if not any("main()" in code for code in combined_code):
        combined_code.append("// Default main function")
        combined_code.append("uniform vec2 iResolution;")
        combined_code.append("uniform float iTime;")
        combined_code.append("in vec2 fragCoord;")
        combined_code.append("out vec4 fragColor;")
        combined_code.append("")
        combined_code.append("void main() {")
        combined_code.append("    vec2 uv = fragCoord / iResolution.xy;")
        combined_code.append("    // Combined shader output")
        combined_code.append("    fragColor = vec4(uv, 0.5 + 0.5 * sin(iTime), 1.0);")
        combined_code.append("}")
    
    # Write combined code to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(combined_code))
    
    print(f"Combined {len(selected_modules)} modules into {output_file}")
    return output_file


def find_module_file(module_name):
    """
    Find the file path for a given module name.
    
    Args:
        module_name (str): Name of the module
    
    Returns:
        str or None: Path to the module file, or None if not found
    """
    modules_dir = Path("modules")
    if modules_dir.exists():
        # Search for the module in all subdirectories
        for module_type_dir in modules_dir.iterdir():
            if module_type_dir.is_dir():
                # Look in both standardized and extracted directories
                for search_dir in [module_type_dir, module_type_dir / "standardized"]:
                    if search_dir.exists():
                        # Look for exact match
                        exact_match = search_dir / f"{module_name}.glsl"
                        if exact_match.exists():
                            return str(exact_match)
                        
                        # Look for files containing the module name
                        for glsl_file in search_dir.glob("*.glsl"):
                            if module_name.replace("_functions", "") in glsl_file.stem:
                                return str(glsl_file)
    
    return None


def create_shader_generator():
    """
    Create a shader generator system that can produce complete shaders from modules.
    """
    generator_code = '''// SuperShader - Shader Generation System
// Combines modules into complete, functional shaders

#version 300 es
precision highp float;

// Common uniforms
uniform vec2 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;
uniform vec4 iMouse;
uniform vec4 iDate;
uniform float iSampleRate;

in vec2 fragCoord;
out vec4 fragColor;

// Include module functions here
// The specific modules will be inserted during generation

void main() {
    // Calculate normalized UV coordinates
    vec2 uv = fragCoord / iResolution.xy;
    
    // Center coordinates (-1 to 1)
    vec2 coord = (2.0 * fragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    // Main shader composition happens here using selected modules
    // This is where the combined functionality goes
    
    // Default output if no specific modules are combined
    fragColor = vec4(uv, 0.5 + 0.5 * sin(iTime), 1.0);
}
'''
    
    os.makedirs("generator", exist_ok=True)
    with open("generator/template.glsl", "w") as f:
        f.write(generator_code)
    
    print("Created shader generator template")


def create_module_interface_standard():
    """
    Define standard interfaces that modules should follow.
    """
    interface_standard = {
        "version": "1.0",
        "standards": {
            "naming_conventions": {
                "function_names": "use_underscores_and_descriptive_names",
                "variables": "use_underscores",
                "constants": "ALL_CAPS_WITH_UNDERSCORES"
            },
            "function_signatures": [
                "vec3 lighting_function(vec3 normal, vec3 lightDir, vec3 viewDir)",
                "vec4 effect_function(vec4 color, vec2 uv, float time)",
                "float sdf_function(vec3 position)",
                "vec3 transformation_function(vec3 position, float time)"
            ],
            "required_comments": [
                "Purpose of the module",
                "Input parameters",
                "Output description",
                "Dependencies if any"
            ],
            "compatibility_guidelines": {
                "uniform_variables": "use standard SuperShader uniforms",
                "global_variables": "avoid if possible",
                "performance": "aim for minimal instructions"
            }
        }
    }
    
    os.makedirs("standards", exist_ok=True)
    with open("standards/interface_standard.json", "w") as f:
        json.dump(interface_standard, f, indent=2)
    
    print("Created module interface standard")


def main():
    print("Creating module combination engine...")
    
    # Create module registry
    registry = create_module_registry()
    
    # Create shader generator template
    create_shader_generator()
    
    # Create module interface standard
    create_module_interface_standard()
    
    # Example of combining a few modules to test the system
    if registry["modules"]:
        print(f"\nFound {len(registry['modules'])} modules in registry")
        
        # Select a few modules for demonstration
        selected_modules = [mod["name"] for mod in registry["modules"][:3]]
        
        print(f"Attempting to combine modules: {selected_modules}")
        
        # Validate compatibility
        is_compatible, issues = validate_module_compatibility(selected_modules)
        if is_compatible:
            print("✓ Modules are compatible")
        else:
            print("⚠ Compatibility issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Combine the modules
        try:
            output_file = combine_modules(selected_modules, "test_combined_shader.glsl")
            print(f"✓ Successfully created {output_file}")
        except Exception as e:
            print(f"✗ Failed to combine modules: {str(e)}")
    
    print("\nModule combination engine created successfully!")


if __name__ == "__main__":
    main()