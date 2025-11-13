# SuperShader - Complete Integration Demo

import json
import os
from pathlib import Path
import time

# Import components we've created
from create_module_engine import combine_modules, validate_module_compatibility
from create_pseudocode_translator import PseudocodeTranslator
from search_modules import ModuleSearcher
from performance_profiler import profiler
from shader_optimizer import ShaderOptimizer


class SuperShaderManager:
    def __init__(self):
        self.module_searcher = ModuleSearcher()
        self.translator = PseudocodeTranslator()
        self.optimizer = ShaderOptimizer()
        
        # Load the registry
        with open("registry/modules.json", "r") as f:
            self.registry = json.load(f)
    
    @profiler.profile("generate_shader_from_modules")
    def generate_shader(self, module_names, target_language="glsl", output_file=None):
        """Generate a complete shader from selected modules."""
        print(f"Generating shader with modules: {module_names}")
        
        # Validate compatibility
        is_compatible, issues = validate_module_compatibility(module_names)
        if not is_compatible:
            print("Warning: Compatibility issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Combine modules
        combined_file = combine_modules(module_names, "temp_combined.glsl")
        
        # Read the combined shader
        with open(combined_file, 'r') as f:
            shader_code = f.read()
        
        # Optimize the shader
        optimized_code = self.optimizer.optimize_shader(shader_code)
        
        # Translate if needed
        if target_language != "glsl":
            # This is a simplified approach - in reality, we'd need to parse the GLSL back to pseudocode
            # For demo purposes, we'll just return the GLSL optimized code
            print(f"Note: Translation to {target_language} requires intermediate pseudocode representation")
        else:
            final_code = optimized_code
        
        # Save to output file
        if output_file:
            with open(output_file, 'w') as f:
                f.write(final_code)
            print(f"Shader saved to {output_file}")
        
        # Cleanup temp file
        if os.path.exists("temp_combined.glsl"):
            os.remove("temp_combined.glsl")
        
        return final_code
    
    @profiler.profile("search_and_recommend_modules")
    def recommend_modules(self, requirements):
        """Recommend modules based on requirements."""
        recommendations = []
        
        for req in requirements:
            # Search for modules that match the requirement
            matches = self.module_searcher.search_by_tag(req)
            recommendations.extend([m['name'] for m in matches])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    @profiler.profile("translate_pseudocode_workflow")
    def translate_workflow(self, pseudocode_file, target_language):
        """Complete workflow: load pseudocode, translate, optimize."""
        with open(pseudocode_file, 'r') as f:
            pseudocode = f.read()
        
        translated = self.translator.translate_pseudocode(pseudocode, target_language)
        optimized = self.optimizer.optimize_shader(translated)
        
        return optimized


def main():
    print("SuperShader - Complete Integration Demo")
    print("=" * 40)
    
    manager = SuperShaderManager()
    
    # Demonstrate module recommendation
    print("\n1. Recommending modules for lighting effects:")
    requirements = ["lighting", "diffuse", "specular"]
    recommendations = manager.recommend_modules(requirements)
    print(f"Recommended modules: {recommendations[:5]}...")  # Show first 5
    
    # Demonstrate shader generation (using available modules)
    print("\n2. Generating a sample shader:")
    
    # Get some lighting modules for the demo
    lighting_modules = [m['name'] for m in manager.registry['modules'] 
                       if 'lighting' in m['category'] or 'light' in m['tags'] or 'diffuse' in m['name']]
    
    if lighting_modules:
        selected_modules = lighting_modules[:3]  # Use first 3 lighting modules
        print(f"Using modules: {selected_modules}")
        shader_code = manager.generate_shader(selected_modules, "glsl", "demo_shader.glsl")
        print(f"Generated shader with {shader_code.count(';')} statements")
    else:
        print("No lighting modules found for demonstration")
    
    # Show available modules
    print(f"\n3. Available modules in registry: {len(manager.registry['modules'])}")
    
    # Sample of different categories
    categories = set(m['category'] for m in manager.registry['modules'])
    print(f"Categories: {list(categories)[:10]}...")  # First 10 categories
    
    # Show performance report
    print("\n4. Performance Report:")
    print(profiler.get_report())
    
    print("\nIntegration demo completed successfully!")
    
    # Show what files were created during the process
    print("\n5. Created/updated files:")
    created_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(ext in file for ext in ['.glsl', '.json', '.py', '.md']):
                if 'demo_shader.glsl' in file or 'registry' in root or 'docs' in root or 'translated_examples' in root:
                    created_files.append(os.path.join(root, file))
    
    for file in sorted(created_files)[:10]:  # Show first 10
        print(f"  - {file}")
    
    if len(created_files) > 10:
        print(f"  ... and {len(created_files) - 10} more files")


def demo_pseudocode_translation():
    """Demonstrate the pseudocode translation capability."""
    print("\n" + "="*50)
    print("PSEUDOCODE TRANSLATION DEMO")
    print("="*50)
    
    # Create a sample pseudocode file
    sample_pseudocode = """function calculate_lighting(normal: vec3, light_dir: vec3, view_dir: vec3, albedo: vec3) {
    var ambient: vec3 = albedo * 0.1;
    var diffuse: float = max(dot(normal, light_dir), 0.0);
    var reflect_dir: vec3 = reflect(-light_dir, normal);
    var specular: float = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    var result: vec3 = ambient + diffuse * albedo + specular * vec3(1.0);
    return result;
}"""
    
    with open("temp_sample.pseudo", "w") as f:
        f.write(sample_pseudocode)
    
    manager = SuperShaderManager()
    
    # Translate to different languages
    for lang in ["glsl", "cpp", "python"]:
        try:
            translated = manager.translate_workflow("temp_sample.pseudo", lang)
            print(f"\n{lang.upper()} translation:")
            print(translated[:200] + "..." if len(translated) > 200 else translated)  # Show first 200 chars
        except Exception as e:
            print(f"Error translating to {lang}: {str(e)}")
    
    # Clean up
    if os.path.exists("temp_sample.pseudo"):
        os.remove("temp_sample.pseudo")


if __name__ == "__main__":
    main()
    demo_pseudocode_translation()
