#!/usr/bin/env python3
"""
Lighting Module Combination Engine
Combines multiple lighting modules into a complete shader
"""

from .registry import get_module_by_name, get_module_dependencies, get_module_conflicts


class ModuleCombiner:
    def __init__(self):
        self.selected_modules = []
        self.compatibility_issues = []
        
    def add_module(self, module_name):
        """Add a module to the combination, checking for conflicts and adding dependencies"""
        # Check for conflicts with already selected modules
        for existing_module in self.selected_modules:
            existing_conflicts = get_module_conflicts(existing_module)
            new_conflicts = get_module_conflicts(module_name)
            
            if module_name in existing_conflicts or existing_module in new_conflicts:
                conflict_info = f"Conflict: {module_name} conflicts with {existing_module}"
                self.compatibility_issues.append(conflict_info)
                print(f"WARNING: {conflict_info}")
                return False
        
        # Add dependencies first
        dependencies = get_module_dependencies(module_name)
        for dep in dependencies:
            if dep not in self.selected_modules:
                self.add_module(dep)
        
        # Add the module
        if module_name not in self.selected_modules:
            self.selected_modules.append(module_name)
            print(f"Added module: {module_name}")
        
        return True
    
    def generate_shader(self):
        """Generate GLSL shader code from selected modules"""
        if not self.selected_modules:
            return "// No lighting modules selected\n"
        
        # Start with common definitions
        shader_code = """#version 330 core

// Common lighting definitions
#define PI 3.14159265359

// Input variables (to be provided by vertex shader or calculated)
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

// Uniforms
uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform sampler2D normalMap;
uniform sampler2D shadowMap;

// Output
out vec4 FragColor;

"""
        
        # Add functions from each module
        all_pseudocode = []
        for module_name in self.selected_modules:
            module = get_module_by_name(module_name)
            if module:
                pseudocode = module['pseudocode']
                # Convert pseudocode to GLSL functions
                glsl_functions = self.pseudocode_to_glsl(pseudocode)
                all_pseudocode.append(glsl_functions)
        
        # Combine all GLSL code
        for code in all_pseudocode:
            shader_code += code + "\n"
        
        # Add main function
        shader_code += self.generate_main_function()
        
        return shader_code
    
    def pseudocode_to_glsl(self, pseudocode):
        """Convert pseudocode format to GLSL"""
        # This is a simplified conversion - in a real implementation, 
        # this would be more sophisticated
        lines = pseudocode.split('\n')
        glsl_lines = []
        
        for line in lines:
            # Remove comments that are just explanations
            if line.strip().startswith('//') and '//' in line:
                # Keep the important comments, remove explanations
                if any(keyword in line.lower() for keyword in ['function', 'calculate', 'implementation']):
                    glsl_lines.append(line)
            elif line.strip() and not line.strip().startswith('---'):
                # Add the line as is
                glsl_lines.append(line)
        
        return '\n'.join(glsl_lines)
    
    def generate_main_function(self):
        """Generate the main function that combines all lighting"""
        main_func = """
void main() {
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Initialize color
    vec3 result = vec3(0.0);
    
    // Apply lighting calculations based on selected modules
"""
        
        # Add logic for each selected module
        if 'basic_point_light' in self.selected_modules:
            main_func += """    
    // Basic point light calculation
    vec3 pointLight = calculatePointLight(FragPos, norm, lightPos, lightColor);
    result += pointLight;
"""
        
        if 'diffuse_lighting' in self.selected_modules:
            main_func += """    
    // Diffuse lighting
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 diffuse = calculateDiffuseLambert(lightDir, norm, lightColor);
    result += diffuse;
"""
        
        if 'specular_lighting' in self.selected_modules:
            main_func += """    
    // Specular lighting
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 specular = calculateSpecularPhong(lightDir, viewDir, norm, lightColor, 32.0);
    result += specular;
"""
        
        if 'normal_mapping' in self.selected_modules and 'diffuse_lighting' in self.selected_modules:
            main_func += """    
    // Normal mapping
    vec3 tangentNormal = sampleNormalMap(normalMap, TexCoords);
    vec3 diffuseNormalMapped = calculateDiffuseLambert(tangentNormal, tangentNormal, lightColor);
    result += diffuseNormalMapped;
"""
        
        if 'pbr_lighting' in self.selected_modules:
            main_func += """    
    // PBR lighting (overrides other lighting if present)
    // For simplicity, we'll use a basic PBR calculation
    vec3 albedo = vec3(0.5); 
    float metallic = 0.0;
    float roughness = 0.5;
    vec4 pbrResult = calculatePBRLighting(FragPos, norm, viewDir, lightPos, lightColor, albedo, metallic, roughness);
    result = pbrResult.rgb;  // PBR typically replaces other lighting
"""
        
        if 'cel_shading' in self.selected_modules:
            main_func += """    
    // Apply cel shading to the result
    vec3 celResult = calculateCelShading(norm, normalize(lightPos - FragPos), result, vec3(1.0));
    result = celResult;
"""
        
        main_func += """
    // Final color
    FragColor = vec4(result, 1.0);
}
"""
        
        return main_func
    
    def validate_combination(self):
        """Validate that the selected modules are compatible"""
        validation_result = {
            'valid': len(self.compatibility_issues) == 0,
            'issues': self.compatibility_issues,
            'selected_modules': self.selected_modules
        }
        return validation_result


def combine_lighting_modules(module_names):
    """Convenience function to combine lighting modules"""
    combiner = ModuleCombiner()
    
    for module_name in module_names:
        combiner.add_module(module_name)
    
    shader_code = combiner.generate_shader()
    validation = combiner.validate_combination()
    
    return {
        'shader_code': shader_code,
        'validation': validation,
        'combiner': combiner
    }


# Example usage
if __name__ == "__main__":
    # Example: combine basic lighting modules
    modules_to_combine = ['diffuse_lighting', 'specular_lighting', 'normal_mapping']
    result = combine_lighting_modules(modules_to_combine)
    
    print("Generated Shader Code:")
    print(result['shader_code'])
    
    print("\nValidation Results:")
    print(f"Valid: {result['validation']['valid']}")
    print(f"Issues: {result['validation']['issues']}")
    print(f"Selected Modules: {result['validation']['selected_modules']}")