#!/usr/bin/env python3
"""
Complete Shader Generation Pipeline
Combines modules from different genres to produce complete, valid GLSL shaders
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from modules.lighting.registry import get_all_modules as get_lighting_modules
from modules.raymarching.registry import get_all_modules as get_raymarching_modules
from modules.effects.registry import get_all_modules
from modules.texturing.registry import get_all_modules as get_texturing_modules
from modules.geometry.registry import get_all_modules as get_geometry_modules
from create_pseudocode_translator import PseudocodeTranslator
from modules.cross_genre_data_flow_validator import CrossGenreDataFlowValidator
from create_module_registry import ModuleRegistry


class ShaderGenerationPipeline:
    def __init__(self):
        self.translator = PseudocodeTranslator()
        self.validator = CrossGenreDataFlowValidator()
        self.registry = ModuleRegistry()
        self.module_cache = {}

    def load_module_pseudocode(self, module_name: str) -> str:
        """Load pseudocode for a module, with caching"""
        if module_name in self.module_cache:
            return self.module_cache[module_name]

        # Get module from registry
        for genre, modules in self.registry.modules.items():
            for mod_name, mod_info in modules.items():
                full_name = f"{genre}/{mod_name}"
                if full_name == module_name or mod_name.split('/')[-1] == module_name:
                    try:
                        import importlib.util
                        path = mod_info['path']
                        spec = importlib.util.spec_from_file_location(mod_name.split('/')[-1], path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        pseudocode = module.get_pseudocode()
                        self.module_cache[module_name] = pseudocode
                        return pseudocode
                    except Exception as e:
                        print(f"Error loading pseudocode for {module_name}: {e}")
                        return ""

        return ""

    def create_shader_skeleton(self, shader_type: str = "fragment") -> Dict[str, str]:
        """Create the basic structure for different types of shaders"""
        skeleton = {
            'header': f'#version 330 core\n\n',
            'defines': '',
            'uniforms': '// Common uniforms\nuniform vec2 resolution;\nuniform float time;\nuniform vec2 mouse;\nuniform int frame;\n\n',
            'inputs': '',
            'outputs': '',
            'global_variables': '',
            'functions': [],
            'main_function': '',
            'shader_type': shader_type
        }

        if shader_type == "vertex":
            skeleton['inputs'] = '// Vertex attributes\nin vec3 aPos;\nin vec3 aNormal;\nin vec2 aTexCoord;\n\n'
            skeleton['outputs'] = '// Varyings to fragment shader\nout vec3 FragPos;\nout vec3 Normal;\nout vec2 TexCoord;\n\n'
        elif shader_type == "fragment":
            skeleton['inputs'] = '// Inputs from vertex shader\nin vec3 FragPos;\nin vec3 Normal;\nin vec2 TexCoord;\n\n'
            skeleton['outputs'] = '// Output color\nout vec4 FragColor;\n\n'
        elif shader_type == "geometry":
            skeleton['header'] = '#version 330 core\n#extension GL_EXT_geometry_shader4 : enable\n\n'
            skeleton['inputs'] = 'layout(triangles) in;\nin vec3 vertNormal[];\nin vec3 vertFragPos[];\n\n'
            skeleton['outputs'] = 'layout(triangle_strip, max_vertices = 3) out;\nout vec3 geomNormal;\nout vec3 geomFragPos;\n\n'

        return skeleton

    def integrate_modules(self, module_names: List[str], shader_config: Dict[str, Any] = None) -> str:
        """Integrate multiple modules into a shader"""
        if shader_config is None:
            shader_config = {}

        # Determine shader type based on configuration or default
        shader_type = shader_config.get('shader_type', 'fragment')

        # Create shader skeleton
        shader_skeleton = self.create_shader_skeleton(shader_type)

        # Get interfaces for all modules to validate integration
        module_interfaces = {}
        for module_name in module_names:
            interface = self.validator.get_module_interface(module_name)
            module_interfaces[module_name] = interface

        # Load and add functions from all modules
        for module_name in module_names:
            pseudocode = self.load_module_pseudocode(module_name)
            if pseudocode:
                shader_skeleton['functions'].append(pseudocode)

        # Generate appropriate main function based on module types
        shader_skeleton['main_function'] = self._generate_main_function(
            module_names, module_interfaces, shader_config
        )

        # Combine all parts
        shader_code = shader_skeleton['header']
        if shader_skeleton['defines']:
            shader_code += shader_skeleton['defines'] + "\n"
        shader_code += shader_skeleton['uniforms']
        shader_code += shader_skeleton['inputs']
        shader_code += shader_skeleton['outputs']
        if shader_skeleton['global_variables']:
            shader_code += shader_skeleton['global_variables'] + "\n"
        
        # Add all functions
        for func in shader_skeleton['functions']:
            if func.strip():
                shader_code += func + "\n"
        
        shader_code += shader_skeleton['main_function']

        return shader_code

    def _generate_main_function(self, module_names: List[str], 
                               module_interfaces: Dict[str, Dict], 
                               config: Dict[str, Any]) -> str:
        """Generate a main function based on the selected modules"""
        # Determine the primary rendering approach based on modules
        has_raymarching = any('raymarching' in name.lower() for name in module_names)
        has_lighting = any('light' in name.lower() for name in module_names or 
                          any('light' in pattern.lower() for pattern in 
                              module_interfaces.get(name, {}).get('metadata', {}).get('patterns', [])))
        has_texturing = any('texture' in name.lower() or 'uv' in name.lower() for name in module_names)
        has_effects = any('effect' in name.lower() or 'post' in name.lower() for name in module_names)

        if has_raymarching:
            return self._generate_raymarching_main(module_names, config)
        elif has_lighting and has_texturing:
            return self._generate_lit_textured_main(module_names, config)
        elif has_effects:
            return self._generate_effects_main(module_names, config)
        else:
            return self._generate_generic_main(module_names, config)

    def _generate_raymarching_main(self, module_names: List[str], config: Dict[str, Any]) -> str:
        """Generate main function for raymarching-based shaders"""
        main_func = '''void main() {
    // Set up ray
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / resolution.y;
    
    // Camera setup
    vec3 ro = vec3(0, 0, 3);  // Ray origin (camera position)
    vec3 rd = normalize(vec3(uv, -1.0));  // Ray direction
    
    // Apply camera transformation
    vec3 target = vec3(0, 0, 0);
    vec3 forward = normalize(target - ro);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));
    
    rd = normalize(forward + uv.x * right + uv.y * up);
    
    // Perform raymarching
    vec2 result = raymarch(ro, rd, 20.0, 64);
    float dist = result.x;
    float material_id = result.y;
    
    // Calculate final color
    vec3 color = vec3(0.0);
    
    if (dist < 20.0) {  // Hit something
        vec3 pos = ro + rd * dist;
        vec3 normal = calculateNormal(pos, 0.001);
        
        // Apply lighting if lighting modules are included
'''
        
        if any('light' in name.lower() for name in module_names):
            main_func += '''        vec3 viewDir = normalize(ro - pos);
        color = raymarchingLighting(pos, normal, viewDir, vec3(0.8, 0.6, 0.4), 0.2, 0.0);
'''
        else:
            main_func += '''        color = vec3(0.8, 0.6, 0.4);  // Default color
        color = color * max(dot(normal, normalize(vec3(1.0, 2.0, 1.0))), 0.1);  // Simple lighting
'''
        
        main_func += '''    } else {  // Background
        color = vec3(0.05, 0.07, 0.15);
        
        // Add some sky effect
        float sky = pow(1.0 - abs(uv.y), 2.0);
        color += vec3(0.1, 0.2, 0.4) * sky;
    }
    
    // Apply post-processing effects if available
'''
        
        if any('bloom' in name.lower() for name in module_names):
            main_func += '''    // Simplified bloom effect
    color = mix(color, vec3(1.0), 0.1 * length(color));
'''
        
        if any('chromatic' in name.lower() or 'aberration' in name.lower() for name in module_names):
            main_func += '''    // Simplified chromatic aberration
    vec2 offset = vec2(0.005, 0.0) * uv;
    float r = texture(gBuffer, TexCoord + offset).r;
    float g = texture(gBuffer, TexCoord).g;
    float b = texture(gBuffer, TexCoord - offset).b;
    color = vec3(r, g, b);
'''
        
        main_func += '''    
    // Apply gamma correction
    color = pow(color, vec3(0.4545));
    
    FragColor = vec4(color, 1.0);
}
'''
        return main_func

    def _generate_lit_textured_main(self, module_names: List[str], config: Dict[str, Any]) -> str:
        """Generate main function for lit and textured shaders"""
        main_func = '''void main() {
    vec3 color = vec3(0.0);
    
    // Use texture if available
'''
        
        if any('uv' in name.lower() or 'texture' in name.lower() for name in module_names):
            main_func += '''    vec3 texColor = texture(gTexture, TexCoord).rgb;
    color = texColor;
'''
        else:
            main_func += '''    color = vec3(0.8, 0.8, 0.8);  // Default color
'''
        
        if any('light' in name.lower() for name in module_names):
            main_func += '''    
    // Apply lighting
    vec3 lightPos = vec3(2.0, 4.0, 2.0);
    vec3 lightColor = vec3(1.0, 0.95, 0.8);
    vec3 lightDir = normalize(lightPos - FragPos);
    
    // Diffuse lighting
    float diff = max(dot(normalize(Normal), lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Ambient lighting
    vec3 ambient = 0.15 * texColor;
    
    // Combine lighting with texture
    color = (ambient + diffuse) * color;
'''
        
        main_func += '''    
    // Apply gamma correction
    color = pow(color, vec3(0.4545));
    
    FragColor = vec4(color, 1.0);
}
'''
        return main_func

    def _generate_effects_main(self, module_names: List[str], config: Dict[str, Any]) -> str:
        """Generate main function for effect shaders"""
        main_func = '''void main() {
    // Sample from texture (typically a rendered scene)
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    vec3 color = texture(gSceneTexture, uv).rgb;
    
'''
        
        if any('bloom' in name.lower() for name in module_names):
            main_func += '''    // Apply bloom effect
    vec3 brightColor = max(color - 0.5, 0.0);
    vec3 bloomColor = brightColor;  // In a real implementation, this would be the blurred version
    color += bloomColor * 0.5;
    
'''
        
        if any('distort' in name.lower() for name in module_names):
            main_func += '''    // Apply simple distortion
    vec2 distortedUV = uv + 0.01 * sin(uv.yx * 10.0 + time);
    color = texture(gSceneTexture, distortedUV).rgb;
    
'''
        
        if any('vignette' in name.lower() or 'post' in name.lower() for name in module_names):
            main_func += '''    // Apply vignette
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    float vig = 1.0 - dist * 0.5;
    color *= pow(vig, 2.0);
    
'''
        
        main_func += '''    // Apply tone mapping if available
    color = color / (color + vec3(1.0));
    
    // Apply gamma correction
    color = pow(color, vec3(0.4545));
    
    FragColor = vec4(color, 1.0);
}
'''
        return main_func

    def _generate_generic_main(self, module_names: List[str], config: Dict[str, Any]) -> str:
        """Generate a generic main function"""
        main_func = '''void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    
    // Default color based on UV position
    vec3 color = 0.5 + 0.5 * cos(time + uv.xyx + vec3(0, 2, 4));
    
    // Add some animation
    color *= abs(sin(time * 0.5)) * 0.5 + 0.5;
    
    FragColor = vec4(color, 1.0);
}
'''
        return main_func

    def generate_shader_with_validation(self, module_names: List[str], 
                                      shader_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a shader with validation of module compatibility"""
        if shader_config is None:
            shader_config = {}

        # Validate the module combination
        validation_result = self._validate_module_combination(module_names)
        
        if not validation_result['valid']:
            print("Warning: Module combination has validation issues:")
            for issue in validation_result['issues']:
                print(f"  - {issue}")

        # Generate the shader
        shader_code = self.integrate_modules(module_names, shader_config)
        
        return {
            'shader_code': shader_code,
            'validation': validation_result,
            'module_names': module_names,
            'config': shader_config
        }

    def _validate_module_combination(self, module_names: List[str]) -> Dict[str, Any]:
        """Validate that the combination of modules is compatible"""
        issues = []
        
        # Check for genre compatibility using cross-genre validator
        for i, source_module in enumerate(module_names):
            for j, target_module in enumerate(module_names):
                if i != j:
                    # In a full implementation, we would validate specific connections
                    # For now, we'll just check genre compatibility
                    source_interface = self.validator.get_module_interface(source_module)
                    target_interface = self.validator.get_module_interface(target_module)
                    
                    source_genre = source_interface.get('genre', 'unknown')
                    target_genre = target_interface.get('genre', 'unknown')
                    
                    if source_genre != target_genre:
                        if not self.validator._check_genre_compatibility(source_genre, target_genre):
                            issues.append(f"Genre incompatibility: {source_genre} -> {target_genre}")

        # Check for potential conflicts between modules
        all_patterns = set()
        for module_name in module_names:
            module_interface = self.validator.get_module_interface(module_name)
            patterns = module_interface.get('metadata', {}).get('patterns', [])
            all_patterns.update([p.lower() for p in patterns])

        # Check for conflicting patterns (simplified check)
        if 'forward' in all_patterns and 'deferred' in all_patterns:
            issues.append("Potential rendering pipeline conflict: Forward and Deferred rendering patterns")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'module_count': len(module_names)
        }

    def save_shader(self, shader_data: Dict[str, Any], filename: str) -> bool:
        """Save the generated shader to a file"""
        try:
            with open(filename, 'w') as f:
                f.write(shader_data['shader_code'])
            
            # Also save metadata
            metadata_file = filename.replace('.glsl', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'module_names': shader_data['module_names'],
                    'config': shader_data['config'],
                    'validation': shader_data['validation'],
                    'generated_at': __import__('datetime').datetime.now().isoformat()
                }, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving shader to {filename}: {e}")
            return False


def create_sample_shaders():
    """Create sample shaders demonstrating different module combinations"""
    pipeline = ShaderGenerationPipeline()
    
    # Sample 1: Basic lighting shader
    lighting_modules = ['lighting/diffuse/diffuse_lighting', 'lighting/specular/specular_lighting']
    lighting_shader = pipeline.generate_shader_with_validation(lighting_modules)
    
    pipeline.save_shader(lighting_shader, 'sample_lighting_shader.glsl')
    print("✓ Generated sample lighting shader: sample_lighting_shader.glsl")
    
    # Sample 2: Raymarching shader
    if all(module in [m['name'] for m in get_raymarching_modules()] 
           for module in ['raymarching_core', 'sdf_primitives', 'raymarching_lighting']):
        raymarching_modules = ['raymarching/raymarching_core', 'raymarching/sdf_primitives', 'raymarching/raymarching_lighting']
        raymarching_shader = pipeline.generate_shader_with_validation(raymarching_modules)
        
        pipeline.save_shader(raymarching_shader, 'sample_raymarching_shader.glsl')
        print("✓ Generated sample raymarching shader: sample_raymarching_shader.glsl")
    
    # Sample 3: Textured and lit shader
    try:
        textured_modules = [
            'texturing/uv_mapping/uv_mapping', 
            'lighting/pbr/pbr_lighting'
        ]
        textured_shader = pipeline.generate_shader_with_validation(textured_modules)
        
        pipeline.save_shader(textured_shader, 'sample_textured_shader.glsl')
        print("✓ Generated sample textured shader: sample_textured_shader.glsl")
    except:
        print("⚠ Could not generate textured shader (modules might not be loaded)")
    
    # Sample 4: Effects shader
    try:
        effect_modules = ['effects/bloom/bloom_effect', 'effects/postprocessing/post_processing']
        effect_shader = pipeline.generate_shader_with_validation(effect_modules)
        
        pipeline.save_shader(effect_shader, 'sample_effects_shader.glsl')
        print("✓ Generated sample effects shader: sample_effects_shader.glsl")
    except:
        print("⚠ Could not generate effects shader (modules might not be loaded)")
    
    # Sample 5: Complex combination
    try:
        complex_modules = [
            'geometry',  # This would need to exist
            'lighting/diffuse/diffuse_lighting', 
            'effects/bloom/bloom_effect'
        ]
        # Filter to only modules that actually exist
        available_modules = {m['name'] for m in pipeline.registry.get_all_modules()}
        valid_complex_modules = [m for m in complex_modules if 
                                any(m.split('/')[-1] in available_name for available_name in available_modules) or
                                m in available_modules]
        
        if valid_complex_modules:
            complex_shader = pipeline.generate_shader_with_validation(valid_complex_modules)
            pipeline.save_shader(complex_shader, 'sample_complex_shader.glsl')
            print("✓ Generated sample complex shader: sample_complex_shader.glsl")
    except:
        print("⚠ Could not generate complex shader")


def run_pipeline_demo():
    """Run a demonstration of the shader generation pipeline"""
    print("Shader Generation Pipeline Demo")
    print("=" * 40)
    
    pipeline = ShaderGenerationPipeline()
    
    # Show available modules
    all_modules = pipeline.registry.get_all_modules()
    print(f"Available modules: {len(all_modules)}")
    
    # Group by genre
    genres = {}
    for module in all_modules:
        genre = module['genre']
        if genre not in genres:
            genres[genre] = []
        genres[genre].append(module['module_name'])
    
    print("\nModules by genre:")
    for genre, modules in genres.items():
        print(f"  {genre}: {len(modules)} modules")
        # Show first few modules for each genre
        for module in modules[:3]:
            print(f"    - {module}")
        if len(modules) > 3:
            print(f"    ... and {len(modules) - 3} more")
    
    print(f"\nGenerating sample shaders...")
    create_sample_shaders()
    
    print(f"\nPipeline demo completed!")


if __name__ == "__main__":
    run_pipeline_demo()