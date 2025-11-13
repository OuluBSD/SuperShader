#!/usr/bin/env python3
"""
Application Builder for SuperShader
Builds complete 3D applications with configurable scenes, shaders, and features
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add the project to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

from create_module_engine import ModuleEngine
from create_pseudocode_translator import PseudocodeTranslator
from generator.shader_generation_pipeline import ShaderGenerationPipeline
from create_module_registry import ModuleRegistry
from shader_optimizer import ShaderOptimizer, OptimizationLevel


class SceneConfiguration:
    """Configuration class for scene settings"""
    def __init__(self):
        self.entities = []
        self.camera = {
            'type': 'static',  # static, following, free
            'position': [0, 0, 5],
            'target': [0, 0, 0],
            'fov': 45.0,
            'aspect_ratio': 16/9
        }
        self.lighting = {
            'models': ['pbr'],  # pbr, phong, blinn_phong, cel_shading
            'lights': [],
            'shadows': True
        }
        self.post_effects = {
            'bloom': False,
            'motion_blur': False,
            'depth_of_field': False,
            'ssao': False,
            'tonemapping': True
        }
        self.neural_models = {
            'enabled': False,
            'models': [],
            'integration_type': 'post_process'  # pre_process, post_process, real_time
        }
        self.rendering = {
            'width': 1024,
            'height': 768,
            'msaa_samples': 4,
            'target_language': 'glsl'
        }
        self.physics = {
            'enabled': True,
            'gravity': [0, -9.81, 0],
            'solver': 'verlet'  # verlet, euler, rk4
        }


class ApplicationBuilder:
    """Main application builder class"""
    def __init__(self):
        self.shader_pipeline = ShaderGenerationPipeline()
        self.module_engine = ModuleEngine()
        self.translator = PseudocodeTranslator()
        self.registry = ModuleRegistry()
        self.config = SceneConfiguration()
        self.scene_modules = []
        self.output_dir = "builds"
        
    def configure_scene(self, config: SceneConfiguration):
        """Set the scene configuration"""
        self.config = config
        
    def add_entity(self, entity_type: str, position: List[float], rotation: List[float] = None, scale: List[float] = None):
        """Add an entity to the scene"""
        entity = {
            'type': entity_type,
            'position': position,
            'rotation': rotation or [0, 0, 0],
            'scale': scale or [1, 1, 1],
            'materials': [],
            'shaders': []
        }
        self.config.entities.append(entity)
        
    def add_light(self, light_type: str, position: List[float], color: List[float] = None, intensity: float = 1.0):
        """Add a light to the scene"""
        light = {
            'type': light_type,  # point, directional, spot, area
            'position': position,
            'color': color or [1, 1, 1],
            'intensity': intensity
        }
        self.config.lighting['lights'].append(light)
        
    def enable_post_effect(self, effect_name: str, enabled: bool = True):
        """Enable or disable a post-processing effect"""
        if effect_name in self.config.post_effects:
            self.config.post_effects[effect_name] = enabled
            
    def select_lighting_model(self, model_name: str):
        """Select a lighting model"""
        if model_name not in self.config.lighting['models']:
            self.config.lighting['models'].append(model_name)
            
    def enable_neural_model(self, model_name: str, integration_type: str = 'post_process'):
        """Enable neural network model integration"""
        self.config.neural_models['enabled'] = True
        self.config.neural_models['models'].append(model_name)
        self.config.neural_models['integration_type'] = integration_type
        
    def setup_camera(self, camera_type: str, position: List[float], target: List[float] = None):
        """Setup camera configuration"""
        self.config.camera['type'] = camera_type
        self.config.camera['position'] = position
        if target:
            self.config.camera['target'] = target
            
    def select_target_language(self, language: str):
        """Select target shader language"""
        self.config.rendering['target_language'] = language
        
    def build_shader_modules(self) -> Dict[str, str]:
        """Build shaders based on configuration"""
        shaders = {}
        
        # Build lighting shaders based on selected models
        lighting_modules = []
        for model in self.config.lighting['models']:
            if model == 'pbr':
                lighting_modules.extend([
                    'lighting/pbr/pbr_lighting',
                    'lighting/normal_mapping/normal_mapping',
                    'lighting/shadow_mapping/shadow_mapping'
                ])
            elif model == 'phong':
                lighting_modules.extend([
                    'lighting/diffuse/diffuse_lighting',
                    'lighting/specular/specular_lighting'
                ])
            elif model == 'blinn_phong':
                lighting_modules.extend([
                    'lighting/diffuse/diffuse_lighting',
                    'lighting/specular/specular_lighting'
                ])
            elif model == 'cel_shading':
                lighting_modules.extend([
                    'lighting/cel_shading/cel_shading',
                    'lighting/diffuse/diffuse_lighting'
                ])
        
        # Build post-processing shaders
        post_modules = []
        if self.config.post_effects['bloom']:
            post_modules.extend(['effects/bloom/bloom_effect'])
        if self.config.post_effects['tonemapping']:
            post_modules.extend(['effects/postprocessing/post_processing'])
        
        # Generate shaders for each type
        target_lang = self.config.rendering['target_language']
        
        if lighting_modules:
            lighting_shader = self.shader_pipeline.generate_shader_with_validation(
                lighting_modules, 
                {'shader_type': 'fragment'}
            )
            shaders['lighting'] = self.translator.translate(
                lighting_shader['shader_code'], 
                target_lang
            )
        
        if post_modules:
            post_shader = self.shader_pipeline.generate_shader_with_validation(
                post_modules,
                {'shader_type': 'fragment'}
            )
            shaders['post_processing'] = self.translator.translate(
                post_shader['shader_code'],
                target_lang
            )
        
        # Add raymarching if needed for certain effects
        if any('raymarching' in entity['type'] for entity in self.config.entities):
            raymarching_modules = [
                'raymarching/raymarching_core',
                'raymarching/sdf_primitives',
                'raymarching/raymarching_lighting'
            ]
            ray_shader = self.shader_pipeline.generate_shader_with_validation(
                raymarching_modules,
                {'shader_type': 'fragment'}
            )
            shaders['raymarching'] = self.translator.translate(
                ray_shader['shader_code'],
                target_lang
            )
        
        return shaders
        
    def generate_scene_code(self) -> str:
        """Generate the main scene/application code based on configuration"""
        # This would generate code in the target language (C++, GLSL, etc.)
        # For now, we'll generate a basic C++/OpenGL structure
        target_lang = self.config.rendering['target_language']
        
        if target_lang == 'glsl':
            # Generate OpenGL/C++ application code that uses the shaders
            app_code = self._generate_opengl_app_code()
        elif target_lang == 'hlsl':
            # Generate DirectX application code
            app_code = self._generate_directx_app_code()
        elif target_lang == 'metal':
            # Generate Metal application code
            app_code = self._generate_metal_app_code()
        else:
            # Default to OpenGL/C++
            app_code = self._generate_opengl_app_code()
            
        return app_code
    
    def _generate_opengl_app_code(self) -> str:
        """Generate OpenGL/C++ application code"""
        width = self.config.rendering['width']
        height = self.config.rendering['height']
        
        code = f'''#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"
#include "model.h"

#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

// Settings
const unsigned int SCR_WIDTH = {width};
const unsigned int SCR_HEIGHT = {height};

// Camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main()
{{
    // glfw: initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "SuperShader Generated App", NULL, NULL);
    if (window == NULL)
    {{
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }}
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {{
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }}

    // configure global opengl state
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader program
    Shader lightingShader("lighting_shader.vs", "lighting_shader.fs");
    Shader postProcessingShader("post_processing.vs", "post_processing.fs");
'''
        
        # Add entity rendering based on configuration
        for i, entity in enumerate(self.config.entities):
            code += f'''
    // load models
    Model ourModel("{entity['type']}.obj");
'''
        
        # Add light configuration
        for i, light in enumerate(self.config.lighting['lights']):
            code += f'''
    lightingShader.setVec3("lightPositions[{i}]", {light['position'][0]}, {light['position'][1]}, {light['position'][2]});
    lightingShader.setVec3("lightColors[{i}]", {light['color'][0]}, {light['color'][1]}, {light['color'][2]});
'''
        
        # Add rendering loop
        code += f'''
    // render loop
    while (!glfwWindowShouldClose(window))
    {{
        // per-frame time logic
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        processInput(window);

        // render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // be sure to activate shader when setting uniforms/drawing objects
        lightingShader.use();
        lightingShader.setVec3("viewPos", camera.Position.x, camera.Position.y, camera.Position.z);
        lightingShader.setFloat("material.shininess", 32.0f);

        // view/projection transformations
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        lightingShader.setMat4("projection", projection);
        lightingShader.setMat4("view", view);

        // render the loaded model
'''
        
        for i, entity in enumerate(self.config.entities):
            code += f'''
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3({entity['position'][0]}, {entity['position'][1]}, {entity['position'][2]})); // translate it down so it's at the center of the scene
        model = glm::scale(model, glm::vec3({entity['scale'][0]}, {entity['scale'][1]}, {entity['scale'][2]}));    // it's a bit too big for our scene, so scale it down
        lightingShader.setMat4("model", model);
        ourModel.Draw(lightingShader);
'''
        
        code += f'''
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }}

    // glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    return 0;
}}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}}

// glfw: whenever the mouse moves, this callback is called
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{{
    if (firstMouse)
    {{
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }}

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{{
    camera.ProcessMouseScroll(yoffset);
}}
'''
        return code
    
    def _generate_directx_app_code(self) -> str:
        """Generate DirectX application code (placeholder)"""
        return f'''// DirectX {self.config.rendering['target_language']} Application
// Generated by SuperShader Build System

#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

// Placeholder DirectX application code
// Width: {self.config.rendering['width']}, Height: {self.config.rendering['height']}

// TODO: Implement DirectX rendering pipeline with configured shaders and scene
'''
    
    def _generate_metal_app_code(self) -> str:
        """Generate Metal application code (placeholder)"""
        return f'''// Metal {self.config.rendering['target_language']} Application
// Generated by SuperShader Build System

#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>

// Placeholder Metal application code
// Width: {self.config.rendering['width']}, Height: {self.config.rendering['height']}

// TODO: Implement Metal rendering pipeline with configured shaders and scene
'''
    
    def build(self, output_dir: str = None) -> bool:
        """Build the complete application"""
        if output_dir:
            self.output_dir = output_dir
            
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate shaders
        print("Generating shaders...")
        shaders = self.build_shader_modules()
        
        # Save shaders to files
        for shader_type, shader_code in shaders.items():
            shader_file = Path(self.output_dir) / f"{shader_type}_shader.glsl"
            with open(shader_file, 'w') as f:
                f.write(shader_code)
            print(f"  Saved {shader_type} shader to {shader_file}")
        
        # Generate application code
        print("Generating application code...")
        app_code = self.generate_scene_code()
        
        # Save application code
        app_file = Path(self.output_dir) / "main_application.cpp"
        with open(app_file, 'w') as f:
            f.write(app_code)
        print(f"  Saved application code to {app_file}")
        
        # Generate configuration file
        config_file = Path(self.output_dir) / "scene_config.json"
        config_data = {
            'scene': self.config.__dict__,
            'shaders': list(shaders.keys()),
            'build_timestamp': __import__('datetime').datetime.now().isoformat()
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"  Saved configuration to {config_file}")
        
        print(f"\nApplication built successfully in {self.output_dir}/")
        return True


def create_default_config() -> SceneConfiguration:
    """Create a default configuration"""
    config = SceneConfiguration()
    
    # Add a default cube entity
    config.entities = [{
        'type': 'cube',
        'position': [0, 0, 0],
        'rotation': [0, 0, 0],
        'scale': [1, 1, 1],
        'materials': [],
        'shaders': []
    }]
    
    # Add a default light
    config.lighting['lights'] = [{
        'type': 'directional',
        'position': [0, 0, 5],
        'color': [1, 1, 1],
        'intensity': 1.0
    }]
    
    return config


def main():
    """Main function to demonstrate the application builder"""
    parser = argparse.ArgumentParser(description='SuperShader Application Builder')
    parser.add_argument('--output', '-o', type=str, default='builds/default_app', help='Output directory for built application')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    parser.add_argument('--width', type=int, default=1024, help='Window width')
    parser.add_argument('--height', type=int, default=768, help='Window height')
    parser.add_argument('--camera', type=str, choices=['static', 'following', 'free'], default='free', help='Camera type')
    parser.add_argument('--lighting', type=str, choices=['pbr', 'phong', 'blinn_phong', 'cel_shading'], default='pbr', help='Lighting model')
    parser.add_argument('--bloom', action='store_true', help='Enable bloom effect')
    parser.add_argument('--ssao', action='store_true', help='Enable SSAO effect')
    parser.add_argument('--shadows', action='store_true', help='Enable shadows')
    parser.add_argument('--target', type=str, choices=['glsl', 'hlsl', 'metal', 'wgsl'], default='glsl', help='Target language')
    
    args = parser.parse_args()
    
    # Create application builder
    builder = ApplicationBuilder()
    
    # Configure from arguments
    config = create_default_config()
    config.rendering['width'] = args.width
    config.rendering['height'] = args.height
    config.camera['type'] = args.camera
    config.lighting['models'] = [args.lighting]
    config.post_effects['bloom'] = args.bloom
    config.post_effects['ssao'] = args.ssao
    config.lighting['shadows'] = args.shadows
    config.rendering['target_language'] = args.target
    
    builder.configure_scene(config)
    
    # Build the application
    success = builder.build(args.output)
    
    if success:
        print(f"\nApplication built successfully in {args.output}/")
        print("Next steps:")
        print(f"  1. Check the generated files in {args.output}/")
        print("  2. Review the scene_config.json for configuration details")
        print("  3. Compile the main_application.cpp with appropriate graphics libraries")
        print("  4. Link with the generated shaders")
    else:
        print("Build failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())