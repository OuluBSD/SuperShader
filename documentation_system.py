"""
Documentation and Tutorial System
Part of SuperShader Project - Phase 9: User Interface and Developer Experience

This module generates interactive documentation for all modules,
creates step-by-step tutorials for common use cases, implements
an example browser with live previews, and adds API reference with usage examples.
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import webbrowser
import http.server
import socketserver
import threading
from jinja2 import Template


@dataclass
class ModuleDocumentation:
    """Documentation for a single shader module"""
    module_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    conflicts: List[str]
    code_example: str
    usage_examples: List[str]
    performance_notes: List[str]


@dataclass
class Tutorial:
    """A step-by-step tutorial"""
    id: str
    title: str
    description: str
    steps: List[Dict[str, Any]]
    estimated_time: int  # in minutes
    difficulty: str  # beginner, intermediate, advanced
    related_modules: List[str]


@dataclass
class APIReference:
    """API reference entry"""
    function_name: str
    parameters: List[Dict[str, str]]  # {name, type, description}
    return_type: str
    description: str
    example: str


class DocumentationGenerator:
    """
    System for generating documentation for shader modules
    """
    
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.modules_docs: Dict[str, ModuleDocumentation] = {}
        self.tutorials: Dict[str, Tutorial] = {}
        self.api_references: Dict[str, APIReference] = {}
    
    def add_module_documentation(self, doc: ModuleDocumentation) -> None:
        """Add documentation for a module"""
        self.modules_docs[doc.module_id] = doc
    
    def add_tutorial(self, tutorial: Tutorial) -> None:
        """Add a tutorial"""
        self.tutorials[tutorial.id] = tutorial
    
    def add_api_reference(self, ref: APIReference) -> None:
        """Add an API reference"""
        self.api_references[ref.function_name] = ref
    
    def generate_module_documentation(self, module_id: str) -> str:
        """Generate HTML documentation for a specific module"""
        if module_id not in self.modules_docs:
            return f"<h1>Module {module_id} not found</h1>"
        
        doc = self.modules_docs[module_id]
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ name }} - Module Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .param-table { width: 100%; border-collapse: collapse; }
        .param-table th, .param-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .param-table th { background-color: #f2f2f2; }
        .code-block { background-color: #f5f5f5; padding: 10px; border-radius: 3px; font-family: monospace; }
        .warning { background-color: #ffdddd; padding: 10px; border-left: 5px solid #ff0000; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ name }}</h1>
        <p>{{ description }}</p>
    </div>
    
    <div class="section">
        <h2>Parameters</h2>
        <table class="param-table">
            <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Default</th>
                <th>Description</th>
            </tr>
            {% for name, param in parameters.items() %}
            <tr>
                <td>{{ name }}</td>
                <td>{{ param.get('type', 'unknown') }}</td>
                <td>{{ param.get('default', 'N/A') }}</td>
                <td>{{ param.get('description', '') }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Dependencies</h2>
        <ul>
            {% for dep in dependencies %}
            <li>{{ dep }}</li>
            {% endfor %}
            {% if not dependencies %}<li>None</li>{% endif %}
        </ul>
    </div>
    
    <div class="section">
        <h2>Conflicts</h2>
        <ul>
            {% for conflict in conflicts %}
            <li>{{ conflict }}</li>
            {% endfor %}
            {% if not conflicts %}<li>None</li>{% endif %}
        </ul>
    </div>
    
    <div class="section">
        <h2>Code Example</h2>
        <div class="code-block">
            <pre>{{ code_example }}</pre>
        </div>
    </div>
    
    <div class="section">
        <h2>Usage Examples</h2>
        {% for example in usage_examples %}
        <div class="code-block">
            <pre>{{ example }}</pre>
        </div>
        {% endfor %}
        {% if not usage_examples %}<p>No usage examples available.</p>{% endif %}
    </div>
    
    <div class="section">
        <h2>Performance Notes</h2>
        {% if performance_notes %}
        <ul>
            {% for note in performance_notes %}
            <li>{{ note }}</li>
            {% endfor %}
        </ul>
        {% else %}
        <p>No performance notes available.</p>
        {% endif %}
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(
            name=doc.name,
            description=doc.description,
            parameters=doc.parameters,
            dependencies=doc.dependencies,
            conflicts=doc.conflicts,
            code_example=doc.code_example,
            usage_examples=doc.usage_examples,
            performance_notes=doc.performance_notes
        )
    
    def generate_tutorial_html(self, tutorial_id: str) -> str:
        """Generate HTML for a tutorial"""
        if tutorial_id not in self.tutorials:
            return f"<h1>Tutorial {tutorial_id} not found</h1>"
        
        tutorial = self.tutorials[tutorial_id]
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - Tutorial</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background-color: #e8f4fd; padding: 20px; border-radius: 5px; }
        .step { margin: 30px 0; padding: 15px; border-left: 4px solid #007acc; }
        .step-number { display: inline-block; background-color: #007acc; color: white; width: 30px; height: 30px; text-align: center; border-radius: 50%; line-height: 30px; margin-right: 10px; }
        .code-block { background-color: #f5f5f5; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; }
        .difficulty-advanced { color: #d9534f; font-weight: bold; }
        .difficulty-intermediate { color: #f0ad4e; font-weight: bold; }
        .difficulty-beginner { color: #5cb85c; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>{{ description }}</p>
        <p><strong>Estimated Time:</strong> {{ estimated_time }} minutes | 
           <strong>Difficulty:</strong> <span class="difficulty-{{ difficulty }}">{{ difficulty }}</span> |
           <strong>Related Modules:</strong> {{ related_modules|join(", ") }}</p>
    </div>
    
    {% for step in steps %}
    <div class="step">
        <div>
            <span class="step-number">{{ loop.index }}</span>
            <h3>{{ step.title }}</h3>
        </div>
        <p>{{ step.description }}</p>
        {% if step.code_example %}
        <div class="code-block">
            <pre>{{ step.code_example }}</pre>
        </div>
        {% endif %}
    </div>
    {% endfor %}
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(
            title=tutorial.title,
            description=tutorial.description,
            steps=tutorial.steps,
            estimated_time=tutorial.estimated_time,
            difficulty=tutorial.difficulty,
            related_modules=tutorial.related_modules
        )
    
    def generate_api_reference_html(self, func_name: str) -> str:
        """Generate HTML for an API reference"""
        if func_name not in self.api_references:
            return f"<h1>API Reference for {func_name} not found</h1>"
        
        ref = self.api_references[func_name]
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ function_name }} - API Reference</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f8f8f8; padding: 15px; border-radius: 5px; }
        .param-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .param-table th, .param-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .param-table th { background-color: #f2f2f2; }
        .code-block { background-color: #f5f5f5; padding: 10px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ function_name }}</h1>
        <p>{{ description }}</p>
    </div>
    
    <h2>Parameters</h2>
    {% if parameters %}
    <table class="param-table">
        <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Description</th>
        </tr>
        {% for param in parameters %}
        <tr>
            <td>{{ param.name }}</td>
            <td>{{ param.type }}</td>
            <td>{{ param.description }}</td>
        </tr>
        {% endfor %}
    </table>
    {% else %}
    <p>No parameters required.</p>
    {% endif %}
    
    <h2>Return Type</h2>
    <p>{{ return_type }}</p>
    
    <h2>Example Usage</h2>
    <div class="code-block">
        <pre>{{ example }}</pre>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(
            function_name=ref.function_name,
            parameters=ref.parameters,
            return_type=ref.return_type,
            description=ref.description,
            example=ref.example
        )
    
    def generate_index_page(self) -> str:
        """Generate the main index page for documentation"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>SuperShader Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; margin-bottom: 30px; }
        .module-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .module-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .module-card h3 { margin-top: 0; color: #007acc; }
        .category-section { margin: 40px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SuperShader Documentation</h1>
        <p>Comprehensive documentation for shader modules, tutorials, and API references</p>
    </div>
    
    <div class="category-section">
        <h2>Shader Modules</h2>
        <div class="module-grid">
            {% for module_id, module in modules_docs.items() %}
            <div class="module-card">
                <h3><a href="modules/{{ module_id }}.html">{{ module.name }}</a></h3>
                <p>{{ module.description }}</p>
                <p><small>Type: {{ module_id.split('_')[0] if '_' in module_id else 'general' }}</small></p>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div class="category-section">
        <h2>Tutorials</h2>
        <div class="module-grid">
            {% for tutorial_id, tutorial in tutorials.items() %}
            <div class="module-card">
                <h3><a href="tutorials/{{ tutorial_id }}.html">{{ tutorial.title }}</a></h3>
                <p>{{ tutorial.description }}</p>
                <p><small>Difficulty: {{ tutorial.difficulty }} | Time: {{ tutorial.estimated_time }} min</small></p>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div class="category-section">
        <h2>API References</h2>
        <div class="module-grid">
            {% for func_name, ref in api_references.items() %}
            <div class="module-card">
                <h3><a href="api/{{ func_name }}.html">{{ ref.function_name }}</a></h3>
                <p>{{ ref.description }}</p>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(
            modules_docs=self.modules_docs,
            tutorials=self.tutorials,
            api_references=self.api_references
        )
    
    def generate_all_documentation(self):
        """Generate all documentation files"""
        # Create subdirectories
        (self.output_dir / "modules").mkdir(exist_ok=True)
        (self.output_dir / "tutorials").mkdir(exist_ok=True)
        (self.output_dir / "api").mkdir(exist_ok=True)
        
        # Generate individual module docs
        for module_id in self.modules_docs:
            html_content = self.generate_module_documentation(module_id)
            with open(self.output_dir / "modules" / f"{module_id}.html", "w") as f:
                f.write(html_content)
        
        # Generate tutorial docs
        for tutorial_id in self.tutorials:
            html_content = self.generate_tutorial_html(tutorial_id)
            with open(self.output_dir / "tutorials" / f"{tutorial_id}.html", "w") as f:
                f.write(html_content)
        
        # Generate API reference docs
        for func_name in self.api_references:
            html_content = self.generate_api_reference_html(func_name)
            with open(self.output_dir / "api" / f"{func_name}.html", "w") as f:
                f.write(html_content)
        
        # Generate index page
        index_content = self.generate_index_page()
        with open(self.output_dir / "index.html", "w") as f:
            f.write(index_content)
        
        print(f"Generated documentation in {self.output_dir}")


class TutorialSystem:
    """
    System for creating and managing tutorials
    """
    
    def __init__(self):
        self.tutorials: Dict[str, Tutorial] = {}
    
    def create_basic_lighting_tutorial(self):
        """Create a basic lighting tutorial"""
        tutorial = Tutorial(
            id="basic_lighting",
            title="Basic Lighting Setup",
            description="Learn how to set up basic lighting in your shaders using SuperShader modules",
            estimated_time=15,
            difficulty="beginner",
            related_modules=["diffuse_lighting", "specular_lighting"],
            steps=[
                {
                    "title": "Introduction",
                    "description": "In this tutorial, you'll learn to create a basic lighting setup using the Diffuse and Specular lighting modules.",
                    "code_example": ""
                },
                {
                    "title": "Setting up the Scene",
                    "description": "First, create a new shader composition and add the Diffuse Lighting module.",
                    "code_example": "// In the visual editor, add the 'Diffuse Lighting' module\n// Set light color to vec3(1.0, 1.0, 1.0) for white light"
                },
                {
                    "title": "Configuring Diffuse Lighting",
                    "description": "Configure the diffuse lighting parameters such as light color and ambient factor.",
                    "code_example": """
// Diffuse Lighting Parameters
uniform vec3 light_color = vec3(1.0, 1.0, 1.0);  // White light
uniform float ambient_factor = 0.2;               // 20% ambient contribution
                    """
                },
                {
                    "title": "Adding Specular Highlights",
                    "description": "Add the Specular Lighting module to create shiny highlights on surfaces.",
                    "code_example": """
// Specular Lighting Parameters
uniform float shininess = 32.0;            // Material shininess
uniform float specular_strength = 0.5;     // Strength of highlights
                    """
                },
                {
                    "title": "Connecting Modules",
                    "description": "Connect the output of one lighting calculation to the next to combine effects.",
                    "code_example": """
// Combining Diffuse and Specular
vec3 final_color = diffuse_color + specular_color;
                    """
                },
                {
                    "title": "Final Result",
                    "description": "You now have a basic lighting setup that combines diffuse and specular components for realistic lighting.",
                    "code_example": """
void main() {
    // Calculate lighting components
    vec3 diffuse = calculate_diffuse_lighting(normal, light_dir, light_color, ambient_factor);
    vec3 specular = calculate_specular_lighting(normal, light_dir, view_dir, shininess, specular_strength);
    
    // Combine for final color
    vec3 final_color = base_color * diffuse + specular;
    gl_FragColor = vec4(final_color, 1.0);
}
                    """
                }
            ]
        )
        self.tutorials[tutorial.id] = tutorial
    
    def create_post_processing_tutorial(self):
        """Create a post-processing tutorial"""
        tutorial = Tutorial(
            id="post_processing",
            title="Post-Processing Effects",
            description="Learn how to apply post-processing effects using SuperShader modules",
            estimated_time=20,
            difficulty="intermediate",
            related_modules=["bloom_effect", "color_correction"],
            steps=[
                {
                    "title": "Introduction",
                    "description": "In this tutorial, you'll learn to apply post-processing effects like bloom and color correction to enhance your rendered scenes.",
                    "code_example": ""
                },
                {
                    "title": "Creating a Post-Processing Pass",
                    "description": "Set up a new shader composition specifically for post-processing that will receive the rendered scene as input.",
                    "code_example": """
// Post-processing fragment shader receives rendered texture
uniform sampler2D scene_texture;  // The rendered scene
uniform vec2 resolution;          // Screen resolution
                    """
                },
                {
                    "title": "Adding Bloom Effect",
                    "description": "Add the Bloom Effect module to create glowing highlights.",
                    "code_example": """
// Bloom Effect Parameters
uniform float bloom_threshold = 0.8;   // Brightness threshold for bloom
uniform float bloom_intensity = 1.0;   // Strength of the bloom effect
                    """
                },
                {
                    "title": "Color Correction",
                    "description": "Apply color correction to adjust the overall look and feel of the image.",
                    "code_example": """
// Color Correction Module
vec3 apply_color_correction(vec3 color, vec3 saturation, vec3 contrast, vec3 brightness) {
    // Apply saturation, contrast, brightness adjustments
    return color;
}
                    """
                },
                {
                    "title": "Combining Effects",
                    "description": "Combine multiple post-processing effects for a polished look.",
                    "code_example": """
void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    vec3 color = texture2D(scene_texture, uv).rgb;
    
    // Apply color correction first
    color = apply_color_correction(color, saturation, contrast, brightness);
    
    // Then apply bloom effect
    color = apply_bloom(color, bloom_intensity, bloom_threshold);
    
    gl_FragColor = vec4(color, 1.0);
}
                    """
                },
                {
                    "title": "Optimization Tips",
                    "description": "Learn how to optimize post-processing effects for better performance.",
                    "code_example": """
// Performance considerations:
// - Use lower quality settings on mobile devices
// - Consider performing effects at lower resolution
// - Use separable filters when possible
                    """
                }
            ]
        )
        self.tutorials[tutorial.id] = tutorial
    
    def get_all_tutorials(self) -> List[Tutorial]:
        """Get all available tutorials"""
        return list(self.tutorials.values())


class ExampleBrowser:
    """
    System for browsing examples with live previews
    """
    
    def __init__(self):
        self.examples: Dict[str, Dict[str, Any]] = {}
    
    def add_example(self, example_id: str, name: str, description: str, 
                   shader_code: str, preview_image_path: str = None):
        """Add an example to the browser"""
        self.examples[example_id] = {
            'name': name,
            'description': description,
            'shader_code': shader_code,
            'preview_image': preview_image_path
        }
    
    def create_shader_examples(self):
        """Create example shaders for the browser"""
        # Basic diffuse lighting
        self.add_example(
            'diffuse_lighting',
            'Diffuse Lighting',
            'Simple diffuse lighting calculation based on surface normal and light direction',
            '''
// Diffuse Lighting Example
uniform vec3 light_position;
uniform vec3 light_color;
varying vec3 normal;
varying vec3 world_position;

void main() {
    vec3 light_dir = normalize(light_position - world_position);
    float diff = max(dot(normalize(normal), light_dir), 0.0);
    vec3 diffuse = diff * light_color;
    gl_FragColor = vec4(diffuse, 1.0);
}
            '''.strip()
        )
        
        # Specular lighting
        self.add_example(
            'specular_lighting',
            'Specular Lighting',
            'Phong/Blinn-Phong specular highlights',
            '''
// Specular Lighting Example
uniform vec3 light_position;
uniform vec3 view_position;
uniform vec3 light_color;
uniform float shininess;
varying vec3 normal;
varying vec3 world_position;

void main() {
    vec3 norm = normalize(normal);
    vec3 light_dir = normalize(light_position - world_position);
    vec3 view_dir = normalize(view_position - world_position);
    vec3 reflect_dir = reflect(-light_dir, norm);
    
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    vec3 specular = spec * light_color;
    gl_FragColor = vec4(specular, 1.0);
}
            '''.strip()
        )
        
        # Texturing
        self.add_example(
            'texturing',
            'Basic Texturing',
            'Simple texture sampling with UV coordinates',
            '''
// Basic Texturing Example
uniform sampler2D texture_sampler;
uniform vec2 uv_scale;
uniform vec2 uv_offset;
varying vec2 v_uv;

void main() {
    vec2 uv = v_uv * uv_scale + uv_offset;
    vec4 tex_color = texture2D(texture_sampler, uv);
    gl_FragColor = tex_color;
}
            '''.strip()
        )
        
        # Post-process bloom
        self.add_example(
            'bloom_effect',
            'Bloom Effect',
            'Post-processing bloom effect that makes bright areas glow',
            '''
// Bloom Effect Example
uniform sampler2D scene_texture;
uniform vec2 resolution;
uniform float threshold;
uniform float intensity;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    vec3 color = texture2D(scene_texture, uv).rgb;
    
    // Extract bright areas
    vec3 bloom = vec3(0.0);
    if (color.r > threshold || color.g > threshold || color.b > threshold) {
        bloom = color * intensity;
    }
    
    gl_FragColor = vec4(color + bloom, 1.0);
}
            '''.strip()
        )
    
    def get_examples_list(self) -> List[Dict[str, str]]:
        """Get a list of all examples with basic info"""
        return [
            {
                'id': example_id,
                'name': info['name'],
                'description': info['description']
            }
            for example_id, info in self.examples.items()
        ]


class DocumentationTutorialSystem:
    """
    Main system for documentation and tutorials
    """
    
    def __init__(self):
        self.doc_generator = DocumentationGenerator()
        self.tutorial_system = TutorialSystem()
        self.example_browser = ExampleBrowser()
        
        # Create default content
        self._create_default_content()
    
    def _create_default_content(self):
        """Create default documentation, tutorials, and examples"""
        # Add module documentation
        self.doc_generator.add_module_documentation(ModuleDocumentation(
            module_id="diffuse_lighting",
            name="Diffuse Lighting",
            description="Basic diffuse lighting calculation using the Lambert cosine law",
            parameters={
                "light_color": {"type": "vec3", "default": [1.0, 1.0, 1.0], "description": "Color of the light source"},
                "ambient_factor": {"type": "float", "default": 0.2, "description": "Ambient light contribution"}
            },
            dependencies=[],
            conflicts=[],
            code_example="""
vec3 calculate_diffuse_lighting(vec3 normal, vec3 light_dir, vec3 light_color, float ambient_factor) {
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 ambient = ambient_factor * light_color;
    vec3 diffuse = diff * light_color;
    return ambient + diffuse;
}
            """,
            usage_examples=[
                """
// Simple usage
vec3 lighting = calculate_diffuse_lighting(v_normal, light_direction, light_color, ambient_factor);
vec3 final_color = material_color * lighting;
                """
            ],
            performance_notes=[
                "Normal vector should be normalized for accurate results",
                "Use lower precision floats on mobile devices if possible"
            ]
        ))
        
        self.doc_generator.add_module_documentation(ModuleDocumentation(
            module_id="specular_lighting",
            name="Specular Lighting",
            description="Phong/Blinn-Phong specular highlights for shiny surfaces",
            parameters={
                "shininess": {"type": "float", "default": 32.0, "description": "Material shininess exponent"},
                "specular_strength": {"type": "float", "default": 0.5, "description": "Strength of specular highlights"}
            },
            dependencies=["diffuse_lighting"],
            conflicts=[],
            code_example="""
vec3 calculate_specular_lighting(vec3 normal, vec3 light_dir, vec3 view_dir, float shininess, float specular_strength) {
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    return specular_strength * spec * vec3(1.0);
}
            """,
            usage_examples=[
                """
// Phong specular
vec3 view_dir = normalize(view_position - frag_position);
vec3 specular = calculate_specular_lighting(normal, light_dir, view_dir, shininess, specular_strength);
                """
            ],
            performance_notes=[
                "The pow() function is computationally expensive",
                "Consider using Blinn-Phong for better performance"
            ]
        ))
        
        # Create tutorials
        self.tutorial_system.create_basic_lighting_tutorial()
        self.tutorial_system.create_post_processing_tutorial()
        
        # Create examples
        self.example_browser.create_shader_examples()
        
        # Add API references
        self.doc_generator.add_api_reference(APIReference(
            function_name="combine_lighting",
            parameters=[
                {"name": "diffuse", "type": "vec3", "description": "Diffuse lighting component"},
                {"name": "specular", "type": "vec3", "description": "Specular lighting component"},
                {"name": "ambient", "type": "vec3", "description": "Ambient lighting component"}
            ],
            return_type="vec3",
            description="Combines different lighting components into a final lighting result",
            example="""
vec3 final_lighting = combine_lighting(diffuse_light, specular_light, ambient_light);
vec3 final_color = material_color * final_lighting + emissive_color;
            """
        ))
        
        self.doc_generator.add_api_reference(APIReference(
            function_name="transform_normal",
            parameters=[
                {"name": "normal", "type": "vec3", "description": "Vertex normal in object space"},
                {"name": "normal_matrix", "type": "mat3", "description": "Normal transformation matrix"}
            ],
            return_type="vec3",
            description="Transforms a normal vector from object space to world/view space",
            example="""
vec3 world_normal = transform_normal(v_normal, normal_matrix);
            """
        ))
    
    def generate_all_documentation(self):
        """Generate all documentation"""
        self.doc_generator.generate_all_documentation()
    
    def start_documentation_server(self, port: int = 8000):
        """Start a local server to serve documentation"""
        os.chdir(self.doc_generator.output_dir)
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            pass
        
        server = socketserver.TCPServer(("", port), Handler)
        
        print(f"Documentation server starting on port {port}")
        print(f"Open your browser to http://localhost:{port}")
        
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        try:
            webbrowser.open(f"http://localhost:{port}")
            print("Press Ctrl+C to stop the server")
            server_thread.join()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()
            server.server_close()


def main():
    """
    Example usage of the Documentation and Tutorial System
    """
    print("Documentation and Tutorial System")
    print("Part of SuperShader Project - Phase 9")
    
    # Create the documentation system
    docs_system = DocumentationTutorialSystem()
    
    # Generate all documentation
    print("Generating all documentation...")
    docs_system.generate_all_documentation()
    
    # Show available tutorials and examples
    print("\nAvailable Tutorials:")
    for tutorial in docs_system.tutorial_system.get_all_tutorials():
        print(f"  - {tutorial.title}: {tutorial.description}")
    
    print("\nAvailable Examples:")
    for example in docs_system.example_browser.get_examples_list():
        print(f"  - {example['name']}: {example['description']}")
    
    # Note: The documentation server would be started here if we wanted interactive viewing
    print(f"\nDocumentation generated in: {docs_system.doc_generator.output_dir}")
    print("To view documentation interactively, run 'python -m http.server 8000' in the docs directory")


if __name__ == "__main__":
    main()