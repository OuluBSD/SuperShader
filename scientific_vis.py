"""
Scientific and Research Applications
Part of SuperShader Project - Phase 10: Advanced Applications and Specialized Domains

This module develops specialized modules for scientific visualization,
creates shaders for data visualization and analysis, implements modules
for medical imaging applications, and adds support for physics simulation visualization.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class ScientificVisualizationConfig:
    """Configuration for scientific visualization modules"""
    visualization_type: str  # 'isosurface', 'volume_rendering', 'flow_field', 'scatter_plot', 'medical_imaging'
    data_format: str  # 'scalar_field', 'vector_field', 'tensor_field', 'image_sequence'
    data_range: Tuple[float, float]
    color_map: str  # 'viridis', 'plasma', 'jet', 'grayscale', 'medical'
    resolution: Tuple[int, int, int]
    transfer_function: Optional[Dict[str, Any]] = None


class IsosurfaceRenderer:
    """
    Isosurface rendering for scalar field visualization
    """
    
    def __init__(self, config: ScientificVisualizationConfig):
        self.config = config
        self.shader_template = self._generate_isosurface_shader()
    
    def _generate_isosurface_shader(self) -> str:
        """Generate shader code for isosurface rendering"""
        iso_value = (self.config.data_range[0] + self.config.data_range[1]) / 2.0  # Midpoint as example
        
        shader_code = f"""
// Isosurface Rendering Module
uniform sampler3D data_volume;
uniform float iso_value = {iso_value};  // Isovalue for surface extraction
uniform vec3 volume_dimensions;  // Size of the 3D volume
uniform mat4 model_view_matrix;
uniform mat4 projection_matrix;

// Marching cubes table (simplified)
vec3 interpolate(vec3 p1, vec3 p2, float val1, float val2, float iso) {{
    if (abs(val1 - val2) < 0.0001) return p1;
    float mu = (iso - val1) / (val2 - val1);
    return p1 + mu * (p2 - p1);
}}

// Scalar field lookup
float get_scalar_value(vec3 pos) {{
    // Normalize position to texture coordinates [0, 1]
    vec3 normalized_pos = pos / volume_dimensions;
    return texture(data_volume, normalized_pos).r;
}}

vec3 calculate_normal(vec3 pos) {{
    // Calculate gradient using central differences
    vec3 h = vec3(1.0, 0.0, 0.0);
    float dx = get_scalar_value(pos + h.xyy) - get_scalar_value(pos - h.xyy);
    float dy = get_scalar_value(pos + h.yxy) - get_scalar_value(pos - h.yxy);
    float dz = get_scalar_value(pos + h.yyx) - get_scalar_value(pos - h.yyx);
    return normalize(vec3(dx, dy, dz));
}}

// Simple ray marching for isosurface (for fragment shader)
float ray_march_iso(vec3 ray_start, vec3 ray_dir) {{
    const int MAX_STEPS = 100;
    const float STEP_SIZE = 0.01;
    
    vec3 current_pos = ray_start;
    for (int i = 0; i < MAX_STEPS; i++) {{
        float value = get_scalar_value(current_pos);
        if (value >= iso_value) {{
            return length(current_pos - ray_start);  // Distance to surface
        }}
        current_pos += ray_dir * STEP_SIZE;
    }}
    return -1.0;  // Did not hit surface
}}

vec4 render_isosurface(vec3 view_ray_origin, vec3 view_ray_direction) {{
    float distance_to_surface = ray_march_iso(view_ray_origin, view_ray_direction);
    
    if (distance_to_surface < 0.0) {{
        return vec4(0.0);  // No intersection
    }}
    
    vec3 surface_pos = view_ray_origin + view_ray_direction * distance_to_surface;
    vec3 normal = calculate_normal(surface_pos);
    
    // Simple lighting
    vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 color = vec3(0.8) * diff + vec3(0.2);  // Diffuse + ambient
    
    return vec4(color, 1.0);
}}
        """
        
        return shader_code
    
    def get_shader_code(self) -> str:
        """Return the generated shader code"""
        return self.shader_template


class VolumeRenderer:
    """
    Volume rendering for 3D scalar field visualization
    """
    
    def __init__(self, config: ScientificVisualizationConfig):
        self.config = config
        self.shader_template = self._generate_volume_shader()
    
    def _generate_volume_shader(self) -> str:
        """Generate shader code for volume rendering"""
        shader_code = f"""
// Volume Rendering Module
uniform sampler3D data_volume;
uniform vec3 volume_dimensions;
uniform vec3 view_ray_step_size;
uniform float absorption_coefficient = 0.1;
uniform int max_steps = 100;
uniform float iso_value_min = {self.config.data_range[0]};
uniform float iso_value_max = {self.config.data_range[1]};

// Transfer function parameters (simplified)
uniform vec3 color_min = vec3(0.0, 0.0, 1.0);  // Blue
uniform vec3 color_max = vec3(1.0, 0.0, 0.0);  // Red

// Convert scalar value to color based on transfer function
vec3 transfer_function(float value) {{
    // Normalize value to [0, 1] range
    float normalized = (value - {self.config.data_range[0]}) / ({self.config.data_range[1]} - {self.config.data_range[0]});
    return mix(color_min, color_max, normalized);
}}

// Simple ray marching through volume
vec4 ray_march_volume(vec3 ray_start, vec3 ray_dir) {{
    vec3 pos = ray_start;
    vec4 accumulated_color = vec4(0.0);
    float accumulated_alpha = 0.0;
    
    for (int i = 0; i < max_steps; i++) {{
        // Sample the volume
        vec3 normalized_pos = pos / volume_dimensions;
        float sample_value = texture(data_volume, normalized_pos).r;
        
        // Get color from transfer function
        vec3 sample_color = transfer_function(sample_value);
        float sample_alpha = (sample_value - {self.config.data_range[0]}) / ({self.config.data_range[1]} - {self.config.data_range[0]});
        
        // Apply absorption
        sample_alpha *= exp(-absorption_coefficient * length(view_ray_step_size));
        
        // Alpha compositing (front-to-back)
        vec4 color_alpha = vec4(sample_color * sample_alpha, sample_alpha);
        accumulated_color = accumulated_color + color_alpha * (1.0 - accumulated_alpha);
        accumulated_alpha = accumulated_alpha + sample_alpha * (1.0 - accumulated_alpha);
        
        // Advance ray
        pos += ray_dir * length(view_ray_step_size);
        
        // Check bounds
        if (pos.x < 0.0 || pos.x > volume_dimensions.x ||
            pos.y < 0.0 || pos.y > volume_dimensions.y ||
            pos.z < 0.0 || pos.z > volume_dimensions.z) {{
            break;
        }}
        
        // Early termination if fully opaque
        if (accumulated_alpha > 0.99) {{
            break;
        }}
    }}
    
    // Normalize alpha
    if (accumulated_alpha > 0.0) {{
        accumulated_color.rgb /= accumulated_alpha;
    }}
    
    return vec4(accumulated_color.rgb, accumulated_alpha);
}}
        """
        
        return shader_code
    
    def get_shader_code(self) -> str:
        """Return the generated shader code"""
        return self.shader_template


class FlowFieldVisualizer:
    """
    Flow field visualization for vector field data
    """
    
    def __init__(self, config: ScientificVisualizationConfig):
        self.config = config
        self.shader_template = self._generate_flow_shader()
    
    def _generate_flow_shader(self) -> str:
        """Generate shader code for flow field visualization"""
        shader_code = f"""
// Flow Field Visualization Module
uniform sampler3D vector_field;
uniform vec3 field_dimensions;
uniform float time;
uniform float step_size = 0.05;
uniform int max_integration_steps = 20;
uniform vec3 color_start = vec3(0.0, 0.0, 1.0);
uniform vec3 color_end = vec3(1.0, 1.0, 0.0);

// Get vector value at position
vec3 get_vector_value(vec3 pos) {{
    vec3 normalized_pos = pos / field_dimensions;
    return texture(vector_field, normalized_pos).xyz * 2.0 - 1.0;  // Convert from [0,1] to [-1,1]
}}

// Advect a particle through the vector field
vec3 integrate_pathline(vec3 start_pos, float direction) {{
    vec3 current_pos = start_pos;
    
    for (int i = 0; i < max_integration_steps; i++) {{
        vec3 velocity = get_vector_value(current_pos);
        current_pos += velocity * step_size * direction;
        
        // Check bounds
        if (current_pos.x < 0.0 || current_pos.x > field_dimensions.x ||
            current_pos.y < 0.0 || current_pos.y > field_dimensions.y ||
            current_pos.z < 0.0 || current_pos.z > field_dimensions.z) {{
            break;
        }}
    }}
    
    return current_pos;
}}

// Visualize streamlines or pathlines
vec4 visualize_flow(vec2 fragCoord) {{
    // Convert fragment coordinates to world coordinates
    vec3 world_pos = vec3(
        mod(fragCoord.x, field_dimensions.x),
        mod(fragCoord.y, field_dimensions.y),
        field_dimensions.z * 0.5  // Start in the middle of the volume
    );
    
    // Integrate forward to get endpoint
    vec3 end_pos = integrate_pathline(world_pos, 1.0);
    
    // Calculate normalized position along streamline
    float dist = length(end_pos - world_pos);
    float normalized_pos = mod(dist * 0.1 + time * 0.1, 1.0);
    
    // Color based on position and velocity magnitude
    vec3 velocity = get_vector_value(world_pos);
    float speed = length(velocity);
    vec3 color = mix(color_start, color_end, normalized_pos);
    color *= (0.5 + 0.5 * speed);  // Brighten based on speed
    
    // Fade out if not on a streamline
    float on_streamline = step(0.1, speed);  // Show only where there's flow
    
    return vec4(color, on_streamline);
}}

// Animated particles following flow
vec4 visualize_flow_particles(vec2 fragCoord) {{
    // Create animated particle positions
    vec2 seed = vec2(fragCoord.x * 0.01, fragCoord.y * 0.01);
    vec2 particle_pos = seed + vec2(sin(time + seed.x * 10.0), cos(time + seed.y * 10.0)) * 0.1;
    
    // Convert to volume coordinates
    vec3 vol_pos = vec3(
        particle_pos.x * field_dimensions.x,
        particle_pos.y * field_dimensions.y,
        field_dimensions.z * 0.5
    );
    
    // Get velocity at particle position
    vec3 velocity = get_vector_value(vol_pos);
    float speed = length(velocity);
    
    // Color based on velocity
    vec3 color = vec3(1.0, 0.5, 0.0) * (0.5 + 0.5 * speed);
    
    return vec4(color, speed > 0.1 ? 1.0 : 0.0);
}}
        """
        
        return shader_code
    
    def get_shader_code(self) -> str:
        """Return the generated shader code"""
        return self.shader_template


class MedicalImagingProcessor:
    """
    Medical imaging visualization and processing
    """
    
    def __init__(self, config: ScientificVisualizationConfig):
        self.config = config
        self.shader_template = self._generate_medical_shader()
    
    def _generate_medical_shader(self) -> str:
        """Generate shader code for medical imaging"""
        shader_code = f"""
// Medical Imaging Module
uniform sampler3D medical_volume;  // CT, MRI, etc.
uniform vec3 volume_dimensions;
uniform float window_center = 0.5;  // For windowing (CT values)
uniform float window_width = 0.3;   // For windowing
uniform bool apply_hounsfield = true;  // Whether to apply HU windowing (CT)
uniform float transparency = 0.8;

// Convert raw value to display value (e.g., Hounsfield units for CT)
float convert_to_display_value(float raw_value) {{
    if (apply_hounsfield) {{
        // Convert to HU and apply windowing
        float hu_value = raw_value * 2000.0 - 1000.0;  // Raw to HU
        float normalized = (hu_value - (window_center - window_width/2.0)) / window_width;
        return clamp(normalized, 0.0, 1.0);
    }} else {{
        return raw_value;  // Already normalized
    }}
}}

// Tissue-specific coloring
vec3 tissue_coloring(float value) {{
    // Simple tissue classification
    if (value < 0.1) return vec3(0.0, 0.0, 0.0);      // Air
    if (value < 0.2) return vec3(0.7, 0.7, 0.7);      // Lung
    if (value < 0.4) return vec3(0.9, 0.9, 0.9);      // Soft tissue
    if (value < 0.6) return vec3(0.9, 0.8, 0.8);      // Blood
    if (value < 0.8) return vec3(0.9, 0.7, 0.7);      // Bone
    return vec3(1.0, 1.0, 1.0);                        // Dense bone
}}

// Maximum Intensity Projection (MIP)
vec4 mip_render(vec3 ray_start, vec3 ray_dir, int steps) {{
    vec3 pos = ray_start;
    float max_value = -1.0;
    vec3 max_pos = vec3(0.0);
    
    for (int i = 0; i < steps; i++) {{
        vec3 normalized_pos = pos / volume_dimensions;
        float value = convert_to_display_value(texture(medical_volume, normalized_pos).r);
        
        if (value > max_value) {{
            max_value = value;
            max_pos = pos;
        }}
        
        pos += ray_dir * 1.0;
        
        // Check bounds
        if (pos.x < 0.0 || pos.x > volume_dimensions.x ||
            pos.y < 0.0 || pos.y > volume_dimensions.y ||
            pos.z < 0.0 || pos.z > volume_dimensions.z) {{
            break;
        }}
    }}
    
    if (max_value > 0.0) {{
        vec3 color = tissue_coloring(max_value);
        return vec4(color, transparency);
    }}
    
    return vec4(0.0, 0.0, 0.0, 0.0);
}}

// Surface rendering for medical volumes
vec4 surface_render(vec3 world_pos) {{
    // Sample the volume
    vec3 normalized_pos = world_pos / volume_dimensions;
    float value = convert_to_display_value(texture(medical_volume, normalized_pos).r);
    
    // Apply surface extraction based on threshold
    float threshold = 0.3;  // Adjustable threshold
    
    if (value > threshold) {{
        vec3 tissue_col = tissue_coloring(value);
        
        // Calculate normal for lighting
        vec3 h = vec3(1.0, 0.0, 0.0);
        float dx = convert_to_display_value(texture(medical_volume, (world_pos + h.xyy) / volume_dimensions).r) - 
                  convert_to_display_value(texture(medical_volume, (world_pos - h.xyy) / volume_dimensions).r);
        float dy = convert_to_display_value(texture(medical_volume, (world_pos + h.yxy) / volume_dimensions).r) - 
                  convert_to_display_value(texture(medical_volume, (world_pos - h.yxy) / volume_dimensions).r);
        float dz = convert_to_display_value(texture(medical_volume, (world_pos + h.yyx) / volume_dimensions).r) - 
                  convert_to_display_value(texture(medical_volume, (world_pos - h.yyx) / volume_dimensions).r);
        vec3 normal = normalize(vec3(dx, dy, dz));
        
        // Simple lighting
        vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(normal, light_dir), 0.0);
        vec3 lit_color = tissue_col * diff + tissue_col * 0.2;
        
        return vec4(lit_color, transparency);
    }}
    
    return vec4(0.0);
}}
        """
        
        return shader_code
    
    def get_shader_code(self) -> str:
        """Return the generated shader code"""
        return self.shader_template


class PhysicsSimulationVisualizer:
    """
    Physics simulation visualization
    """
    
    def __init__(self, config: ScientificVisualizationConfig):
        self.config = config
        self.shader_template = self._generate_physics_shader()
    
    def _generate_physics_shader(self) -> str:
        """Generate shader code for physics simulation visualization"""
        shader_code = f"""
// Physics Simulation Visualization Module
uniform sampler2D particle_positions;  // Texture with particle positions
uniform sampler2D particle_attributes; // Texture with particle velocities, temperatures, etc.
uniform int num_particles;
uniform vec2 viewport_size;
uniform float particle_size = 2.0;
uniform float time;
uniform vec3 color_by = vec3(1.0, 0.0, 0.0);  // How to color particles (velocity, temperature, etc.)

// Get particle position and attribute
vec4 get_particle_data(int index) {{
    // Convert index to texture coordinates
    float inv_particles = 1.0 / float(num_particles);
    vec2 coord = vec2(mod(float(index), viewport_size.x * inv_particles), 
                      floor(float(index) / viewport_size.x) * inv_particles);
    return texture2D(particle_positions, coord);
}}

// Visualize particle system
vec4 visualize_particles(vec2 fragCoord) {{
    vec4 final_color = vec4(0.0);
    
    // Go through nearby particles and draw them
    for (int i = 0; i < num_particles && i < 100; i++) {{ // Limit to 100 for performance
        vec4 particle_data = get_particle_data(i);
        vec2 particle_pos = particle_data.xy; // Position in viewport space
        float particle_w = particle_data.w;   // Additional data (e.g., temperature)
        
        vec2 diff = fragCoord - particle_pos;
        float dist = length(diff);
        
        // If pixel is within particle radius
        if (dist < particle_size) {{
            float intensity = 1.0 - dist / particle_size;
            vec3 particle_color = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), particle_w);
            final_color += vec4(particle_color * intensity, intensity);
        }}
    }}
    
    return final_color;
}}

// Visualize force fields
vec4 visualize_force_field(vec2 fragCoord) {{
    // Calculate force vector at this point
    // This would be based on your physics simulation
    vec2 field_pos = fragCoord / viewport_size;
    
    // Example force calculation (e.g., gravity field with some point masses)
    vec2 total_force = vec2(0.0);
    
    // Add forces from various sources
    vec2 point_mass_pos = vec2(0.5, 0.5); // Center of screen
    vec2 diff = field_pos - point_mass_pos;
    float dist_squared = dot(diff, diff);
    float dist = sqrt(dist_squared);
    
    if (dist > 0.01) {{
        vec2 force_dir = normalize(diff);
        float force_mag = 0.1 / dist_squared;  // Inverse square law
        total_force += force_dir * force_mag;
    }}
    
    // Color based on force direction and magnitude
    float magnitude = length(total_force);
    vec3 force_color = vec3(0.5) + 0.5 * normalize(vec3(total_force, 0.0));
    
    // Fade based on magnitude
    return vec4(force_color, magnitude * 0.5);
}}

// Visualize fluid dynamics (simplified)
vec4 visualize_fluid_flow(vec2 fragCoord) {{
    // Simulate fluid flow patterns
    vec2 uv = fragCoord / viewport_size;
    vec2 flow_dir = vec2(sin(uv.y * 10.0 + time), cos(uv.x * 8.0 + time * 1.2));
    
    // Color based on flow velocity and direction
    float speed = length(flow_dir);
    vec3 flow_color = vec3(
        0.5 + 0.5 * flow_dir.x,
        0.5 + 0.5 * flow_dir.y,
        0.2 + 0.3 * speed
    );
    
    return vec4(flow_color, 0.7);
}}
        """
        
        return shader_code
    
    def get_shader_code(self) -> str:
        """Return the generated shader code"""
        return self.shader_template


class ScientificVisualizationSystem:
    """
    Main system for scientific and research visualization
    """
    
    def __init__(self):
        self.isosurface_renderer: Optional[IsosurfaceRenderer] = None
        self.volume_renderer: Optional[VolumeRenderer] = None
        self.flow_visualizer: Optional[FlowFieldVisualizer] = None
        self.medical_processor: Optional[MedicalImagingProcessor] = None
        self.physics_visualizer: Optional[PhysicsSimulationVisualizer] = None
    
    def create_isosurface_renderer(self, config: ScientificVisualizationConfig) -> IsosurfaceRenderer:
        """Create an isosurface renderer"""
        self.isosurface_renderer = IsosurfaceRenderer(config)
        return self.isosurface_renderer
    
    def create_volume_renderer(self, config: ScientificVisualizationConfig) -> VolumeRenderer:
        """Create a volume renderer"""
        self.volume_renderer = VolumeRenderer(config)
        return self.volume_renderer
    
    def create_flow_visualizer(self, config: ScientificVisualizationConfig) -> FlowFieldVisualizer:
        """Create a flow field visualizer"""
        self.flow_visualizer = FlowFieldVisualizer(config)
        return self.flow_visualizer
    
    def create_medical_processor(self, config: ScientificVisualizationConfig) -> MedicalImagingProcessor:
        """Create a medical imaging processor"""
        self.medical_processor = MedicalImagingProcessor(config)
        return self.medical_processor
    
    def create_physics_visualizer(self, config: ScientificVisualizationConfig) -> PhysicsSimulationVisualizer:
        """Create a physics visualization processor"""
        self.physics_visualizer = PhysicsSimulationVisualizer(config)
        return self.physics_visualizer
    
    def generate_visualization_shader(self, vis_type: str, config: ScientificVisualizationConfig) -> str:
        """Generate shader code for a specific visualization type"""
        if vis_type == 'isosurface':
            renderer = self.create_isosurface_renderer(config)
            return renderer.get_shader_code()
        elif vis_type == 'volume_rendering':
            renderer = self.create_volume_renderer(config)
            return renderer.get_shader_code()
        elif vis_type == 'flow_field':
            visualizer = self.create_flow_visualizer(config)
            return visualizer.get_shader_code()
        elif vis_type == 'medical_imaging':
            processor = self.create_medical_processor(config)
            return processor.get_shader_code()
        elif vis_type == 'physics_simulation':
            visualizer = self.create_physics_visualizer(config)
            return visualizer.get_shader_code()
        else:
            raise ValueError(f"Unknown visualization type: {vis_type}")


def main():
    """
    Example usage of the Scientific Visualization System
    """
    print("Scientific and Research Applications")
    print("Part of SuperShader Project - Phase 10")
    
    # Create visualization system
    vis_system = ScientificVisualizationSystem()
    
    # Example configuration for data visualization
    config = ScientificVisualizationConfig(
        visualization_type='volume_rendering',
        data_format='scalar_field',
        data_range=(0.0, 1.0),
        color_map='viridis',
        resolution=(256, 256, 256),
        transfer_function={'mapping': 'linear', 'opacity': 0.8}
    )
    
    # Create a volume renderer
    print("\n--- Creating Volume Renderer ---")
    volume_renderer = vis_system.create_volume_renderer(config)
    shader_code = volume_renderer.get_shader_code()
    print(f"Generated volume rendering shader with {len(shader_code.split())} lines")
    
    # Example configuration for flow visualization
    flow_config = ScientificVisualizationConfig(
        visualization_type='flow_field',
        data_format='vector_field',
        data_range=(-1.0, 1.0),
        color_map='plasma',
        resolution=(128, 128, 64)
    )
    
    print("\n--- Creating Flow Field Visualizer ---")
    flow_visualizer = vis_system.create_flow_visualizer(flow_config)
    flow_shader = flow_visualizer.get_shader_code()
    print(f"Generated flow visualization shader with {len(flow_shader.split())} lines")
    
    # Example configuration for medical imaging
    med_config = ScientificVisualizationConfig(
        visualization_type='medical_imaging',
        data_format='scalar_field',
        data_range=(0.0, 1.0),
        color_map='medical',
        resolution=(512, 512, 100)
    )
    
    print("\n--- Creating Medical Imaging Processor ---")
    med_processor = vis_system.create_medical_processor(med_config)
    med_shader = med_processor.get_shader_code()
    print(f"Generated medical imaging shader with {len(med_shader.split())} lines")
    
    # Example configuration for physics simulation
    physics_config = ScientificVisualizationConfig(
        visualization_type='physics_simulation',
        data_format='vector_field',
        data_range=(0.0, 100.0),  # For particle positions
        color_map='jet',
        resolution=(1024, 768, 1)
    )
    
    print("\n--- Creating Physics Simulation Visualizer ---")
    physics_visualizer = vis_system.create_physics_visualizer(physics_config)
    physics_shader = physics_visualizer.get_shader_code()
    print(f"Generated physics visualization shader with {len(physics_shader.split())} lines")
    
    # Show all visualization types available
    print("\n--- Available Visualization Types ---")
    vis_types = [
        'isosurface', 'volume_rendering', 
        'flow_field', 'medical_imaging', 'physics_simulation'
    ]
    
    print("Visualization types implemented:")
    for vis_type in vis_types:
        print(f"  - {vis_type}")


if __name__ == "__main__":
    main()