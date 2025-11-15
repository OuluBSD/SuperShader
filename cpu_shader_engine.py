#!/usr/bin/env python3
"""
CPU-based Shader Execution Engine
Implements a complete software rendering pipeline that can execute shaders on CPU
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import time


class ShaderType(Enum):
    """Types of shaders supported"""
    VERTEX = "vertex"
    FRAGMENT = "fragment" 
    GEOMETRY = "geometry"
    COMPUTE = "compute"


@dataclass
class VertexInput:
    """Input data for vertex shader"""
    position: np.ndarray  # vec3 or vec4
    normal: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    tangent: Optional[np.ndarray] = None


@dataclass
class VertexOutput:
    """Output data from vertex shader"""
    position: np.ndarray  # gl_Position
    normal: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    tangent: Optional[np.ndarray] = None


@dataclass
class FragmentInput:
    """Input data for fragment shader (interpolated from vertex outputs)"""
    position: np.ndarray  # screen space position
    normal: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    depth: float = 0.0


@dataclass
class FragmentOutput:
    """Output data from fragment shader"""
    color: np.ndarray  # vec4 color
    depth: float = 0.0


class UniformBuffer:
    """Uniform buffer for shader data"""
    def __init__(self):
        self.data: Dict[str, Any] = {}
    
    def set_uniform(self, name: str, value: Any):
        self.data[name] = value
    
    def get_uniform(self, name: str) -> Any:
        return self.data.get(name)


class Texture2D:
    """CPU-based 2D texture"""
    def __init__(self, width: int, height: int, channels: int = 4):
        self.width = width
        self.height = height
        self.channels = channels
        self.data = np.zeros((height, width, channels), dtype=np.float32)
    
    def set_data(self, data: np.ndarray):
        """Set texture data"""
        if data.shape == (self.height, self.width, self.channels):
            self.data = data.astype(np.float32)
        else:
            # Resize if necessary
            from scipy.ndimage import zoom
            scale_factors = (self.height / data.shape[0], self.width / data.shape[1], 1)
            self.data = zoom(data, scale_factors, order=1).astype(np.float32)
    
    def sample(self, u: float, v: float) -> np.ndarray:
        """Sample texture at UV coordinates"""
        # Wrap coordinates
        u = u % 1.0
        v = v % 1.0
        
        # Convert to pixel coordinates
        x = int(u * (self.width - 1))
        y = int(v * (self.height - 1))
        
        # Clamp to bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        
        return self.data[y, x]
    
    def sample_linear(self, u: float, v: float) -> np.ndarray:
        """Sample texture with bilinear interpolation"""
        # Wrap coordinates
        u = u % 1.0
        v = v % 1.0
        
        # Convert to pixel coordinates
        x = u * (self.width - 1)
        y = v * (self.height - 1)
        
        # Get integer and fractional parts
        x0 = int(x)
        y0 = int(y)
        x1 = min(x0 + 1, self.width - 1)
        y1 = min(y0 + 1, self.height - 1)
        
        # Get fractional part for interpolation
        fx = x - x0
        fy = y - y0
        
        # Sample four corners
        c00 = self.data[y0, x0]
        c01 = self.data[y0, x1]
        c10 = self.data[y1, x0]
        c11 = self.data[y1, x1]
        
        # Interpolate
        c0 = c00 * (1 - fx) + c01 * fx
        c1 = c10 * (1 - fx) + c11 * fx
        result = c0 * (1 - fy) + c1 * fy
        
        return result


class CPURasterizer:
    """CPU-based rasterizer that processes primitives"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.framebuffer = np.zeros((height, width, 4), dtype=np.float32)  # RGBA
        self.depth_buffer = np.ones((height, width), dtype=np.float32) * np.inf
    
    def clear(self):
        """Clear the framebuffer and depth buffer"""
        self.framebuffer.fill(0.0)
        self.depth_buffer.fill(np.inf)
    
    def draw_triangle(self, vertices: List[VertexOutput], 
                     fragment_shader: Callable[[FragmentInput], FragmentOutput]):
        """Draw a triangle using scanline algorithm"""
        # Sort vertices by Y coordinate
        sorted_vertices = sorted(vertices, key=lambda v: v.position[1])
        v0, v1, v2 = sorted_vertices
        
        # Calculate bounding box
        min_y = max(0, int(min(v.position[1] for v in vertices)))
        max_y = min(self.height - 1, int(max(v.position[1] for v in vertices)))
        
        # Process each scanline
        for y in range(min_y, max_y + 1):
            # Find X intersections with current scanline
            x_intersections = []
            
            # Check each edge
            for v_start, v_end in [(v0, v1), (v1, v2), (v2, v0)]:
                if v_start.position[1] <= y < v_end.position[1] or v_end.position[1] <= y < v_start.position[1]:
                    # Calculate intersection
                    if v_end.position[1] != v_start.position[1]:
                        t = (y - v_start.position[1]) / (v_end.position[1] - v_start.position[1])
                        x = v_start.position[0] + t * (v_end.position[0] - v_start.position[0])
                        x_intersections.append(x)
            
            if len(x_intersections) >= 2:
                x_start = int(min(x_intersections))
                x_end = int(max(x_intersections))
                
                # Clamp to screen bounds
                x_start = max(0, min(x_start, self.width - 1))
                x_end = max(0, min(x_end, self.width - 1))
                
                # Draw horizontal line
                for x in range(x_start, x_end + 1):
                    # Calculate barycentric coordinates for interpolation
                    # Simple approach: just use the triangle edges
                    if x < self.width and y < self.height:
                        # Create fragment input by interpolating vertex attributes
                        # For simplicity, just use vertex 0's attributes for now
                        frag_input = FragmentInput(
                            position=np.array([float(x), float(y), 0.0, 1.0]),
                            normal=v0.normal,
                            uv=v0.uv,
                            color=v0.color,
                            depth=v0.position[2] if len(v0.position) > 2 else 0.0
                        )
                        
                        # Execute fragment shader
                        frag_output = fragment_shader(frag_input)
                        
                        # Write to framebuffer if depth test passes
                        if frag_output.depth < self.depth_buffer[y, x]:
                            self.depth_buffer[y, x] = frag_output.depth
                            self.framebuffer[y, x] = frag_output.color


class CPUBasedShaderEngine:
    """CPU-based shader execution engine"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.rasterizer = CPURasterizer(width, height)
        self.uniforms = UniformBuffer()
        self.textures: Dict[str, Texture2D] = {}
        self.max_threads = 4  # Maximum threads for parallel processing
        
    def add_texture(self, name: str, texture: Texture2D):
        """Add a texture to the engine"""
        self.textures[name] = texture
    
    def set_viewport(self, x: int, y: int, width: int, height: int):
        """Set viewport for rendering"""
        # For now, just store for later use
        pass
    
    def execute_vertex_shader(self, vertex_shader: Callable[[VertexInput], VertexOutput], 
                            vertices: List[VertexInput]) -> List[VertexOutput]:
        """Execute vertex shader on a list of vertices"""
        results = []
        for vertex in vertices:
            result = vertex_shader(vertex)
            results.append(result)
        return results
    
    def execute_fragment_shader(self, fragment_shader: Callable[[FragmentInput], FragmentOutput],
                              fragment_inputs: List[FragmentInput]) -> List[FragmentOutput]:
        """Execute fragment shader on a list of fragment inputs"""
        results = []
        for frag_input in fragment_inputs:
            result = fragment_shader(frag_input)
            results.append(result)
        return results
    
    def execute_compute_shader(self, compute_shader: Callable[[int, int, int], Any],
                             grid_size: Tuple[int, int, int]) -> np.ndarray:
        """Execute compute shader on a grid"""
        # Create a result buffer
        result = np.zeros((grid_size[0], grid_size[1], grid_size[2]), dtype=np.float32)
        
        # Execute compute shader for each grid location
        for z in range(grid_size[2]):
            for y in range(grid_size[1]):
                for x in range(grid_size[0]):
                    result[x, y, z] = compute_shader(x, y, z)
        
        return result
    
    def render_frame(self, vertices: List[VertexInput], 
                    vertex_shader: Callable[[VertexInput], VertexOutput],
                    fragment_shader: Callable[[FragmentInput], FragmentOutput],
                    topology: str = "triangles"):
        """Render a frame using the provided shaders"""
        # Clear the framebuffer
        self.rasterizer.clear()
        
        # Execute vertex shader
        vertex_outputs = self.execute_vertex_shader(vertex_shader, vertices)
        
        # Process primitives based on topology
        if topology == "triangles" and len(vertex_outputs) >= 3:
            # Process triangles in groups of 3
            for i in range(0, len(vertex_outputs), 3):
                if i + 2 < len(vertex_outputs):
                    triangle = vertex_outputs[i:i+3]
                    
                    # Rasterize the triangle
                    self.rasterizer.draw_triangle(triangle, fragment_shader)
        
        return self.rasterizer.framebuffer
    
    def get_rendered_image(self) -> np.ndarray:
        """Get the rendered image as a numpy array"""
        return self.rasterizer.framebuffer.copy()
    
    def save_image(self, filename: str):
        """Save the rendered image to a file"""
        try:
            import PIL.Image as Image
            # Convert from 0-1 range to 0-255 range
            img_data = (self.rasterizer.framebuffer * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_data, 'RGBA')
            img.save(filename)
            print(f"Image saved to {filename}")
        except ImportError:
            print("PIL not available. Install Pillow to save images.")
            # Just print some stats
            print(f"Framebuffer shape: {self.rasterizer.framebuffer.shape}")
            print(f"Framebuffer range: {self.rasterizer.framebuffer.min()} to {self.rasterizer.framebuffer.max()}")


def example_vertex_shader(vertex_input: VertexInput) -> VertexOutput:
    """Example vertex shader function"""
    # Simple transformation (just pass through with some modifications)
    position = vertex_input.position.copy()
    position[2] += 0.1  # Add some depth
    
    return VertexOutput(
        position=position,
        normal=vertex_input.normal,
        uv=vertex_input.uv,
        color=vertex_input.color
    )


def example_fragment_shader(fragment_input: FragmentInput) -> FragmentOutput:
    """Example fragment shader function"""
    # Simple color based on position
    r = abs(fragment_input.position[0] / 800.0)  # Normalize to 0-1
    g = abs(fragment_input.position[1] / 600.0)  # Normalize to 0-1
    b = 0.5  # Constant blue
    
    return FragmentOutput(
        color=np.array([r, g, b, 1.0]),
        depth=fragment_input.depth
    )


def create_cpu_shader_engine_demo():
    """Create a demonstration of the CPU shader engine"""
    print("Creating CPU-based shader execution engine...")
    
    # Create an engine instance
    engine = CPUBasedShaderEngine(800, 600)
    
    # Create some test vertices (a simple triangle)
    test_vertices = [
        VertexInput(position=np.array([100.0, 100.0, 0.0, 1.0])),
        VertexInput(position=np.array([400.0, 100.0, 0.0, 1.0])),
        VertexInput(position=np.array([250.0, 400.0, 0.0, 1.0]))
    ]
    
    # Render the frame
    start_time = time.time()
    result = engine.render_frame(
        test_vertices,
        example_vertex_shader,
        example_fragment_shader,
        "triangles"
    )
    end_time = time.time()
    
    print(f"Rendered frame in {end_time - start_time:.4f} seconds")
    print(f"Framebuffer shape: {result.shape}")
    
    # Try to save the image
    engine.save_image("cpu_shader_demo.png")
    
    print("CPU-based shader execution engine created successfully!")


if __name__ == "__main__":
    create_cpu_shader_engine_demo()