#!/usr/bin/env python3
"""
Software Rendering Pipeline
Implements a complete software rendering pipeline that runs on CPU
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum


class PrimitiveType(Enum):
    """Types of primitives supported by the pipeline"""
    POINTS = "points"
    LINES = "lines"
    TRIANGLES = "triangles"
    LINE_STRIP = "line_strip"
    TRIANGLE_STRIP = "triangle_strip"


@dataclass
class Vertex:
    """Vertex structure with position and attributes"""
    position: np.ndarray  # vec4 - homogeneous coordinates
    normal: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    tangent: Optional[np.ndarray] = None
    bitangent: Optional[np.ndarray] = None


@dataclass
class Fragment:
    """Fragment structure for pixel processing"""
    position: np.ndarray  # vec3 - screen space coordinates
    normal: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    depth: float = 0.0


@dataclass
class RenderTarget:
    """Render target for the pipeline"""
    width: int
    height: int
    color_buffer: np.ndarray  # RGBA
    depth_buffer: np.ndarray  # depth values
    stencil_buffer: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.color_buffer is None:
            self.color_buffer = np.zeros((self.height, self.width, 4), dtype=np.float32)
        if self.depth_buffer is None:
            self.depth_buffer = np.ones((self.height, self.width), dtype=np.float32) * np.inf


class TransformPipeline:
    """Handles transformation of vertices through the pipeline"""
    
    def __init__(self):
        self.model_matrix = np.eye(4, dtype=np.float32)
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.projection_matrix = np.eye(4, dtype=np.float32)
        self.viewport_matrix = np.eye(4, dtype=np.float32)
    
    def set_model_matrix(self, matrix: np.ndarray):
        self.model_matrix = matrix.astype(np.float32)
    
    def set_view_matrix(self, matrix: np.ndarray):
        self.view_matrix = matrix.astype(np.float32)
    
    def set_projection_matrix(self, matrix: np.ndarray):
        self.projection_matrix = matrix.astype(np.float32)
    
    def set_viewport(self, x: int, y: int, width: int, height: int):
        """Set viewport transformation"""
        # Viewport matrix to transform from NDC to screen coordinates
        self.viewport_matrix = np.array([
            [width/2, 0, 0, x + width/2],
            [0, height/2, 0, y + height/2],
            [0, 0, 0.5, 0.5],  # Depth from [-1,1] to [0,1]
            [0, 0, 0, 1]
        ], dtype=np.float32)
    
    def transform_vertex(self, vertex: Vertex) -> Vertex:
        """Transform a vertex through the pipeline"""
        # Apply MVP (Model-View-Projection) transformation
        world_pos = self.model_matrix @ vertex.position
        view_pos = self.view_matrix @ world_pos
        clip_pos = self.projection_matrix @ view_pos
        
        # Perspective division to get normalized device coordinates
        if clip_pos[3] != 0:
            ndc_pos = clip_pos[:3] / clip_pos[3]
        else:
            ndc_pos = clip_pos[:3]  # Avoid division by zero
        
        # Convert to screen space using viewport transformation
        screen_pos = self.viewport_matrix @ np.append(ndc_pos, 1.0)
        
        # Return transformed vertex with updated position
        return Vertex(
            position=screen_pos,
            normal=vertex.normal,
            uv=vertex.uv,
            color=vertex.color,
            tangent=vertex.tangent,
            bitangent=vertex.bitangent
        )


class Rasterizer:
    """Software rasterizer that converts primitives to fragments"""
    
    def __init__(self, render_target: RenderTarget):
        self.render_target = render_target
        self.width = render_target.width
        self.height = render_target.height
    
    def rasterize_triangle(self, v0: Vertex, v1: Vertex, v2: Vertex) -> List[Fragment]:
        """Rasterize a triangle using barycentric coordinates"""
        fragments = []
        
        # Get bounding box of the triangle
        min_x = max(0, int(min(v0.position[0], v1.position[0], v2.position[0])))
        max_x = min(self.width - 1, int(max(v0.position[0], v1.position[0], v2.position[0])))
        min_y = max(0, int(min(v0.position[1], v1.position[1], v2.position[1])))
        max_y = min(self.height - 1, int(max(v0.position[1], v1.position[1], v2.position[1])))
        
        # Calculate area of the triangle
        area = self._triangle_area_2d(v0.position, v1.position, v2.position)
        
        if area == 0:
            return fragments  # Degenerate triangle
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Calculate barycentric coordinates
                w0 = self._triangle_area_2d(np.array([x, y, 0, 1]), v1.position, v2.position) / area
                w1 = self._triangle_area_2d(np.array([x, y, 0, 1]), v2.position, v0.position) / area
                w2 = self._triangle_area_2d(np.array([x, y, 0, 1]), v0.position, v1.position) / area
                
                # Only render if point is inside triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Interpolate depth using barycentric coordinates
                    depth = w0 * v0.position[2] + w1 * v1.position[2] + w2 * v2.position[2]
                    
                    # Check depth buffer
                    if depth < self.render_target.depth_buffer[y, x]:
                        # Interpolate attributes using barycentric coordinates
                        pos = np.array([float(x), float(y), depth, 1.0])
                        
                        # For now, simple interpolation of just position
                        fragment = Fragment(
                            position=pos,
                            depth=depth
                        )
                        fragments.append(fragment)
                        
                        # Update depth buffer
                        self.render_target.depth_buffer[y, x] = depth
        
        return fragments
    
    def _triangle_area_2d(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate 2D area of a triangle using cross product"""
        return abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))


class SoftwareRenderer:
    """Main software rendering pipeline"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        
        # Create render target
        self.render_target = RenderTarget(
            width=width,
            height=height,
            color_buffer=np.zeros((height, width, 4), dtype=np.float32),
            depth_buffer=np.ones((height, width), dtype=np.float32) * np.inf
        )
        
        # Initialize pipeline components
        self.transform_pipeline = TransformPipeline()
        self.rasterizer = Rasterizer(self.render_target)
        
        # Set default viewport
        self.transform_pipeline.set_viewport(0, 0, width, height)
        
        # Default matrices
        self._setup_default_matrices()
    
    def _setup_default_matrices(self):
        """Set up default transformation matrices"""
        # Default projection - orthographic for simplicity
        self.transform_pipeline.set_projection_matrix(np.array([
            [2.0/800.0, 0, 0, 0],
            [0, 2.0/600.0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32))
    
    def clear(self, color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0), 
              depth: float = 1.0):
        """Clear the render target"""
        self.render_target.color_buffer.fill(0)
        self.render_target.color_buffer[:, :] = color
        self.render_target.depth_buffer.fill(depth)
    
    def draw_primitives(self, vertices: List[Vertex], primitive_type: PrimitiveType):
        """Draw primitives using the software rendering pipeline"""
        if primitive_type == PrimitiveType.TRIANGLES:
            # Process triangles in groups of 3
            for i in range(0, len(vertices), 3):
                if i + 2 < len(vertices):
                    # Transform vertices
                    v0 = self.transform_pipeline.transform_vertex(vertices[i])
                    v1 = self.transform_pipeline.transform_vertex(vertices[i+1])
                    v2 = self.transform_pipeline.transform_vertex(vertices[i+2])
                    
                    # Rasterize triangle
                    fragments = self.rasterizer.rasterize_triangle(v0, v1, v2)
                    
                    # Process fragments (for now, just render them)
                    for fragment in fragments:
                        x, y = int(fragment.position[0]), int(fragment.position[1])
                        if 0 <= x < self.width and 0 <= y < self.height:
                            # Write color to framebuffer (simple white for now)
                            self.render_target.color_buffer[y, x] = [1.0, 0.0, 0.0, 1.0]  # Red
        elif primitive_type == PrimitiveType.LINES:
            # Process lines in groups of 2
            for i in range(0, len(vertices), 2):
                if i + 1 < len(vertices):
                    # Transform vertices
                    v0 = self.transform_pipeline.transform_vertex(vertices[i])
                    v1 = self.transform_pipeline.transform_vertex(vertices[i+1])
                    
                    # Draw line using Bresenham's algorithm
                    self._draw_line(v0, v1)
        elif primitive_type == PrimitiveType.POINTS:
            # Process points
            for vertex in vertices:
                v = self.transform_pipeline.transform_vertex(vertex)
                x, y = int(v.position[0]), int(v.position[1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.render_target.color_buffer[y, x] = [1.0, 1.0, 1.0, 1.0]  # White
    
    def _draw_line(self, v0: Vertex, v1: Vertex):
        """Draw a line between two transformed vertices using Bresenham's algorithm"""
        x0, y0 = int(v0.position[0]), int(v0.position[1])
        x1, y1 = int(v1.position[0]), int(v1.position[1])
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        
        error = dx - dy
        
        x, y = x0, y0
        while True:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.render_target.color_buffer[y, x] = [0.0, 1.0, 0.0, 1.0]  # Green
            
            if x == x1 and y == y1:
                break
                
            error2 = 2 * error
            if error2 > -dy:
                error -= dy
                x += x_step
            if error2 < dx:
                error += dx
                y += y_step
    
    def get_color_buffer(self) -> np.ndarray:
        """Get the color buffer as a numpy array"""
        return self.render_target.color_buffer.copy()
    
    def save_framebuffer(self, filename: str):
        """Save the rendered framebuffer to a file"""
        try:
            import PIL.Image as Image
            # Convert from 0-1 range to 0-255 range
            img_data = (self.render_target.color_buffer * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_data, 'RGBA')
            img.save(filename)
            print(f"Framebuffer saved to {filename}")
        except ImportError:
            print("PIL not available. Install Pillow to save images.")
            print(f"Framebuffer shape: {self.render_target.color_buffer.shape}")
            print(f"Framebuffer range: {self.render_target.color_buffer.min()} to {self.render_target.color_buffer.max()}")


def create_software_rendering_pipeline():
    """Create and demonstrate the software rendering pipeline"""
    print("Creating software rendering pipeline...")
    
    # Create a renderer
    renderer = SoftwareRenderer(800, 600)
    
    # Create some test vertices (a simple triangle)
    test_vertices = [
        Vertex(position=np.array([100.0, 100.0, 0.0, 1.0])),
        Vertex(position=np.array([400.0, 100.0, 0.0, 1.0])),
        Vertex(position=np.array([250.0, 400.0, 0.0, 1.0]))
    ]
    
    # Clear and render
    renderer.clear((0.2, 0.3, 0.4, 1.0))  # Clear with a blue-ish color
    renderer.draw_primitives(test_vertices, PrimitiveType.TRIANGLES)
    
    # Save the result
    renderer.save_framebuffer("software_rendering_demo.png")
    
    print("Software rendering pipeline created successfully!")


if __name__ == "__main__":
    create_software_rendering_pipeline()