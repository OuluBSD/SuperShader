#!/usr/bin/env python3
"""
Compute Shader Equivalents in C/C++
Generates C/C++ code equivalents for compute shaders that can be executed on CPU
"""

import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ComputeShaderConfig:
    """Configuration for compute shader generation"""
    local_size_x: int = 8
    local_size_y: int = 8
    local_size_z: int = 8
    use_simd: bool = False
    memory_layout: str = "aos"  # "aos" (array of structures) or "soa" (structure of arrays)


class ComputeShaderGenerator:
    """Generator for C/C++ compute shader equivalents"""
    
    def __init__(self):
        self.config = ComputeShaderConfig()
        self.cpp_templates = self._load_cpp_templates()
    
    def _load_cpp_templates(self) -> Dict[str, str]:
        """Load C++ code templates for compute operations"""
        return {
            "header": '''#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <cmath>
#include <iostream>

// Compute shader utilities
struct ComputeBuffer {{
    std::vector<float> data;
    size_t width, height, depth;
    
    ComputeBuffer(size_t w, size_t h, size_t d = 1) : width(w), height(h), depth(d) {{
        data.resize(w * h * d * 4); // RGBA per element
    }}
    
    float& operator()(size_t x, size_t y, size_t z = 0, size_t channel = 0) {{
        size_t index = ((z * height + y) * width + x) * 4 + channel;
        if (index >= data.size()) {{
            static float dummy = 0.0f;
            return dummy;
        }}
        return data[index];
    }}
    
    const float& operator()(size_t x, size_t y, size_t z = 0, size_t channel = 0) const {{
        size_t index = ((z * height + y) * width + x) * 4 + channel;
        if (index >= data.size()) {{
            static const float dummy = 0.0f;
            return dummy;
        }}
        return data[index];
    }}
}};

struct ComputeContext {{
    size_t global_id_x, global_id_y, global_id_z;
    size_t local_id_x, local_id_y, local_id_z;
    size_t group_id_x, group_id_y, group_id_z;
    size_t num_groups_x, num_groups_y, num_groups_z;
    
    ComputeContext(size_t gx, size_t gy, size_t gz, 
                   size_t lx, size_t ly, size_t lz,
                   size_t ggx, size_t ggy, size_t ggz,
                   size_t ngx, size_t ngy, size_t ngz)
        : global_id_x(gx), global_id_y(gy), global_id_z(gz),
          local_id_x(lx), local_id_y(ly), local_id_z(lz),
          group_id_x(ggx), group_id_y(ggy), group_id_z(ggz),
          num_groups_x(ngx), num_groups_y(ngy), num_groups_z(ngz) {{}}
}};

// Math functions equivalent to GLSL
static inline float clamp(float x, float minVal, float maxVal) {{
    return std::min(std::max(x, minVal), maxVal);
}}

static inline float smoothstep(float edge0, float edge1, float x) {{
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}}

static inline float mix(float x, float y, float a) {{
    return x * (1.0f - a) + y * a;
}}

// Vector operations
struct vec2 {{ float x, y; }};
struct vec3 {{ float x, y, z; }};
struct vec4 {{ float x, y, z, w; }};

static inline float dot(const vec3& a, const vec3& b) {{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}}

static inline vec3 normalize(const vec3& v) {{
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.0001f) {{
        return {{v.x / len, v.y / len, v.z / len}};
    }}
    return {{0.0f, 0.0f, 0.0f}};
}}

static inline float length(const vec3& v) {{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}}

static inline vec3 cross(const vec3& a, const vec3& b) {{
    return {{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}};
}}
''',
            "compute_function_wrapper": '''
// Compute shader function wrapper
void compute_shader_function(ComputeContext& ctx, {params}) {{
    // User-defined compute shader code goes here
    {compute_code}
}}

// Dispatch function to execute compute shader
void dispatch_compute(ComputeBuffer& output_buffer, 
                     size_t num_groups_x, size_t num_groups_y, size_t num_groups_z,
                     {thread_params}) {{
    std::vector<std::thread> threads;
    
    for (size_t gz = 0; gz < num_groups_z; ++gz) {{
        for (size_t gy = 0; gy < num_groups_y; ++gy) {{
            for (size_t gx = 0; gx < num_groups_x; ++gx) {{
                threads.emplace_back([=, &output_buffer, {capture_params}]() {{
                    const size_t local_size_x = {local_size_x};
                    const size_t local_size_y = {local_size_y};
                    const size_t local_size_z = {local_size_z};
                    
                    for (size_t lz = 0; lz < local_size_z; ++lz) {{
                        for (size_t ly = 0; ly < local_size_y; ++ly) {{
                            for (size_t lx = 0; lx < local_size_x; ++lx) {{
                                size_t global_x = gx * local_size_x + lx;
                                size_t global_y = gy * local_size_y + ly;
                                size_t global_z = gz * local_size_z + lz;
                                
                                if (global_x < output_buffer.width && 
                                    global_y < output_buffer.height && 
                                    global_z < output_buffer.depth) {{
                                    
                                    ComputeContext ctx(global_x, global_y, global_z,
                                                     lx, ly, lz,
                                                     gx, gy, gz,
                                                     num_groups_x, num_groups_y, num_groups_z);
                                    
                                    compute_shader_function(ctx, {param_values});
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}
    }}
    
    for (auto& t : threads) {{
        t.join();
    }}
}}
'''
        }
    
    def generate_compute_shader_cpp(self, compute_operations: List[Dict[str, Any]]) -> str:
        """Generate C++ equivalent for compute operations"""
        cpp_code = self.cpp_templates["header"]
        
        for i, operation in enumerate(compute_operations):
            # Generate compute function for each operation
            func_name = f"compute_operation_{i}"
            params = operation.get("params", "")
            compute_code = operation.get("code", "// No compute code provided")
            
            # Create compute function
            compute_func = f'''
// Compute operation {i}
void {func_name}(ComputeContext& ctx, {params}) {{
    // User-defined compute shader code
    {compute_code}
}}
'''
            cpp_code += compute_func
        
        # Add a sample compute operation
        sample_compute = '''
// Sample compute operation: simple image processing
void compute_image_blur(ComputeContext& ctx, ComputeBuffer& input_buffer, ComputeBuffer& output_buffer) {
    // Get current pixel position
    int x = ctx.global_id_x;
    int y = ctx.global_id_y;
    int z = ctx.global_id_z;  // For 3D textures
    
    // Simple 3x3 blur
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    int count = 0;
    
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            
            // Boundary check
            if (nx >= 0 && nx < input_buffer.width && 
                ny >= 0 && ny < input_buffer.height) {
                sum_r += input_buffer(nx, ny, z, 0);  // R channel
                sum_g += input_buffer(nx, ny, z, 1);  // G channel
                sum_b += input_buffer(nx, ny, z, 2);  // B channel
                count++;
            }
        }
    }
    
    // Average the values
    if (count > 0) {
        float avg_r = sum_r / count;
        float avg_g = sum_g / count;
        float avg_b = sum_b / count;
        
        // Write to output
        output_buffer(x, y, z, 0) = avg_r;
        output_buffer(x, y, z, 1) = avg_g;
        output_buffer(x, y, z, 2) = avg_b;
        output_buffer(x, y, z, 3) = input_buffer(x, y, z, 3);  // Preserve alpha
    }
}
'''
        cpp_code += sample_compute
        
        # Add compute blur dispatch function
        blur_dispatch = '''
// Dispatch function for image blur operation
void dispatch_image_blur(ComputeBuffer& input_buffer, ComputeBuffer& output_buffer,
                        size_t width, size_t height, size_t depth = 1) {
    // Calculate number of groups needed
    const size_t local_size_x = 8;
    const size_t local_size_y = 8;
    const size_t local_size_z = 1;  // 2D operation
    
    size_t num_groups_x = (width + local_size_x - 1) / local_size_x;
    size_t num_groups_y = (height + local_size_y - 1) / local_size_y;
    size_t num_groups_z = (depth + local_size_z - 1) / local_size_z;
    
    std::vector<std::thread> threads;
    
    for (size_t gz = 0; gz < num_groups_z; ++gz) {
        for (size_t gy = 0; gy < num_groups_y; ++gy) {
            for (size_t gx = 0; gx < num_groups_x; ++gx) {
                threads.emplace_back([=, &input_buffer, &output_buffer]() {
                    for (size_t lz = 0; lz < local_size_z; ++lz) {
                        for (size_t ly = 0; ly < local_size_y; ++ly) {
                            for (size_t lx = 0; lx < local_size_x; ++lx) {
                                size_t global_x = gx * local_size_x + lx;
                                size_t global_y = gy * local_size_y + ly;
                                size_t global_z = gz * local_size_z + lz;
                                
                                if (global_x < width && global_y < height && global_z < depth) {
                                    ComputeContext ctx(global_x, global_y, global_z,
                                                     lx, ly, lz,
                                                     gx, gy, gz,
                                                     num_groups_x, num_groups_y, num_groups_z);
                                    
                                    compute_image_blur(ctx, input_buffer, output_buffer);
                                }
                            }
                        }
                    }
                });
            }
        }
    }
    
    for (auto& t : threads) {
        t.join();
    }
}
'''
        cpp_code += blur_dispatch
        
        # Add physics simulation compute function
        physics_compute = '''
// Compute operation for physics simulation
void compute_physics_step(ComputeContext& ctx, 
                         ComputeBuffer& position_buffer, 
                         ComputeBuffer& velocity_buffer,
                         float deltaTime) {
    size_t idx = (ctx.global_id_z * position_buffer.height + ctx.global_id_y) * position_buffer.width + ctx.global_id_x;
    
    // Get current position and velocity
    vec3 pos = {position_buffer(idx, 0, 0, 0), 
                position_buffer(idx, 0, 0, 1), 
                position_buffer(idx, 0, 0, 2)};
    vec3 vel = {velocity_buffer(idx, 0, 0, 0), 
                velocity_buffer(idx, 0, 0, 1), 
                velocity_buffer(idx, 0, 0, 2)};
    
    // Simple physics integration (Euler method)
    // Apply gravity
    float gravity = -9.81f;
    vel.y += gravity * deltaTime;
    
    // Update position
    pos.x += vel.x * deltaTime;
    pos.y += vel.y * deltaTime;
    pos.z += vel.z * deltaTime;
    
    // Simple ground collision
    if (pos.y < 0.0f) {
        pos.y = 0.0f;
        vel.y = -vel.y * 0.7f; // Bounce with damping
    }
    
    // Write back
    position_buffer(idx, 0, 0, 0) = pos.x;
    position_buffer(idx, 0, 0, 1) = pos.y;
    position_buffer(idx, 0, 0, 2) = pos.z;
    velocity_buffer(idx, 0, 0, 0) = vel.x;
    velocity_buffer(idx, 0, 0, 1) = vel.y;
    velocity_buffer(idx, 0, 0, 2) = vel.z;
}
'''
        cpp_code += physics_compute
        
        # Add a general compute dispatcher
        general_dispatcher = '''
// Generic compute dispatcher for different operations
template<typename ComputeFunc>
void dispatch_compute_generic(size_t width, size_t height, size_t depth,
                            ComputeFunc compute_func) {
    const size_t local_size_x = 8;
    const size_t local_size_y = 8;
    const size_t local_size_z = 8;
    
    size_t num_groups_x = (width + local_size_x - 1) / local_size_x;
    size_t num_groups_y = (height + local_size_y - 1) / local_size_y;
    size_t num_groups_z = (depth + local_size_z - 1) / local_size_z;
    
    std::vector<std::thread> threads;
    
    for (size_t gz = 0; gz < num_groups_z; ++gz) {
        for (size_t gy = 0; gy < num_groups_y; ++gy) {
            for (size_t gx = 0; gx < num_groups_x; ++gx) {
                threads.emplace_back([=]() {
                    for (size_t lz = 0; lz < local_size_z; ++lz) {
                        for (size_t ly = 0; ly < local_size_y; ++ly) {
                            for (size_t lx = 0; lx < local_size_x; ++lx) {
                                size_t global_x = gx * local_size_x + lx;
                                size_t global_y = gy * local_size_y + ly;
                                size_t global_z = gz * local_size_z + lz;
                                
                                if (global_x < width && 
                                    global_y < height && 
                                    global_z < depth) {
                                    ComputeContext ctx(global_x, global_y, global_z,
                                                     lx, ly, lz,
                                                     gx, gy, gz,
                                                     num_groups_x, num_groups_y, num_groups_z);
                                    
                                    compute_func(ctx);
                                }
                            }
                        }
                    }
                });
            }
        }
    }
    
    for (auto& t : threads) {
        t.join();
    }
}
'''
        cpp_code += general_dispatcher
        
        return cpp_code
    
    def generate_from_pseudocode(self, pseudocode: str) -> str:
        """Generate C++ compute shader equivalent from pseudocode"""
        # Convert common GLSL patterns to C++ equivalents
        cpp_code = pseudocode
        
        # Replace GLSL-specific keywords with C++ equivalents
        replacements = {
            r'\bgl_GlobalInvocationID\.x\b': 'ctx.global_id_x',
            r'\bgl_GlobalInvocationID\.y\b': 'ctx.global_id_y', 
            r'\bgl_GlobalInvocationID\.z\b': 'ctx.global_id_z',
            r'\bgl_LocalInvocationID\.x\b': 'ctx.local_id_x',
            r'\bgl_LocalInvocationID\.y\b': 'ctx.local_id_y',
            r'\bgl_LocalInvocationID\.z\b': 'ctx.local_id_z',
            r'\bgl_WorkGroupID\.x\b': 'ctx.group_id_x',
            r'\bgl_WorkGroupID\.y\b': 'ctx.group_id_y',
            r'\bgl_WorkGroupID\.z\b': 'ctx.group_id_z',
            r'\bsqrt\b': 'std::sqrt',
            r'\bsin\b': 'std::sin',
            r'\bcos\b': 'std::cos',
            r'\btan\b': 'std::tan',
            r'\bpow\b': 'std::pow',
            r'\babs\b': 'std::abs',
            r'\bfloor\b': 'std::floor',
            r'\bceil\b': 'std::ceil',
            r'\bmax\b': 'std::max',
            r'\bmin\b': 'std::min',
            r'\btexture\b': 'texture_sample',
            r'\btexture2D\b': 'texture2D_sample',
        }
        
        for pattern, replacement in replacements.items():
            import re
            cpp_code = re.sub(pattern, replacement, cpp_code)
        
        # Wrap the converted code in a compute shader function
        wrapped_code = f'''
// Converted from pseudocode
void compute_from_pseudocode(ComputeContext& ctx) {{
{cpp_code}
}}
'''
        
        return self.cpp_templates["header"] + wrapped_code


def create_compute_shader_equivalents():
    """Create C/C++ compute shader equivalents"""
    print("Creating compute shader equivalents in C/C++...")
    
    generator = ComputeShaderGenerator()
    
    # Define some compute operations
    compute_operations = [
        {
            "name": "image_blur",
            "params": "ComputeBuffer& input_buffer, ComputeBuffer& output_buffer",
            "code": '''// Simple image blur operation
    int x = ctx.global_id_x;
    int y = ctx.global_id_y;
    
    // Sample neighboring pixels and average
    float avg_r = 0.0f, avg_g = 0.0f, avg_b = 0.0f;
    int count = 0;
    
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < input_buffer.width && 
                ny >= 0 && ny < input_buffer.height) {
                avg_r += input_buffer(nx, ny, 0, 0);
                avg_g += input_buffer(nx, ny, 0, 1);
                avg_b += input_buffer(nx, ny, 0, 2);
                count++;
            }
        }
    }
    
    if (count > 0) {
        output_buffer(x, y, 0, 0) = avg_r / count;
        output_buffer(x, y, 0, 1) = avg_g / count;
        output_buffer(x, y, 0, 2) = avg_b / count;
        output_buffer(x, y, 0, 3) = input_buffer(x, y, 0, 3); // Alpha
    }'''
        },
        {
            "name": "particle_physics",
            "params": "ComputeBuffer& position_buffer, ComputeBuffer& velocity_buffer, float deltaTime",
            "code": '''// Particle physics simulation
    size_t idx = ctx.global_id_x;
    if (idx < position_buffer.width) {
        // Update position based on velocity
        float px = position_buffer(idx, 0, 0, 0);
        float py = position_buffer(idx, 0, 0, 1);
        float pz = position_buffer(idx, 0, 0, 2);
        
        float vx = velocity_buffer(idx, 0, 0, 0);
        float vy = velocity_buffer(idx, 0, 0, 1);
        float vz = velocity_buffer(idx, 0, 0, 2);
        
        // Apply gravity
        vy += -9.81f * deltaTime;
        
        // Update position
        px += vx * deltaTime;
        py += vy * deltaTime;
        pz += vz * deltaTime;
        
        // Simple boundary check
        if (py < 0.0f) {
            py = 0.0f;
            vy *= -0.8f; // Bounce
        }
        
        // Write back
        position_buffer(idx, 0, 0, 0) = px;
        position_buffer(idx, 0, 0, 1) = py;
        position_buffer(idx, 0, 0, 2) = pz;
        velocity_buffer(idx, 0, 0, 0) = vx;
        velocity_buffer(idx, 0, 0, 1) = vy;
        velocity_buffer(idx, 0, 0, 2) = vz;
    }'''
        }
    ]
    
    # Generate the C++ code
    cpp_code = generator.generate_compute_shader_cpp(compute_operations)
    
    # Save to file
    with open("compute_shader_equivalents.cpp", "w") as f:
        f.write(cpp_code)
    
    print("Generated compute_shader_equivalents.cpp")
    
    # Also generate from pseudocode example
    pseudocode_example = """
    // Compute shader to update particle positions
    vec3 pos = particle_positions[gl_GlobalInvocationID.x];
    vec3 vel = particle_velocities[gl_GlobalInvocationID.x];
    
    // Apply physics
    vel.y -= 9.81 * deltaTime;  // gravity
    pos += vel * deltaTime;
    
    // Store new position
    particle_positions[gl_GlobalInvocationID.x] = pos;
    particle_velocities[gl_GlobalInvocationID.x] = vel;
    """
    
    cpp_from_pseudocode = generator.generate_from_pseudocode(pseudocode_example)
    
    # Save the pseudocode-generated version
    with open("compute_from_pseudocode.cpp", "w") as f:
        f.write(cpp_from_pseudocode)
    
    print("Generated compute_from_pseudocode.cpp")
    print("Compute shader equivalents in C/C++ created successfully!")


if __name__ == "__main__":
    create_compute_shader_equivalents()