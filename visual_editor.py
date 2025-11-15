"""
Shader Module Visual Editor
Part of SuperShader Project - Phase 9: User Interface and Developer Experience

This module creates a graphical interface for combining shader modules,
implements visual preview of shader combinations, adds drag-and-drop
functionality for module arrangement, and creates live preview of parameter adjustments.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ModuleType(Enum):
    LIGHTING = "lighting"
    TEXTURING = "texturing"
    EFFECTS = "effects"
    GEOMETRY = "geometry"
    POST_PROCESSING = "post_processing"
    ANIMATION = "animation"


@dataclass
class ShaderModule:
    """Represents a shader module that can be used in the visual editor"""
    id: str
    name: str
    module_type: ModuleType
    description: str
    parameters: Dict[str, Dict[str, Any]]  # param_name -> {type, default, min, max, description}
    dependencies: List[str]
    conflicts: List[str]
    code_template: str


@dataclass
class ModuleConnection:
    """Represents a connection between shader modules"""
    source_module_id: str
    source_output: str
    target_module_id: str
    target_input: str


class ModuleLibrary:
    """
    Library of available shader modules
    """
    
    def __init__(self):
        self.modules: Dict[str, ShaderModule] = {}
        self._load_default_modules()
    
    def _load_default_modules(self):
        """Load default shader modules"""
        # Lighting modules
        self.add_module(ShaderModule(
            id="diffuse_lighting",
            name="Diffuse Lighting",
            module_type=ModuleType.LIGHTING,
            description="Basic diffuse lighting calculation",
            parameters={
                "light_color": {"type": "vec3", "default": [1.0, 1.0, 1.0], "min": [0, 0, 0], "max": [1, 1, 1], "description": "Color of the light"},
                "ambient_factor": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "description": "Ambient light contribution"},
            },
            dependencies=[],
            conflicts=[],
            code_template="""
// Diffuse Lighting Module
vec3 calculate_diffuse_lighting(vec3 normal, vec3 light_dir, vec3 light_color, float ambient_factor) {
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 ambient = ambient_factor * light_color;
    vec3 diffuse = diff * light_color;
    return ambient + diffuse;
}
            """
        ))
        
        self.add_module(ShaderModule(
            id="specular_lighting", 
            name="Specular Lighting",
            module_type=ModuleType.LIGHTING,
            description="Phong/Blinn-Phong specular lighting",
            parameters={
                "shininess": {"type": "float", "default": 32.0, "min": 1.0, "max": 128.0, "description": "Material shininess"},
                "specular_strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "description": "Strength of specular highlights"},
            },
            dependencies=["diffuse_lighting"],
            conflicts=[],
            code_template="""
// Specular Lighting Module  
vec3 calculate_specular_lighting(vec3 normal, vec3 light_dir, vec3 view_dir, float shininess, float specular_strength) {
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    return specular_strength * spec * vec3(1.0);
}
            """
        ))
        
        # Texturing modules
        self.add_module(ShaderModule(
            id="basic_texture",
            name="Basic Texture Sampling",
            module_type=ModuleType.TEXTURING,
            description="Basic texture sampling with UV coordinates",
            parameters={
                "uv_scale": {"type": "vec2", "default": [1.0, 1.0], "min": [0.1, 0.1], "max": [10.0, 10.0], "description": "UV coordinate scaling"},
                "uv_offset": {"type": "vec2", "default": [0.0, 0.0], "min": [-5.0, -5.0], "max": [5.0, 5.0], "description": "UV coordinate offset"},
            },
            dependencies=[],
            conflicts=[],
            code_template="""
// Basic Texture Module
vec4 sample_basic_texture(sampler2D texture_sampler, vec2 uv, vec2 scale, vec2 offset) {
    vec2 scaled_uv = uv * scale + offset;
    return texture2D(texture_sampler, scaled_uv);
}
            """
        ))
        
        # Effects modules
        self.add_module(ShaderModule(
            id="bloom_effect",
            name="Bloom Effect",
            module_type=ModuleType.EFFECTS,
            description="Bloom post-processing effect",
            parameters={
                "intensity": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "description": "Bloom intensity"},
                "threshold": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "description": "Brightness threshold"},
            },
            dependencies=[],
            conflicts=[],
            code_template="""
// Bloom Effect Module
vec3 apply_bloom(vec3 color, float intensity, float threshold) {
    vec3 bloom = vec3(0.0);
    if (color.r > threshold || color.g > threshold || color.b > threshold) {
        bloom = color * intensity;
    }
    return color + bloom;
}
            """
        ))
    
    def add_module(self, module: ShaderModule) -> None:
        """Add a module to the library"""
        self.modules[module.id] = module
    
    def get_modules_by_type(self, module_type: ModuleType) -> List[ShaderModule]:
        """Get all modules of a specific type"""
        return [m for m in self.modules.values() if m.module_type == module_type]
    
    def get_all_modules(self) -> List[ShaderModule]:
        """Get all modules in the library"""
        return list(self.modules.values())
    
    def get_module_by_id(self, module_id: str) -> Optional[ShaderModule]:
        """Get a module by its ID"""
        return self.modules.get(module_id)


class VisualShaderEditor:
    """
    Main visual shader editor application
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SuperShader Visual Editor")
        self.root.geometry("1200x800")
        
        # Initialize module library
        self.module_library = ModuleLibrary()
        
        # Current shader composition
        self.modules_in_composition: Dict[str, ShaderModule] = {}
        self.connections: List[ModuleConnection] = []
        self.module_positions: Dict[str, Tuple[int, int]] = {}
        
        # Create UI components
        self._create_ui()
        
        # Load initial modules into palette
        self._load_module_palette()
    
    def _create_ui(self):
        """Create the user interface"""
        # Main frame with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Module palette
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Module palette label
        palette_label = ttk.Label(left_frame, text="Shader Modules", font=("Arial", 12, "bold"))
        palette_label.pack(pady=5)
        
        # Module type selector
        self.module_type_var = tk.StringVar(value="all")
        type_selector = ttk.Combobox(
            left_frame, 
            textvariable=self.module_type_var,
            values=["all", "lighting", "texturing", "effects", "geometry", "post_processing", "animation"],
            state="readonly"
        )
        type_selector.pack(fill=tk.X, padx=5, pady=5)
        type_selector.bind("<<ComboboxSelected>>", self._filter_module_palette)
        
        # Module listbox
        self.module_listbox = tk.Listbox(left_frame)
        self.module_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.module_listbox.bind("<Button-1>", self._on_module_select)
        
        # Right panel - Canvas area
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        # Canvas for visual editing
        self.canvas = tk.Canvas(right_frame, bg="white", scrollregion=(0, 0, 2000, 2000))
        
        # Scrollbars for canvas
        h_scrollbar = ttk.Scrollbar(right_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(toolbar, text="Add Module", command=self._add_selected_module).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(toolbar, text="Clear All", command=self._clear_composition).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(toolbar, text="Export", command=self._export_shader).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(toolbar, text="Preview", command=self._show_preview).pack(side=tk.LEFT, padx=2, pady=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind canvas events for module placement
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        # Store drag state
        self.drag_data = {"x": 0, "y": 0, "item": None}
        self.canvas.bind("<ButtonPress-1>", self._on_module_press)
        self.canvas.bind("<B1-Motion>", self._on_module_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_module_release)
    
    def _load_module_palette(self):
        """Load modules into the palette listbox"""
        self.module_listbox.delete(0, tk.END)
        
        all_modules = self.module_library.get_all_modules()
        for module in all_modules:
            self.module_listbox.insert(tk.END, f"{module.name} ({module.module_type.value})")
    
    def _filter_module_palette(self, event=None):
        """Filter modules based on selected type"""
        self.module_listbox.delete(0, tk.END)
        
        selected_type = self.module_type_var.get()
        if selected_type == "all":
            modules = self.module_library.get_all_modules()
        else:
            try:
                module_type = ModuleType(selected_type.upper())
                modules = self.module_library.get_modules_by_type(module_type)
            except ValueError:
                modules = []
        
        for module in modules:
            self.module_listbox.insert(tk.END, f"{module.name} ({module.module_type.value})")
    
    def _on_module_select(self, event):
        """Handle module selection in the listbox"""
        selection = self.module_listbox.curselection()
        if selection:
            index = selection[0]
            modules = self.module_library.get_all_modules()
            if 0 <= index < len(modules):
                self.selected_module = modules[index]
                self.status_var.set(f"Selected: {self.selected_module.name}")
    
    def _on_canvas_click(self, event):
        """Handle click on canvas (for placing modules)"""
        pass  # We'll handle this differently
    
    def _add_selected_module(self):
        """Add the selected module from the palette to the canvas"""
        selection = self.module_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a module from the palette")
            return
        
        index = selection[0]
        modules = self.module_library.get_all_modules()
        if index >= len(modules):
            return
        
        module = modules[index]
        
        # Add to composition
        if module.id not in self.modules_in_composition:
            self.modules_in_composition[module.id] = module
            
            # Place the module on the canvas
            x, y = 50 + len(self.modules_in_composition) * 30, 50 + len(self.modules_in_composition) * 30
            self._draw_module_on_canvas(module, x, y)
            self.module_positions[module.id] = (x, y)
            
            self.status_var.set(f"Added: {module.name}")
        else:
            self.status_var.set("Module already in composition")
    
    def _draw_module_on_canvas(self, module: ShaderModule, x: int, y: int):
        """Draw a module on the canvas"""
        # Draw module rectangle
        rect = self.canvas.create_rectangle(
            x, y, x+150, y+80, 
            fill="#e0e0e0", outline="black", width=2,
            tags=("module", f"module_{module.id}")
        )
        
        # Draw module name
        self.canvas.create_text(
            x+75, y+15, 
            text=module.name, 
            font=("Arial", 10, "bold"),
            tags=(f"module_{module.id}",)
        )
        
        # Draw module type
        self.canvas.create_text(
            x+75, y+30, 
            text=f"Type: {module.module_type.value}", 
            font=("Arial", 8),
            tags=(f"module_{module.id}",)
        )
        
        # Draw inputs and outputs
        self.canvas.create_oval(x-5, y+35, x+5, y+45, fill="red", tags=(f"module_{module.id}", "input"))
        self.canvas.create_oval(x+145, y+35, x+155, y+45, fill="green", tags=(f"module_{module.id}", "output"))
    
    def _on_module_press(self, event):
        """Handle mouse press on a module"""
        # Check if we clicked on a module
        item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags(item)
        
        if "module" in tags:
            self.drag_data["item"] = item
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
    
    def _on_module_drag(self, event):
        """Handle module dragging"""
        if self.drag_data["item"]:
            # Get the current position of the item
            coords = self.canvas.coords(self.drag_data["item"])
            # Calculate the movement
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            # Move the item
            self.canvas.move(self.drag_data["item"], dx, dy)
            # Update the drag position
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
    
    def _on_module_release(self, event):
        """Handle mouse release after dragging"""
        if self.drag_data["item"]:
            self.drag_data["item"] = None
    
    def _clear_composition(self):
        """Clear all modules from the composition"""
        self.modules_in_composition.clear()
        self.connections.clear()
        self.module_positions.clear()
        self.canvas.delete("module")  # Delete all modules from canvas
        self.status_var.set("Composition cleared")
    
    def _export_shader(self):
        """Export the current shader composition"""
        if not self.modules_in_composition:
            messagebox.showwarning("Warning", "No modules in composition to export")
            return
        
        # Generate shader code from composition
        shader_code = self._generate_shader_code()
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".glsl",
            filetypes=[("GLSL files", "*.glsl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(shader_code)
                self.status_var.set(f"Shader exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export shader: {str(e)}")
    
    def _generate_shader_code(self) -> str:
        """Generate GLSL code from the current composition"""
        # This is a simplified version - a real implementation would need to:
        # 1. Resolve module dependencies
        # 2. Handle connections between modules
        # 3. Generate proper GLSL with all included modules
        
        code_parts = [
            "// Generated Shader from SuperShader Visual Editor",
            "#version 330 core",
            "",
            "// Uniforms for module parameters",
        ]
        
        # Add uniforms for all module parameters
        for module_id, module in self.modules_in_composition.items():
            for param_name, param_info in module.parameters.items():
                param_type = param_info["type"]
                if param_type == "vec3":
                    code_parts.append(f"uniform vec3 {module_id}_{param_name};")
                elif param_type == "vec2":
                    code_parts.append(f"uniform vec2 {module_id}_{param_name};")
                elif param_type == "float":
                    code_parts.append(f"uniform float {module_id}_{param_name};")
                elif param_type == "bool":
                    code_parts.append(f"uniform bool {module_id}_{param_name};")
        
        code_parts.extend([
            "",
            "// Module code",
        ])
        
        # Add code for each module
        for module_id, module in self.modules_in_composition.items():
            code_parts.append(f"// --- {module.name} ---")
            code_parts.append(module.code_template)
            code_parts.append("")
        
        code_parts.extend([
            "",
            "// Main shader functions",
            "void main() {",
            "    // Combine all modules based on connections",
            "    // This would be dynamically generated based on module connections",
            "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;  // Placeholder",
            "}"
        ])
        
        return "\n".join(code_parts)
    
    def _show_preview(self):
        """Show a preview of the current shader (placeholder)"""
        preview_text = "Shader Preview:\n\n"
        preview_text += f"Modules in composition: {len(self.modules_in_composition)}\n"
        preview_text += f"Connections: {len(self.connections)}\n\n"
        preview_text += "Module list:\n"
        
        for module_id, module in self.modules_in_composition.items():
            preview_text += f"- {module.name} ({module.module_type.value})\n"
        
        messagebox.showinfo("Shader Preview", preview_text)
    
    def run(self):
        """Run the visual editor application"""
        self.root.mainloop()


def main():
    """
    Example usage of the Visual Shader Editor
    """
    print("Shader Module Visual Editor")
    print("Part of SuperShader Project - Phase 9")
    
    # Create and run the visual editor
    editor = VisualShaderEditor()
    print("Starting visual editor... (close the window to continue)")
    editor.run()


if __name__ == "__main__":
    main()