#!/usr/bin/env python3
"""
Create Initial Module Library for SuperShader Distribution
Generates a standard library of shader modules for distribution
"""

import os
import sys
import json
from pathlib import Path
import shutil
from typing import Dict, List, Any


def create_module_library():
    """Create the initial module library for distribution"""
    
    # Define the module library structure
    module_library_dir = Path("distributions/module_library")
    module_library_dir.mkdir(exist_ok=True, parents=True)
    
    # Define core modules to include in the initial library
    core_modules = {
        'procedural': {
            'perlin_noise': {
                'path': 'modules/procedural/standardized/perlin_noise.py',
                'description': 'Classic Perlin noise with FBM implementation',
                'patterns': ['Perlin Noise', 'Fractal Brownian Motion'],
                'type': 'procedural'
            },
            'simplex_noise': {
                'path': 'modules/procedural/standardized/simplex_noise.py',
                'description': 'Improved Simplex noise with better performance',
                'patterns': ['Simplex Noise', 'Gradient Noise'],
                'type': 'procedural'
            }
        },
        'raymarching': {
            'raymarching_core': {
                'path': 'modules/raymarching/standardized/raymarching_core.py',
                'description': 'Core raymarching algorithms with distance estimation',
                'patterns': ['Raymarching', 'Distance Field', 'Signed Distance Function'],
                'type': 'raymarching'
            },
            'sdf_primitives': {
                'path': 'modules/raymarching/standardized/sdf_primitives.py',
                'description': 'Standard SDF primitives for raymarching scenes',
                'patterns': ['Sphere', 'Box', 'Plane', 'Cylinder', 'Torus'],
                'type': 'raymarching'
            }
        },
        'physics': {
            'verlet_integration': {
                'path': 'modules/physics/standardized/verlet_integration.py',
                'description': 'Verlet integration for physics simulations',
                'patterns': ['Verlet Integration', 'Physics Simulation', 'Position Based Dynamics'],
                'type': 'physics'
            },
            'spring_system': {
                'path': 'modules/physics/standardized/spring_system.py',
                'description': 'Spring-based physics system for cloth and soft bodies',
                'patterns': ['Spring Physics', 'Mass-Spring System', 'Elasticity'],
                'type': 'physics'
            }
        },
        'texturing': {
            'uv_mapping': {
                'path': 'modules/texturing/standardized/uv_mapping.py',
                'description': 'UV coordinate mapping for texture application',
                'patterns': ['UV Mapping', 'Texture Coordinates', 'Planar Mapping'],
                'type': 'texturing'
            },
            'triplanar_mapping': {
                'path': 'modules/texturing/standardized/triplanar_mapping.py',
                'description': 'Triplanar texture mapping for complex geometries',
                'patterns': ['Triplanar Mapping', 'Axis-aligned Mapping', 'Seamless Texturing'],
                'type': 'texturing'
            }
        },
        'audio': {
            'beat_detection': {
                'path': 'modules/audio/standardized/beat_detection.py',
                'description': 'Audio reactive beat detection algorithms',
                'patterns': ['Beat Detection', 'Frequency Analysis', 'Audio Visualization'],
                'type': 'audio'
            },
            'spectrum_analysis': {
                'path': 'modules/audio/standardized/spectrum_analysis.py',
                'description': 'Spectrum analysis for audio visualization',
                'patterns': ['Fourier Transform', 'Spectrum Display', 'Frequency Bands'],
                'type': 'audio'
            }
        },
        'game': {
            'input_handling': {
                'path': 'modules/game/standardized/input_handling.py',
                'description': 'Game input handling and interaction processing',
                'patterns': ['Input Handling', 'Mouse Detection', 'Interactive Elements'],
                'type': 'game'
            },
            'particle_system': {
                'path': 'modules/game/standardized/particle_system.py',
                'description': 'GPU-based particle system simulation',
                'patterns': ['Particle System', 'GPU Particles', 'Simulation'],
                'type': 'game'
            }
        },
        'ui': {
            'basic_shapes': {
                'path': 'modules/ui/standardized/basic_shapes.py',
                'description': 'Basic 2D shape rendering for UI elements',
                'patterns': ['Basic Shapes', 'UI Rendering', '2D Graphics'],
                'type': 'ui'
            },
            'button_component': {
                'path': 'modules/ui/standardized/button_component.py',
                'description': 'Interactive button component with hover/click effects',
                'patterns': ['UI Buttons', 'Visual Feedback', 'Interaction'],
                'type': 'ui'
            }
        }
    }
    
    # Create module library directory structure
    lib_modules_dir = module_library_dir / 'modules'
    lib_modules_dir.mkdir(exist_ok=True)
    
    # Copy modules to the library and create a manifest
    manifest = {
        'library_name': 'SuperShader Standard Module Library',
        'version': '1.0.0',
        'description': 'Standard library of shader modules for SuperShader distribution',
        'generated_at': __import__('datetime').datetime.now().isoformat(),
        'modules': {}
    }
    
    copied_modules = 0
    
    for category, modules in core_modules.items():
        category_dir = lib_modules_dir / category
        category_dir.mkdir(exist_ok=True)
        
        for module_name, module_info in modules.items():
            # Check if module file exists
            module_path = Path(module_info['path'])
            if module_path.exists():
                dest_path = category_dir / f"{module_name}.py"
                
                # Copy the module file
                shutil.copy2(module_path, dest_path)
                
                # Add to manifest
                manifest['modules'][f"{category}/{module_name}"] = {
                    'name': module_name,
                    'category': category,
                    'description': module_info['description'],
                    'patterns': module_info['patterns'],
                    'type': module_info['type'],
                    'source_path': str(module_path)
                }
                
                copied_modules += 1
                print(f"Copied {category}/{module_name}")
            else:
                print(f"Warning: Module file {module_path} not found")
    
    # Create the manifest file
    manifest_path = module_library_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create a README for the module library
    readme_content = f"""# SuperShader Standard Module Library
    
This is the initial module library for SuperShader distribution, containing {copied_modules} essential shader modules across {len(core_modules)} categories.

## Library Contents

The library includes modules for:
"""
    
    for category, modules in core_modules.items():
        readme_content += f"- {category.title()}: {len(modules)} modules\n"
    
    readme_content += """

## Module Categories

### Procedural
- Perlin Noise: Classic Perlin noise with fractal Brownian motion
- Simplex Noise: Improved Simplex noise with better performance

### Raymarching
- Raymarching Core: Core algorithms for distance field rendering
- SDF Primitives: Standard signed distance functions for common shapes

### Physics
- Verlet Integration: Stable physics simulation using Verlet integration
- Spring System: Mass-spring system for cloth and elastic materials

### Texturing
- UV Mapping: Standard UV coordinate mapping techniques
- Triplanar Mapping: Seamless texturing for complex geometries

### Audio
- Beat Detection: Audio reactive algorithms for visualization
- Spectrum Analysis: Frequency domain analysis for audio

### Game
- Input Handling: Interactive element processing
- Particle System: GPU-accelerated particle simulation

### UI
- Basic Shapes: 2D shape rendering for UI components
- Button Component: Interactive button with visual feedback

## Installation

To use these modules with SuperShader:

```
# Copy the modules directory to your SuperShader installation
cp -r modules/* /path/to/supershader/modules/
```

## Usage

See the documentation in each module for specific usage instructions.

For more information about SuperShader, visit the main project repository.
"""
    
    readme_path = module_library_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\nModule library created successfully!")
    print(f"Location: {module_library_dir.absolute()}")
    print(f"Modules included: {copied_modules}")
    print(f"Categories covered: {len(core_modules)}")
    print(f"Manifest created: {manifest_path}")
    
    # Create a summary file
    summary_path = module_library_dir / 'SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write(f"SuperShader Module Library Summary\n")
        f.write(f"==================================\n")
        f.write(f"Date: {manifest['generated_at']}\n")
        f.write(f"Version: {manifest['version']}\n")
        f.write(f"Total modules: {copied_modules}\n\n")
        
        for category, modules in core_modules.items():
            f.write(f"{category.title()} ({len(modules)} modules):\n")
            for module_name in modules.keys():
                f.write(f"  - {module_name}\n")
            f.write("\n")
    
    print(f"Summary created: {summary_path}")
    
    return {
        'library_path': str(module_library_dir.absolute()),
        'modules_count': copied_modules,
        'categories_count': len(core_modules),
        'manifest_path': str(manifest_path),
        'readme_path': str(readme_path)
    }


def main():
    """Main entry point to create the initial module library"""
    print("Creating Initial SuperShader Module Library for Distribution...")
    
    library_info = create_module_library()
    
    print(f"\n✅ Initial module library created successfully!")
    print(f"   Location: {library_info['library_path']}")
    print(f"   Modules: {library_info['modules_count']}")
    print(f"   Categories: {library_info['categories_count']}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)