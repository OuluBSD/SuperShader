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
    
    # Define modules that actually exist in the project
    available_modules = {
        'procedural': {
            'perlin_noise': {
                'path': 'modules/procedural/perlin_noise/perlin_noise.py',
                'description': 'Perlin noise generation with FBM',
                'patterns': ['Perlin Noise', 'Fractal Brownian Motion'],
                'type': 'procedural'
            },
            'noise_functions_branching': {
                'path': 'modules/procedural/standardized/noise_functions_branching.py',
                'description': 'Procedural noise generation with branching for different algorithms',
                'patterns': ['Perlin Noise', 'Simplex Noise', 'Value Noise', 'Fractal Brownian Motion'],
                'type': 'procedural'
            }
        },
        'raymarching': {
            'raymarching_core': {
                'path': 'modules/raymarching/raymarching_core/raymarching_core.py',
                'description': 'Core raymarching algorithms with distance field operations',
                'patterns': ['Raymarching', 'Distance Field', 'Signed Distance Function'],
                'type': 'raymarching'
            },
            'raymarching_advanced_branching': {
                'path': 'modules/raymarching/standardized/advanced_raymarching_branching.py',
                'description': 'Advanced raymarching with branching for different algorithms',
                'patterns': ['Raymarching', 'Adaptive Steps', 'Cone Tracing', 'Multi-Raymarching'],
                'type': 'raymarching'
            }
        },
        'physics': {
            'verlet_integration': {
                'path': 'modules/physics/verlet_integration/verlet_integration.py',
                'description': 'Verlet integration for physics simulations',
                'patterns': ['Verlet Integration', 'Physics Simulation', 'Position Based Dynamics'],
                'type': 'physics'
            },
            'physics_advanced_branching': {
                'path': 'modules/physics/standardized/advanced_physics_branching.py',
                'description': 'Advanced physics with branching for different integration methods',
                'patterns': ['Euler Integration', 'Verlet Integration', 'RK4 Integration', 'Semi-Implicit Euler'],
                'type': 'physics'
            }
        },
        'texturing': {
            'uv_mapping': {
                'path': 'modules/texturing/uv_mapping/uv_mapping.py',
                'description': 'UV mapping and texturing functions',
                'patterns': ['UV Mapping', 'Texture Coordinates', 'Triplanar'],
                'type': 'texturing'
            },
            'advanced_texturing_branching': {
                'path': 'modules/texturing/standardized/advanced_texturing_branching.py',
                'description': 'Advanced texturing with branching for different mapping approaches',
                'patterns': ['Planar UV Mapping', 'Spherical UV Mapping', 'Cylindrical UV Mapping', 'Triplanar UV Mapping'],
                'type': 'texturing'
            }
        },
        'audio': {
            'beat_detection': {
                'path': 'modules/audio/beat_detection/beat_detection.py',
                'description': 'Audio reactive beat detection algorithms',
                'patterns': ['Beat Detection', 'Frequency Analysis', 'Audio Visualization'],
                'type': 'audio'
            },
            'audio_advanced_branching': {
                'path': 'modules/audio/standardized/advanced_audio_branching.py',
                'description': 'Advanced audio processing with branching for different analysis methods',
                'patterns': ['Fourier Transform Analysis', 'Wavelet Analysis', 'Autocorrelation Analysis'],
                'type': 'audio'
            }
        },
        'game': {
            'input_handling': {
                'path': 'modules/game/input_handling/input_handling.py',
                'description': 'Game input handling and interaction processing',
                'patterns': ['Input Handling', 'Mouse Detection', 'Interactive Elements'],
                'type': 'game'
            },
            'game_advanced_branching': {
                'path': 'modules/game/standardized/advanced_game_branching.py',
                'description': 'Advanced game modules with branching for different UI approaches',
                'patterns': ['Minimalist HUD', 'Detailed HUD', 'Themed HUD', 'Click-based Interaction', 'Hover-based Interaction'],
                'type': 'game'
            }
        },
        'ui': {
            'basic_shapes': {
                'path': 'modules/ui/basic_shapes/basic_shapes.py',
                'description': 'Basic 2D shape rendering for UI elements',
                'patterns': ['Basic Shapes', 'UI Rendering', '2D Graphics'],
                'type': 'ui'
            },
            'ui_advanced_branching': {
                'path': 'modules/ui/standardized/advanced_ui_branching.py',
                'description': 'Advanced UI with branching for different design approaches',
                'patterns': ['Flat UI', 'Material Design', 'Neumorphic UI', 'Glassmorphism UI'],
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
    
    for category, modules in available_modules.items():
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
    
This is the initial module library for SuperShader distribution, containing {copied_modules} essential shader modules across {len(available_modules)} categories.

## Library Contents

The library includes modules for:
"""
    
    for category, modules in available_modules.items():
        readme_content += f"- {category.title()}: {len(modules)} modules\n"
    
    readme_content += """

## Module Categories

### Procedural
- Perlin Noise: Classic Perlin noise with fractal Brownian motion
- Noise Functions Branching: Procedural noise with branching for different algorithms (Perlin, Simplex, Value)

### Raymarching
- Raymarching Core: Core algorithms for distance field rendering
- Raymarching Advanced Branching: Advanced raymarching with branching for different approaches

### Physics
- Verlet Integration: Verlet integration for physics simulations
- Physics Advanced Branching: Advanced physics with branching for different integration methods

### Texturing
- UV Mapping: UV coordinate mapping for texture application
- Advanced Texturing Branching: Advanced texturing with branching for different mapping approaches

### Audio
- Beat Detection: Audio reactive beat detection algorithms
- Audio Advanced Branching: Advanced audio processing with branching for different analysis methods

### Game
- Input Handling: Game input handling and interaction processing
- Game Advanced Branching: Advanced game modules with branching for different UI approaches

### UI
- Basic Shapes: Basic 2D shape rendering for UI elements
- UI Advanced Branching: Advanced UI with branching for different design approaches

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
    print(f"Categories covered: {len(available_modules)}")
    print(f"Manifest created: {manifest_path}")
    
    # Create a summary file
    summary_path = module_library_dir / 'SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write(f"SuperShader Module Library Summary\n")
        f.write(f"==================================\n")
        f.write(f"Date: {manifest['generated_at']}\n")
        f.write(f"Version: {manifest['version']}\n")
        f.write(f"Total modules: {copied_modules}\n\n")
        
        for category, modules in available_modules.items():
            f.write(f"{category.title()} ({len(modules)} modules):\n")
            for module_name in modules.keys():
                f.write(f"  - {module_name}\n")
            f.write("\n")
    
    print(f"Summary created: {summary_path}")
    
    return {
        'library_path': str(module_library_dir.absolute()),
        'modules_count': copied_modules,
        'categories_count': len(available_modules),
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