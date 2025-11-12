# SuperShader

Tongue-in-cheek project that combines all shaders into one unified system using generic modules.

## Goal

The SuperShader project aims to:
- Break down all shaders into generic, reusable modules
- Identify and eliminate duplicate code
- Find similar modules and group them logically
- Create a management system to combine modules into useful shaders
- Store code in a universal format (pseudocode) that can be translated to multiple languages and rendering APIs

## Approach

- Modules are organized by genre, with conflicting features separated into branches
- Management code (Python) combines modules into functional shaders
- Use assimp for 3D model loading, replacing built-in 3D primitives when appropriate
- Utilize platform-appropriate math libraries (GLM for C/C++, DirectX math, etc.) instead of built-in mathematical code
- Recognize code that duplicates system library functionality for potential replacement
- Universal pseudocode serves as storage format, easily translatable to:
  - Programming languages: C/C++, Java, Python, JavaScript, C#
  - Graphics APIs: OpenGL, DirectX, Vulkan, Metal, Software rendering

## Structure

- `modules/` - Contains generic shader modules organized by genre
- `management/` - Python scripts to combine modules into shaders
- `extract_glsl.py` - Utility to extract GLSL code from JSON files
- `search.py` - Search functionality for shader analysis
- `TASKS.md` - Detailed project tasks
- `PLAN.md` - Project phases and implementation plan
- `QWEN.md` - Project context and instructions for AI assistants
- `json/` - Original shader data files in JSON format
- `common/` - Common resources and assets

## Features

- Extract GLSL code from JSON shader files
- Analyze and categorize shader modules
- Identify duplicate and similar code segments
- Create generic, reusable modules from specific implementations
- Generate shaders by combining modules
- Tag analysis for organizing shaders by genre
- Feature cataloging system for identifying common shader components
- Data processing pipeline for batch analysis of shaders
- Universal pseudocode system for cross-platform shaders

## License

The SuperShader code (excluding the `json/` directory) is licensed under the BSD 3-Clause License (see LICENSE file).
The JSON files in the `json/` directory are not covered by this license and remain the property of their respective copyright holders.