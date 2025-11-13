#!/usr/bin/env python3
"""
Pseudocode translation system for SuperShader project.

This script creates translators from pseudocode to various target languages
including GLSL, C/C++, Java, Python, and different graphics APIs.
"""

import os
import re
from pathlib import Path


class PseudocodeTranslator:
    def __init__(self):
        self.translators = {
            'glsl': self.translate_to_glsl,
            'cpp': self.translate_to_cpp,
            'java': self.translate_to_java,
            'python': self.translate_to_python
        }
        
        # Define pseudocode to target language mappings
        self.mappings = {
            'glsl': {
                'function_def': (r'function\s+(\w+)\s*\(([^)]*)\)\s*{', r'{} {}({}) {{'),
                'variable_decl': (r'var\s+(\w+)\s*:\s*(\w+)', r'{} {};'),
                'return_stmt': (r'return\s+(.+?);', r'return {};'),
                'if_stmt': (r'if\s*\((.+?)\)\s*{', r'if ({}) {{'),
                'for_loop': (r'for\s+(\w+)\s+in\s+range\s*\((.+?)\)\s*{', r'for (int {} = 0; {} < {}; {}++) {{'),
                'while_loop': (r'while\s*\((.+?)\)\s*{', r'while ({}) {{'),
                'vec3_type': r'vec3',
                'vec2_type': r'vec2',
                'float_type': r'float',
                'int_type': r'int'
            },
            'cpp': {
                'function_def': (r'function\s+(\w+)\s*\(([^)]*)\)\s*{', r'auto {}({}) -> {} {{'),
                'variable_decl': (r'var\s+(\w+)\s*:\s*(\w+)', r'{} {};'),
                'return_stmt': (r'return\s+(.+?);', r'return {};'),
                'if_stmt': (r'if\s*\((.+?)\)\s*{', r'if ({}) {{'),
                'for_loop': (r'for\s+(\w+)\s+in\s+range\s*\((.+?)\)\s*{', r'for (int {} = 0; {} < {}; {}++) {{'),
                'while_loop': (r'while\s*\((.+?)\)\s*{', r'while ({}) {{'),
                'vec3_type': r'glm::vec3',  # Using GLM library
                'vec2_type': r'glm::vec2',
                'float_type': r'float',
                'int_type': r'int'
            },
            'java': {
                'function_def': (r'function\s+(\w+)\s*\(([^)]*)\)\s*{', r'{} {}({}) {{'),
                'variable_decl': (r'var\s+(\w+)\s*:\s*(\w+)', r'{} {};'),
                'return_stmt': (r'return\s+(.+?);', r'return {};'),
                'if_stmt': (r'if\s*\((.+?)\)\s*{', r'if ({}) {{'),
                'for_loop': (r'for\s+(\w+)\s+in\s+range\s*\((.+?)\)\s*{', r'for (int {} = 0; {} < {}; {}++) {{'),
                'while_loop': (r'while\s*\((.+?)\)\s*{', r'while ({}) {{'),
                'vec3_type': r'Vector3f',
                'vec2_type': r'Vector2f',
                'float_type': r'float',
                'int_type': r'int'
            },
            'python': {
                'function_def': (r'function\s+(\w+)\s*\(([^)]*)\)\s*{', r'def {}({}):'),
                'variable_decl': (r'var\s+(\w+)\s*:\s*(\w+)', r'{} = None  # : {}'),
                'return_stmt': (r'return\s+(.+?);', r'return {}'),
                'if_stmt': (r'if\s*\((.+?)\)\s*{', r'if {}:'),
                'for_loop': (r'for\s+(\w+)\s+in\s+range\s*\((.+?)\)\s*{', r'for {} in range({}):'),
                'while_loop': (r'while\s*\((.+?)\)\s*{', r'while {}:'),
                'vec3_type': r'numpy.array',  # Using numpy
                'vec2_type': r'numpy.array',
                'float_type': r'float',
                'int_type': r'int'
            }
        }
    
    def translate_pseudocode(self, pseudocode, target_language):
        """Translate pseudocode to the target language."""
        if target_language not in self.translators:
            raise ValueError(f"Unsupported target language: {target_language}")
        
        return self.translators[target_language](pseudocode)
    
    def translate_to_glsl(self, pseudocode):
        """Translate pseudocode to GLSL."""
        code = pseudocode
        
        # Replace type definitions
        code = re.sub(r'(\W)vec3(\W)', r'\g<1>vec3\g<2>', code)
        code = re.sub(r'(\W)vec2(\W)', r'\g<1>vec2\g<2>', code)
        code = re.sub(r'(\W)float(\W)', r'\g<1>float\g<2>', code)
        code = re.sub(r'(\W)int(\W)', r'\g<1>int\g<2>', code)
        
        # Replace function definitions
        def replace_function_def(match):
            params = match.group(2)  # Second capture group contains parameters
            func_name = match.group(1)  # First capture group contains function name
            mapped_params = self.map_parameters(params, 'glsl')
            # For GLSL, we need to determine return type - simple heuristic
            return_type = 'void'  # Default
            if 'return ' in pseudocode:  # Simple check - in real implementation, this would be more sophisticated
                # Look at the function to determine return type
                func_part = pseudocode[pseudocode.find(match.group(0)):pseudocode.find('}', pseudocode.find(match.group(0)))]
                if 'vec3' in func_part:
                    return_type = 'vec3'
                elif 'vec2' in func_part:
                    return_type = 'vec2'
                elif 'float' in func_part:
                    return_type = 'float'
                else:
                    return_type = 'void'
            return f"{return_type} {func_name}({mapped_params}) {{"
        
        code = re.sub(self.mappings['glsl']['function_def'][0], replace_function_def, code)
        
        # Replace variable declarations
        code = re.sub(self.mappings['glsl']['variable_decl'][0], 
                     lambda m: self.mappings['glsl']['variable_decl'][1].format(
                         self.map_type(m.group(2), 'glsl'), m.group(1)), 
                     code)
        
        # Replace return statements
        code = re.sub(self.mappings['glsl']['return_stmt'][0], 
                     lambda m: self.mappings['glsl']['return_stmt'][1].format(m.group(1)), 
                     code)
        
        # Replace control structures
        code = re.sub(self.mappings['glsl']['if_stmt'][0], 
                     lambda m: self.mappings['glsl']['if_stmt'][1].format(m.group(1)), 
                     code)
        code = re.sub(self.mappings['glsl']['for_loop'][0], 
                     lambda m: self.mappings['glsl']['for_loop'][1].format(
                         m.group(1), m.group(1), m.group(2), m.group(1)), 
                     code)
        code = re.sub(self.mappings['glsl']['while_loop'][0], 
                     lambda m: self.mappings['glsl']['while_loop'][1].format(m.group(1)), 
                     code)
        
        return code
    
    def translate_to_cpp(self, pseudocode):
        """Translate pseudocode to C++."""
        code = pseudocode
        
        # Add necessary includes
        header = "#include <glm/glm.hpp>\n#include <glm/gtc/matrix_transform.hpp>\n\n"
        
        # Replace type definitions
        code = re.sub(r'(\W)vec3(\W)', r'\g<1>glm::vec3\g<2>', code)
        code = re.sub(r'(\W)vec2(\W)', r'\g<1>glm::vec2\g<2>', code)
        
        # Replace function definitions
        def replace_function_def_cpp(match):
            params = match.group(2)
            func_name = match.group(1)
            mapped_params = self.map_parameters(params, 'cpp')
            # For C++, determine return type similarly to GLSL
            return_type = 'auto'  # Default
            if 'return ' in pseudocode:
                func_part = pseudocode[pseudocode.find(match.group(0)):pseudocode.find('}', pseudocode.find(match.group(0)))]
                if 'vec3' in func_part:
                    return_type = 'glm::vec3'
                elif 'vec2' in func_part:
                    return_type = 'glm::vec2'
                elif 'float' in func_part:
                    return_type = 'float'
                else:
                    return_type = 'void'
            return f"{return_type} {func_name}({mapped_params}) {{"
        
        code = re.sub(self.mappings['cpp']['function_def'][0], replace_function_def_cpp, code)
        
        # Replace variable declarations
        code = re.sub(self.mappings['cpp']['variable_decl'][0], 
                     lambda m: self.mappings['cpp']['variable_decl'][1].format(
                         self.map_type(m.group(2), 'cpp'), m.group(1)), 
                     code)
        
        # Replace return statements
        code = re.sub(self.mappings['cpp']['return_stmt'][0], 
                     lambda m: self.mappings['cpp']['return_stmt'][1].format(m.group(1)), 
                     code)
        
        # Replace control structures
        code = re.sub(self.mappings['cpp']['if_stmt'][0], 
                     lambda m: self.mappings['cpp']['if_stmt'][1].format(m.group(1)), 
                     code)
        code = re.sub(self.mappings['cpp']['for_loop'][0], 
                     lambda m: self.mappings['cpp']['for_loop'][1].format(
                         m.group(1), m.group(1), m.group(2), m.group(1)), 
                     code)
        code = re.sub(self.mappings['cpp']['while_loop'][0], 
                     lambda m: self.mappings['cpp']['while_loop'][1].format(m.group(1)), 
                     code)
        
        return header + code
    
    def translate_to_java(self, pseudocode):
        """Translate pseudocode to Java."""
        code = pseudocode
        
        # Add class wrapper
        class_wrapper = "public class ShaderFunctions {\n\n"
        
        # Replace type definitions
        code = re.sub(r'(\W)vec3(\W)', r'\g<1>Vector3f\g<2>', code)
        code = re.sub(r'(\W)vec2(\W)', r'\g<1>Vector2f\g<2>', code)
        
        # Replace function definitions
        def replace_function_def_java(match):
            params = match.group(2)
            func_name = match.group(1)
            mapped_params = self.map_parameters(params, 'java')
            # For Java, determine return type
            return_type = 'void'  # Default
            if 'return ' in pseudocode:
                func_part = pseudocode[pseudocode.find(match.group(0)):pseudocode.find('}', pseudocode.find(match.group(0)))]
                if 'vec3' in func_part:
                    return_type = 'Vector3f'
                elif 'vec2' in func_part:
                    return_type = 'Vector2f'
                elif 'float' in func_part:
                    return_type = 'float'
                else:
                    return_type = 'void'
            return f"{return_type} {func_name}({mapped_params}) {{"
        
        code = re.sub(self.mappings['java']['function_def'][0], replace_function_def_java, code)
        
        # Replace variable declarations
        code = re.sub(self.mappings['java']['variable_decl'][0], 
                     lambda m: self.mappings['java']['variable_decl'][1].format(
                         self.map_type(m.group(2), 'java'), m.group(1)), 
                     code)
        
        # Replace return statements
        code = re.sub(self.mappings['java']['return_stmt'][0], 
                     lambda m: self.mappings['java']['return_stmt'][1].format(m.group(1)), 
                     code)
        
        # Replace control structures
        code = re.sub(self.mappings['java']['if_stmt'][0], 
                     lambda m: self.mappings['java']['if_stmt'][1].format(m.group(1)), 
                     code)
        code = re.sub(self.mappings['java']['for_loop'][0], 
                     lambda m: self.mappings['java']['for_loop'][1].format(
                         m.group(1), m.group(1), m.group(2), m.group(1)), 
                     code)
        code = re.sub(self.mappings['java']['while_loop'][0], 
                     lambda m: self.mappings['java']['while_loop'][1].format(m.group(1)), 
                     code)
        
        return class_wrapper + code + "\n}"
    
    def translate_to_python(self, pseudocode):
        """Translate pseudocode to Python."""
        code = pseudocode
        
        # Add necessary imports
        header = "import numpy as np\n\n"
        
        # Replace function definitions
        code = re.sub(self.mappings['python']['function_def'][0], 
                     lambda m: self.mappings['python']['function_def'][1].format(
                         m.group(1), self.map_parameters(m.group(2), 'python')), 
                     code)
        
        # Replace variable declarations
        code = re.sub(self.mappings['python']['variable_decl'][0], 
                     lambda m: self.mappings['python']['variable_decl'][1].format(
                         m.group(1), m.group(2)), 
                     code)
        
        # Replace return statements
        code = re.sub(self.mappings['python']['return_stmt'][0], 
                     lambda m: self.mappings['python']['return_stmt'][1].format(m.group(1)), 
                     code)
        
        # Replace control structures
        code = re.sub(self.mappings['python']['if_stmt'][0], 
                     lambda m: self.mappings['python']['if_stmt'][1].format(m.group(1)), 
                     code)
        code = re.sub(self.mappings['python']['for_loop'][0], 
                     lambda m: self.mappings['python']['for_loop'][1].format(
                         m.group(1), m.group(2)), 
                     code)
        code = re.sub(self.mappings['python']['while_loop'][0], 
                     lambda m: self.mappings['python']['while_loop'][1].format(m.group(1)), 
                     code)
        
        return header + code
    
    def map_parameters(self, params, target_language):
        """Map pseudocode parameters to target language parameters."""
        if not params.strip():
            return ""
        
        # Parse parameters: "param1: type1, param2: type2" -> [("param1", "type1"), ("param2", "type2")]
        param_pairs = []
        for p in params.split(","):
            p = p.strip()
            if ":" in p:
                name, ptype = p.split(":", 1)
                name = name.strip()
                ptype = ptype.strip()
                mapped_type = self.map_type(ptype, target_language)
                param_pairs.append(f"{mapped_type} {name}")
        
        return ", ".join(param_pairs)
    
    def map_type(self, ptype, target_language):
        """Map pseudocode type to target language type."""
        if target_language == 'glsl':
            if ptype == 'vec3':
                return 'vec3'
            elif ptype == 'vec2':
                return 'vec2'
            elif ptype == 'float':
                return 'float'
            elif ptype == 'int':
                return 'int'
        elif target_language == 'cpp':
            if ptype == 'vec3':
                return 'glm::vec3'
            elif ptype == 'vec2':
                return 'glm::vec2'
            elif ptype == 'float':
                return 'float'
            elif ptype == 'int':
                return 'int'
        elif target_language == 'java':
            if ptype == 'vec3':
                return 'Vector3f'
            elif ptype == 'vec2':
                return 'Vector2f'
            elif ptype == 'float':
                return 'float'
            elif ptype == 'int':
                return 'int'
        elif target_language == 'python':
            if ptype == 'vec3':
                return 'np.ndarray'
            elif ptype == 'vec2':
                return 'np.ndarray'
            elif ptype == 'float':
                return 'float'
            elif ptype == 'int':
                return 'int'
        
        return ptype  # Return original if no mapping found
    
    def map_return_type(self, params, target_language):
        """Map return type from parameters."""
        # Simple heuristic: return the type of the first parameter
        if params.strip():
            parts = params.split(":")
            if len(parts) > 1:
                return self.map_type(parts[1].strip().split()[0], target_language)
        
        return self.map_type('float', target_language)  # Default to float


def create_pseudocode_examples():
    """Create example pseudocode files for testing the translator."""
    examples = {
        'lighting.pseudo': '''function calculate_diffuse_lighting(normal: vec3, light_dir: vec3, light_color: vec3) {
    var intensity: float = max(dot(normal, light_dir), 0.0);
    var diffuse: vec3 = intensity * light_color;
    return diffuse;
}''',
        
        'sdf_sphere.pseudo': '''function sdf_sphere(position: vec3, center: vec3, radius: float) {
    var distance: float = length(position - center) - radius;
    return distance;
}''',
        
        'color_adjustment.pseudo': '''function adjust_brightness_contrast(color: vec3, brightness: float, contrast: float) {
    var adjusted: vec3 = color + brightness;
    adjusted = (adjusted - 0.5) * contrast + 0.5;
    return adjusted;
}'''
    }
    
    os.makedirs('pseudocode_examples', exist_ok=True)
    
    for filename, code in examples.items():
        with open(f'pseudocode_examples/{filename}', 'w') as f:
            f.write(code)
    
    print(f"Created {len(examples)} pseudocode examples")


def translate_examples():
    """Translate example pseudocode to different languages."""
    translator = PseudocodeTranslator()
    languages = ['glsl', 'cpp', 'java', 'python']
    
    os.makedirs('translated_examples', exist_ok=True)
    
    example_files = list(Path('pseudocode_examples').glob('*.pseudo'))
    
    for example_file in example_files:
        pseudocode = example_file.read_text()
        
        for lang in languages:
            try:
                translated = translator.translate_pseudocode(pseudocode, lang)
                output_file = f"translated_examples/{example_file.stem}_to_{lang}.{get_file_extension(lang)}"
                with open(output_file, 'w') as f:
                    f.write(translated)
                print(f"Translated {example_file.name} to {lang} -> {output_file}")
            except Exception as e:
                print(f"Error translating {example_file.name} to {lang}: {str(e)}")


def get_file_extension(language):
    """Get appropriate file extension for the language."""
    extensions = {
        'glsl': 'glsl',
        'cpp': 'cpp',
        'java': 'java',
        'python': 'py'
    }
    return extensions.get(language, 'txt')


def create_translation_api():
    """Create a simple API for pseudocode translation."""
    api_code = '''# Pseudocode Translation API

from pseudocode_translator import PseudocodeTranslator

def translate_pseudocode_file(input_file, target_language, output_file):
    """
    Translate a pseudocode file to a target language.
    
    Args:
        input_file (str): Path to the input pseudocode file
        target_language (str): Target language ('glsl', 'cpp', 'java', 'python')
        output_file (str): Path to the output file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(input_file, 'r') as f:
            pseudocode = f.read()
        
        translator = PseudocodeTranslator()
        translated_code = translator.translate_pseudocode(pseudocode, target_language)
        
        with open(output_file, 'w') as f:
            f.write(translated_code)
        
        return True
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return False


def translate_pseudocode_string(pseudocode, target_language):
    """
    Translate pseudocode string to a target language.
    
    Args:
        pseudocode (str): Pseudocode string to translate
        target_language (str): Target language ('glsl', 'cpp', 'java', 'python')
    
    Returns:
        str: Translated code, or empty string if error
    """
    try:
        translator = PseudocodeTranslator()
        return translator.translate_pseudocode(pseudocode, target_language)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return ""


# Example usage
if __name__ == "__main__":
    # Example of translating a small piece of pseudocode
    sample_pseudocode = """function simple_function(input: float) {
    var result: float = input * 2.0;
    return result;
}"""

    glsl_code = translate_pseudocode_string(sample_pseudocode, 'glsl')
    print("Translated to GLSL:")
    print(glsl_code)
'''
    
    with open('pseudocode_api.py', 'w') as f:
        f.write(api_code)
    
    print("Created pseudocode translation API")


def main():
    print("Creating pseudocode translation system...")
    
    # Create example pseudocode files
    create_pseudocode_examples()
    
    # Translate examples to different languages
    translate_examples()
    
    # Create translation API
    create_translation_api()
    
    print("\nPseudocode translation system created successfully!")
    
    # Show example translation
    translator = PseudocodeTranslator()
    sample_code = '''function calculate_normal(position: vec3) {
    var epsilon: float = 0.001;
    var dx: float = sdf(position + vec3(epsilon, 0, 0)) - sdf(position - vec3(epsilon, 0, 0));
    var dy: float = sdf(position + vec3(0, epsilon, 0)) - sdf(position - vec3(0, epsilon, 0));
    var dz: float = sdf(position + vec3(0, 0, epsilon)) - sdf(position - vec3(0, 0, epsilon));
    return normalize(vec3(dx, dy, dz));
}'''
    
    print("\nExample translation:")
    print("Original pseudocode:")
    print(sample_code)
    print("\nTranslated to GLSL:")
    try:
        print(translator.translate_pseudocode(sample_code, 'glsl'))
    except Exception as e:
        print(f"Error in translation: {str(e)}")


if __name__ == "__main__":
    main()