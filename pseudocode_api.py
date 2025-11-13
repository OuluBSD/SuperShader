# Pseudocode Translation API

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
