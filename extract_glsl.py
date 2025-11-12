#!/usr/bin/env python3
"""
GLSL Extraction Utility for SuperShader Project

This script extracts GLSL code from JSON shader files for analysis.
It can output GLSL code to stdout for easy processing and analysis.
"""

import json
import os
import sys
import argparse
from pathlib import Path


def extract_glsl_from_file(json_file_path, shader_type=None):
    """
    Extract GLSL code from a single JSON shader file
    
    Args:
        json_file_path (str): Path to the JSON shader file
        shader_type (str): Type of shader to extract ('vertex', 'fragment', 'geometry', etc.)
    
    Returns:
        str: GLSL code extracted from the file
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        shader_data = json.load(f)
    
    glsl_code = []
    
    # Extract GLSL code based on shader passes
    if 'renderpass' in shader_data:
        for i, pass_data in enumerate(shader_data['renderpass']):
            if 'code' in pass_data:
                code = pass_data['code']
                pass_type = pass_data.get('type', 'fragment')
                
                if shader_type is None or pass_type.lower() == shader_type.lower():
                    glsl_code.append(f"// Pass {i} ({pass_type})")
                    glsl_code.append(code)
                    glsl_code.append("")  # Empty line separator
    else:
        # Fallback: try to find shader code in other possible fields
        possible_fields = ['fragment_shader', 'vertex_shader', 'shader', 'code', 'main']
        for field in possible_fields:
            if field in shader_data:
                code = shader_data[field]
                if isinstance(code, str):
                    glsl_code.append(f"// From field: {field}")
                    glsl_code.append(code)
                    glsl_code.append("")
    
    return "\n".join(glsl_code)


def extract_glsl_from_directory(json_dir, shader_type=None):
    """
    Extract GLSL code from all JSON files in a directory
    
    Args:
        json_dir (str): Directory containing JSON shader files
        shader_type (str): Type of shader to extract
    
    Yields:
        tuple: (file_path, glsl_code)
    """
    json_dir = Path(json_dir)
    for json_file in json_dir.glob("*.json"):
        try:
            glsl_code = extract_glsl_from_file(json_file, shader_type)
            if glsl_code.strip():  # Only yield if there's actual code
                yield str(json_file), glsl_code
        except Exception as e:
            print(f"Error processing {json_file}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Extract GLSL code from JSON shader files")
    parser.add_argument("--json-file", type=str, help="Single JSON file to extract GLSL from")
    parser.add_argument("--json-dir", type=str, help="Directory containing JSON files to process")
    parser.add_argument("--type", type=str, help="Specific shader type to extract (vertex, fragment, etc.)")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    if not args.json_file and not args.json_dir:
        print("Error: Must specify either --json-file or --json-dir", file=sys.stderr)
        sys.exit(1)
    
    output_file = None
    if args.output:
        output_file = open(args.output, 'w')
    
    def write_output(text):
        if output_file:
            output_file.write(text)
        else:
            print(text)
    
    try:
        if args.json_file:
            # Process single file
            if not os.path.exists(args.json_file):
                print(f"Error: File {args.json_file} does not exist", file=sys.stderr)
                sys.exit(1)
            
            glsl_code = extract_glsl_from_file(args.json_file, args.type)
            write_output(glsl_code)
        
        elif args.json_dir:
            # Process directory
            if not os.path.exists(args.json_dir):
                print(f"Error: Directory {args.json_dir} does not exist", file=sys.stderr)
                sys.exit(1)
            
            for file_path, glsl_code in extract_glsl_from_directory(args.json_dir, args.type):
                write_output(f"// File: {file_path}\n")
                write_output(glsl_code)
                write_output("\n// --- End of file ---\n\n")
    
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()