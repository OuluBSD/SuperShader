#!/usr/bin/env python3
"""
Simple test for the pseudocode translator to debug issues.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the AST."""
    PROGRAM = "program"
    FUNCTION = "function"
    VARIABLE_DECLARATION = "variable_declaration"
    ASSIGNMENT = "assignment"
    BINARY_OPERATION = "binary_operation"
    UNARY_OPERATION = "unary_operation"
    FUNCTION_CALL = "function_call"
    IF_STATEMENT = "if_statement"
    FOR_LOOP = "for_loop"
    WHILE_LOOP = "while_loop"
    RETURN_STATEMENT = "return_statement"
    VARIABLE_REFERENCE = "variable_reference"
    LITERAL = "literal"


@dataclass
class Node:
    """Base class for AST nodes."""
    node_type: NodeType
    children: List['Node']
    value: Any = None
    line_number: Optional[int] = None


class SimplePseudocodeParser:
    """Simplified parser to test basic functionality."""
    
    def __init__(self):
        self.tokens = []
        self.position = 0
        
        # Define token patterns
        self.token_patterns = [
            (r'//.*?$', 'COMMENT'),  # Single-line comment
            (r'/\*.*?\*/', 'COMMENT'),  # Multi-line comment (simplified)
            (r'int|float|bool|vec2|vec3|vec4|mat2|mat3|mat4|sampler2D|samplerCube', 'TYPE'),
            (r'if|else|for|while|return', 'KEYWORD'),
            (r'\+\+|--|\+=|-=|\*=|/=|==|!=|<=|>=|&&|\|\||[+\-*/%=<>!&|]', 'OPERATOR'),
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),
            (r'\d+\.\d*|\.\d+|\d+', 'NUMBER'),
            (r'"[^"]*"', 'STRING'),
            (r'[{}();,\[\]]', 'PUNCTUATION'),
            (r'\s+', 'WHITESPACE'),  # Will be ignored
        ]
    
    def tokenize(self, code: str) -> List[tuple]:
        """Convert code string to tokens."""
        tokens = []
        pos = 0
        
        while pos < len(code):
            matched = False
            for pattern, token_type in self.token_patterns:
                regex = re.compile(pattern, re.DOTALL)
                match = regex.match(code, pos)
                
                if match:
                    value = match.group(0)
                    if token_type != 'WHITESPACE':  # Skip whitespace tokens
                        tokens.append((token_type, value))
                    pos = match.end()
                    matched = True
                    break
            
            if not matched:
                print(f"Unexpected character at position {pos}: {repr(code[pos])}")
                pos += 1  # Skip the problematic character
        
        return tokens
    
    def parse(self, code: str) -> Node:
        """Parse pseudocode into an AST."""
        self.tokens = self.tokenize(code)
        self.position = 0
        
        # Create root program node
        program_node = Node(NodeType.PROGRAM, [])
        
        # For this simple test, we'll just create a basic function node
        # that represents the phong lighting function
        func_body = [
            # float ambientStrength = 0.1;
            Node(NodeType.VARIABLE_DECLARATION, [
                Node(NodeType.LITERAL, [], value=0.1)
            ], value={'type': 'float', 'name': 'ambientStrength'}),
            
            # vec3 ambient = ambientStrength * lightColor;
            Node(NodeType.ASSIGNMENT, [
                Node(NodeType.BINARY_OPERATION, [
                    Node(NodeType.VARIABLE_REFERENCE, [], value='ambientStrength'),
                    Node(NodeType.VARIABLE_REFERENCE, [], value='lightColor')
                ], value={'operator': '*'})
            ], value={'variable': 'ambient'}),
            
            # return ambient + diffuse + specular;
            Node(NodeType.RETURN_STATEMENT, [
                Node(NodeType.BINARY_OPERATION, [
                    Node(NodeType.BINARY_OPERATION, [
                        Node(NodeType.VARIABLE_REFERENCE, [], value='ambient'),
                        Node(NodeType.VARIABLE_REFERENCE, [], value='diffuse')
                    ], value={'operator': '+'}),
                    Node(NodeType.VARIABLE_REFERENCE, [], value='specular')
                ], value={'operator': '+'})
            ])
        ]
        
        func_node = Node(NodeType.FUNCTION, func_body, value={
            'name': 'phongLighting',
            'return_type': 'vec3',
            'params': [
                ('vec3', 'position'),
                ('vec3', 'normal'),
                ('vec3', 'viewDir'),
                ('vec3', 'lightPos'),
                ('vec3', 'lightColor')
            ]
        })
        
        program_node.children.append(func_node)
        
        return program_node


class SimplePseudocodeTranslator:
    """Simplified translator to demonstrate the concept."""
    
    def __init__(self):
        # Mapping of pseudocode functions to target language equivalents
        self.function_mappings = {
            'length': {
                'glsl': 'length',
                'hlsl': 'length',
                'vulkan': 'length',
                'metal': 'length'
            },
            'normalize': {
                'glsl': 'normalize',
                'hlsl': 'normalize',
                'vulkan': 'normalize',
                'metal': 'normalize'
            },
            'distance': {
                'glsl': 'distance',
                'hlsl': 'distance',
                'vulkan': 'distance',
                'metal': 'distance'
            },
            'dot': {
                'glsl': 'dot',
                'hlsl': 'dot',
                'vulkan': 'dot',
                'metal': 'dot'
            },
            'cross': {
                'glsl': 'cross',
                'hlsl': 'cross',
                'vulkan': 'cross',
                'metal': 'cross'
            },
            'max': {
                'glsl': 'max',
                'hlsl': 'max',
                'vulkan': 'max',
                'metal': 'max'
            },
            'pow': {
                'glsl': 'pow',
                'hlsl': 'pow',
                'vulkan': 'pow',
                'metal': 'pow'
            }
        }
        
        # Type mappings
        self.type_mappings = {
            'vec2': {
                'glsl': 'vec2',
                'hlsl': 'float2',
                'vulkan': 'vec2',
                'metal': 'float2'
            },
            'vec3': {
                'glsl': 'vec3',
                'hlsl': 'float3',
                'vulkan': 'vec3',
                'metal': 'float3'
            },
            'vec4': {
                'glsl': 'vec4',
                'hlsl': 'float4',
                'vulkan': 'vec4',
                'metal': 'float4'
            },
            'mat2': {
                'glsl': 'mat2',
                'hlsl': 'float2x2',
                'vulkan': 'mat2',
                'metal': 'float2x2'
            },
            'mat3': {
                'glsl': 'mat3',
                'hlsl': 'float3x3',
                'vulkan': 'mat3',
                'metal': 'float3x3'
            },
            'mat4': {
                'glsl': 'mat4',
                'hlsl': 'float4x4',
                'vulkan': 'mat4',
                'metal': 'float4x4'
            },
            'int': {
                'glsl': 'int',
                'hlsl': 'int',
                'vulkan': 'int',
                'metal': 'int'
            },
            'float': {
                'glsl': 'float',
                'hlsl': 'float',
                'vulkan': 'float',
                'metal': 'float'
            },
            'bool': {
                'glsl': 'bool',
                'hlsl': 'bool',
                'vulkan': 'bool',
                'metal': 'bool'
            }
        }
    
    def translate(self, ast: Node, target: str) -> str:
        """Translate AST to target language."""
        if target not in ['glsl', 'hlsl', 'vulkan', 'metal']:
            raise ValueError(f"Unsupported target: {target}")
        
        return self._translate_node(ast, target, 0)
    
    def _translate_node(self, node: Node, target: str, indent_level: int = 0) -> str:
        """Recursively translate AST nodes."""
        indent = "    " * indent_level
        
        if node.node_type == NodeType.PROGRAM:
            result = ""
            for i, child in enumerate(node.children):
                result += self._translate_node(child, target, indent_level)
                if i < len(node.children) - 1:
                    result += "\n"
            return result
        
        elif node.node_type == NodeType.FUNCTION:
            func_info = node.value
            params = ", ".join([f"{self._map_type(p[0], target)} {p[1]}" for p in func_info['params']])
            result = f"{self._map_type(func_info['return_type'], target)} {func_info['name']}({params}) {{\n"
            
            for i, child in enumerate(node.children):
                result += "    " + self._translate_node(child, target, indent_level + 1)
                if i < len(node.children) - 1:
                    result += "\n"
            
            result += "\n}"
            return result
        
        elif node.node_type == NodeType.VARIABLE_DECLARATION:
            var_info = node.value
            type_name = self._map_type(var_info['type'], target)
            result = f"{type_name} {var_info['name']}"
            
            if node.children:  # Has initialization
                result += f" = {self._translate_node(node.children[0], target, 0)}"
            
            result += ";"
            return result
        
        elif node.node_type == NodeType.ASSIGNMENT:
            var_name = node.value['variable']
            value = self._translate_node(node.children[0], target, 0)
            return f"{var_name} = {value};"
        
        elif node.node_type == NodeType.BINARY_OPERATION:
            left = self._translate_node(node.children[0], target, 0)
            right = self._translate_node(node.children[1], target, 0)
            op = node.value.get('operator', '+')  # Default to + if not specified
            return f"({left} {op} {right})"
        
        elif node.node_type == NodeType.UNARY_OPERATION:
            operand = self._translate_node(node.children[0], target, 0)
            op = node.value.get('operator', '-')  # Default to - if not specified
            return f"({op}{operand})"
        
        elif node.node_type == NodeType.FUNCTION_CALL:
            func_name = node.value['function']
            mapped_name = self._map_function(func_name, target)
            
            args = []
            for child in node.children:
                args.append(self._translate_node(child, target, 0))
            
            return f"{mapped_name}({', '.join(args)})"
        
        elif node.node_type == NodeType.LITERAL:
            if isinstance(node.value, float):
                # Ensure decimal for floats
                return f"{node.value}" if '.' in str(node.value) else f"{node.value}.0"
            else:
                return str(node.value)
        
        elif node.node_type == NodeType.VARIABLE_REFERENCE:
            return node.value
        
        elif node.node_type == NodeType.RETURN_STATEMENT:
            value = self._translate_node(node.children[0], target, 0)
            return f"return {value};"
        
        else:
            # For other node types, return a placeholder
            return f"/* {node.node_type.value} */"
    
    def _map_function(self, func_name: str, target: str) -> str:
        """Map a pseudocode function name to the target language equivalent."""
        if func_name in self.function_mappings:
            if target in self.function_mappings[func_name]:
                return self.function_mappings[func_name][target]
        
        # If not found, return the original name
        return func_name
    
    def _map_type(self, type_name: str, target: str) -> str:
        """Map a pseudocode type to the target language equivalent."""
        if type_name in self.type_mappings:
            if target in self.type_mappings[type_name]:
                return self.type_mappings[type_name][target]
        
        # If not found, return the original name
        return type_name


def main():
    """Test the pseudocode translator."""
    print("SuperShader Pseudocode Translator - Basic Test")
    print("=" * 50)
    
    # Test with simple code
    pseudocode = """
    // Simple function for testing
    vec3 testFunction(vec3 input) {
        float scale = 2.0;
        vec3 result = input * scale;
        return result;
    }
    """
    
    print("Original pseudocode:")
    print(pseudocode)
    print()
    
    # Parse the pseudocode
    parser = SimplePseudocodeParser()
    ast = parser.parse(pseudocode)
    print("Parsed AST successfully!")
    print(f"AST root type: {ast.node_type}")
    print(f"Number of children: {len(ast.children)}")
    if ast.children:
        print(f"First child type: {ast.children[0].node_type}")
        print(f"First child function name: {ast.children[0].value['name']}")
    print()
    
    # Translate to different targets
    translator = SimplePseudocodeTranslator()
    
    for target in ['glsl', 'hlsl']:
        print(f"Translated to {target.upper()}:")
        try:
            translated = translator.translate(ast, target)
            print(translated)
            print()
        except Exception as e:
            print(f"Error translating to {target}: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()