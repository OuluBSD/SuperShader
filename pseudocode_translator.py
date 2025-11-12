#!/usr/bin/env python3
"""
Pseudocode Translator for SuperShader Project

This module provides the foundation for translating pseudocode to various
target languages and graphics APIs. It includes a basic parser and
translation system as outlined in the pseudocode specification.
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


class PseudocodeParser:
    """Parses pseudocode into an abstract syntax tree."""
    
    def __init__(self):
        self.tokens = []
        self.position = 0
        self.line_number = 1
        
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
                        tokens.append((token_type, value, self.line_number))
                    pos = match.end()
                    
                    # Update line number based on newlines
                    self.line_number += value.count('\n')
                    matched = True
                    break
            
            if not matched:
                raise SyntaxError(f"Unexpected character at position {pos}: {code[pos]}")
        
        return tokens
    
    def parse(self, code: str) -> Node:
        """Parse pseudocode into an AST."""
        self.tokens = self.tokenize(code)
        self.position = 0
        self.line_number = 1
        
        # Create root program node
        program_node = Node(NodeType.PROGRAM, [])
        
        # Parse top-level elements (functions, global variables)
        while self.position < len(self.tokens):
            token_type, value, line_num = self.current_token()
            
            if token_type == 'TYPE':
                # This could be a function or variable declaration
                node = self.parse_declaration()
                if node:
                    program_node.children.append(node)
            elif token_type == 'COMMENT':
                self.position += 1  # Skip comment
            elif token_type == 'WHITESPACE':
                self.position += 1  # Skip whitespace
            else:
                raise SyntaxError(f"Unexpected token: {value} at line {line_num}")
        
        return program_node
    
    def current_token(self):
        """Get the current token."""
        if self.position >= len(self.tokens):
            return None, None, None
        return self.tokens[self.position]
    
    def peek_token(self, offset=1):
        """Peek at a token without advancing position."""
        pos = self.position + offset
        if pos >= len(self.tokens):
            return None, None, None
        return self.tokens[pos]
    
    def consume_token(self, expected_type=None):
        """Consume and return the current token."""
        if self.position >= len(self.tokens):
            return None, None, None
        
        token = self.tokens[self.position]
        if expected_type and token[0] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token[0]}")
        
        self.position += 1
        return token
    
    def parse_declaration(self):
        """Parse a type declaration (function or variable)."""
        type_token, type_name, line_num = self.consume_token('TYPE')
        
        # Check if it's a function (next token is identifier followed by '(')
        if self.current_token()[0] == 'IDENTIFIER':
            next_token = self.peek_token(1)
            if next_token and next_token[0] == 'PUNCTUATION' and next_token[1] == '(':
                return self.parse_function(type_name, line_num)
        
        # Otherwise, it's a variable declaration
        return self.parse_variable_declaration(type_name, line_num)
    
    def parse_function(self, return_type: str, line_num: int):
        """Parse a function declaration."""
        func_name, name_val, _ = self.consume_token('IDENTIFIER')
        self.consume_token('PUNCTUATION')  # '('
        
        # Parse parameters
        params = []
        while self.current_token()[1] != ')':
            param_type, param_type_val, _ = self.consume_token('TYPE')
            param_name, param_name_val, _ = self.consume_token('IDENTIFIER')
            params.append((param_type_val, param_name_val))
            
            if self.current_token()[1] == ',':
                self.consume_token('PUNCTUATION')  # ','
        
        self.consume_token('PUNCTUATION')  # ')'
        self.consume_token('PUNCTUATION')  # '{'
        
        # Parse function body
        body = []
        while self.current_token()[1] != '}':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        
        self.consume_token('PUNCTUATION')  # '}'
        
        return Node(NodeType.FUNCTION, body, value={'name': name_val, 'return_type': return_type, 'params': params})
    
    def parse_variable_declaration(self, var_type: str, line_num: int):
        """Parse a variable declaration."""
        var_name, name_val, _ = self.consume_token('IDENTIFIER')
        
        # Check if there's an assignment
        if self.current_token()[1] == '=':
            self.consume_token('OPERATOR')  # '='
            value = self.parse_expression()
            return Node(NodeType.VARIABLE_DECLARATION, [value], value={'type': var_type, 'name': name_val})
        
        return Node(NodeType.VARIABLE_DECLARATION, [], value={'type': var_type, 'name': name_val})
    
    def parse_statement(self):
        """Parse a statement."""
        token_type, value, line_num = self.current_token()
        
        if value == 'if':
            return self.parse_if_statement()
        elif value == 'for':
            return self.parse_for_loop()
        elif value == 'while':
            return self.parse_while_loop()
        elif value == 'return':
            return self.parse_return_statement()
        elif token_type == 'IDENTIFIER':
            # Could be assignment or function call
            next_token = self.peek_token(1)
            if next_token and next_token[1] == '=':
                return self.parse_assignment()
            else:
                # For now, treat as an expression (could be function call without assignment)
                return self.parse_expression()
        else:
            # Try to parse as expression
            return self.parse_expression()
    
    def parse_if_statement(self):
        """Parse an if statement."""
        self.consume_token('KEYWORD')  # 'if'
        self.consume_token('PUNCTUATION')  # '('
        condition = self.parse_expression()
        self.consume_token('PUNCTUATION')  # ')'
        self.consume_token('PUNCTUATION')  # '{'
        
        # Parse if body
        if_body = []
        while self.current_token()[1] != '}':
            stmt = self.parse_statement()
            if stmt:
                if_body.append(stmt)
        self.consume_token('PUNCTUATION')  # '}'
        
        # Check for else
        has_else = False
        else_body = []
        if self.current_token()[0] == 'KEYWORD' and self.current_token()[1] == 'else':
            has_else = True
            self.consume_token('KEYWORD')  # 'else'
            self.consume_token('PUNCTUATION')  # '{'
            while self.current_token()[1] != '}':
                stmt = self.parse_statement()
                if stmt:
                    else_body.append(stmt)
            self.consume_token('PUNCTUATION')  # '}'
        
        return Node(NodeType.IF_STATEMENT, [condition] + if_body + else_body, value={'has_else': has_else, 'if_count': len(if_body)})
    
    def parse_assignment(self):
        """Parse an assignment statement."""
        var_name, name_val, _ = self.consume_token('IDENTIFIER')
        self.consume_token('OPERATOR')  # '='
        value = self.parse_expression()
        
        return Node(NodeType.ASSIGNMENT, [value], value={'variable': name_val})
    
    def parse_for_loop(self):
        """Parse a for loop."""
        self.consume_token('KEYWORD')  # 'for'
        self.consume_token('PUNCTUATION')  # '('
        
        # Parse initialization
        init = self.parse_statement()
        self.consume_token('PUNCTUATION')  # ';'
        
        # Parse condition
        condition = self.parse_expression()
        self.consume_token('PUNCTUATION')  # ';'
        
        # Parse increment
        increment = self.parse_statement()
        self.consume_token('PUNCTUATION')  # ')'
        self.consume_token('PUNCTUATION')  # '{'
        
        # Parse loop body
        body = []
        while self.current_token()[1] != '}':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume_token('PUNCTUATION')  # '}'
        
        return Node(NodeType.FOR_LOOP, [init, condition, increment] + body)
    
    def parse_while_loop(self):
        """Parse a while loop."""
        self.consume_token('KEYWORD')  # 'while'
        self.consume_token('PUNCTUATION')  # '('
        condition = self.parse_expression()
        self.consume_token('PUNCTUATION')  # ')'
        self.consume_token('PUNCTUATION')  # '{'
        
        # Parse loop body
        body = []
        while self.current_token()[1] != '}':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.consume_token('PUNCTUATION')  # '}'
        
        return Node(NodeType.WHILE_LOOP, [condition] + body)
    
    def parse_return_statement(self):
        """Parse a return statement."""
        self.consume_token('KEYWORD')  # 'return'
        value = self.parse_expression()
        
        return Node(NodeType.RETURN_STATEMENT, [value])
    
    def parse_expression(self):
        """Parse an expression - simplified version for basic cases."""
        # Handle unary operators
        if self.current_token()[0] == 'OPERATOR' and self.current_token()[1] in ['+', '-']:
            op_token = self.consume_token('OPERATOR')
            operand = self.parse_expression()
            return Node(NodeType.UNARY_OPERATION, [operand], value={'operator': op_token[1]})
        
        # Parse the left operand
        left = self.parse_primary()
        
        # Handle binary operations
        while self.current_token()[0] == 'OPERATOR':
            op_token = self.consume_token('OPERATOR')
            right = self.parse_primary()
            left = Node(NodeType.BINARY_OPERATION, [left, right], value={'operator': op_token[1]})
        
        return left
    
    def parse_primary(self):
        """Parse primary expressions (literals, variables, function calls, parenthesized expressions)."""
        token_type, value, line_num = self.current_token()
        
        if token_type == 'NUMBER':
            self.consume_token('NUMBER')
            return Node(NodeType.LITERAL, [], value=float(value))
        elif token_type == 'IDENTIFIER':
            # Check if it's a function call
            if self.peek_token(1)[1] == '(':
                return self.parse_function_call()
            else:
                self.consume_token('IDENTIFIER')
                return Node(NodeType.VARIABLE_REFERENCE, [], value=value)
        elif value == '(':
            self.consume_token('PUNCTUATION')  # '('
            expr = self.parse_expression()
            self.consume_token('PUNCTUATION')  # ')'
            return expr
        else:
            # For now, return a simple placeholder
            return Node(NodeType.LITERAL, [], value=0.0)
    
    def parse_function_call(self):
        """Parse a function call."""
        func_name, name_val, _ = self.consume_token('IDENTIFIER')
        self.consume_token('PUNCTUATION')  # '('
        
        args = []
        while self.current_token()[1] != ')':
            arg = self.parse_expression()
            args.append(arg)
            
            if self.current_token()[1] == ',':
                self.consume_token('PUNCTUATION')  # ','
        
        self.consume_token('PUNCTUATION')  # ')'
        
        return Node(NodeType.FUNCTION_CALL, args, value={'function': name_val})


class PseudocodeTranslator:
    """Translates pseudocode AST to target languages and APIs."""
    
    def __init__(self):
        # Mapping of pseudocode functions to target language equivalents
        self.function_mappings = {
            # Mathematical functions
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
            # Texture functions
            'texture': {
                'glsl': 'texture',
                'hlsl': 'tex.Sample(sampler, coord)',
                'vulkan': 'texture',  # Via SPIR-V
                'metal': 'texture.sample(sampler, coord)'
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
            
            for child in node.children:
                result += "    " + self._translate_node(child, target, indent_level + 1) + "\n"
            
            result += "}\n"
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
            return f"{var_name} = {value}"
        
        elif node.node_type == NodeType.BINARY_OPERATION:
            left = self._translate_node(node.children[0], target, 0)
            right = self._translate_node(node.children[1], target, 0)
            op = node.value.get('operator', '+')  # Default to + if not specified
            return f"({left} {op} {right})"
        
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
    """Example usage of the pseudocode translator."""
    print("SuperShader Pseudocode Translator")
    print("=" * 40)
    
    # Example pseudocode
    pseudocode = """
// Function to calculate lighting with Phong model
vec3 phongLighting(vec3 position, vec3 normal, vec3 viewDir, vec3 lightPos, vec3 lightColor) {
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse 
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - position);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    
    return ambient + diffuse + specular;
}
"""
    
    print("Original pseudocode:")
    print(pseudocode)
    print()
    
    # Parse the pseudocode
    parser = PseudocodeParser()
    try:
        ast = parser.parse(pseudocode)
        print("Parsed AST successfully!")
        print()
        
        # Translate to different targets
        translator = PseudocodeTranslator()
        
        for target in ['glsl', 'hlsl']:
            print(f"Translated to {target.upper()}:")
            translated = translator.translate(ast, target)
            print(translated)
            print()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()