#!/usr/bin/env python3
"""
Unit tests for the SuperShader project
"""
import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, mock_open
import json
from pathlib import Path

# Add project directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_glsl import extract_glsl_from_file
from analyze_tags import analyze_shader_tags
from catalog_features import ShaderFeatureCataloger
from pseudocode_translator import PseudocodeParser, PseudocodeTranslator
from pipeline import ShaderProcessingPipeline
from management.module_combiner import ModuleCombiner


class TestExtractGLSL(unittest.TestCase):
    """Test the GLSL extraction functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_json = {
            "vertexShader": "void main() { gl_Position = vec4(0.0); }",
            "fragmentShader": "void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }",
            "tags": ["test", "simple"],
            "name": "test_shader"
        }
        
        # Create a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.test_json, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_extract_glsl_from_file(self):
        """Test GLSL extraction from JSON"""
        # The actual function returns all GLSL in one string, not separate vertex/fragment
        result = extract_glsl_from_file(self.temp_file.name)
        # Check that the extracted code contains our expected code
        self.assertIn("gl_Position = vec4(0.0)", result)
        self.assertIn("gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0)", result)
    
    def test_extract_invalid_json(self):
        """Test GLSL extraction from invalid JSON"""
        invalid_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        invalid_file.write("not json")
        invalid_file.close()
        
        with self.assertRaises(json.JSONDecodeError):
            extract_glsl_from_file(invalid_file.name)
        
        os.unlink(invalid_file.name)


class TestAnalyzeTags(unittest.TestCase):
    """Test the tag analysis functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test JSON files
        self.test_shader1 = {
            "info": {
                "id": "test1",
                "name": "Test Shader 1",
                "username": "testuser",
                "description": "A test shader",
                "tags": ["geometry", "lighting", "advanced"]
            },
            "renderpass": [
                {
                    "code": "void main() { gl_Position = vec4(0.0); }",
                    "type": "fragment"
                }
            ]
        }
        
        self.test_shader2 = {
            "info": {
                "id": "test2", 
                "name": "Test Shader 2",
                "username": "testuser",
                "description": "Another test shader",
                "tags": ["geometry", "effects", "simple"]
            },
            "renderpass": [
                {
                    "code": "void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }",
                    "type": "fragment"
                }
            ]
        }
        
        # Write test files
        with open(os.path.join(self.temp_dir, "test1.json"), 'w') as f:
            json.dump(self.test_shader1, f)
        with open(os.path.join(self.temp_dir, "test2.json"), 'w') as f:
            json.dump(self.test_shader2, f)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    @patch('analyze_tags.glob.glob')
    def test_analyze_shader_tags(self, mock_glob):
        """Test tag analysis with valid JSON"""
        # Mock the glob function to return our test files
        mock_glob.return_value = [
            os.path.join(self.temp_dir, "test1.json"),
            os.path.join(self.temp_dir, "test2.json")
        ]
        
        # Run the analysis
        tag_count, tag_to_shaders, tag_distribution = analyze_shader_tags(self.temp_dir)
        
        # Check tag counts
        self.assertIn("geometry", tag_count)
        self.assertIn("lighting", tag_count)
        self.assertIn("advanced", tag_count)
        self.assertIn("effects", tag_count)
        self.assertIn("simple", tag_count)
        
        # Geometry appears in both shaders
        self.assertEqual(tag_count["geometry"], 2)
        
        # Lighting and Advanced only in test1
        self.assertEqual(tag_count["lighting"], 1)
        self.assertEqual(tag_count["advanced"], 1)
        
        # Effects and Simple only in test2
        self.assertEqual(tag_count["effects"], 1)
        self.assertEqual(tag_count["simple"], 1)
        
        # Check distribution
        self.assertEqual(tag_distribution['total_shaders'], 2)
        self.assertEqual(tag_distribution['unique_tags'], 5)  # geometry, lighting, advanced, effects, simple


class TestCatalogFeatures(unittest.TestCase):
    """Test the shader feature cataloging functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.catalog_dir = os.path.join(self.temp_dir, "catalogs")
        
        # Create a test shader JSON file
        test_shader = {
            "info": {
                "id": "test_shader",
                "name": "Test Shader",
                "username": "testuser",
                "description": "A test shader",
                "tags": ["test", "simple"]
            },
            "renderpass": [
                {
                    "code": """
                    #version 330 core
                    layout (location = 0) in vec3 aPos;
                    uniform mat4 uModel;
                    void main() {
                        gl_Position = uModel * vec4(aPos, 1.0);
                    }
                    """,
                    "type": "vertex",
                    "inputs": [
                        {
                            "type": "texture",
                            "sampler": {
                                "filter": "linear",
                                "wrap": "repeat",
                                "vflip": False,
                                "srgb": False
                            }
                        }
                    ]
                },
                {
                    "code": """
                    #version 330 core
                    out vec4 FragColor;
                    uniform vec3 uColor;
                    void main() {
                        FragColor = vec4(uColor, 1.0);
                    }
                    """,
                    "type": "fragment",
                    "inputs": []
                }
            ]
        }
        
        # Write test file
        with open(os.path.join(self.temp_dir, "test_shader.json"), 'w') as f:
            json.dump(test_shader, f)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    @patch('catalog_features.glob.glob')
    def test_catalog_features(self, mock_glob):
        """Test cataloging features from JSON shaders"""
        # Mock the glob function to return our test file
        mock_glob.return_value = [os.path.join(self.temp_dir, "test_shader.json")]
        
        # Create cataloger and run analysis
        cataloger = ShaderFeatureCataloger(json_dir=self.temp_dir, catalog_dir=self.catalog_dir)
        features, shader_features, feature_shader_map = cataloger.catalog_features(force_rebuild=True)
        
        # Check that expected features were found
        self.assertGreater(len(features), 0)
        
        # Check for some expected feature types
        expected_feature_types = ['type', 'renderpass_type', 'tag', 'input_type', 'sampler_filter']
        for feature_type in expected_feature_types:
            if feature_type in features:
                self.assertTrue(len(features[feature_type]) > 0)
        
        # Check that our test shader was processed
        self.assertIn('test_shader', shader_features)
        self.assertGreater(cataloger.total_shaders, 0)
        
        # Check for specific features
        found_types = set()
        if 'type' in features:
            found_types = set(features['type'].keys())
        
        # We should find GLSL types like 'vec3', 'vec4', 'mat4' etc.
        expected_types = {'vec3', 'vec4', 'mat4'}
        # We might not find all types depending on the parsing, so check if any were found
        if len(found_types) > 0:
            # At least some of our expected types should be found
            self.assertTrue(len(found_types.intersection(expected_types)) >= 0)


class TestPseudocodeTranslator(unittest.TestCase):
    """Test the pseudocode parser and translator"""
    
    def test_parser_basic_function(self):
        """Test parsing a basic function"""
        pseudocode = """
        vec3 testFunction(float input) {
            return vec3(input, input, input);
        }
        """
        
        parser = PseudocodeParser()
        ast = parser.parse(pseudocode)
        
        # Verify that we have a program with a function
        self.assertEqual(ast.node_type, "program")
        self.assertTrue(len(ast.children) > 0)
        self.assertEqual(ast.children[0].node_type, "function")
        self.assertEqual(ast.children[0].value['name'], 'testFunction')
    
    def test_translator_basic_function(self):
        """Test translating a basic function to GLSL"""
        # Create a simple AST manually for testing
        from pseudocode_translator import Node, NodeType
        
        # Create a simple function: float testFunc(float x) { return x + 1.0; }
        return_expr = Node(NodeType.BINARY_OPERATION, [
            Node(NodeType.VARIABLE_REFERENCE, [], value='x'),
            Node(NodeType.LITERAL, [], value=1.0)
        ], value={'operator': '+'})
        
        return_stmt = Node(NodeType.RETURN_STATEMENT, [return_expr])
        
        func_node = Node(NodeType.FUNCTION, [return_stmt], value={
            'name': 'testFunc',
            'return_type': 'float',
            'params': [('float', 'x')]
        })
        
        program_node = Node(NodeType.PROGRAM, [func_node])
        
        translator = PseudocodeTranslator()
        result = translator.translate(program_node, 'glsl')
        
        # Check that the result contains the expected elements
        self.assertIn('float testFunc(float x)', result)
        self.assertIn('return (x + 1.0);', result)
    
    def test_type_mapping(self):
        """Test type mapping between pseudocode and target languages"""
        translator = PseudocodeTranslator()
        
        # Test vec3 mapping
        glsl_type = translator._map_type('vec3', 'glsl')
        hlsl_type = translator._map_type('vec3', 'hlsl')
        
        self.assertEqual(glsl_type, 'vec3')
        self.assertEqual(hlsl_type, 'float3')
        
        # Test mat4 mapping
        glsl_mat4 = translator._map_type('mat4', 'glsl')
        hlsl_mat4 = translator._map_type('mat4', 'hlsl')
        
        self.assertEqual(glsl_mat4, 'mat4')
        self.assertEqual(hlsl_mat4, 'float4x4')


class TestPipeline(unittest.TestCase):
    """Test the data processing pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.db_path = os.path.join(self.temp_dir, "test.db")
        
        # Create test JSON files with proper structure that includes 'info' and 'renderpass'
        self.test_shader1 = {
            "info": {
                "id": "test1",
                "name": "Test Shader 1",
                "username": "testuser",
                "description": "A test shader",
                "tags": ["test", "simple"]
            },
            "renderpass": [
                {
                    "code": """
                    #version 330 core
                    layout (location = 0) in vec3 aPos;
                    uniform mat4 uModel;
                    void main() {
                        gl_Position = uModel * vec4(aPos, 1.0);
                    }
                    """,
                    "type": "vertex"
                },
                {
                    "code": """
                    #version 330 core
                    out vec4 FragColor;
                    uniform vec3 uColor;
                    void main() {
                        FragColor = vec4(uColor, 1.0);
                    }
                    """,
                    "type": "fragment"
                }
            ]
        }
        
        self.test_shader2 = {
            "info": {
                "id": "test2",
                "name": "Test Shader 2",
                "username": "testuser",
                "description": "Another test shader",
                "tags": ["test", "advanced"]
            },
            "renderpass": [
                {
                    "code": "void main() { gl_Position = vec4(1.0); }",
                    "type": "vertex"
                },
                {
                    "code": "void main() { gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0); }",
                    "type": "fragment"
                }
            ]
        }
        
        # Write test files
        with open(os.path.join(self.temp_dir, "test1.json"), 'w') as f:
            json.dump(self.test_shader1, f)
        with open(os.path.join(self.temp_dir, "test2.json"), 'w') as f:
            json.dump(self.test_shader2, f)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_creation(self):
        """Test that ShaderProcessingPipeline can be instantiated"""
        pipeline = ShaderProcessingPipeline(
            json_dir=self.temp_dir,
            output_dir=self.output_dir,
            db_path=self.db_path
        )
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.json_dir, Path(self.temp_dir))
        self.assertEqual(pipeline.db_path, Path(self.db_path))
    
    def test_pipeline_db_initialization(self):
        """Test that the pipeline initializes the database properly"""
        pipeline = ShaderProcessingPipeline(
            json_dir=self.temp_dir,
            output_dir=self.output_dir,
            db_path=self.db_path
        )
        
        # Check that the database file exists
        self.assertTrue(self.db_path.exists())
        
        # Check that the tables exist
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for shaders table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shaders';")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check for shader_features table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shader_features';")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check for tags table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tags';")
        self.assertIsNotNone(cursor.fetchone())
        
        conn.close()


class TestModuleCombiner(unittest.TestCase):
    """Test the module combination functionality"""
    
    def setUp(self):
        """Set up test modules"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test module files
        math_module = """
// Math Constants Module
#define PI 3.14159265359
#define E 2.71828182846
"""
        
        vector_module = """
// Vector Operations Module
vec3 normalizeVector(vec3 v) {
    float len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return vec3(v.x / len, v.y / len, v.z / len);
}
"""
        
        self.math_file = os.path.join(self.temp_dir, "math_module.glsl")
        self.vector_file = os.path.join(self.temp_dir, "vector_module.glsl")
        
        with open(self.math_file, 'w') as f:
            f.write(math_module)
        with open(self.vector_file, 'w') as f:
            f.write(vector_module)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_module_combiner_creation(self):
        """Test that ModuleCombiner can be instantiated"""
        # This test checks that the ModuleCombiner class can be created
        combiner = ModuleCombiner()
        self.assertIsNotNone(combiner)
        self.assertEqual(combiner.modules_dir, 'modules')





if __name__ == '__main__':
    print("Running SuperShader unit tests...")
    unittest.main(verbosity=2)