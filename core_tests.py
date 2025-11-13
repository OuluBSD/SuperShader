#!/usr/bin/env python3
"""
Core unit tests for the SuperShader project
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch
import json
from pathlib import Path


class TestExtractGLSL(unittest.TestCase):
    """Test the GLSL extraction functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_json = {
            "info": {
                "id": "test_shader",
                "name": "Test Shader",
                "tags": ["test", "simple"]
            },
            "renderpass": [
                {
                    "code": "void main() { gl_Position = vec4(0.0); }",
                    "type": "vertex"
                },
                {
                    "code": "void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }",
                    "type": "fragment"
                }
            ]
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
        from extract_glsl import extract_glsl_from_file
        result = extract_glsl_from_file(self.temp_file.name)
        # Check that the extracted code contains our expected code
        self.assertIn("gl_Position = vec4(0.0)", result)
        self.assertIn("gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0)", result)
    
    def test_extract_invalid_json(self):
        """Test GLSL extraction from invalid JSON"""
        from extract_glsl import extract_glsl_from_file
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
        from analyze_tags import analyze_shader_tags
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
        from catalog_features import ShaderFeatureCataloger
        # Mock the glob function to return our test file
        mock_glob.return_value = [os.path.join(self.temp_dir, "test_shader.json")]
        
        # Create cataloger and run analysis
        cataloger = ShaderFeatureCataloger(json_dir=self.temp_dir, catalog_dir=self.catalog_dir)
        features, shader_features, feature_shader_map = cataloger.catalog_features(force_rebuild=True)
        
        # Check that expected features were found
        self.assertGreater(len(features), 0)
        
        # Check that our test shader was processed
        self.assertIn('test_shader', shader_features)
        self.assertGreater(cataloger.total_shaders, 0)


class TestModuleCombiner(unittest.TestCase):
    """Test the module combiner functionality"""
    
    def test_module_combiner_creation(self):
        """Test that ModuleCombiner can be instantiated"""
        from management.module_combiner import ModuleCombiner
        # This test checks that the ModuleCombiner class can be created
        combiner = ModuleCombiner()
        self.assertIsNotNone(combiner)
        self.assertEqual(combiner.modules_dir, 'modules')


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
        
        # Write test file
        with open(os.path.join(self.temp_dir, "test1.json"), 'w') as f:
            json.dump(self.test_shader1, f)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_creation(self):
        """Test that ShaderProcessingPipeline can be instantiated"""
        from pipeline import ShaderProcessingPipeline
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
        from pipeline import ShaderProcessingPipeline
        # Convert to Path object
        db_path = Path(self.db_path)
        pipeline = ShaderProcessingPipeline(
            json_dir=self.temp_dir,
            output_dir=self.output_dir,
            db_path=db_path
        )
        
        # Check that the database file exists
        self.assertTrue(db_path.exists())
        
        # Check that the tables exist
        import sqlite3
        conn = sqlite3.connect(db_path)
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


if __name__ == '__main__':
    print("Running SuperShader core unit tests...")
    unittest.main(verbosity=2)