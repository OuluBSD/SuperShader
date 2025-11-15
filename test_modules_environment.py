#!/usr/bin/env python3
"""
Comprehensive Testing Environment for Individual Modules
Provides a complete testing framework for all module types in the SuperShader project
"""

import sys
import os
import unittest
import json
from typing import Dict, List, Any, Tuple

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from management.module_combiner import ModuleCombiner
from create_pseudocode_translator import PseudocodeTranslator
from modules.procedural.registry import get_module_by_name as get_procedural_module
from modules.raymarching.registry import get_module_by_name as get_raymarching_module
from modules.physics.registry import get_module_by_name as get_physics_module
from modules.texturing.registry import get_module_by_name as get_texturing_module
from modules.audio.registry import get_module_by_name as get_audio_module
from modules.game.registry import get_module_by_name as get_game_module
from modules.ui.registry import get_module_by_name as get_ui_module


class ModuleTestingEnvironment:
    """Main testing environment for all module types"""
    
    def __init__(self):
        self.translator = PseudocodeTranslator()
        self.combiner = ModuleCombiner()
        self.test_results = {
            'procedural': {'passed': 0, 'failed': 0, 'tests': []},
            'raymarching': {'passed': 0, 'failed': 0, 'tests': []},
            'physics': {'passed': 0, 'failed': 0, 'tests': []},
            'texturing': {'passed': 0, 'failed': 0, 'tests': []},
            'audio': {'passed': 0, 'failed': 0, 'tests': []},
            'game': {'passed': 0, 'failed': 0, 'tests': []},
            'ui': {'passed': 0, 'failed': 0, 'tests': []}
        }
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all tests for all module types"""
        print("Starting comprehensive module testing environment...")
        
        # Run tests for each module type
        self._test_procedural_modules()
        self._test_raymarching_modules()
        self._test_physics_modules()
        self._test_texturing_modules()
        self._test_audio_modules()
        self._test_game_modules()
        self._test_ui_modules()
        
        # Print summary
        self._print_test_summary()
        
        return self.test_results
    
    def _test_procedural_modules(self):
        """Test all procedural modules"""
        print("\nTesting Procedural Modules...")
        
        # Get all procedural modules
        modules = [
            'perlin_noise', 
            'noise_functions_branching'
        ]
        
        for module_name in modules:
            try:
                module = get_procedural_module(module_name)
                if module:
                    # Test that module can be loaded
                    if 'pseudocode' in module:
                        # Test pseudocode translation
                        try:
                            glsl_code = self.translator.translate_to_glsl(module['pseudocode'])
                            if glsl_code and len(glsl_code) > 0:
                                self.test_results['procedural']['passed'] += 1
                                self.test_results['procedural']['tests'].append({
                                    'module': module_name,
                                    'test': 'pseudocode_translation',
                                    'status': 'PASS'
                                })
                            else:
                                self.test_results['procedural']['failed'] += 1
                                self.test_results['procedural']['tests'].append({
                                    'module': module_name,
                                    'test': 'pseudocode_translation',
                                    'status': 'FAIL',
                                    'error': 'Empty GLSL translation'
                                })
                        except Exception as e:
                            self.test_results['procedural']['failed'] += 1
                            self.test_results['procedural']['tests'].append({
                                'module': module_name,
                                'test': 'pseudocode_translation',
                                'status': 'FAIL',
                                'error': str(e)
                            })
                    else:
                        self.test_results['procedural']['failed'] += 1
                        self.test_results['procedural']['tests'].append({
                            'module': module_name,
                            'test': 'pseudocode_check',
                            'status': 'FAIL',
                            'error': 'No pseudocode found'
                        })
                else:
                    self.test_results['procedural']['failed'] += 1
                    self.test_results['procedural']['tests'].append({
                        'module': module_name,
                        'test': 'module_load',
                        'status': 'FAIL',
                        'error': 'Module not found'
                    })
            except Exception as e:
                self.test_results['procedural']['failed'] += 1
                self.test_results['procedural']['tests'].append({
                    'module': module_name,
                    'test': 'general',
                    'status': 'FAIL',
                    'error': str(e)
                })
    
    def _test_raymarching_modules(self):
        """Test all raymarching modules"""
        print("\nTesting Raymarching Modules...")
        
        # Get all raymarching modules
        modules = [
            'raymarching_core',
            'raymarching_advanced_branching'
        ]
        
        for module_name in modules:
            try:
                module = get_raymarching_module(module_name)
                if module:
                    # Test that module can be loaded
                    if 'pseudocode' in module:
                        # Test pseudocode translation
                        pseudocode = module['pseudocode']
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_code = self.translator.translate_to_glsl(branch_code)
                                    if glsl_code and len(glsl_code) > 0:
                                        self.test_results['raymarching']['passed'] += 1
                                        self.test_results['raymarching']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'PASS'
                                        })
                                    else:
                                        self.test_results['raymarching']['failed'] += 1
                                        self.test_results['raymarching']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'FAIL',
                                            'error': 'Empty GLSL translation'
                                        })
                                except Exception as e:
                                    self.test_results['raymarching']['failed'] += 1
                                    self.test_results['raymarching']['tests'].append({
                                        'module': f"{module_name}:{branch_name}",
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': str(e)
                                    })
                        else:
                            # Non-branching module
                            try:
                                glsl_code = self.translator.translate_to_glsl(pseudocode)
                                if glsl_code and len(glsl_code) > 0:
                                    self.test_results['raymarching']['passed'] += 1
                                    self.test_results['raymarching']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'PASS'
                                    })
                                else:
                                    self.test_results['raymarching']['failed'] += 1
                                    self.test_results['raymarching']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': 'Empty GLSL translation'
                                    })
                            except Exception as e:
                                self.test_results['raymarching']['failed'] += 1
                                self.test_results['raymarching']['tests'].append({
                                    'module': module_name,
                                    'test': 'pseudocode_translation',
                                    'status': 'FAIL',
                                    'error': str(e)
                                })
                    else:
                        self.test_results['raymarching']['failed'] += 1
                        self.test_results['raymarching']['tests'].append({
                            'module': module_name,
                            'test': 'pseudocode_check',
                            'status': 'FAIL',
                            'error': 'No pseudocode found'
                        })
                else:
                    self.test_results['raymarching']['failed'] += 1
                    self.test_results['raymarching']['tests'].append({
                        'module': module_name,
                        'test': 'module_load',
                        'status': 'FAIL',
                        'error': 'Module not found'
                    })
            except Exception as e:
                self.test_results['raymarching']['failed'] += 1
                self.test_results['raymarching']['tests'].append({
                    'module': module_name,
                    'test': 'general',
                    'status': 'FAIL',
                    'error': str(e)
                })
    
    def _test_physics_modules(self):
        """Test all physics modules"""
        print("\nTesting Physics Modules...")
        
        # Get all physics modules
        modules = [
            'verlet_integration',
            'physics_advanced_branching'
        ]
        
        for module_name in modules:
            try:
                module = get_physics_module(module_name)
                if module:
                    # Test that module can be loaded
                    if 'pseudocode' in module:
                        # Test pseudocode translation
                        pseudocode = module['pseudocode']
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_code = self.translator.translate_to_glsl(branch_code)
                                    if glsl_code and len(glsl_code) > 0:
                                        self.test_results['physics']['passed'] += 1
                                        self.test_results['physics']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'PASS'
                                        })
                                    else:
                                        self.test_results['physics']['failed'] += 1
                                        self.test_results['physics']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'FAIL',
                                            'error': 'Empty GLSL translation'
                                        })
                                except Exception as e:
                                    self.test_results['physics']['failed'] += 1
                                    self.test_results['physics']['tests'].append({
                                        'module': f"{module_name}:{branch_name}",
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': str(e)
                                    })
                        else:
                            # Non-branching module
                            try:
                                glsl_code = self.translator.translate_to_glsl(pseudocode)
                                if glsl_code and len(glsl_code) > 0:
                                    self.test_results['physics']['passed'] += 1
                                    self.test_results['physics']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'PASS'
                                    })
                                else:
                                    self.test_results['physics']['failed'] += 1
                                    self.test_results['physics']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': 'Empty GLSL translation'
                                    })
                            except Exception as e:
                                self.test_results['physics']['failed'] += 1
                                self.test_results['physics']['tests'].append({
                                    'module': module_name,
                                    'test': 'pseudocode_translation',
                                    'status': 'FAIL',
                                    'error': str(e)
                                })
                    else:
                        self.test_results['physics']['failed'] += 1
                        self.test_results['physics']['tests'].append({
                            'module': module_name,
                            'test': 'pseudocode_check',
                            'status': 'FAIL',
                            'error': 'No pseudocode found'
                        })
                else:
                    self.test_results['physics']['failed'] += 1
                    self.test_results['physics']['tests'].append({
                        'module': module_name,
                        'test': 'module_load',
                        'status': 'FAIL',
                        'error': 'Module not found'
                    })
            except Exception as e:
                self.test_results['physics']['failed'] += 1
                self.test_results['physics']['tests'].append({
                    'module': module_name,
                    'test': 'general',
                    'status': 'FAIL',
                    'error': str(e)
                })
    
    def _test_texturing_modules(self):
        """Test all texturing modules"""
        print("\nTesting Texturing Modules...")
        
        # Get all texturing modules
        modules = [
            'uv_mapping',
            'texturing_advanced_branching'
        ]
        
        for module_name in modules:
            try:
                module = get_texturing_module(module_name)
                if module:
                    # Test that module can be loaded
                    if 'pseudocode' in module:
                        # Test pseudocode translation
                        pseudocode = module['pseudocode']
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_code = self.translator.translate_to_glsl(branch_code)
                                    if glsl_code and len(glsl_code) > 0:
                                        self.test_results['texturing']['passed'] += 1
                                        self.test_results['texturing']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'PASS'
                                        })
                                    else:
                                        self.test_results['texturing']['failed'] += 1
                                        self.test_results['texturing']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'FAIL',
                                            'error': 'Empty GLSL translation'
                                        })
                                except Exception as e:
                                    self.test_results['texturing']['failed'] += 1
                                    self.test_results['texturing']['tests'].append({
                                        'module': f"{module_name}:{branch_name}",
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': str(e)
                                    })
                        else:
                            # Non-branching module
                            try:
                                glsl_code = self.translator.translate_to_glsl(pseudocode)
                                if glsl_code and len(glsl_code) > 0:
                                    self.test_results['texturing']['passed'] += 1
                                    self.test_results['texturing']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'PASS'
                                    })
                                else:
                                    self.test_results['texturing']['failed'] += 1
                                    self.test_results['texturing']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': 'Empty GLSL translation'
                                    })
                            except Exception as e:
                                self.test_results['texturing']['failed'] += 1
                                self.test_results['texturing']['tests'].append({
                                    'module': module_name,
                                    'test': 'pseudocode_translation',
                                    'status': 'FAIL',
                                    'error': str(e)
                                })
                    else:
                        self.test_results['texturing']['failed'] += 1
                        self.test_results['texturing']['tests'].append({
                            'module': module_name,
                            'test': 'pseudocode_check',
                            'status': 'FAIL',
                            'error': 'No pseudocode found'
                        })
                else:
                    self.test_results['texturing']['failed'] += 1
                    self.test_results['texturing']['tests'].append({
                        'module': module_name,
                        'test': 'module_load',
                        'status': 'FAIL',
                        'error': 'Module not found'
                    })
            except Exception as e:
                self.test_results['texturing']['failed'] += 1
                self.test_results['texturing']['tests'].append({
                    'module': module_name,
                    'test': 'general',
                    'status': 'FAIL',
                    'error': str(e)
                })
    
    def _test_audio_modules(self):
        """Test all audio modules"""
        print("\nTesting Audio Modules...")
        
        # Get all audio modules
        modules = [
            'beat_detection',
            'audio_advanced_branching'
        ]
        
        for module_name in modules:
            try:
                module = get_audio_module(module_name)
                if module:
                    # Test that module can be loaded
                    if 'pseudocode' in module:
                        # Test pseudocode translation
                        pseudocode = module['pseudocode']
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_code = self.translator.translate_to_glsl(branch_code)
                                    if glsl_code and len(glsl_code) > 0:
                                        self.test_results['audio']['passed'] += 1
                                        self.test_results['audio']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'PASS'
                                        })
                                    else:
                                        self.test_results['audio']['failed'] += 1
                                        self.test_results['audio']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'FAIL',
                                            'error': 'Empty GLSL translation'
                                        })
                                except Exception as e:
                                    self.test_results['audio']['failed'] += 1
                                    self.test_results['audio']['tests'].append({
                                        'module': f"{module_name}:{branch_name}",
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': str(e)
                                    })
                        else:
                            # Non-branching module
                            try:
                                glsl_code = self.translator.translate_to_glsl(pseudocode)
                                if glsl_code and len(glsl_code) > 0:
                                    self.test_results['audio']['passed'] += 1
                                    self.test_results['audio']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'PASS'
                                    })
                                else:
                                    self.test_results['audio']['failed'] += 1
                                    self.test_results['audio']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': 'Empty GLSL translation'
                                    })
                            except Exception as e:
                                self.test_results['audio']['failed'] += 1
                                self.test_results['audio']['tests'].append({
                                    'module': module_name,
                                    'test': 'pseudocode_translation',
                                    'status': 'FAIL',
                                    'error': str(e)
                                })
                    else:
                        self.test_results['audio']['failed'] += 1
                        self.test_results['audio']['tests'].append({
                            'module': module_name,
                            'test': 'pseudocode_check',
                            'status': 'FAIL',
                            'error': 'No pseudocode found'
                        })
                else:
                    self.test_results['audio']['failed'] += 1
                    self.test_results['audio']['tests'].append({
                        'module': module_name,
                        'test': 'module_load',
                        'status': 'FAIL',
                        'error': 'Module not found'
                    })
            except Exception as e:
                self.test_results['audio']['failed'] += 1
                self.test_results['audio']['tests'].append({
                    'module': module_name,
                    'test': 'general',
                    'status': 'FAIL',
                    'error': str(e)
                })
    
    def _test_game_modules(self):
        """Test all game modules"""
        print("\nTesting Game Modules...")
        
        # Get all game modules
        modules = [
            'input_handling',
            'game_advanced_branching'
        ]
        
        for module_name in modules:
            try:
                module = get_game_module(module_name)
                if module:
                    # Test that module can be loaded
                    if 'pseudocode' in module:
                        # Test pseudocode translation
                        pseudocode = module['pseudocode']
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_code = self.translator.translate_to_glsl(branch_code)
                                    if glsl_code and len(glsl_code) > 0:
                                        self.test_results['game']['passed'] += 1
                                        self.test_results['game']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'PASS'
                                        })
                                    else:
                                        self.test_results['game']['failed'] += 1
                                        self.test_results['game']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'FAIL',
                                            'error': 'Empty GLSL translation'
                                        })
                                except Exception as e:
                                    self.test_results['game']['failed'] += 1
                                    self.test_results['game']['tests'].append({
                                        'module': f"{module_name}:{branch_name}",
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': str(e)
                                    })
                        else:
                            # Non-branching module
                            try:
                                glsl_code = self.translator.translate_to_glsl(pseudocode)
                                if glsl_code and len(glsl_code) > 0:
                                    self.test_results['game']['passed'] += 1
                                    self.test_results['game']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'PASS'
                                    })
                                else:
                                    self.test_results['game']['failed'] += 1
                                    self.test_results['game']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': 'Empty GLSL translation'
                                    })
                            except Exception as e:
                                self.test_results['game']['failed'] += 1
                                self.test_results['game']['tests'].append({
                                    'module': module_name,
                                    'test': 'pseudocode_translation',
                                    'status': 'FAIL',
                                    'error': str(e)
                                })
                    else:
                        self.test_results['game']['failed'] += 1
                        self.test_results['game']['tests'].append({
                            'module': module_name,
                            'test': 'pseudocode_check',
                            'status': 'FAIL',
                            'error': 'No pseudocode found'
                        })
                else:
                    self.test_results['game']['failed'] += 1
                    self.test_results['game']['tests'].append({
                        'module': module_name,
                        'test': 'module_load',
                        'status': 'FAIL',
                        'error': 'Module not found'
                    })
            except Exception as e:
                self.test_results['game']['failed'] += 1
                self.test_results['game']['tests'].append({
                    'module': module_name,
                    'test': 'general',
                    'status': 'FAIL',
                    'error': str(e)
                })
    
    def _test_ui_modules(self):
        """Test all UI modules"""
        print("\nTesting UI Modules...")
        
        # Get all UI modules
        modules = [
            'basic_shapes',
            'ui_advanced_branching'
        ]
        
        for module_name in modules:
            try:
                module = get_ui_module(module_name)
                if module:
                    # Test that module can be loaded
                    if 'pseudocode' in module:
                        # Test pseudocode translation
                        pseudocode = module['pseudocode']
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_code = self.translator.translate_to_glsl(branch_code)
                                    if glsl_code and len(glsl_code) > 0:
                                        self.test_results['ui']['passed'] += 1
                                        self.test_results['ui']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'PASS'
                                        })
                                    else:
                                        self.test_results['ui']['failed'] += 1
                                        self.test_results['ui']['tests'].append({
                                            'module': f"{module_name}:{branch_name}",
                                            'test': 'pseudocode_translation',
                                            'status': 'FAIL',
                                            'error': 'Empty GLSL translation'
                                        })
                                except Exception as e:
                                    self.test_results['ui']['failed'] += 1
                                    self.test_results['ui']['tests'].append({
                                        'module': f"{module_name}:{branch_name}",
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': str(e)
                                    })
                        else:
                            # Non-branching module
                            try:
                                glsl_code = self.translator.translate_to_glsl(pseudocode)
                                if glsl_code and len(glsl_code) > 0:
                                    self.test_results['ui']['passed'] += 1
                                    self.test_results['ui']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'PASS'
                                    })
                                else:
                                    self.test_results['ui']['failed'] += 1
                                    self.test_results['ui']['tests'].append({
                                        'module': module_name,
                                        'test': 'pseudocode_translation',
                                        'status': 'FAIL',
                                        'error': 'Empty GLSL translation'
                                    })
                            except Exception as e:
                                self.test_results['ui']['failed'] += 1
                                self.test_results['ui']['tests'].append({
                                    'module': module_name,
                                    'test': 'pseudocode_translation',
                                    'status': 'FAIL',
                                    'error': str(e)
                                })
                    else:
                        self.test_results['ui']['failed'] += 1
                        self.test_results['ui']['tests'].append({
                            'module': module_name,
                            'test': 'pseudocode_check',
                            'status': 'FAIL',
                            'error': 'No pseudocode found'
                        })
                else:
                    self.test_results['ui']['failed'] += 1
                    self.test_results['ui']['tests'].append({
                        'module': module_name,
                        'test': 'module_load',
                        'status': 'FAIL',
                        'error': 'Module not found'
                    })
            except Exception as e:
                self.test_results['ui']['failed'] += 1
                self.test_results['ui']['tests'].append({
                    'module': module_name,
                    'test': 'general',
                    'status': 'FAIL',
                    'error': str(e)
                })
    
    def _print_test_summary(self):
        """Print a summary of all test results"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODULE TESTING SUMMARY")
        print("="*60)
        
        total_passed = 0
        total_failed = 0
        
        for module_type, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total = passed + failed
            
            total_passed += passed
            total_failed += failed
            
            print(f"\n{module_type.upper()} MODULES:")
            print(f"  Total Tests: {total}")
            print(f"  Passed: {passed}")
            print(f"  Failed: {failed}")
            print(f"  Success Rate: {passed/total*100:.1f}%" if total > 0 else "  Success Rate: 0%")
        
        print("\n" + "="*60)
        print("OVERALL SUMMARY:")
        print(f"  Total Tests: {total_passed + total_failed}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_failed}")
        print(f"  Success Rate: {total_passed/(total_passed + total_failed)*100:.1f}%" if (total_passed + total_failed) > 0 else "  Success Rate: 0%")
        print("="*60)


def main():
    """Main entry point for the testing environment"""
    print("Initializing SuperShader Module Testing Environment...")
    
    tester = ModuleTestingEnvironment()
    results = tester.run_all_tests()
    
    # Return success code based on test results
    total_tests = sum([r['passed'] + r['failed'] for r in results.values()])
    failed_tests = sum([r['failed'] for r in results.values()])
    
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)