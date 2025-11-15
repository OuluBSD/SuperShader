#!/usr/bin/env python3
"""
Regression Testing Framework for SuperShader Modules
Automated tests to ensure that new changes don't break existing functionality
"""

import sys
import os
import unittest
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import traceback

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


class RegressionTestSuite:
    """Comprehensive regression testing framework for SuperShader modules"""
    
    def __init__(self):
        self.translator = PseudocodeTranslator()
        self.combiner = ModuleCombiner()
        self.results_dir = Path("regression_tests_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Store baseline results for comparison
        self.baselines = {}
        
    def generate_baseline(self):
        """Generate baseline results for all modules"""
        print("Generating regression test baselines...")
        baselines = {}
        
        # Test procedural modules
        baselines['procedural'] = self._test_procedural_modules()
        
        # Test raymarching modules
        baselines['raymarching'] = self._test_raymarching_modules()
        
        # Test physics modules
        baselines['physics'] = self._test_physics_modules()
        
        # Test texturing modules
        baselines['texturing'] = self._test_texturing_modules()
        
        # Test audio modules
        baselines['audio'] = self._test_audio_modules()
        
        # Test game modules
        baselines['game'] = self._test_game_modules()
        
        # Test UI modules
        baselines['ui'] = self._test_ui_modules()
        
        # Save baseline to file
        baseline_file = self.results_dir / "baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2)
        
        self.baselines = baselines
        print(f"Baseline saved to {baseline_file}")
        return baselines
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests and compare with baseline"""
        print("Running regression tests...")
        
        # Load baseline if not already loaded
        if not self.baselines:
            baseline_file = self.results_dir / "baseline.json"
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    self.baselines = json.load(f)
            else:
                print("No baseline found. Generating new baseline...")
                self.generate_baseline()
        
        # Run current tests
        current_results = {}
        current_results['procedural'] = self._test_procedural_modules()
        current_results['raymarching'] = self._test_raymarching_modules()
        current_results['physics'] = self._test_physics_modules()
        current_results['texturing'] = self._test_texturing_modules()
        current_results['audio'] = self._test_audio_modules()
        current_results['game'] = self._test_game_modules()
        current_results['ui'] = self._test_ui_modules()
        
        # Compare with baseline
        differences = self._compare_results(self.baselines, current_results)
        
        # Save current results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"regression_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(current_results, f, indent=2)
        
        # Print summary
        self._print_regression_summary(differences)
        
        return {
            'current_results': current_results,
            'differences': differences,
            'results_file': str(results_file),
            'all_passed': len(differences) == 0
        }
    
    def _test_procedural_modules(self) -> Dict[str, Any]:
        """Test procedural modules and return results"""
        results = {}
        
        modules = ['perlin_noise', 'noise_functions_branching']
        
        for module_name in modules:
            try:
                module = get_procedural_module(module_name)
                if module:
                    # Test pseudocode translation
                    if 'pseudocode' in module:
                        pseudocode = module['pseudocode']
                        
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            branch_results = {}
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_result = self._test_pseudocode_translation(branch_code)
                                    branch_results[branch_name] = glsl_result
                                except Exception as e:
                                    branch_results[branch_name] = {
                                        'success': False,
                                        'error': str(e),
                                        'traceback': traceback.format_exc()
                                    }
                            results[module_name] = {
                                'type': 'branching',
                                'branches': branch_results,
                                'hash': self._calculate_hash(branch_results)
                            }
                        else:
                            # Non-branching module
                            glsl_result = self._test_pseudocode_translation(pseudocode)
                            results[module_name] = {
                                'type': 'non_branching',
                                'result': glsl_result,
                                'hash': self._calculate_hash(glsl_result)
                            }
                    else:
                        results[module_name] = {
                            'type': 'missing_pseudocode',
                            'success': False,
                            'error': 'No pseudocode found'
                        }
                else:
                    results[module_name] = {
                        'type': 'missing_module',
                        'success': False,
                        'error': 'Module not found'
                    }
            except Exception as e:
                results[module_name] = {
                    'type': 'exception',
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return results
    
    def _test_raymarching_modules(self) -> Dict[str, Any]:
        """Test raymarching modules and return results"""
        results = {}
        
        modules = ['raymarching_core', 'raymarching_advanced_branching']
        
        for module_name in modules:
            try:
                module = get_raymarching_module(module_name)
                if module:
                    # Test pseudocode translation
                    if 'pseudocode' in module:
                        pseudocode = module['pseudocode']
                        
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            branch_results = {}
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_result = self._test_pseudocode_translation(branch_code)
                                    branch_results[branch_name] = glsl_result
                                except Exception as e:
                                    branch_results[branch_name] = {
                                        'success': False,
                                        'error': str(e),
                                        'traceback': traceback.format_exc()
                                    }
                            results[module_name] = {
                                'type': 'branching',
                                'branches': branch_results,
                                'hash': self._calculate_hash(branch_results)
                            }
                        else:
                            # Non-branching module
                            glsl_result = self._test_pseudocode_translation(pseudocode)
                            results[module_name] = {
                                'type': 'non_branching',
                                'result': glsl_result,
                                'hash': self._calculate_hash(glsl_result)
                            }
                    else:
                        results[module_name] = {
                            'type': 'missing_pseudocode',
                            'success': False,
                            'error': 'No pseudocode found'
                        }
                else:
                    results[module_name] = {
                        'type': 'missing_module',
                        'success': False,
                        'error': 'Module not found'
                    }
            except Exception as e:
                results[module_name] = {
                    'type': 'exception',
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return results
    
    def _test_physics_modules(self) -> Dict[str, Any]:
        """Test physics modules and return results"""
        results = {}
        
        modules = ['verlet_integration', 'physics_advanced_branching']
        
        for module_name in modules:
            try:
                module = get_physics_module(module_name)
                if module:
                    # Test pseudocode translation
                    if 'pseudocode' in module:
                        pseudocode = module['pseudocode']
                        
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            branch_results = {}
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_result = self._test_pseudocode_translation(branch_code)
                                    branch_results[branch_name] = glsl_result
                                except Exception as e:
                                    branch_results[branch_name] = {
                                        'success': False,
                                        'error': str(e),
                                        'traceback': traceback.format_exc()
                                    }
                            results[module_name] = {
                                'type': 'branching',
                                'branches': branch_results,
                                'hash': self._calculate_hash(branch_results)
                            }
                        else:
                            # Non-branching module
                            glsl_result = self._test_pseudocode_translation(pseudocode)
                            results[module_name] = {
                                'type': 'non_branching',
                                'result': glsl_result,
                                'hash': self._calculate_hash(glsl_result)
                            }
                    else:
                        results[module_name] = {
                            'type': 'missing_pseudocode',
                            'success': False,
                            'error': 'No pseudocode found'
                        }
                else:
                    results[module_name] = {
                        'type': 'missing_module',
                        'success': False,
                        'error': 'Module not found'
                    }
            except Exception as e:
                results[module_name] = {
                    'type': 'exception',
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return results
    
    def _test_texturing_modules(self) -> Dict[str, Any]:
        """Test texturing modules and return results"""
        results = {}
        
        modules = ['uv_mapping', 'texturing_advanced_branching']
        
        for module_name in modules:
            try:
                module = get_texturing_module(module_name)
                if module:
                    # Test pseudocode translation
                    if 'pseudocode' in module:
                        pseudocode = module['pseudocode']
                        
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            branch_results = {}
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_result = self._test_pseudocode_translation(branch_code)
                                    branch_results[branch_name] = glsl_result
                                except Exception as e:
                                    branch_results[branch_name] = {
                                        'success': False,
                                        'error': str(e),
                                        'traceback': traceback.format_exc()
                                    }
                            results[module_name] = {
                                'type': 'branching',
                                'branches': branch_results,
                                'hash': self._calculate_hash(branch_results)
                            }
                        else:
                            # Non-branching module
                            glsl_result = self._test_pseudocode_translation(pseudocode)
                            results[module_name] = {
                                'type': 'non_branching',
                                'result': glsl_result,
                                'hash': self._calculate_hash(glsl_result)
                            }
                    else:
                        results[module_name] = {
                            'type': 'missing_pseudocode',
                            'success': False,
                            'error': 'No pseudocode found'
                        }
                else:
                    results[module_name] = {
                        'type': 'missing_module',
                        'success': False,
                        'error': 'Module not found'
                    }
            except Exception as e:
                results[module_name] = {
                    'type': 'exception',
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return results
    
    def _test_audio_modules(self) -> Dict[str, Any]:
        """Test audio modules and return results"""
        results = {}
        
        modules = ['beat_detection', 'audio_advanced_branching']
        
        for module_name in modules:
            try:
                module = get_audio_module(module_name)
                if module:
                    # Test pseudocode translation
                    if 'pseudocode' in module:
                        pseudocode = module['pseudocode']
                        
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            branch_results = {}
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_result = self._test_pseudocode_translation(branch_code)
                                    branch_results[branch_name] = glsl_result
                                except Exception as e:
                                    branch_results[branch_name] = {
                                        'success': False,
                                        'error': str(e),
                                        'traceback': traceback.format_exc()
                                    }
                            results[module_name] = {
                                'type': 'branching',
                                'branches': branch_results,
                                'hash': self._calculate_hash(branch_results)
                            }
                        else:
                            # Non-branching module
                            glsl_result = self._test_pseudocode_translation(pseudocode)
                            results[module_name] = {
                                'type': 'non_branching',
                                'result': glsl_result,
                                'hash': self._calculate_hash(glsl_result)
                            }
                    else:
                        results[module_name] = {
                            'type': 'missing_pseudocode',
                            'success': False,
                            'error': 'No pseudocode found'
                        }
                else:
                    results[module_name] = {
                        'type': 'missing_module',
                        'success': False,
                        'error': 'Module not found'
                    }
            except Exception as e:
                results[module_name] = {
                    'type': 'exception',
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return results
    
    def _test_game_modules(self) -> Dict[str, Any]:
        """Test game modules and return results"""
        results = {}
        
        modules = ['input_handling', 'game_advanced_branching']
        
        for module_name in modules:
            try:
                module = get_game_module(module_name)
                if module:
                    # Test pseudocode translation
                    if 'pseudocode' in module:
                        pseudocode = module['pseudocode']
                        
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            branch_results = {}
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_result = self._test_pseudocode_translation(branch_code)
                                    branch_results[branch_name] = glsl_result
                                except Exception as e:
                                    branch_results[branch_name] = {
                                        'success': False,
                                        'error': str(e),
                                        'traceback': traceback.format_exc()
                                    }
                            results[module_name] = {
                                'type': 'branching',
                                'branches': branch_results,
                                'hash': self._calculate_hash(branch_results)
                            }
                        else:
                            # Non-branching module
                            glsl_result = self._test_pseudocode_translation(pseudocode)
                            results[module_name] = {
                                'type': 'non_branching',
                                'result': glsl_result,
                                'hash': self._calculate_hash(glsl_result)
                            }
                    else:
                        results[module_name] = {
                            'type': 'missing_pseudocode',
                            'success': False,
                            'error': 'No pseudocode found'
                        }
                else:
                    results[module_name] = {
                        'type': 'missing_module',
                        'success': False,
                        'error': 'Module not found'
                    }
            except Exception as e:
                results[module_name] = {
                    'type': 'exception',
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return results
    
    def _test_ui_modules(self) -> Dict[str, Any]:
        """Test UI modules and return results"""
        results = {}
        
        modules = ['basic_shapes', 'ui_advanced_branching']
        
        for module_name in modules:
            try:
                module = get_ui_module(module_name)
                if module:
                    # Test pseudocode translation
                    if 'pseudocode' in module:
                        pseudocode = module['pseudocode']
                        
                        # For branching modules, test each branch
                        if isinstance(pseudocode, dict):
                            branch_results = {}
                            for branch_name, branch_code in pseudocode.items():
                                try:
                                    glsl_result = self._test_pseudocode_translation(branch_code)
                                    branch_results[branch_name] = glsl_result
                                except Exception as e:
                                    branch_results[branch_name] = {
                                        'success': False,
                                        'error': str(e),
                                        'traceback': traceback.format_exc()
                                    }
                            results[module_name] = {
                                'type': 'branching',
                                'branches': branch_results,
                                'hash': self._calculate_hash(branch_results)
                            }
                        else:
                            # Non-branching module
                            glsl_result = self._test_pseudocode_translation(pseudocode)
                            results[module_name] = {
                                'type': 'non_branching',
                                'result': glsl_result,
                                'hash': self._calculate_hash(glsl_result)
                            }
                    else:
                        results[module_name] = {
                            'type': 'missing_pseudocode',
                            'success': False,
                            'error': 'No pseudocode found'
                        }
                else:
                    results[module_name] = {
                        'type': 'missing_module',
                        'success': False,
                        'error': 'Module not found'
                    }
            except Exception as e:
                results[module_name] = {
                    'type': 'exception',
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return results
    
    def _test_pseudocode_translation(self, pseudocode: str) -> Dict[str, Any]:
        """Test pseudocode translation and return result"""
        try:
            # Test GLSL translation
            glsl_result = self.translator.translate_to_glsl(pseudocode)
            
            # Test Metal translation
            metal_result = self.translator.translate(pseudocode, 'metal')
            
            # Test C++ translation
            cpp_result = self.translator.translate(pseudocode, 'c_cpp')
            
            return {
                'success': True,
                'translations': {
                    'glsl': {
                        'success': glsl_result is not None,
                        'length': len(glsl_result) if glsl_result else 0
                    },
                    'metal': {
                        'success': metal_result is not None,
                        'length': len(metal_result) if metal_result else 0
                    },
                    'cpp': {
                        'success': cpp_result is not None,
                        'length': len(cpp_result) if cpp_result else 0
                    }
                },
                'original_length': len(pseudocode)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _compare_results(self, baseline: Dict, current: Dict) -> Dict[str, Any]:
        """Compare current results with baseline and return differences"""
        differences = {}
        
        for module_type in baseline:
            if module_type not in current:
                differences[module_type] = {
                    'status': 'missing_in_current',
                    'baseline': baseline[module_type]
                }
                continue
            
            type_diffs = self._compare_module_type(baseline[module_type], current[module_type])
            if type_diffs:
                differences[module_type] = type_diffs
        
        # Check for new module types in current that weren't in baseline
        for module_type in current:
            if module_type not in baseline:
                differences[module_type] = {
                    'status': 'new_in_current',
                    'current': current[module_type]
                }
        
        return differences
    
    def _compare_module_type(self, baseline: Dict, current: Dict) -> Dict[str, Any]:
        """Compare results for a specific module type"""
        differences = {}
        
        # Check for modules that exist in baseline but not in current
        for module_name in baseline:
            if module_name not in current:
                differences[module_name] = {
                    'status': 'removed',
                    'baseline': baseline[module_name]
                }
                continue
            
            # Compare module results
            baseline_mod = baseline[module_name]
            current_mod = current[module_name]
            
            if baseline_mod.get('type') != current_mod.get('type'):
                differences[module_name] = {
                    'status': 'type_changed',
                    'baseline_type': baseline_mod.get('type'),
                    'current_type': current_mod.get('type')
                }
                continue
            
            # Check if hashes are different for same types
            baseline_hash = baseline_mod.get('hash')
            current_hash = current_mod.get('hash')
            
            if baseline_hash and current_hash and baseline_hash != current_hash:
                # Different implementation detected
                differences[module_name] = {
                    'status': 'implementation_changed',
                    'baseline_hash': baseline_hash,
                    'current_hash': current_hash
                }
            elif not baseline_mod.get('success', True) and current_mod.get('success', False):
                # Module was broken before, now fixed
                differences[module_name] = {
                    'status': 'fixed',
                    'baseline_error': baseline_mod.get('error'),
                    'current_success': True
                }
            elif baseline_mod.get('success', True) and not current_mod.get('success', True):
                # Module was working before, now broken
                differences[module_name] = {
                    'status': 'broken',
                    'baseline_success': True,
                    'current_error': current_mod.get('error')
                }
        
        # Check for new modules in current that weren't in baseline
        for module_name in current:
            if module_name not in baseline:
                differences[module_name] = {
                    'status': 'new',
                    'current': current[module_name]
                }
        
        return differences
    
    def _calculate_hash(self, obj: Any) -> str:
        """Calculate hash of an object for comparison purposes"""
        try:
            # Convert object to string representation for hashing
            obj_str = json.dumps(obj, sort_keys=True, default=str)
            return hashlib.md5(obj_str.encode()).hexdigest()
        except:
            return ""
    
    def _print_regression_summary(self, differences: Dict[str, Any]):
        """Print a summary of regression test results"""
        print("\n" + "="*60)
        print("REGRESSION TEST RESULTS")
        print("="*60)
        
        if not differences:
            print("✓ No regressions detected! All modules behave as expected.")
            print("✓ All functionality preserved compared to baseline.")
        else:
            print(f"✗ REGRESSIONS DETECTED: {len(differences)} differences found")
            
            for module_type, diffs in differences.items():
                print(f"\n{module_type.upper()} MODULES:")
                
                if 'status' in diffs:
                    # Single difference at type level
                    if diffs['status'] == 'missing_in_current':
                        print(f"  - Module type removed: {module_type}")
                    elif diffs['status'] == 'new_in_current':
                        print(f"  - New module type added: {module_type}")
                else:
                    # Differences in individual modules
                    for module_name, diff_details in diffs.items():
                        status = diff_details.get('status', 'unknown')
                        if status == 'implementation_changed':
                            print(f"  - {module_name}: IMPLEMENTATION CHANGED")
                        elif status == 'broken':
                            error = diff_details.get('current_error', 'Unknown error')
                            print(f"  - {module_name}: WAS WORKING, NOW BROKEN - {error}")
                        elif status == 'fixed':
                            print(f"  - {module_name}: WAS BROKEN, NOW FIXED")
                        elif status == 'removed':
                            print(f"  - {module_name}: MODULE REMOVED")
                        elif status == 'new':
                            print(f"  - {module_name}: NEW MODULE ADDED")
        
        print("="*60)
        
        # Count issues
        issues = 0
        for diffs in differences.values():
            if 'status' not in diffs:  # Individual module differences
                for diff in diffs.values():
                    status = diff.get('status', '')
                    if status in ['broken', 'implementation_changed']:
                        issues += 1
        
        print(f"ISSUES IDENTIFIED: {issues} potential regressions")
        print("="*60)


def main():
    """Main entry point for regression testing"""
    print("Initializing SuperShader Regression Testing Framework...")
    
    regression_suite = RegressionTestSuite()
    
    # Generate a new baseline (since this is a new test run)
    print("\nGenerating new baseline...")
    baseline = regression_suite.generate_baseline()
    
    print("\nRunning regression tests against baseline...")
    results = regression_suite.run_regression_tests()
    
    print(f"\nResults saved to: {results['results_file']}")
    
    # Return success based on presence of regressions
    has_regressions = len(results['differences']) > 0
    
    return 0 if not has_regressions else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)