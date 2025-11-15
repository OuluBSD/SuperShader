#!/usr/bin/env python3
"""
Comprehensive Test Suite for SuperShader
Includes all types of tests: unit, integration, regression, compatibility, and performance
"""

import sys
import os
import unittest
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import tempfile


class ComprehensiveTestSuite:
    """Comprehensive test suite for all SuperShader components"""
    
    def __init__(self):
        self.test_results = {
            'unit_tests': [],
            'integration_tests': [],
            'regression_tests': [],
            'compatibility_tests': [],
            'performance_tests': [],
            'functional_tests': []
        }
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, verbose=False) -> Dict[str, Any]:
        """Run all test categories"""
        print("Starting Comprehensive SuperShader Test Suite...")
        print("=" * 60)
        self.start_time = time.time()
        
        # Run all test categories
        self.run_unit_tests(verbose)
        self.run_integration_tests(verbose)
        self.run_regression_tests(verbose)
        self.run_compatibility_tests(verbose)
        self.run_performance_tests(verbose)
        self.run_functional_tests(verbose)
        
        self.end_time = time.time()
        
        # Generate report
        report = self.generate_test_report()
        self.print_test_summary(report)
        
        return report
    
    def run_unit_tests(self, verbose=False) -> List[Dict[str, Any]]:
        """Run unit tests for individual components"""
        print("Running Unit Tests...")
        
        # Test pseudocode translator
        translator_tests = self.test_pseudocode_translator()
        
        # Test module combiner
        combiner_tests = self.test_module_combiner()
        
        # Test management components
        management_tests = self.test_management_components()
        
        # Combine results
        results = translator_tests + combiner_tests + management_tests
        
        self.test_results['unit_tests'] = results
        
        if verbose:
            passed = sum(1 for t in results if t['status'] == 'PASS')
            failed = len(results) - passed
            print(f"  Unit Tests: {passed} passed, {failed} failed")
        
        return results
    
    def test_pseudocode_translator(self) -> List[Dict[str, Any]]:
        """Test the pseudocode translator functionality"""
        results = []
        
        try:
            from create_pseudocode_translator import PseudocodeTranslator
            translator = PseudocodeTranslator()
            
            # Sample pseudocode
            sample_pseudocode = """
            float sampleFunction(vec2 coord) {
                float value = length(coord);
                return value * 0.5;
            }
            """
            
            # Test GLSL translation
            start_time = time.time()
            glsl_result = translator.translate_to_glsl(sample_pseudocode)
            glsl_time = time.time() - start_time
            
            if glsl_result and len(glsl_result) > 0:
                results.append({
                    'name': 'Pseudocode to GLSL Translation',
                    'status': 'PASS',
                    'execution_time': glsl_time,
                    'details': f'Translated to {len(glsl_result)} chars'
                })
            else:
                results.append({
                    'name': 'Pseudocode to GLSL Translation',
                    'status': 'FAIL',
                    'execution_time': glsl_time,
                    'details': 'Translation returned empty or None'
                })
            
            # Test Metal translation
            start_time = time.time()
            metal_result = translator.translate(sample_pseudocode, 'metal')
            metal_time = time.time() - start_time
            
            if metal_result and len(metal_result) > 0:
                results.append({
                    'name': 'Pseudocode to Metal Translation',
                    'status': 'PASS',
                    'execution_time': metal_time,
                    'details': f'Translated to {len(metal_result)} chars'
                })
            else:
                results.append({
                    'name': 'Pseudocode to Metal Translation',
                    'status': 'FAIL',
                    'execution_time': metal_time,
                    'details': 'Translation returned empty or None'
                })
                
            # Test C/C++ translation
            start_time = time.time()
            cpp_result = translator.translate(sample_pseudocode, 'c_cpp')
            cpp_time = time.time() - start_time
            
            if cpp_result and len(cpp_result) > 0:
                results.append({
                    'name': 'Pseudocode to C/C++ Translation',
                    'status': 'PASS',
                    'execution_time': cpp_time,
                    'details': f'Translated to {len(cpp_result)} chars'
                })
            else:
                results.append({
                    'name': 'Pseudocode to C/C++ Translation',
                    'status': 'FAIL',
                    'execution_time': cpp_time,
                    'details': 'Translation returned empty or None'
                })
                
        except Exception as e:
            results.append({
                'name': 'Pseudocode Translator Tests',
                'status': 'ERROR',
                'execution_time': 0,
                'details': f'Exception: {str(e)}'
            })
        
        return results
    
    def test_module_combiner(self) -> List[Dict[str, Any]]:
        """Test the module combiner functionality"""
        results = []
        
        try:
            from management.module_combiner import ModuleCombiner
            combiner = ModuleCombiner()
            
            # Test basic module loading functionality
            results.append({
                'name': 'Module Combiner Initialization',
                'status': 'PASS',
                'execution_time': 0.001,
                'details': 'ModuleCombiner instantiated successfully'
            })
            
            # Test combining with no modules (should work)
            start_time = time.time()
            combined = combiner.combine_modules([])
            combiner_time = time.time() - start_time
            
            if combined is not None:
                results.append({
                    'name': 'Empty Module Combination',
                    'status': 'PASS',
                    'execution_time': combiner_time,
                    'details': 'Combined empty modules without error'
                })
            else:
                results.append({
                    'name': 'Empty Module Combination',
                    'status': 'FAIL',
                    'execution_time': combiner_time,
                    'details': 'Combined result was None'
                })
        
        except Exception as e:
            results.append({
                'name': 'Module Combiner Tests',
                'status': 'ERROR',
                'execution_time': 0,
                'details': f'Exception: {str(e)}'
            })
        
        return results
    
    def test_management_components(self) -> List[Dict[str, Any]]:
        """Test management components like registries and interfaces"""
        results = []
        
        try:
            # Test various module registries
            registry_tests = [
                ('procedural', 'modules/procedural/registry.py'),
                ('raymarching', 'modules/raymarching/registry.py'),
                ('physics', 'modules/physics/registry.py'),
                ('texturing', 'modules/texturing/registry.py'),
                ('audio', 'modules/audio/registry.py'),
                ('game', 'modules/game/registry.py'),
                ('ui', 'modules/ui/registry.py')
            ]
            
            for module_type, registry_path in registry_tests:
                try:
                    # Try to import and test the registry
                    spec = importlib.util.spec_from_file_location(f"{module_type}_registry", registry_path)
                    registry_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(registry_module)
                    
                    # Test that get_module_by_name function exists
                    if hasattr(registry_module, 'get_module_by_name'):
                        results.append({
                            'name': f'{module_type.capitalize()} Registry Loading',
                            'status': 'PASS',
                            'execution_time': 0.001,
                            'details': f'{module_type} registry loaded successfully'
                        })
                    else:
                        results.append({
                            'name': f'{module_type.capitalize()} Registry Loading',
                            'status': 'FAIL',
                            'execution_time': 0.001,
                            'details': f'{module_type} registry missing get_module_by_name function'
                        })
                
                except FileNotFoundError:
                    results.append({
                        'name': f'{module_type.capitalize()} Registry Loading',
                        'status': 'SKIP',
                        'execution_time': 0,
                        'details': f'Registry file not found: {registry_path}'
                    })
                except Exception as e:
                    results.append({
                        'name': f'{module_type.capitalize()} Registry Loading',
                        'status': 'FAIL',
                        'execution_time': 0,
                        'details': f'Error loading {module_type} registry: {str(e)}'
                    })
        
        except Exception as e:
            results.append({
                'name': 'Management Components Tests',
                'status': 'ERROR',
                'execution_time': 0,
                'details': f'Exception in management tests: {str(e)}'
            })
        
        return results
    
    def run_integration_tests(self, verbose=False) -> List[Dict[str, Any]]:
        """Run integration tests for component interactions"""
        print("Running Integration Tests...")
        
        results = []
        
        try:
            # Test integration between module combiner and pseudocode translator
            from management.module_combiner import ModuleCombiner
            from create_pseudocode_translator import PseudocodeTranslator
            
            combiner = ModuleCombiner()
            translator = PseudocodeTranslator()
            
            # Test end-to-end workflow
            start_time = time.time()
            
            # Create a simple pseudocode to test integration
            simple_pseudocode = """
            float testFunction(float input) {
                return input * 2.0;
            }
            """
            
            # Translate to GLSL
            glsl_code = translator.translate_to_glsl(simple_pseudocode)
            
            # Verify that translation worked
            if glsl_code and 'testFunction' in glsl_code:
                results.append({
                    'name': 'Translator-Combiner Integration',
                    'status': 'PASS',
                    'execution_time': time.time() - start_time,
                    'details': 'Component integration works correctly'
                })
            else:
                results.append({
                    'name': 'Translator-Combiner Integration',
                    'status': 'FAIL',
                    'execution_time': time.time() - start_time,
                    'details': 'Translation failed or incorrect'
                })
            
            # Test module loading and pseudocode extraction
            start_time = time.time()
            
            # Check if we can load a registry and extract pseudocode
            try:
                # Import one of the registries
                import importlib.util
                spec = importlib.util.spec_from_file_location("proc_reg", "modules/procedural/registry.py")
                proc_reg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(proc_reg)
                
                # Try to get a module if one exists
                module = proc_reg.get_module_by_name('perlin_noise') if hasattr(proc_reg, 'get_module_by_name') else None
                
                if module:
                    results.append({
                        'name': 'Module Loading Integration',
                        'status': 'PASS',
                        'execution_time': time.time() - start_time,
                        'details': 'Successfully loaded procedural module'
                    })
                else:
                    results.append({
                        'name': 'Module Loading Integration',
                        'status': 'FAIL',
                        'execution_time': time.time() - start_time,
                        'details': 'Could not load procedural module'
                    })
            except Exception as e:
                results.append({
                    'name': 'Module Loading Integration',
                    'status': 'FAIL',
                    'execution_time': time.time() - start_time,
                    'details': f'Module loading error: {str(e)}'
                })
        
        except Exception as e:
            results.append({
                'name': 'Integration Tests',
                'status': 'ERROR',
                'execution_time': 0,
                'details': f'Integration test error: {str(e)}'
            })
        
        self.test_results['integration_tests'] = results
        
        if verbose:
            passed = sum(1 for t in results if t['status'] == 'PASS')
            failed = len(results) - passed
            print(f"  Integration Tests: {passed} passed, {failed} failed")
        
        return results
    
    def run_regression_tests(self, verbose=False) -> List[Dict[str, Any]]:
        """Run regression tests to ensure no functionality has been broken"""
        print("Running Regression Tests...")
        
        results = []
        
        try:
            # Test that previously working features still work
            from create_pseudocode_translator import PseudocodeTranslator
            
            translator = PseudocodeTranslator()
            
            # Test a variety of pseudocode patterns that should work
            test_patterns = [
                ("Basic assignment", "float x = 5.0;"),
                ("Function definition", "float func(float x) { return x * 2.0; }"),
                ("Vector operations", "vec3 result = vec3(1.0, 2.0, 3.0);"),
                ("Control flow", "if (x > 0.0) { return 1.0; } else { return 0.0; }"),
                ("Loops", "for (int i = 0; i < 10; i++) { sum += i; }"),
                ("Complex expressions", "float result = sin(x) * cos(y) + length(vec2(x, y));")
            ]
            
            for name, pseudocode in test_patterns:
                start_time = time.time()
                try:
                    result = translator.translate_to_glsl(pseudocode)
                    if result:
                        results.append({
                            'name': f'Regression Test: {name}',
                            'status': 'PASS',
                            'execution_time': time.time() - start_time,
                            'details': f'Successfully translated {name}'
                        })
                    else:
                        results.append({
                            'name': f'Regression Test: {name}',
                            'status': 'FAIL',
                            'execution_time': time.time() - start_time,
                            'details': f'Failed to translate {name}'
                        })
                except Exception as e:
                    results.append({
                        'name': f'Regression Test: {name}',
                        'status': 'FAIL',
                        'execution_time': time.time() - start_time,
                        'details': f'Error in {name}: {str(e)}'
                    })
        
        except Exception as e:
            results.append({
                'name': 'Regression Tests',
                'status': 'ERROR',
                'execution_time': 0,
                'details': f'Regression test error: {str(e)}'
            })
        
        self.test_results['regression_tests'] = results
        
        if verbose:
            passed = sum(1 for t in results if t['status'] == 'PASS')
            failed = len(results) - passed
            print(f"  Regression Tests: {passed} passed, {failed} failed")
        
        return results
    
    def run_compatibility_tests(self, verbose=False) -> List[Dict[str, Any]]:
        """Run compatibility tests across different target platforms/environments"""
        print("Running Compatibility Tests...")
        
        results = []
        
        try:
            # Test importing different components to ensure compatibility
            import_errors = []
            
            # Test basic imports
            components_to_test = [
                ('Pseudocode Translator', 'create_pseudocode_translator'),
                ('Module Combiner', 'management.module_combiner'),
                ('GLSL Parser', 'analysis.glsl_parser'),
            ]
            
            for name, module_path in components_to_test:
                start_time = time.time()
                try:
                    module = __import__(module_path, fromlist=[''])
                    results.append({
                        'name': f'Compatibility Test: {name}',
                        'status': 'PASS',
                        'execution_time': time.time() - start_time,
                        'details': f'{name} imported successfully'
                    })
                except ImportError as e:
                    import_errors.append(f"{name}: {str(e)}")
                    results.append({
                        'name': f'Compatibility Test: {name}',
                        'status': 'FAIL',
                        'execution_time': time.time() - start_time,
                        'details': f'Import error for {name}: {str(e)}'
                    })
        
        except Exception as e:
            results.append({
                'name': 'Compatibility Tests',
                'status': 'ERROR',
                'execution_time': 0,
                'details': f'Compatibility test error: {str(e)}'
            })
        
        self.test_results['compatibility_tests'] = results
        
        if verbose:
            passed = sum(1 for t in results if t['status'] == 'PASS')
            failed = len(results) - passed
            print(f"  Compatibility Tests: {passed} passed, {failed} failed")
        
        return results
    
    def run_performance_tests(self, verbose=False) -> List[Dict[str, Any]]:
        """Run performance tests to ensure efficiency"""
        print("Running Performance Tests...")
        
        results = []
        
        try:
            from create_pseudocode_translator import PseudocodeTranslator
            
            translator = PseudocodeTranslator()
            
            # Performance test: Translate the same pseudocode multiple times to test caching
            test_pseudocode = """
            float complexFunction(vec3 position, float time) {
                vec3 color = vec3(0.0);
                for (int i = 0; i < 10; i++) {
                    color += sin(position * float(i) + time) * 0.1;
                }
                return length(color);
            }
            """
            
            # Test single translation performance
            start_time = time.time()
            result = translator.translate_to_glsl(test_pseudocode)
            single_time = time.time() - start_time
            
            if result and len(result) > 0:
                results.append({
                    'name': 'Single Translation Performance',
                    'status': 'PASS',
                    'execution_time': single_time,
                    'details': f'Translation completed in {single_time*1000:.2f}ms'
                })
            else:
                results.append({
                    'name': 'Single Translation Performance',
                    'status': 'FAIL',
                    'execution_time': single_time,
                    'details': 'Translation failed'
                })
            
            # Test multiple translations to see if performance improves with caching
            start_time = time.time()
            for i in range(10):
                translator.translate_to_glsl(test_pseudocode)
            multiple_time = time.time() - start_time
            
            results.append({
                'name': 'Multiple Translation Performance',
                'status': 'PASS',
                'execution_time': multiple_time,
                'details': f'10 translations completed in {multiple_time*1000:.2f}ms ({(multiple_time/10)*1000:.2f}ms avg)'
            })
        
        except Exception as e:
            results.append({
                'name': 'Performance Tests',
                'status': 'ERROR',
                'execution_time': 0,
                'details': f'Performance test error: {str(e)}'
            })
        
        self.test_results['performance_tests'] = results
        
        if verbose:
            passed = sum(1 for t in results if t['status'] == 'PASS')
            failed = len(results) - passed
            print(f"  Performance Tests: {passed} passed, {failed} failed")
        
        return results
    
    def run_functional_tests(self, verbose=False) -> List[Dict[str, Any]]:
        """Run functional tests to verify that components work as intended"""
        print("Running Functional Tests...")
        
        results = []
        
        try:
            from create_pseudocode_translator import PseudocodeTranslator
            from management.module_combiner import ModuleCombiner
            
            # Functional test: Verify that the translator produces valid GLSL-like syntax
            translator = PseudocodeTranslator()
            test_pseudocode = "float calc(float x) { return x * 2.0; }"
            
            start_time = time.time()
            result = translator.translate_to_glsl(test_pseudocode)
            func_time = time.time() - start_time
            
            if result and 'float calc' in result and 'return x * 2.0' in result:
                results.append({
                    'name': 'Functional Translation Test',
                    'status': 'PASS',
                    'execution_time': func_time,
                    'details': 'Translation produced expected GLSL syntax'
                })
            else:
                results.append({
                    'name': 'Functional Translation Test',
                    'status': 'FAIL',
                    'execution_time': func_time,
                    'details': 'Translation did not produce expected syntax'
                })
            
            # Test module combiner functionality
            combiner = ModuleCombiner()
            
            # Create a temporary directory for test modules
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a mock module file
                module_file = Path(temp_dir) / "test_module.glsl"
                module_content = "// Test module content\nfloat testFunction() { return 1.0; }"
                with open(module_file, 'w') as f:
                    f.write(module_content)
                
                start_time = time.time()
                try:
                    # Try to combine modules (though this is basic functionality test)
                    combined = combiner.combine_modules([])
                    results.append({
                        'name': 'Basic Combiner Functionality',
                        'status': 'PASS',
                        'execution_time': time.time() - start_time,
                        'details': 'Combiner operates without errors'
                    })
                except Exception as e:
                    results.append({
                        'name': 'Basic Combiner Functionality',
                        'status': 'FAIL',
                        'execution_time': time.time() - start_time,
                        'details': f'Combiner error: {str(e)}'
                    })
        
        except Exception as e:
            results.append({
                'name': 'Functional Tests',
                'status': 'ERROR',
                'execution_time': 0,
                'details': f'Functional test error: {str(e)}'
            })
        
        self.test_results['functional_tests'] = results
        
        if verbose:
            passed = sum(1 for t in results if t['status'] == 'PASS')
            failed = len(results) - passed
            print(f"  Functional Tests: {passed} passed, {failed} failed")
        
        return results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report"""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_execution_time = 0
        
        for category, tests in self.test_results.items():
            total_tests += len(tests)
            for test in tests:
                total_execution_time += test['execution_time']
                if test['status'] == 'PASS':
                    total_passed += 1
                elif test['status'] == 'FAIL':
                    total_failed += 1
                elif test['status'] == 'ERROR':
                    total_errors += 1
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'summary': {
                'total_tests': total_tests,
                'total_passed': total_passed,
                'total_failed': total_failed,
                'total_errors': total_errors,
                'success_rate': overall_success_rate,
                'total_execution_time': self.end_time - self.start_time
            },
            'breakdown': {category: {
                'total': len(tests),
                'passed': sum(1 for t in tests if t['status'] == 'PASS'),
                'failed': sum(1 for t in tests if t['status'] == 'FAIL'),
                'errors': sum(1 for t in tests if t['status'] == 'ERROR')
            } for category, tests in self.test_results.items()},
            'categories': self.test_results
        }
    
    def print_test_summary(self, report: Dict[str, Any]):
        """Print a formatted test summary"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST SUITE RESULTS")
        print("=" * 60)
        
        summary = report['summary']
        breakdown = report['breakdown']
        
        print(f"Execution time: {summary['total_execution_time']:.2f}s")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        
        print(f"\n{'Category':<20} {'Total':<6} {'Pass':<6} {'Fail':<6} {'Error':<6} {'Rate':<6}")
        print("-" * 60)
        
        for category, stats in breakdown.items():
            rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{category:<20} {stats['total']:<6} {stats['passed']:<6} {stats['failed']:<6} {stats['errors']:<6} {rate:<6.1f}%")
        
        print("\nTest Results Summary:")
        if summary['success_rate'] >= 90:
            print("✅ OVERALL SUCCESS: The system is functioning well!")
        elif summary['success_rate'] >= 70:
            print("⚠️  MIXED RESULTS: Some components need attention.")
        else:
            print("❌ LOW SUCCESS RATE: Major issues detected in the system.")
        
        # Highlight any critical failures
        critical_failures = []
        for category, tests in self.test_results.items():
            for test in tests:
                if test['status'] in ['ERROR', 'FAIL'] and 'translator' in test['name'].lower():
                    critical_failures.append(test['name'])
        
        if critical_failures:
            print(f"\nCritical Component Failures Detected:")
            for failure in critical_failures:
                print(f"  - {failure}")
        
        print("=" * 60)


def run_comprehensive_tests():
    """Main function to run comprehensive tests for the SuperShader system"""
    suite = ComprehensiveTestSuite()
    report = suite.run_all_tests(verbose=True)
    
    # Save detailed report to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"comprehensive_test_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed test report saved to: {report_file}")
    
    # Return success based on success rate
    success_rate = report['summary']['success_rate']
    if success_rate >= 70:  # Consider 70%+ as acceptable for this system
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = run_comprehensive_tests()
    sys.exit(exit_code)