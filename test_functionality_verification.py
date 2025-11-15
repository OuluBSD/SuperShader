#!/usr/bin/env python3
"""
Functionality Verification for SuperShader Modules
Tests the actual functionality and expected behavior of modules
"""

import sys
import os
import math
from typing import Dict, Any, Optional

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from create_pseudocode_translator import PseudocodeTranslator
from modules.procedural.registry import get_module_by_name as get_procedural_module
from modules.raymarching.registry import get_module_by_name as get_raymarching_module
from modules.physics.registry import get_module_by_name as get_physics_module
from modules.texturing.registry import get_module_by_name as get_texturing_module
from modules.audio.registry import get_module_by_name as get_audio_module
from modules.game.registry import get_module_by_name as get_game_module
from modules.ui.registry import get_module_by_name as get_ui_module


class FunctionalityVerifier:
    """Verifies the actual functionality of modules"""

    def __init__(self):
        self.translator = PseudocodeTranslator()

    def _get_branches(self, module):
        """Helper to get branches from either top level or nested in metadata"""
        if not module:
            return {}
        
        # Check if branches are at top level (procedural modules)
        if 'branches' in module:
            return module['branches']
        
        # Check if branches are in metadata (other modules)
        if 'metadata' in module and 'branches' in module['metadata']:
            return module['metadata']['branches']
        
        # No branches found
        return {}

    def verify_procedural_functionality(self) -> Dict[str, Any]:
        """Verify functionality of procedural modules"""
        results = {'passed': 0, 'failed': 0, 'tests': []}

        # Test noise module functionality
        try:
            noise_module = get_procedural_module('noise_functions_branching')
            if noise_module:
                # Verify that the module has expected functionality
                branches = self._get_branches(noise_module)

                if 'noise_algorithm' in branches and 'octave_mode' in branches:
                    # Check for expected noise types
                    noise_algorithms = branches['noise_algorithm']
                    expected_algorithms = ['perlin', 'simplex', 'value']

                    for algo in expected_algorithms:
                        if algo not in noise_algorithms:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'noise_functions_branching',
                                'test': f'noise_algorithm_{algo}',
                                'status': 'FAIL',
                                'error': f'Missing {algo} algorithm'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'noise_functions_branching',
                                'test': f'noise_algorithm_{algo}',
                                'status': 'PASS'
                            })

                    # Verify conflicts are properly defined
                    perlin_conflicts = noise_algorithms['perlin'].get('conflicts', [])
                    if 'simplex' in perlin_conflicts and 'value' in perlin_conflicts:
                        results['passed'] += 1
                        results['tests'].append({
                            'module': 'noise_functions_branching',
                            'test': 'perlin_conflicts_verification',
                            'status': 'PASS'
                        })
                    else:
                        results['failed'] += 1
                        results['tests'].append({
                            'module': 'noise_functions_branching',
                            'test': 'perlin_conflicts_verification',
                            'status': 'FAIL',
                            'error': f'Perlin conflicts not properly defined: {perlin_conflicts}'
                        })

                    # Check octave modes as well
                    octave_modes = branches['octave_mode']
                    expected_modes = ['fbm', 'ridged', 'turbulence']
                    for mode in expected_modes:
                        if mode not in octave_modes:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'noise_functions_branching',
                                'test': f'octave_mode_{mode}',
                                'status': 'FAIL',
                                'error': f'Missing {mode} octave mode'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'noise_functions_branching',
                                'test': f'octave_mode_{mode}',
                                'status': 'PASS'
                            })
                else:
                    results['failed'] += 1
                    results['tests'].append({
                        'module': 'noise_functions_branching',
                        'test': 'branch_structure',
                        'status': 'FAIL',
                        'error': 'Missing expected branch structure'
                    })
            else:
                results['failed'] += 1
                results['tests'].append({
                    'module': 'noise_functions_branching',
                    'test': 'module_load',
                    'status': 'FAIL',
                    'error': 'Module not found'
                })
        except Exception as e:
            results['failed'] += 1
            results['tests'].append({
                'module': 'noise_functions_branching',
                'test': 'general_verification',
                'status': 'FAIL',
                'error': str(e)
            })

        return results

    def verify_raymarching_functionality(self) -> Dict[str, Any]:
        """Verify functionality of raymarching modules"""
        results = {'passed': 0, 'failed': 0, 'tests': []}

        try:
            raymarching_module = get_raymarching_module('raymarching_advanced_branching')
            if raymarching_module:
                # Verify that the module has expected functionality
                branches = self._get_branches(raymarching_module)

                if 'algorithm_type' in branches and 'normal_calculation' in branches:
                    # Check for expected algorithm types
                    algorithms = branches['algorithm_type']
                    expected_algorithms = ['basic', 'adaptive', 'cone', 'multi']

                    for algo in expected_algorithms:
                        if algo not in algorithms:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'raymarching_advanced_branching',
                                'test': f'algorithm_type_{algo}',
                                'status': 'FAIL',
                                'error': f'Missing {algo} algorithm type'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'raymarching_advanced_branching',
                                'test': f'algorithm_type_{algo}',
                                'status': 'PASS'
                            })

                    # Verify conflicts are properly defined
                    basic_conflicts = algorithms['basic'].get('conflicts', [])
                    if 'adaptive' in basic_conflicts and 'cone' in basic_conflicts and 'multi' in basic_conflicts:
                        results['passed'] += 1
                        results['tests'].append({
                            'module': 'raymarching_advanced_branching',
                            'test': 'basic_conflicts_verification',
                            'status': 'PASS'
                        })
                    else:
                        results['failed'] += 1
                        results['tests'].append({
                            'module': 'raymarching_advanced_branching',
                            'test': 'basic_conflicts_verification',
                            'status': 'FAIL',
                            'error': f'Basic algorithm conflicts not properly defined: {basic_conflicts}'
                        })

                    # Check other branch types as well
                    normal_calculations = branches['normal_calculation']
                    expected_calculations = ['standard', 'analytical', 'hybrid']
                    for calc in expected_calculations:
                        if calc not in normal_calculations:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'raymarching_advanced_branching',
                                'test': f'normal_calculation_{calc}',
                                'status': 'FAIL',
                                'error': f'Missing {calc} normal calculation'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'raymarching_advanced_branching',
                                'test': f'normal_calculation_{calc}',
                                'status': 'PASS'
                            })
                else:
                    results['failed'] += 1
                    results['tests'].append({
                        'module': 'raymarching_advanced_branching',
                        'test': 'branch_structure',
                        'status': 'FAIL',
                        'error': 'Missing expected branch structure'
                    })
            else:
                results['failed'] += 1
                results['tests'].append({
                    'module': 'raymarching_advanced_branching',
                    'test': 'module_load',
                    'status': 'FAIL',
                    'error': 'Module not found'
                })
        except Exception as e:
            results['failed'] += 1
            results['tests'].append({
                'module': 'raymarching_advanced_branching',
                'test': 'general_verification',
                'status': 'FAIL',
                'error': str(e)
            })

        return results

    def verify_physics_functionality(self) -> Dict[str, Any]:
        """Verify functionality of physics modules"""
        results = {'passed': 0, 'failed': 0, 'tests': []}

        try:
            physics_module = get_physics_module('physics_advanced_branching')
            if physics_module:
                # Verify that the module has expected functionality
                branches = self._get_branches(physics_module)

                if 'integration_method' in branches and 'collision_handling' in branches:
                    # Check for expected integration methods
                    integration_methods = branches['integration_method']
                    expected_methods = ['euler', 'verlet', 'rk4', 'semi_implicit']

                    for method in expected_methods:
                        if method not in integration_methods:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'physics_advanced_branching',
                                'test': f'integration_method_{method}',
                                'status': 'FAIL',
                                'error': f'Missing {method} integration method'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'physics_advanced_branching',
                                'test': f'integration_method_{method}',
                                'status': 'PASS'
                            })

                    # Verify conflicts are properly defined
                    euler_conflicts = integration_methods['euler'].get('conflicts', [])
                    if 'verlet' in euler_conflicts and 'rk4' in euler_conflicts and 'semi_implicit' in euler_conflicts:
                        results['passed'] += 1
                        results['tests'].append({
                            'module': 'physics_advanced_branching',
                            'test': 'euler_conflicts_verification',
                            'status': 'PASS'
                        })
                    else:
                        results['failed'] += 1
                        results['tests'].append({
                            'module': 'physics_advanced_branching',
                            'test': 'euler_conflicts_verification',
                            'status': 'FAIL',
                            'error': f'Euler integration conflicts not properly defined: {euler_conflicts}'
                        })

                    # Check other branch types as well
                    collision_handlings = branches['collision_handling']
                    expected_handlings = ['simple', 'constraint_based', 'impulse']
                    for handling in expected_handlings:
                        if handling not in collision_handlings:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'physics_advanced_branching',
                                'test': f'collision_handling_{handling}',
                                'status': 'FAIL',
                                'error': f'Missing {handling} collision handling'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'physics_advanced_branching',
                                'test': f'collision_handling_{handling}',
                                'status': 'PASS'
                            })
                else:
                    results['failed'] += 1
                    results['tests'].append({
                        'module': 'physics_advanced_branching',
                        'test': 'branch_structure',
                        'status': 'FAIL',
                        'error': 'Missing expected branch structure'
                    })
            else:
                results['failed'] += 1
                results['tests'].append({
                    'module': 'physics_advanced_branching',
                    'test': 'module_load',
                    'status': 'FAIL',
                    'error': 'Module not found'
                })
        except Exception as e:
            results['failed'] += 1
            results['tests'].append({
                'module': 'physics_advanced_branching',
                'test': 'general_verification',
                'status': 'FAIL',
                'error': str(e)
            })

        return results

    def verify_texturing_functionality(self) -> Dict[str, Any]:
        """Verify functionality of texturing modules"""
        results = {'passed': 0, 'failed': 0, 'tests': []}

        try:
            texturing_module = get_texturing_module('texturing_advanced_branching')
            if texturing_module:
                # Verify that the module has expected functionality
                branches = self._get_branches(texturing_module)

                if 'uv_mapping_method' in branches and 'texture_filtering' in branches:
                    # Check for expected mapping methods
                    mapping_methods = branches['uv_mapping_method']
                    expected_methods = ['planar', 'spherical', 'cylindrical', 'triplanar']

                    for method in expected_methods:
                        if method not in mapping_methods:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'texturing_advanced_branching',
                                'test': f'uv_mapping_method_{method}',
                                'status': 'FAIL',
                                'error': f'Missing {method} UV mapping method'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'texturing_advanced_branching',
                                'test': f'uv_mapping_method_{method}',
                                'status': 'PASS'
                            })

                    # Verify conflicts are properly defined
                    planar_conflicts = mapping_methods['planar'].get('conflicts', [])
                    if 'spherical' in planar_conflicts and 'cylindrical' in planar_conflicts and 'triplanar' in planar_conflicts:
                        results['passed'] += 1
                        results['tests'].append({
                            'module': 'texturing_advanced_branching',
                            'test': 'planar_conflicts_verification',
                            'status': 'PASS'
                        })
                    else:
                        results['failed'] += 1
                        results['tests'].append({
                            'module': 'texturing_advanced_branching',
                            'test': 'planar_conflicts_verification',
                            'status': 'FAIL',
                            'error': f'Planar mapping conflicts not properly defined: {planar_conflicts}'
                        })

                    # Check other branch types as well
                    filtering_types = branches['texture_filtering']
                    expected_filterings = ['nearest', 'bilinear', 'trilinear', 'anisotropic']
                    for filtering in expected_filterings:
                        if filtering not in filtering_types:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'texturing_advanced_branching',
                                'test': f'texture_filtering_{filtering}',
                                'status': 'FAIL',
                                'error': f'Missing {filtering} texture filtering'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'texturing_advanced_branching',
                                'test': f'texture_filtering_{filtering}',
                                'status': 'PASS'
                            })
                else:
                    results['failed'] += 1
                    results['tests'].append({
                        'module': 'texturing_advanced_branching',
                        'test': 'branch_structure',
                        'status': 'FAIL',
                        'error': 'Missing expected branch structure'
                    })
            else:
                results['failed'] += 1
                results['tests'].append({
                    'module': 'texturing_advanced_branching',
                    'test': 'module_load',
                    'status': 'FAIL',
                    'error': 'Module not found'
                })
        except Exception as e:
            results['failed'] += 1
            results['tests'].append({
                'module': 'texturing_advanced_branching',
                'test': 'general_verification',
                'status': 'FAIL',
                'error': str(e)
            })

        return results

    def verify_audio_functionality(self) -> Dict[str, Any]:
        """Verify functionality of audio modules"""
        results = {'passed': 0, 'failed': 0, 'tests': []}

        try:
            audio_module = get_audio_module('audio_processing_branching')
            if audio_module:
                # Verify that the module has expected functionality
                branches = self._get_branches(audio_module)

                if 'spectrum_analysis' in branches and 'synthesis_method' in branches:
                    # Check for expected analysis methods
                    analysis_methods = branches['spectrum_analysis']
                    expected_methods = ['fourier', 'wavelet', 'autocorrelation']

                    for method in expected_methods:
                        if method not in analysis_methods:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'audio_processing_branching',
                                'test': f'spectrum_analysis_{method}',
                                'status': 'FAIL',
                                'error': f'Missing {method} spectrum analysis method'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'audio_processing_branching',
                                'test': f'spectrum_analysis_{method}',
                                'status': 'PASS'
                            })

                    # Verify conflicts are properly defined
                    fourier_conflicts = analysis_methods['fourier'].get('conflicts', [])
                    if 'wavelet' in fourier_conflicts and 'autocorrelation' in fourier_conflicts:
                        results['passed'] += 1
                        results['tests'].append({
                            'module': 'audio_processing_branching',
                            'test': 'fourier_conflicts_verification',
                            'status': 'PASS'
                        })
                    else:
                        results['failed'] += 1
                        results['tests'].append({
                            'module': 'audio_processing_branching',
                            'test': 'fourier_conflicts_verification',
                            'status': 'FAIL',
                            'error': f'Fourier analysis conflicts not properly defined: {fourier_conflicts}'
                        })

                    # Check other branch types as well
                    synthesis_methods = branches['synthesis_method']
                    expected_syntheses = ['additive', 'subtractive', 'fm', 'wavetable']
                    for synthesis in expected_syntheses:
                        if synthesis not in synthesis_methods:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'audio_processing_branching',
                                'test': f'synthesis_method_{synthesis}',
                                'status': 'FAIL',
                                'error': f'Missing {synthesis} synthesis method'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'audio_processing_branching',
                                'test': f'synthesis_method_{synthesis}',
                                'status': 'PASS'
                            })
                else:
                    results['failed'] += 1
                    results['tests'].append({
                        'module': 'audio_processing_branching',
                        'test': 'branch_structure',
                        'status': 'FAIL',
                        'error': 'Missing expected branch structure'
                    })
            else:
                results['failed'] += 1
                results['tests'].append({
                    'module': 'audio_processing_branching',
                    'test': 'module_load',
                    'status': 'FAIL',
                    'error': 'Module not found'
                })
        except Exception as e:
            results['failed'] += 1
            results['tests'].append({
                'module': 'audio_processing_branching',
                'test': 'general_verification',
                'status': 'FAIL',
                'error': str(e)
            })

        return results

    def verify_game_functionality(self) -> Dict[str, Any]:
        """Verify functionality of game modules"""
        results = {'passed': 0, 'failed': 0, 'tests': []}

        try:
            game_module = get_game_module('game_advanced_branching')
            if game_module:
                # Verify that the module has expected functionality
                branches = self._get_branches(game_module)

                if 'hud_style' in branches and 'interaction_mode' in branches:
                    # Check for expected HUD styles
                    hud_styles = branches['hud_style']
                    expected_styles = ['minimalist', 'detailed', 'themed']

                    for style in expected_styles:
                        if style not in hud_styles:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'game_advanced_branching',
                                'test': f'hud_style_{style}',
                                'status': 'FAIL',
                                'error': f'Missing {style} HUD style'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'game_advanced_branching',
                                'test': f'hud_style_{style}',
                                'status': 'PASS'
                            })

                    # Verify conflicts are properly defined
                    minimalist_conflicts = hud_styles['minimalist'].get('conflicts', [])
                    if 'detailed' in minimalist_conflicts and 'themed' in minimalist_conflicts:
                        results['passed'] += 1
                        results['tests'].append({
                            'module': 'game_advanced_branching',
                            'test': 'minimalist_conflicts_verification',
                            'status': 'PASS'
                        })
                    else:
                        results['failed'] += 1
                        results['tests'].append({
                            'module': 'game_advanced_branching',
                            'test': 'minimalist_conflicts_verification',
                            'status': 'FAIL',
                            'error': f'Minimalist HUD conflicts not properly defined: {minimalist_conflicts}'
                        })

                    # Check other branch types as well
                    interaction_modes = branches['interaction_mode']
                    expected_modes = ['click', 'hover', 'touch']
                    for mode in expected_modes:
                        if mode not in interaction_modes:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'game_advanced_branching',
                                'test': f'interaction_mode_{mode}',
                                'status': 'FAIL',
                                'error': f'Missing {mode} interaction mode'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'game_advanced_branching',
                                'test': f'interaction_mode_{mode}',
                                'status': 'PASS'
                            })
                else:
                    results['failed'] += 1
                    results['tests'].append({
                        'module': 'game_advanced_branching',
                        'test': 'branch_structure',
                        'status': 'FAIL',
                        'error': 'Missing expected branch structure'
                    })
            else:
                results['failed'] += 1
                results['tests'].append({
                    'module': 'game_advanced_branching',
                    'test': 'module_load',
                    'status': 'FAIL',
                    'error': 'Module not found'
                })
        except Exception as e:
            results['failed'] += 1
            results['tests'].append({
                'module': 'game_advanced_branching',
                'test': 'general_verification',
                'status': 'FAIL',
                'error': str(e)
            })

        return results

    def verify_ui_functionality(self) -> Dict[str, Any]:
        """Verify functionality of UI modules"""
        results = {'passed': 0, 'failed': 0, 'tests': []}

        try:
            ui_module = get_ui_module('ui_advanced_branching')
            if ui_module:
                # Verify that the module has expected functionality
                branches = self._get_branches(ui_module)

                if 'widget_style' in branches and 'animation_style' in branches:
                    # Check for expected widget styles
                    widget_styles = branches['widget_style']
                    expected_styles = ['flat', 'material', 'neumorphic', 'glassmorphism']

                    for style in expected_styles:
                        if style not in widget_styles:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'ui_advanced_branching',
                                'test': f'widget_style_{style}',
                                'status': 'FAIL',
                                'error': f'Missing {style} widget style'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'ui_advanced_branching',
                                'test': f'widget_style_{style}',
                                'status': 'PASS'
                            })

                    # Verify conflicts are properly defined
                    flat_conflicts = widget_styles['flat'].get('conflicts', [])
                    if 'material' in flat_conflicts and 'neumorphic' in flat_conflicts and 'glassmorphism' in flat_conflicts:
                        results['passed'] += 1
                        results['tests'].append({
                            'module': 'ui_advanced_branching',
                            'test': 'flat_conflicts_verification',
                            'status': 'PASS'
                        })
                    else:
                        results['failed'] += 1
                        results['tests'].append({
                            'module': 'ui_advanced_branching',
                            'test': 'flat_conflicts_verification',
                            'status': 'FAIL',
                            'error': f'Flat UI conflicts not properly defined: {flat_conflicts}'
                        })

                    # Check other branch types as well
                    animation_styles = branches['animation_style']
                    expected_animations = ['static', 'subtle', 'dynamic']
                    for animation in expected_animations:
                        if animation not in animation_styles:
                            results['failed'] += 1
                            results['tests'].append({
                                'module': 'ui_advanced_branching',
                                'test': f'animation_style_{animation}',
                                'status': 'FAIL',
                                'error': f'Missing {animation} animation style'
                            })
                        else:
                            results['passed'] += 1
                            results['tests'].append({
                                'module': 'ui_advanced_branching',
                                'test': f'animation_style_{animation}',
                                'status': 'PASS'
                            })
                else:
                    results['failed'] += 1
                    results['tests'].append({
                        'module': 'ui_advanced_branching',
                        'test': 'branch_structure',
                        'status': 'FAIL',
                        'error': 'Missing expected branch structure'
                    })
            else:
                results['failed'] += 1
                results['tests'].append({
                    'module': 'ui_advanced_branching',
                    'test': 'module_load',
                    'status': 'FAIL',
                    'error': 'Module not found'
                })
        except Exception as e:
            results['failed'] += 1
            results['tests'].append({
                'module': 'ui_advanced_branching',
                'test': 'general_verification',
                'status': 'FAIL',
                'error': str(e)
            })

        return results

    def run_all_functionality_verifications(self) -> Dict[str, Dict[str, Any]]:
        """Run all functionality verification tests"""
        print("Starting functionality verification for all modules...")

        results = {
            'procedural': self.verify_procedural_functionality(),
            'raymarching': self.verify_raymarching_functionality(),
            'physics': self.verify_physics_functionality(),
            'texturing': self.verify_texturing_functionality(),
            'audio': self.verify_audio_functionality(),
            'game': self.verify_game_functionality(),
            'ui': self.verify_ui_functionality()
        }

        # Print summary
        self._print_functionality_summary(results)

        return results

    def _print_functionality_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print a summary of functionality verification results"""
        print("\n" + "="*60)
        print("FUNCTIONALITY VERIFICATION SUMMARY")
        print("="*60)

        total_passed = 0
        total_failed = 0

        for module_type, result in results.items():
            passed = result['passed']
            failed = result['failed']
            total = passed + failed

            total_passed += passed
            total_failed += failed

            print(f"\n{module_type.upper()} FUNCTIONALITY:")
            print(f"  Total Tests: {total}")
            print(f"  Passed: {passed}")
            print(f"  Failed: {failed}")
            print(f"  Success Rate: {passed/total*100:.1f}%" if total > 0 else "  Success Rate: 0%")

        print("\n" + "="*60)
        print("OVERALL FUNCTIONALITY SUMMARY:")
        print(f"  Total Tests: {total_passed + total_failed}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_failed}")
        print(f"  Success Rate: {total_passed/(total_passed + total_failed)*100:.1f}%" if (total_passed + total_failed) > 0 else "  Success Rate: 0%")
        print("="*60)


def main():
    """Main entry point for functionality verification"""
    print("Initializing SuperShader Functionality Verification...")

    verifier = FunctionalityVerifier()
    results = verifier.run_all_functionality_verifications()

    # Return success code based on test results
    total_tests = sum([r['passed'] + r['failed'] for r in results.values()])
    failed_tests = sum([r['failed'] for r in results.values()])

    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)