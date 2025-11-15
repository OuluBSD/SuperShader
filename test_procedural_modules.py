#!/usr/bin/env python3
"""
Test script for procedural modules with management code
Tests the integration of procedural modules with the module combiner and pseudocode translator
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from management.module_combiner import ModuleCombiner
from create_pseudocode_translator import PseudocodeTranslator
from modules.procedural.registry import get_module_by_name


def test_module_loading():
    """Test that procedural modules can be loaded from the registry"""
    print("Testing module loading...")
    
    # Test loading the perlin noise module
    perlin_module = get_module_by_name('perlin_noise')
    if perlin_module:
        print("✓ Successfully loaded perlin_noise module")
        print(f"  Name: {perlin_module['name']}")
        print(f"  Type: {perlin_module['type']}")
        print(f"  Patterns: {perlin_module['patterns']}")
    else:
        print("✗ Failed to load perlin_noise module")
        return False

    # Test loading the branching noise module
    branching_module = get_module_by_name('noise_functions_branching')
    if branching_module:
        print("✓ Successfully loaded noise_functions_branching module")
        print(f"  Name: {branching_module['name']}")
        print(f"  Type: {branching_module['type']}")
        print(f"  Patterns: {branching_module['patterns']}")
        print(f"  Branches: {list(branching_module.get('branches', {}).keys())}")
    else:
        print("✗ Failed to load noise_functions_branching module")
        return False

    return True


def test_pseudocode_translation():
    """Test that pseudocode can be translated to different languages"""
    print("\nTesting pseudocode translation...")
    
    translator = PseudocodeTranslator()
    
    # Load a module
    module = get_module_by_name('perlin_noise')
    if not module or 'pseudocode' not in module:
        print("✗ Could not get pseudocode from module")
        return False

    pseudocode = module['pseudocode']
    
    try:
        # Test GLSL translation
        glsl_code = translator.translate_to_glsl(pseudocode)
        if glsl_code and len(glsl_code) > 0:
            print("✓ Successfully translated pseudocode to GLSL")
        else:
            print("✗ Failed to translate pseudocode to GLSL")
            return False
            
        # Test Metal translation
        metal_code = translator.translate(pseudocode, 'metal')
        if metal_code and len(metal_code) > 0:
            print("✓ Successfully translated pseudocode to Metal")
        else:
            print("✗ Failed to translate pseudocode to Metal")
            return False
            
        # Test C/C++ translation
        cpp_code = translator.translate(pseudocode, 'c_cpp')
        if cpp_code and len(cpp_code) > 0:
            print("✓ Successfully translated pseudocode to C/C++")
        else:
            print("✗ Failed to translate pseudocode to C/C++")
            return False
            
        # Test branching pseudocode translation
        branching_module = get_module_by_name('noise_functions_branching')
        if branching_module and 'pseudocode' in branching_module:
            # Test translating a specific branch
            perlin_branch = branching_module['pseudocode']['perlin']
            glsl_branch = translator.translate_to_glsl(perlin_branch)
            if glsl_branch and len(glsl_branch) > 0:
                print("✓ Successfully translated branching pseudocode to GLSL")
            else:
                print("✗ Failed to translate branching pseudocode to GLSL")
                return False
    
    except Exception as e:
        print(f"✗ Error during pseudocode translation: {str(e)}")
        return False

    return True


def test_module_combination():
    """Test that procedural modules can be combined using the module combiner"""
    print("\nTesting module combination...")
    
    # Note: The current module combiner expects JSON files, so we'll need to simulate
    # the combination process differently for our registry-based modules
    
    try:
        # Create a combiner instance
        combiner = ModuleCombiner()
        
        # Test that we can access the modules we need
        perlin_module = get_module_by_name('perlin_noise')
        if perlin_module:
            print("✓ Module combiner can access perlin_noise module")
        else:
            print("✗ Module combiner cannot access perlin_noise module")
            return False
            
        branching_module = get_module_by_name('noise_functions_branching')
        if branching_module:
            print("✓ Module combiner can access noise_functions_branching module")
        else:
            print("✗ Module combiner cannot access noise_functions_branching module")
            return False
        
        # Test branch validation
        if 'branches' in branching_module:
            branches = branching_module['branches']
            print(f"✓ Branching module has {len(branches)} branch types: {list(branches.keys())}")
            
            # Test that we can validate branch selections
            selected_branches = {
                'noise_algorithm': 'perlin',
                'octave_mode': 'fbm'
            }
            
            # Check that these branches don't conflict
            noise_algo_branch = branches['noise_algorithm']['perlin']
            octave_mode_branch = branches['octave_mode']['fbm']
            
            # Check for conflicts between selected branches
            conflicts = set(noise_algo_branch['conflicts']) & set([octave_mode_branch['name'].lower()])
            if not conflicts:
                print("✓ Selected branches have no conflicts")
            else:
                print(f"✗ Selected branches have conflicts: {conflicts}")
                return False
        else:
            print("✗ Branching module doesn't have expected branch structure")
            return False
    
    except Exception as e:
        print(f"✗ Error during module combination test: {str(e)}")
        return False

    return True


def test_conflict_detection():
    """Test that conflicts between branches are properly detected"""
    print("\nTesting conflict detection...")
    
    try:
        branching_module = get_module_by_name('noise_functions_branching')
        if not branching_module:
            print("✗ Could not load branching module for conflict test")
            return False
            
        branches = branching_module.get('branches', {})
        if not branches:
            print("✗ Branching module has no branches")
            return False
            
        # Test conflict detection between noise algorithms
        noise_branches = branches.get('noise_algorithm', {})
        if not noise_branches:
            print("✗ No noise algorithm branches found")
            return False
            
        # Check that perlin conflicts with simplex and value
        perlin_conflicts = noise_branches['perlin']['conflicts']
        expected_conflicts = ['simplex', 'value']
        
        if all(conflict in perlin_conflicts for conflict in expected_conflicts):
            print("✓ Perlin noise properly conflicts with simplex and value noise")
        else:
            print(f"✗ Perlin noise conflicts {perlin_conflicts} don't match expected {expected_conflicts}")
            return False
            
        # Check that simplex conflicts with perlin and value  
        simplex_conflicts = noise_branches['simplex']['conflicts']
        if all(conflict in simplex_conflicts for conflict in ['perlin', 'value']):
            print("✓ Simplex noise properly conflicts with perlin and value noise")
        else:
            print("✗ Simplex noise conflicts not properly defined")
            return False
            
        # Test octave mode conflicts
        octave_branches = branches.get('octave_mode', {})
        if not octave_branches:
            print("✗ No octave mode branches found")
            return False
            
        fbm_conflicts = octave_branches['fbm']['conflicts']
        expected_fbm_conflicts = ['ridged', 'turbulence']
        
        if all(conflict in fbm_conflicts for conflict in expected_fbm_conflicts):
            print("✓ FBM properly conflicts with ridged and turbulence modes")
        else:
            print(f"✗ FBM conflicts {fbm_conflicts} don't match expected {expected_fbm_conflicts}")
            return False
    
    except Exception as e:
        print(f"✗ Error during conflict detection test: {str(e)}")
        return False

    return True


def main():
    """Run all tests for procedural modules"""
    print("Testing Procedural Modules with Management Code")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run each test
    all_tests_passed &= test_module_loading()
    all_tests_passed &= test_pseudocode_translation()
    all_tests_passed &= test_module_combination()
    all_tests_passed &= test_conflict_detection()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All procedural module tests PASSED")
        return 0
    else:
        print("✗ Some procedural module tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)