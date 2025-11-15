#!/usr/bin/env python3
"""
Test script for raymarching modules with management code
Tests the integration of raymarching modules with the module combiner and pseudocode translator
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from management.module_combiner import ModuleCombiner
from create_pseudocode_translator import PseudocodeTranslator
from modules.raymarching.registry import get_module_by_name


def test_raymarching_module_loading():
    """Test that raymarching modules can be loaded from the registry"""
    print("Testing raymarching module loading...")
    
    # Test loading the core raymarching module
    core_module = get_module_by_name('raymarching_core')
    if core_module:
        print("✓ Successfully loaded raymarching_core module")
        print(f"  Name: {core_module['name']}")
        print(f"  Type: {core_module['metadata']['type']}")
        print(f"  Patterns: {core_module['metadata']['patterns']}")
    else:
        print("✗ Failed to load raymarching_core module")
        return False

    # Test loading the advanced branching raymarching module
    branching_module = get_module_by_name('raymarching_advanced_branching')
    if branching_module:
        print("✓ Successfully loaded raymarching_advanced_branching module")
        print(f"  Name: {branching_module['name']}")
        print(f"  Type: {branching_module['metadata']['type']}")
        print(f"  Patterns: {branching_module['metadata']['patterns']}")
        print(f"  Branches: {list(branching_module['metadata'].get('branches', {}).keys())}")
    else:
        print("✗ Failed to load raymarching_advanced_branching module")
        return False

    return True


def test_raymarching_pseudocode_translation():
    """Test that raymarching pseudocode can be translated to different languages"""
    print("\nTesting raymarching pseudocode translation...")
    
    translator = PseudocodeTranslator()
    
    # Load the branching raymarching module
    module = get_module_by_name('raymarching_advanced_branching')
    if not module or 'pseudocode' in module:
        print("✓ Successfully got branching raymarching module with pseudocode")
    else:
        print("✗ Could not get pseudocode from raymarching module")
        return False

    # Test translating one of the pseudocode branches
    pseudocodes = module['pseudocode']
    if isinstance(pseudocodes, dict) and 'basic' in pseudocodes:
        basic_pseudocode = pseudocodes['basic']
        
        try:
            # Test GLSL translation
            glsl_code = translator.translate_to_glsl(basic_pseudocode)
            if glsl_code and len(glsl_code) > 0:
                print("✓ Successfully translated basic raymarching pseudocode to GLSL")
            else:
                print("✗ Failed to translate basic raymarching pseudocode to GLSL")
                return False
                
            # Test Metal translation
            metal_code = translator.translate(basic_pseudocode, 'metal')
            if metal_code and len(metal_code) > 0:
                print("✓ Successfully translated basic raymarching pseudocode to Metal")
            else:
                print("✗ Failed to translate basic raymarching pseudocode to Metal")
                return False
                
            # Test C/C++ translation
            cpp_code = translator.translate(basic_pseudocode, 'c_cpp')
            if cpp_code and len(cpp_code) > 0:
                print("✓ Successfully translated basic raymarching pseudocode to C/C++")
            else:
                print("✗ Failed to translate basic raymarching pseudocode to C/C++")
                return False
        except Exception as e:
            print(f"✗ Error during basic pseudocode translation: {str(e)}")
            return False
    else:
        print("✗ Branching module doesn't have expected pseudocode structure")
        return False

    return True


def test_raymarching_conflict_detection():
    """Test that conflicts between raymarching branches are properly detected"""
    print("\nTesting raymarching conflict detection...")
    
    try:
        branching_module = get_module_by_name('raymarching_advanced_branching')
        if not branching_module:
            print("✗ Could not load branching module for conflict test")
            return False
            
        branches = branching_module['metadata'].get('branches', {})
        if not branches:
            print("✗ Branching module has no branches")
            return False
            
        # Test algorithm_type conflicts
        alg_branches = branches.get('algorithm_type', {})
        if not alg_branches:
            print("✗ No algorithm_type branches found")
            return False
            
        # Check that basic algorithm conflicts with others
        basic_conflicts = alg_branches['basic']['conflicts']
        expected_basic_conflicts = ['adaptive', 'cone', 'multi']
        
        if all(conflict in basic_conflicts for conflict in expected_basic_conflicts):
            print("✓ Basic raymarching properly conflicts with adaptive, cone, and multi")
        else:
            print(f"✗ Basic raymarching conflicts {basic_conflicts} don't match expected {expected_basic_conflicts}")
            return False
            
        # Check that adaptive algorithm conflicts with others
        adaptive_conflicts = alg_branches['adaptive']['conflicts']
        expected_adaptive_conflicts = ['basic', 'cone', 'multi']
        if all(conflict in adaptive_conflicts for conflict in expected_adaptive_conflicts):
            print("✓ Adaptive raymarching properly conflicts with basic, cone, and multi")
        else:
            print("✗ Adaptive raymarching conflicts not properly defined")
            return False
            
        # Test normal calculation conflicts
        normal_branches = branches.get('normal_calculation', {})
        if not normal_branches:
            print("✗ No normal_calculation branches found")
            return False
            
        standard_conflicts = normal_branches['standard']['conflicts']
        expected_standard_conflicts = ['analytical', 'hybrid']
        
        if all(conflict in standard_conflicts for conflict in expected_standard_conflicts):
            print("✓ Standard normal calculation properly conflicts with analytical and hybrid")
        else:
            print(f"✗ Standard normal conflicts {standard_conflicts} don't match expected {expected_standard_conflicts}")
            return False
            
        # Test optimization conflicts
        opt_branches = branches.get('optimization', {})
        if not opt_branches:
            print("✗ No optimization branches found")
            return False
            
        none_conflicts = opt_branches['none']['conflicts']
        expected_none_conflicts = ['early_exit', 'adaptive_threshold']
        
        if all(conflict in none_conflicts for conflict in expected_none_conflicts):
            print("✓ No optimization properly conflicts with early_exit and adaptive_threshold")
        else:
            print(f"✗ None optimization conflicts {none_conflicts} don't match expected {expected_none_conflicts}")
            return False
    
    except Exception as e:
        print(f"✗ Error during raymarching conflict detection test: {str(e)}")
        return False

    return True


def main():
    """Run all tests for raymarching modules"""
    print("Testing Raymarching Modules with Management Code")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run each test
    all_tests_passed &= test_raymarching_module_loading()
    all_tests_passed &= test_raymarching_pseudocode_translation()
    all_tests_passed &= test_raymarching_conflict_detection()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All raymarching module tests PASSED")
        return 0
    else:
        print("✗ Some raymarching module tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)