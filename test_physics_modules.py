#!/usr/bin/env python3
"""
Test script for physics modules with management code
Tests the integration of physics modules with the module combiner and pseudocode translator
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from management.module_combiner import ModuleCombiner
from create_pseudocode_translator import PseudocodeTranslator
from modules.physics.registry import get_module_by_name


def test_physics_module_loading():
    """Test that physics modules can be loaded from the registry"""
    print("Testing physics module loading...")
    
    # Test loading the verlet integration module
    verlet_module = get_module_by_name('verlet_integration')
    if verlet_module:
        print("✓ Successfully loaded verlet_integration module")
        print(f"  Name: {verlet_module['name']}")
        print(f"  Type: {verlet_module['metadata']['type']}")
        print(f"  Patterns: {verlet_module['metadata']['patterns']}")
    else:
        print("✗ Failed to load verlet_integration module")
        return False

    # Test loading the advanced branching physics module
    branching_module = get_module_by_name('physics_advanced_branching')
    if branching_module:
        print("✓ Successfully loaded physics_advanced_branching module")
        print(f"  Name: {branching_module['name']}")
        print(f"  Type: {branching_module['metadata']['type']}")
        print(f"  Patterns: {branching_module['metadata']['patterns']}")
        print(f"  Branches: {list(branching_module['metadata'].get('branches', {}).keys())}")
    else:
        print("✗ Failed to load physics_advanced_branching module")
        return False

    return True


def test_physics_pseudocode_translation():
    """Test that physics pseudocode can be translated to different languages"""
    print("\nTesting physics pseudocode translation...")
    
    translator = PseudocodeTranslator()
    
    # Load the branching physics module
    module = get_module_by_name('physics_advanced_branching')
    if not module or 'pseudocode' in module:
        print("✓ Successfully got branching physics module with pseudocode")
    else:
        print("✗ Could not get pseudocode from physics module")
        return False

    # Test translating one of the pseudocode branches
    pseudocodes = module['pseudocode']
    if isinstance(pseudocodes, dict) and 'euler_integration' in pseudocodes:
        euler_pseudocode = pseudocodes['euler_integration']
        
        try:
            # Test GLSL translation
            glsl_code = translator.translate_to_glsl(euler_pseudocode)
            if glsl_code and len(glsl_code) > 0:
                print("✓ Successfully translated Euler integration pseudocode to GLSL")
            else:
                print("✗ Failed to translate Euler integration pseudocode to GLSL")
                return False
                
            # Test Metal translation
            metal_code = translator.translate(euler_pseudocode, 'metal')
            if metal_code and len(metal_code) > 0:
                print("✓ Successfully translated Euler integration pseudocode to Metal")
            else:
                print("✗ Failed to translate Euler integration pseudocode to Metal")
                return False
                
            # Test C/C++ translation
            cpp_code = translator.translate(euler_pseudocode, 'c_cpp')
            if cpp_code and len(cpp_code) > 0:
                print("✓ Successfully translated Euler integration pseudocode to C/C++")
            else:
                print("✗ Failed to translate Euler integration pseudocode to C/C++")
                return False
        except Exception as e:
            print(f"✗ Error during Euler pseudocode translation: {str(e)}")
            return False
    else:
        print("✗ Branching module doesn't have expected pseudocode structure")
        return False

    return True


def test_physics_conflict_detection():
    """Test that conflicts between physics branches are properly detected"""
    print("\nTesting physics conflict detection...")
    
    try:
        branching_module = get_module_by_name('physics_advanced_branching')
        if not branching_module:
            print("✗ Could not load branching module for conflict test")
            return False
            
        branches = branching_module['metadata'].get('branches', {})
        if not branches:
            print("✗ Branching module has no branches")
            return False
            
        # Test integration_method conflicts
        alg_branches = branches.get('integration_method', {})
        if not alg_branches:
            print("✗ No integration_method branches found")
            return False
            
        # Check that Euler integration conflicts with others
        euler_conflicts = alg_branches['euler']['conflicts']
        expected_euler_conflicts = ['verlet', 'rk4', 'semi_implicit']
        
        if all(conflict in euler_conflicts for conflict in expected_euler_conflicts):
            print("✓ Euler integration properly conflicts with Verlet, RK4, and Semi-Implicit")
        else:
            print(f"✗ Euler integration conflicts {euler_conflicts} don't match expected {expected_euler_conflicts}")
            return False
            
        # Check that Verlet integration conflicts with others
        verlet_conflicts = alg_branches['verlet']['conflicts']
        expected_verlet_conflicts = ['euler', 'rk4', 'semi_implicit']
        if all(conflict in verlet_conflicts for conflict in expected_verlet_conflicts):
            print("✓ Verlet integration properly conflicts with Euler, RK4, and Semi-Implicit")
        else:
            print("✗ Verlet integration conflicts not properly defined")
            return False
            
        # Test collision handling conflicts
        collision_branches = branches.get('collision_handling', {})
        if not collision_branches:
            print("✗ No collision_handling branches found")
            return False
            
        simple_conflicts = collision_branches['simple']['conflicts']
        expected_simple_conflicts = ['constraint_based', 'impulse']
        
        if all(conflict in simple_conflicts for conflict in expected_simple_conflicts):
            print("✓ Simple collision handling properly conflicts with constraint-based and impulse")
        else:
            print(f"✗ Simple collision conflicts {simple_conflicts} don't match expected {expected_simple_conflicts}")
            return False
            
        # Test force calculation conflicts
        force_branches = branches.get('force_calculation', {})
        if not force_branches:
            print("✗ No force_calculation branches found")
            return False
            
        newtonian_conflicts = force_branches['newtonian']['conflicts']
        expected_newtonian_conflicts = ['field_based', 'potential']
        
        if all(conflict in newtonian_conflicts for conflict in expected_newtonian_conflicts):
            print("✓ Newtonian forces properly conflict with field-based and potential")
        else:
            print(f"✗ Newtonian forces conflicts {newtonian_conflicts} don't match expected {expected_newtonian_conflicts}")
            return False
    
    except Exception as e:
        print(f"✗ Error during physics conflict detection test: {str(e)}")
        return False

    return True


def main():
    """Run all tests for physics modules"""
    print("Testing Physics Modules with Management Code")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run each test
    all_tests_passed &= test_physics_module_loading()
    all_tests_passed &= test_physics_pseudocode_translation()
    all_tests_passed &= test_physics_conflict_detection()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All physics module tests PASSED")
        return 0
    else:
        print("✗ Some physics module tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)