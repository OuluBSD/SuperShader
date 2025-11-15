#!/usr/bin/env python3
"""
Test script for texturing modules with management code
Tests the integration of texturing modules with the module combiner and pseudocode translator
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from management.module_combiner import ModuleCombiner
from create_pseudocode_translator import PseudocodeTranslator
from modules.texturing.registry import get_module_by_name


def test_texturing_module_loading():
    """Test that texturing modules can be loaded from the registry"""
    print("Testing texturing module loading...")
    
    # Test loading the UV mapping module
    uv_module = get_module_by_name('uv_mapping')
    if uv_module:
        print("✓ Successfully loaded uv_mapping module")
        print(f"  Name: {uv_module['name']}")
        print(f"  Type: {uv_module['metadata']['type']}")
        print(f"  Patterns: {uv_module['metadata']['patterns']}")
    else:
        print("✗ Failed to load uv_mapping module")
        return False

    # Test loading the advanced branching texturing module
    branching_module = get_module_by_name('texturing_advanced_branching')
    if branching_module:
        print("✓ Successfully loaded texturing_advanced_branching module")
        print(f"  Name: {branching_module['name']}")
        print(f"  Type: {branching_module['metadata']['type']}")
        print(f"  Patterns: {branching_module['metadata']['patterns']}")
        print(f"  Branches: {list(branching_module['metadata'].get('branches', {}).keys())}")
    else:
        print("✗ Failed to load texturing_advanced_branching module")
        return False

    return True


def test_texturing_pseudocode_translation():
    """Test that texturing pseudocode can be translated to different languages"""
    print("\nTesting texturing pseudocode translation...")
    
    translator = PseudocodeTranslator()
    
    # Load the branching texturing module
    module = get_module_by_name('texturing_advanced_branching')
    if not module or 'pseudocode' in module:
        print("✓ Successfully got branching texturing module with pseudocode")
    else:
        print("✗ Could not get pseudocode from texturing module")
        return False

    # Test translating one of the pseudocode branches
    pseudocodes = module['pseudocode']
    if isinstance(pseudocodes, dict) and 'planar_mapping' in pseudocodes:
        planar_pseudocode = pseudocodes['planar_mapping']
        
        try:
            # Test GLSL translation
            glsl_code = translator.translate_to_glsl(planar_pseudocode)
            if glsl_code and len(glsl_code) > 0:
                print("✓ Successfully translated planar mapping pseudocode to GLSL")
            else:
                print("✗ Failed to translate planar mapping pseudocode to GLSL")
                return False
                
            # Test Metal translation
            metal_code = translator.translate(planar_pseudocode, 'metal')
            if metal_code and len(metal_code) > 0:
                print("✓ Successfully translated planar mapping pseudocode to Metal")
            else:
                print("✗ Failed to translate planar mapping pseudocode to Metal")
                return False
                
            # Test C/C++ translation
            cpp_code = translator.translate(planar_pseudocode, 'c_cpp')
            if cpp_code and len(cpp_code) > 0:
                print("✓ Successfully translated planar mapping pseudocode to C/C++")
            else:
                print("✗ Failed to translate planar mapping pseudocode to C/C++")
                return False
        except Exception as e:
            print(f"✗ Error during planar pseudocode translation: {str(e)}")
            return False
    else:
        print("✗ Branching module doesn't have expected pseudocode structure")
        return False

    return True


def test_texturing_conflict_detection():
    """Test that conflicts between texturing branches are properly detected"""
    print("\nTesting texturing conflict detection...")
    
    try:
        branching_module = get_module_by_name('texturing_advanced_branching')
        if not branching_module:
            print("✗ Could not load branching module for conflict test")
            return False
            
        branches = branching_module['metadata'].get('branches', {})
        if not branches:
            print("✗ Branching module has no branches")
            return False
            
        # Test uv_mapping_method conflicts
        alg_branches = branches.get('uv_mapping_method', {})
        if not alg_branches:
            print("✗ No uv_mapping_method branches found")
            return False
            
        # Check that Planar mapping conflicts with others
        planar_conflicts = alg_branches['planar']['conflicts']
        expected_planar_conflicts = ['spherical', 'cylindrical', 'triplanar']
        
        if all(conflict in planar_conflicts for conflict in expected_planar_conflicts):
            print("✓ Planar UV mapping properly conflicts with spherical, cylindrical, and triplanar")
        else:
            print(f"✗ Planar UV mapping conflicts {planar_conflicts} don't match expected {expected_planar_conflicts}")
            return False
            
        # Check that Triplanar mapping conflicts with others
        triplanar_conflicts = alg_branches['triplanar']['conflicts']
        expected_triplanar_conflicts = ['planar', 'spherical', 'cylindrical']
        if all(conflict in triplanar_conflicts for conflict in expected_triplanar_conflicts):
            print("✓ Triplanar UV mapping properly conflicts with planar, spherical, and cylindrical")
        else:
            print("✗ Triplanar UV mapping conflicts not properly defined")
            return False
            
        # Test texture filtering conflicts
        filtering_branches = branches.get('texture_filtering', {})
        if not filtering_branches:
            print("✗ No texture_filtering branches found")
            return False
            
        nearest_conflicts = filtering_branches['nearest']['conflicts']
        expected_nearest_conflicts = ['bilinear', 'trilinear', 'anisotropic']
        
        if all(conflict in nearest_conflicts for conflict in expected_nearest_conflicts):
            print("✓ Nearest neighbor filtering properly conflicts with bilinear, trilinear, and anisotropic")
        else:
            print(f"✗ Nearest neighbor conflicts {nearest_conflicts} don't match expected {expected_nearest_conflicts}")
            return False
            
        # Test blending mode conflicts
        blending_branches = branches.get('blending_mode', {})
        if not blending_branches:
            print("✗ No blending_mode branches found")
            return False
            
        multiply_conflicts = blending_branches['multiply']['conflicts']
        expected_multiply_conflicts = ['overlay', 'soft_light', 'additive']
        
        if all(conflict in multiply_conflicts for conflict in expected_multiply_conflicts):
            print("✓ Multiply blending properly conflicts with overlay, soft_light, and additive")
        else:
            print(f"✗ Multiply blending conflicts {multiply_conflicts} don't match expected {expected_multiply_conflicts}")
            return False
    
    except Exception as e:
        print(f"✗ Error during texturing conflict detection test: {str(e)}")
        return False

    return True


def main():
    """Run all tests for texturing modules"""
    print("Testing Texturing Modules with Management Code")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run each test
    all_tests_passed &= test_texturing_module_loading()
    all_tests_passed &= test_texturing_pseudocode_translation()
    all_tests_passed &= test_texturing_conflict_detection()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All texturing module tests PASSED")
        return 0
    else:
        print("✗ Some texturing module tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)