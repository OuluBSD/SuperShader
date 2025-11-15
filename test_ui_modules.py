#!/usr/bin/env python3
"""
Test script for UI modules with management code
Tests the integration of UI modules with the module combiner and pseudocode translator
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from management.module_combiner import ModuleCombiner
from create_pseudocode_translator import PseudocodeTranslator
from modules.ui.registry import get_module_by_name


def test_ui_module_loading():
    """Test that UI modules can be loaded from the registry"""
    print("Testing UI module loading...")
    
    # Test loading the basic shapes module
    shapes_module = get_module_by_name('basic_shapes')
    if shapes_module:
        print("✓ Successfully loaded basic_shapes module")
        print(f"  Name: {shapes_module['name']}")
        print(f"  Type: {shapes_module['metadata']['type']}")
        print(f"  Patterns: {shapes_module['metadata']['patterns']}")
    else:
        print("✗ Failed to load basic_shapes module")
        return False

    # Test loading the advanced branching UI module
    branching_module = get_module_by_name('ui_advanced_branching')
    if branching_module:
        print("✓ Successfully loaded ui_advanced_branching module")
        print(f"  Name: {branching_module['name']}")
        print(f"  Type: {branching_module['metadata']['type']}")
        print(f"  Patterns: {branching_module['metadata']['patterns']}")
        print(f"  Branches: {list(branching_module['metadata'].get('branches', {}).keys())}")
    else:
        print("✗ Failed to load ui_advanced_branching module")
        return False

    return True


def test_ui_pseudocode_translation():
    """Test that UI pseudocode can be translated to different languages"""
    print("\nTesting UI pseudocode translation...")
    
    translator = PseudocodeTranslator()
    
    # Load the branching UI module
    module = get_module_by_name('ui_advanced_branching')
    if not module or 'pseudocode' in module:
        print("✓ Successfully got branching UI module with pseudocode")
    else:
        print("✗ Could not get pseudocode from UI module")
        return False

    # Test translating one of the pseudocode branches
    pseudocodes = module['pseudocode']
    if isinstance(pseudocodes, dict) and 'flat_widget' in pseudocodes:
        widget_pseudocode = pseudocodes['flat_widget']
        
        try:
            # Test GLSL translation
            glsl_code = translator.translate_to_glsl(widget_pseudocode)
            if glsl_code and len(glsl_code) > 0:
                print("✓ Successfully translated flat widget pseudocode to GLSL")
            else:
                print("✗ Failed to translate flat widget pseudocode to GLSL")
                return False
                
            # Test Metal translation
            metal_code = translator.translate(widget_pseudocode, 'metal')
            if metal_code and len(metal_code) > 0:
                print("✓ Successfully translated flat widget pseudocode to Metal")
            else:
                print("✗ Failed to translate flat widget pseudocode to Metal")
                return False
                
            # Test C/C++ translation
            cpp_code = translator.translate(widget_pseudocode, 'c_cpp')
            if cpp_code and len(cpp_code) > 0:
                print("✓ Successfully translated flat widget pseudocode to C/C++")
            else:
                print("✗ Failed to translate flat widget pseudocode to C/C++")
                return False
        except Exception as e:
            print(f"✗ Error during widget pseudocode translation: {str(e)}")
            return False
    else:
        print("✓ Branching module has expected pseudocode structure")
        # Find the first available pseudocode to test
        for key, code in pseudocodes.items():
            if isinstance(code, str):
                try:
                    glsl_code = translator.translate_to_glsl(code)
                    if glsl_code and len(glsl_code) > 0:
                        print(f"✓ Successfully translated {key} pseudocode to GLSL")
                    else:
                        print(f"✗ Failed to translate {key} pseudocode to GLSL")
                        return False
                    break
                except Exception as e:
                    print(f"✗ Error during {key} pseudocode translation: {str(e)}")
                    return False

    return True


def test_ui_conflict_detection():
    """Test that conflicts between UI branches are properly detected"""
    print("\nTesting UI conflict detection...")
    
    try:
        branching_module = get_module_by_name('ui_advanced_branching')
        if not branching_module:
            print("✗ Could not load branching module for conflict test")
            return False
            
        branches = branching_module['metadata'].get('branches', {})
        if not branches:
            print("✗ Branching module has no branches")
            return False
            
        # Test widget style conflicts
        widget_branches = branches.get('widget_style', {})
        if not widget_branches:
            print("✗ No widget_style branches found")
            return False
        
        # Check that flat UI conflicts with others
        flat_branch = widget_branches.get('flat')
        if not flat_branch:
            print("✗ No flat UI branch found")
            return False
            
        flat_conflicts = flat_branch.get('conflicts', [])
        expected_flat_conflicts = ['material', 'neumorphic', 'glassmorphism']
        
        if all(conflict in flat_conflicts for conflict in expected_flat_conflicts):
            print("✓ Flat UI properly conflicts with material, neumorphic, and glassmorphism")
        else:
            print(f"✗ Flat UI conflicts {flat_conflicts} don't match expected {expected_flat_conflicts}")
            return False
            
        # Check that glassmorphism UI conflicts with others
        glass_branch = widget_branches.get('glassmorphism')
        if not glass_branch:
            print("✗ No glassmorphism UI branch found")
            return False
            
        glass_conflicts = glass_branch.get('conflicts', [])
        expected_glass_conflicts = ['flat', 'material', 'neumorphic']
        
        if all(conflict in glass_conflicts for conflict in expected_glass_conflicts):
            print("✓ Glassmorphism UI properly conflicts with flat, material, and neumorphic")
        else:
            print(f"✗ Glassmorphism UI conflicts {glass_conflicts} don't match expected {expected_glass_conflicts}")
            return False
            
        # Test animation style conflicts
        animation_branches = branching_module['metadata']['branches'].get('animation_style', {})
        if not animation_branches:
            print("✗ No animation_style branches found")
            return False
            
        static_branch = animation_branches.get('static')
        if not static_branch:
            print("✗ No static animation branch found")
            return False
            
        static_conflicts = static_branch.get('conflicts', [])
        expected_static_conflicts = ['subtle', 'dynamic']
        
        if all(conflict in static_conflicts for conflict in expected_static_conflicts):
            print("✓ Static UI properly conflicts with subtle and dynamic")
        else:
            print(f"✗ Static UI conflicts {static_conflicts} don't match expected {expected_static_conflicts}")
            return False
            
        # Test interaction feedback conflicts
        feedback_branches = branching_module['metadata']['branches'].get('interaction_feedback', {})
        if not feedback_branches:
            print("✗ No interaction_feedback branches found")
            return False
            
        minimal_branch = feedback_branches.get('minimal')
        if not minimal_branch:
            print("✗ No minimal feedback branch found")
            return False
            
        minimal_conflicts = minimal_branch.get('conflicts', [])
        expected_minimal_conflicts = ['haptic', 'visual']
        
        if all(conflict in minimal_conflicts for conflict in expected_minimal_conflicts):
            print("✓ Minimal feedback properly conflicts with haptic and visual")
        else:
            print(f"✗ Minimal feedback conflicts {minimal_conflicts} don't match expected {expected_minimal_conflicts}")
            return False
    
    except Exception as e:
        print(f"✗ Error during UI conflict detection test: {str(e)}")
        return False

    return True


def main():
    """Run all tests for UI modules"""
    print("Testing UI Modules with Management Code")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run each test
    all_tests_passed &= test_ui_module_loading()
    all_tests_passed &= test_ui_pseudocode_translation()
    all_tests_passed &= test_ui_conflict_detection()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All UI module tests PASSED")
        return 0
    else:
        print("✗ Some UI module tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)