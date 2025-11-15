#!/usr/bin/env python3
"""
Test script for game modules with management code
Tests the integration of game modules with the module combiner and pseudocode translator
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from management.module_combiner import ModuleCombiner
from create_pseudocode_translator import PseudocodeTranslator
from modules.game.registry import get_module_by_name


def test_game_module_loading():
    """Test that game modules can be loaded from the registry"""
    print("Testing game module loading...")

    # Test loading the input handling module
    input_module = get_module_by_name('input_handling')
    if input_module:
        print("✓ Successfully loaded input_handling module")
        print(f"  Name: {input_module['name']}")
        print(f"  Type: {input_module['metadata']['type']}")
        print(f"  Patterns: {input_module['metadata']['patterns']}")
    else:
        print("✗ Failed to load input_handling module")
        return False

    # Test loading the advanced branching game module
    branching_module = get_module_by_name('game_advanced_branching')
    if branching_module:
        print("✓ Successfully loaded game_advanced_branching module")
        print(f"  Name: {branching_module['name']}")
        print(f"  Type: {branching_module['metadata']['type']}")
        print(f"  Patterns: {branching_module['metadata']['patterns']}")
        print(f"  Branches: {list(branching_module['metadata'].get('branches', {}).keys())}")
    else:
        print("✗ Failed to load game_advanced_branching module")
        return False

    return True


def test_game_pseudocode_translation():
    """Test that game pseudocode can be translated to different languages"""
    print("\nTesting game pseudocode translation...")

    translator = PseudocodeTranslator()

    # Load the branching game module
    module = get_module_by_name('game_advanced_branching')
    if not module or 'pseudocode' not in module:
        print("✗ Could not get pseudocode from game module")
        return False
    else:
        print("✓ Successfully got branching game module with pseudocode")

    # Test translating one of the pseudocode branches
    pseudocodes = module['pseudocode']
    if isinstance(pseudocodes, dict) and 'minimalist_hud' in pseudocodes:
        hud_pseudocode = pseudocodes['minimalist_hud']

        try:
            # Test GLSL translation
            glsl_code = translator.translate_to_glsl(hud_pseudocode)
            if glsl_code and len(glsl_code) > 0:
                print("✓ Successfully translated minimalist HUD pseudocode to GLSL")
            else:
                print("✗ Failed to translate minimalist HUD pseudocode to GLSL")
                return False

            # Test Metal translation
            metal_code = translator.translate(hud_pseudocode, 'metal')
            if metal_code and len(metal_code) > 0:
                print("✓ Successfully translated minimalist HUD pseudocode to Metal")
            else:
                print("✗ Failed to translate minimalist HUD pseudocode to Metal")
                return False

            # Test C/C++ translation
            cpp_code = translator.translate(hud_pseudocode, 'c_cpp')
            if cpp_code and len(cpp_code) > 0:
                print("✓ Successfully translated minimalist HUD pseudocode to C/C++")
            else:
                print("✗ Failed to translate minimalist HUD pseudocode to C/C++")
                return False
        except Exception as e:
            print(f"✗ Error during HUD pseudocode translation: {str(e)}")
            return False
    else:
        print("✗ Branching module doesn't have expected pseudocode structure")
        return False

    return True


def test_game_conflict_detection():
    """Test that conflicts between game branches are properly detected"""
    print("\nTesting game conflict detection...")

    try:
        branching_module = get_module_by_name('game_advanced_branching')
        if not branching_module:
            print("✗ Could not load branching module for conflict test")
            return False

        branches = branching_module['metadata'].get('branches', {})
        if not branches:
            print("✗ Branching module has no branches")
            return False

        # Test HUD style conflicts
        hud_branches = branches.get('hud_style', {})
        if not hud_branches:
            print("✗ No hud_style branches found")
            return False

        # Check that minimalist HUD conflicts with others
        minimalist_conflicts = hud_branches['minimalist']['conflicts']
        expected_minimalist_conflicts = ['detailed', 'themed']

        if all(conflict in minimalist_conflicts for conflict in expected_minimalist_conflicts):
            print("✓ Minimalist HUD properly conflicts with detailed and themed")
        else:
            print(f"✗ Minimalist HUD conflicts {minimalist_conflicts} don't match expected {expected_minimalist_conflicts}")
            return False

        # Check themed HUD conflicts
        themed_conflicts = hud_branches['themed']['conflicts']
        expected_themed_conflicts = ['minimalist', 'detailed']

        if all(conflict in themed_conflicts for conflict in expected_themed_conflicts):
            print("✓ Themed HUD properly conflicts with minimalist and detailed")
        else:
            print(f"✗ Themed HUD conflicts {themed_conflicts} don't match expected {expected_themed_conflicts}")
            return False

        # Test interaction mode conflicts
        interaction_branches = branches.get('interaction_mode', {})
        if not interaction_branches:
            print("✗ No interaction_mode branches found")
            return False

        # Check that click-based interaction conflicts with others
        click_branch = interaction_branches.get('click')
        if not click_branch:
            print("✗ No click interaction branch found")
            return False

        click_conflicts = click_branch['conflicts']
        expected_click_conflicts = ['hover', 'touch']

        if all(conflict in click_conflicts for conflict in expected_click_conflicts):
            print("✓ Click-based interaction properly conflicts with hover and touch")
        else:
            print(f"✗ Click-based interaction conflicts {click_conflicts} don't match expected {expected_click_conflicts}")
            return False

        # Test visual feedback conflicts
        feedback_branches = branches.get('visual_feedback', {})
        if not feedback_branches:
            print("✗ No visual_feedback branches found")
            return False

        subtle_branch = feedback_branches.get('subtle')
        if not subtle_branch:
            print("✗ No subtle feedback branch found")
            return False

        subtle_conflicts = subtle_branch['conflicts']
        expected_subtle_conflicts = ['flashy', 'animated']

        if all(conflict in subtle_conflicts for conflict in expected_subtle_conflicts):
            print("✓ Subtle feedback properly conflicts with flashy and animated")
        else:
            print(f"✗ Subtle feedback conflicts {subtle_conflicts} don't match expected {expected_subtle_conflicts}")
            return False

    except Exception as e:
        print(f"✗ Error during game conflict detection test: {str(e)}")
        return False

    return True


def main():
    """Run all tests for game modules"""
    print("Testing Game Modules with Management Code")
    print("=" * 50)

    all_tests_passed = True

    # Run each test
    all_tests_passed &= test_game_module_loading()
    all_tests_passed &= test_game_pseudocode_translation()
    all_tests_passed &= test_game_conflict_detection()

    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All game module tests PASSED")
        return 0
    else:
        print("✗ Some game module tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)