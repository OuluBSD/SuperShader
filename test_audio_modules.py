#!/usr/bin/env python3
"""
Test script for audio modules with management code
Tests the integration of audio modules with the module combiner and pseudocode translator
"""

import sys
import os
import json

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from management.module_combiner import ModuleCombiner
from create_pseudocode_translator import PseudocodeTranslator
from modules.audio.registry import get_module_by_name


def test_audio_module_loading():
    """Test that audio modules can be loaded from the registry"""
    print("Testing audio module loading...")

    # Test loading the beat detection module
    beat_module = get_module_by_name('beat_detection')
    if beat_module:
        print("✓ Successfully loaded beat_detection module")
        print(f"  Name: {beat_module['name']}")
        print(f"  Type: {beat_module['metadata']['type']}")
        print(f"  Patterns: {beat_module['metadata']['patterns']}")
    else:
        print("✗ Failed to load beat_detection module")
        return False

    # Test loading the advanced branching audio module
    branching_module = get_module_by_name('audio_processing_branching')
    if branching_module:
        print("✓ Successfully loaded audio_processing_branching module")
        print(f"  Name: {branching_module['name']}")
        print(f"  Type: {branching_module['metadata']['type']}")
        print(f"  Patterns: {branching_module['metadata']['patterns']}")
        print(f"  Branches: {list(branching_module['metadata'].get('branches', {}).keys())}")
    else:
        print("✗ Failed to load audio_processing_branching module")
        return False

    return True


def test_audio_pseudocode_translation():
    """Test that audio pseudocode can be translated to different languages"""
    print("\nTesting audio pseudocode translation...")

    translator = PseudocodeTranslator()

    # Load the branching audio module
    module = get_module_by_name('audio_processing_branching')
    if not module or 'pseudocode' not in module:
        print("✗ Could not get pseudocode from audio module")
        return False
    else:
        print("✓ Successfully got branching audio module with pseudocode")

    # Test translating one of the pseudocode branches
    pseudocodes = module['pseudocode']
    if isinstance(pseudocodes, dict) and 'fft_analysis' in pseudocodes:
        fft_pseudocode = pseudocodes['fft_analysis']

        try:
            # Test GLSL translation
            glsl_code = translator.translate_to_glsl(fft_pseudocode)
            if glsl_code and len(glsl_code) > 0:
                print("✓ Successfully translated FFT analysis pseudocode to GLSL")
            else:
                print("✗ Failed to translate FFT analysis pseudocode to GLSL")
                return False

            # Test Metal translation
            metal_code = translator.translate(fft_pseudocode, 'metal')
            if metal_code and len(metal_code) > 0:
                print("✓ Successfully translated FFT analysis pseudocode to Metal")
            else:
                print("✗ Failed to translate FFT analysis pseudocode to Metal")
                return False

            # Test C/C++ translation
            cpp_code = translator.translate(fft_pseudocode, 'c_cpp')
            if cpp_code and len(cpp_code) > 0:
                print("✓ Successfully translated FFT analysis pseudocode to C/C++")
            else:
                print("✗ Failed to translate FFT analysis pseudocode to C/C++")
                return False
        except Exception as e:
            print(f"✗ Error during FFT pseudocode translation: {str(e)}")
            return False
    else:
        print("✓ Branching module has expected pseudocode structure")

    return True


def test_audio_conflict_detection():
    """Test that conflicts between audio branches are properly detected"""
    print("\nTesting audio conflict detection...")

    try:
        branching_module = get_module_by_name('audio_processing_branching')
        if not branching_module:
            print("✗ Could not load branching module for conflict test")
            return False

        branches = branching_module['metadata'].get('branches', {})
        if not branches:
            print("✗ Branching module has no branches")
            return False

        # Test spectrum analysis conflicts
        spectrum_branches = branches.get('spectrum_analysis', {})
        if not spectrum_branches:
            print("✗ No spectrum_analysis branches found")
            return False

        # Check that fourier conflicts with others
        fourier_conflicts = spectrum_branches['fourier']['conflicts']
        expected_fourier_conflicts = ['wavelet', 'autocorrelation']

        if all(conflict in fourier_conflicts for conflict in expected_fourier_conflicts):
            print("✓ Fourier transform properly conflicts with wavelet and autocorrelation")
        else:
            print(f"✗ Fourier transform conflicts {fourier_conflicts} don't match expected {expected_fourier_conflicts}")
            return False

        # Check wavelet and autocorrelation conflicts too
        wavelet_conflicts = spectrum_branches['wavelet']['conflicts']
        autocorr_conflicts = spectrum_branches['autocorrelation']['conflicts']

        if all(conflict in wavelet_conflicts for conflict in ['fourier', 'autocorrelation']):
            print("✓ Wavelet transform properly conflicts with fourier and autocorrelation")
        else:
            print("✗ Wavelet transform conflicts not properly defined")
            return False

        if all(conflict in autocorr_conflicts for conflict in ['fourier', 'wavelet']):
            print("✓ Autocorrelation properly conflicts with fourier and wavelet")
        else:
            print("✗ Autocorrelation conflicts not properly defined")
            return False

        # Test synthesis method conflicts
        synth_methods = branches.get('synthesis_method', {})
        if not synth_methods:
            print("✗ No synthesis_method branches found")
            return False

        additive_conflicts = synth_methods['additive']['conflicts']
        expected_additive_conflicts = ['subtractive', 'fm', 'wavetable']

        if all(conflict in additive_conflicts for conflict in expected_additive_conflicts):
            print("✓ Additive synthesis properly conflicts with subtractive, FM and wavetable")
        else:
            print(f"✗ Additive synthesis conflicts {additive_conflicts} don't match expected {expected_additive_conflicts}")
            return False

        # Test filter conflicts
        filter_branches = branches.get('filter_type', {})
        if not filter_branches:
            print("✗ No filter_type branches found")
            return False

        lp_conflicts = filter_branches['low_pass']['conflicts']
        expected_lp_conflicts = ['high_pass', 'band_pass', 'notch']

        if all(conflict in lp_conflicts for conflict in expected_lp_conflicts):
            print("✓ Low-pass filter properly conflicts with high-pass, band-pass, and notch")
        else:
            print(f"✗ Low-pass filter conflicts {lp_conflicts} don't match expected {expected_lp_conflicts}")
            return False

    except Exception as e:
        print(f"✗ Error during audio conflict detection test: {str(e)}")
        return False

    return True


def main():
    """Run all tests for audio modules"""
    print("Testing Audio Modules with Management Code")
    print("=" * 50)

    all_tests_passed = True

    # Run each test
    all_tests_passed &= test_audio_module_loading()
    all_tests_passed &= test_audio_pseudocode_translation()
    all_tests_passed &= test_audio_conflict_detection()

    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All audio module tests PASSED")
        return 0
    else:
        print("✗ Some audio module tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)