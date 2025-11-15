#!/usr/bin/env python3
"""
Enhanced Module Verification System for SuperShader
Provides comprehensive functionality verification and testing capabilities for all modules
"""

import sys
import os
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

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


class VerificationResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass
class VerificationStep:
    module_type: str
    module_name: str
    test_name: str
    result: VerificationResult
    details: str = ""
    error: str = ""


class ModuleVerificationSystem:
    """Comprehensive verification system for SuperShader modules"""
    
    def __init__(self):
        self.translator = PseudocodeTranslator()
        self.combiner = ModuleCombiner()
        self.verification_results: List[VerificationStep] = []
        
    def verify_all_modules(self) -> List[VerificationStep]:
        """Run comprehensive verification on all modules"""
        print("Starting comprehensive module verification...")
        
        # Verify each module type
        self.verify_procedural_modules()
        self.verify_raymarching_modules()
        self.verify_physics_modules()
        self.verify_texturing_modules()
        self.verify_audio_modules()
        self.verify_game_modules()
        self.verify_ui_modules()
        
        # Print summary
        self.print_verification_summary()
        
        return self.verification_results
    
    def verify_procedural_modules(self):
        """Verify procedural modules functionality"""
        print("\nVerifying Procedural Modules...")
        
        modules = ['perlin_noise', 'noise_functions_branching']
        
        for module_name in modules:
            print(f"  Verifying {module_name}...")
            
            # Load module
            module = get_procedural_module(module_name)
            if not module:
                self.verification_results.append(VerificationStep(
                    'procedural', module_name, 'load_module', VerificationResult.FAIL,
                    error=f"Module {module_name} not found"
                ))
                continue
            
            # Verify metadata
            if self.verify_metadata(module):
                self.verification_results.append(VerificationStep(
                    'procedural', module_name, 'verify_metadata', VerificationResult.PASS,
                    details="Metadata verification passed"
                ))
            else:
                self.verification_results.append(VerificationStep(
                    'procedural', module_name, 'verify_metadata', VerificationResult.FAIL,
                    error="Metadata verification failed"
                ))
            
            # Verify pseudocode translation
            if 'pseudocode' in module:
                pseudocode = module['pseudocode']
                if self.verify_pseudocode_translation(pseudocode):
                    self.verification_results.append(VerificationStep(
                        'procedural', module_name, 'translate_pseudocode', VerificationResult.PASS,
                        details="Pseudocode translation successful"
                    ))
                else:
                    self.verification_results.append(VerificationStep(
                        'procedural', module_name, 'translate_pseudocode', VerificationResult.FAIL,
                        error="Pseudocode translation failed"
                    ))
            
            # Verify interface if present
            if 'interface' in module.get('metadata', {}):
                interface = module['metadata']['interface']
                if self.verify_interface(interface):
                    self.verification_results.append(VerificationStep(
                        'procedural', module_name, 'verify_interface', VerificationResult.PASS,
                        details="Interface verification passed"
                    ))
                else:
                    self.verification_results.append(VerificationStep(
                        'procedural', module_name, 'verify_interface', VerificationResult.FAIL,
                        error="Interface verification failed"
                    ))
    
    def verify_raymarching_modules(self):
        """Verify raymarching modules functionality"""
        print("\nVerifying Raymarching Modules...")
        
        modules = ['raymarching_core', 'raymarching_advanced_branching']
        
        for module_name in modules:
            print(f"  Verifying {module_name}...")
            
            # Load module
            module = get_raymarching_module(module_name)
            if not module:
                self.verification_results.append(VerificationStep(
                    'raymarching', module_name, 'load_module', VerificationResult.FAIL,
                    error=f"Module {module_name} not found"
                ))
                continue
            
            # Verify metadata
            if self.verify_metadata(module):
                self.verification_results.append(VerificationStep(
                    'raymarching', module_name, 'verify_metadata', VerificationResult.PASS,
                    details="Metadata verification passed"
                ))
            else:
                self.verification_results.append(VerificationStep(
                    'raymarching', module_name, 'verify_metadata', VerificationResult.FAIL,
                    error="Metadata verification failed"
                ))
            
            # Verify pseudocode translation
            if 'pseudocode' in module:
                pseudocode = module['pseudocode']
                if isinstance(pseudocode, dict):
                    # Branching module - test each branch
                    for branch_name, branch_code in pseudocode.items():
                        if self.verify_pseudocode_translation(branch_code):
                            self.verification_results.append(VerificationStep(
                                'raymarching', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.PASS,
                                details="Pseudocode translation successful"
                            ))
                        else:
                            self.verification_results.append(VerificationStep(
                                'raymarching', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.FAIL,
                                error="Pseudocode translation failed"
                            ))
                else:
                    # Non-branching module
                    if self.verify_pseudocode_translation(pseudocode):
                        self.verification_results.append(VerificationStep(
                            'raymarching', module_name, 'translate_pseudocode', VerificationResult.PASS,
                            details="Pseudocode translation successful"
                        ))
                    else:
                        self.verification_results.append(VerificationStep(
                            'raymarching', module_name, 'translate_pseudocode', VerificationResult.FAIL,
                            error="Pseudocode translation failed"
                        ))
    
    def verify_physics_modules(self):
        """Verify physics modules functionality"""
        print("\nVerifying Physics Modules...")
        
        modules = ['verlet_integration', 'physics_advanced_branching']
        
        for module_name in modules:
            print(f"  Verifying {module_name}...")
            
            # Load module
            module = get_physics_module(module_name)
            if not module:
                self.verification_results.append(VerificationStep(
                    'physics', module_name, 'load_module', VerificationResult.FAIL,
                    error=f"Module {module_name} not found"
                ))
                continue
            
            # Verify metadata
            if self.verify_metadata(module):
                self.verification_results.append(VerificationStep(
                    'physics', module_name, 'verify_metadata', VerificationResult.PASS,
                    details="Metadata verification passed"
                ))
            else:
                self.verification_results.append(VerificationStep(
                    'physics', module_name, 'verify_metadata', VerificationResult.FAIL,
                    error="Metadata verification failed"
                ))
            
            # Verify pseudocode translation
            if 'pseudocode' in module:
                pseudocode = module['pseudocode']
                if isinstance(pseudocode, dict):
                    # Branching module - test each branch
                    for branch_name, branch_code in pseudocode.items():
                        if self.verify_pseudocode_translation(branch_code):
                            self.verification_results.append(VerificationStep(
                                'physics', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.PASS,
                                details="Pseudocode translation successful"
                            ))
                        else:
                            self.verification_results.append(VerificationStep(
                                'physics', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.FAIL,
                                error="Pseudocode translation failed"
                            ))
                else:
                    # Non-branching module
                    if self.verify_pseudocode_translation(pseudocode):
                        self.verification_results.append(VerificationStep(
                            'physics', module_name, 'translate_pseudocode', VerificationResult.PASS,
                            details="Pseudocode translation successful"
                        ))
                    else:
                        self.verification_results.append(VerificationStep(
                            'physics', module_name, 'translate_pseudocode', VerificationResult.FAIL,
                            error="Pseudocode translation failed"
                        ))
    
    def verify_texturing_modules(self):
        """Verify texturing modules functionality"""
        print("\nVerifying Texturing Modules...")
        
        modules = ['uv_mapping', 'texturing_advanced_branching']
        
        for module_name in modules:
            print(f"  Verifying {module_name}...")
            
            # Load module
            module = get_texturing_module(module_name)
            if not module:
                self.verification_results.append(VerificationStep(
                    'texturing', module_name, 'load_module', VerificationResult.FAIL,
                    error=f"Module {module_name} not found"
                ))
                continue
            
            # Verify metadata
            if self.verify_metadata(module):
                self.verification_results.append(VerificationStep(
                    'texturing', module_name, 'verify_metadata', VerificationResult.PASS,
                    details="Metadata verification passed"
                ))
            else:
                self.verification_results.append(VerificationStep(
                    'texturing', module_name, 'verify_metadata', VerificationResult.FAIL,
                    error="Metadata verification failed"
                ))
            
            # Verify pseudocode translation
            if 'pseudocode' in module:
                pseudocode = module['pseudocode']
                if isinstance(pseudocode, dict):
                    # Branching module - test each branch
                    for branch_name, branch_code in pseudocode.items():
                        if self.verify_pseudocode_translation(branch_code):
                            self.verification_results.append(VerificationStep(
                                'texturing', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.PASS,
                                details="Pseudocode translation successful"
                            ))
                        else:
                            self.verification_results.append(VerificationStep(
                                'texturing', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.FAIL,
                                error="Pseudocode translation failed"
                            ))
                else:
                    # Non-branching module
                    if self.verify_pseudocode_translation(pseudocode):
                        self.verification_results.append(VerificationStep(
                            'texturing', module_name, 'translate_pseudocode', VerificationResult.PASS,
                            details="Pseudocode translation successful"
                        ))
                    else:
                        self.verification_results.append(VerificationStep(
                            'texturing', module_name, 'translate_pseudocode', VerificationResult.FAIL,
                            error="Pseudocode translation failed"
                        ))
    
    def verify_audio_modules(self):
        """Verify audio modules functionality"""
        print("\nVerifying Audio Modules...")
        
        modules = ['beat_detection', 'audio_advanced_branching']
        
        for module_name in modules:
            print(f"  Verifying {module_name}...")
            
            # Load module
            module = get_audio_module(module_name)
            if not module:
                self.verification_results.append(VerificationStep(
                    'audio', module_name, 'load_module', VerificationResult.FAIL,
                    error=f"Module {module_name} not found"
                ))
                continue
            
            # Verify metadata
            if self.verify_metadata(module):
                self.verification_results.append(VerificationStep(
                    'audio', module_name, 'verify_metadata', VerificationResult.PASS,
                    details="Metadata verification passed"
                ))
            else:
                self.verification_results.append(VerificationStep(
                    'audio', module_name, 'verify_metadata', VerificationResult.FAIL,
                    error="Metadata verification failed"
                ))
            
            # Verify pseudocode translation
            if 'pseudocode' in module:
                pseudocode = module['pseudocode']
                if isinstance(pseudocode, dict):
                    # Branching module - test each branch
                    for branch_name, branch_code in pseudocode.items():
                        if self.verify_pseudocode_translation(branch_code):
                            self.verification_results.append(VerificationStep(
                                'audio', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.PASS,
                                details="Pseudocode translation successful"
                            ))
                        else:
                            self.verification_results.append(VerificationStep(
                                'audio', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.FAIL,
                                error="Pseudocode translation failed"
                            ))
                else:
                    # Non-branching module
                    if self.verify_pseudocode_translation(pseudocode):
                        self.verification_results.append(VerificationStep(
                            'audio', module_name, 'translate_pseudocode', VerificationResult.PASS,
                            details="Pseudocode translation successful"
                        ))
                    else:
                        self.verification_results.append(VerificationStep(
                            'audio', module_name, 'translate_pseudocode', VerificationResult.FAIL,
                            error="Pseudocode translation failed"
                        ))
    
    def verify_game_modules(self):
        """Verify game modules functionality"""
        print("\nVerifying Game Modules...")
        
        modules = ['input_handling', 'game_advanced_branching']
        
        for module_name in modules:
            print(f"  Verifying {module_name}...")
            
            # Load module
            module = get_game_module(module_name)
            if not module:
                self.verification_results.append(VerificationStep(
                    'game', module_name, 'load_module', VerificationResult.FAIL,
                    error=f"Module {module_name} not found"
                ))
                continue
            
            # Verify metadata
            if self.verify_metadata(module):
                self.verification_results.append(VerificationStep(
                    'game', module_name, 'verify_metadata', VerificationResult.PASS,
                    details="Metadata verification passed"
                ))
            else:
                self.verification_results.append(VerificationStep(
                    'game', module_name, 'verify_metadata', VerificationResult.FAIL,
                    error="Metadata verification failed"
                ))
            
            # Verify pseudocode translation
            if 'pseudocode' in module:
                pseudocode = module['pseudocode']
                if isinstance(pseudocode, dict):
                    # Branching module - test each branch
                    for branch_name, branch_code in pseudocode.items():
                        if self.verify_pseudocode_translation(branch_code):
                            self.verification_results.append(VerificationStep(
                                'game', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.PASS,
                                details="Pseudocode translation successful"
                            ))
                        else:
                            self.verification_results.append(VerificationStep(
                                'game', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.FAIL,
                                error="Pseudocode translation failed"
                            ))
                else:
                    # Non-branching module
                    if self.verify_pseudocode_translation(pseudocode):
                        self.verification_results.append(VerificationStep(
                            'game', module_name, 'translate_pseudocode', VerificationResult.PASS,
                            details="Pseudocode translation successful"
                        ))
                    else:
                        self.verification_results.append(VerificationStep(
                            'game', module_name, 'translate_pseudocode', VerificationResult.FAIL,
                            error="Pseudocode translation failed"
                        ))
    
    def verify_ui_modules(self):
        """Verify UI modules functionality"""
        print("\nVerifying UI Modules...")
        
        modules = ['basic_shapes', 'ui_advanced_branching']
        
        for module_name in modules:
            print(f"  Verifying {module_name}...")
            
            # Load module
            module = get_ui_module(module_name)
            if not module:
                self.verification_results.append(VerificationStep(
                    'ui', module_name, 'load_module', VerificationResult.FAIL,
                    error=f"Module {module_name} not found"
                ))
                continue
            
            # Verify metadata
            if self.verify_metadata(module):
                self.verification_results.append(VerificationStep(
                    'ui', module_name, 'verify_metadata', VerificationResult.PASS,
                    details="Metadata verification passed"
                ))
            else:
                self.verification_results.append(VerificationStep(
                    'ui', module_name, 'verify_metadata', VerificationResult.FAIL,
                    error="Metadata verification failed"
                ))
            
            # Verify pseudocode translation
            if 'pseudocode' in module:
                pseudocode = module['pseudocode']
                if isinstance(pseudocode, dict):
                    # Branching module - test each branch
                    for branch_name, branch_code in pseudocode.items():
                        if self.verify_pseudocode_translation(branch_code):
                            self.verification_results.append(VerificationStep(
                                'ui', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.PASS,
                                details="Pseudocode translation successful"
                            ))
                        else:
                            self.verification_results.append(VerificationStep(
                                'ui', f"{module_name}:{branch_name}", 'translate_pseudocode', VerificationResult.FAIL,
                                error="Pseudocode translation failed"
                            ))
                else:
                    # Non-branching module
                    if self.verify_pseudocode_translation(pseudocode):
                        self.verification_results.append(VerificationStep(
                            'ui', module_name, 'translate_pseudocode', VerificationResult.PASS,
                            details="Pseudocode translation successful"
                        ))
                    else:
                        self.verification_results.append(VerificationStep(
                            'ui', module_name, 'translate_pseudocode', VerificationResult.FAIL,
                            error="Pseudocode translation failed"
                        ))

    def verify_metadata(self, module: Dict[str, Any]) -> bool:
        """Verify that module metadata is complete and valid"""
        try:
            required_keys = ['name', 'type', 'patterns', 'frequency', 'dependencies', 'conflicts', 'description']
            for key in required_keys:
                if key not in module.get('metadata', {}):
                    return False
            return True
        except:
            return False
    
    def verify_pseudocode_translation(self, pseudocode: str) -> bool:
        """Verify that pseudocode can be translated to GLSL without errors"""
        try:
            glsl_code = self.translator.translate_to_glsl(pseudocode)
            return glsl_code is not None and len(glsl_code) > 0
        except:
            return False
    
    def verify_interface(self, interface: Dict[str, Any]) -> bool:
        """Verify that interface specification is complete and valid"""
        try:
            required_sections = ['inputs', 'outputs', 'uniforms']
            for section in required_sections:
                if section not in interface:
                    return False
                if not isinstance(interface[section], list):
                    return False
            return True
        except:
            return False
    
    def print_verification_summary(self):
        """Print verification results summary"""
        print("\n" + "="*60)
        print("MODULE VERIFICATION SUMMARY")
        print("="*60)
        
        # Count results by type
        results_by_type = {}
        for result in self.verification_results:
            if result.module_type not in results_by_type:
                results_by_type[result.module_type] = {'PASS': 0, 'FAIL': 0, 'ERROR': 0}
            results_by_type[result.module_type][result.result.value] += 1
        
        total_tests = len(self.verification_results)
        passed_tests = sum(1 for r in self.verification_results if r.result == VerificationResult.PASS)
        failed_tests = sum(1 for r in self.verification_results if r.result == VerificationResult.FAIL)
        error_tests = sum(1 for r in self.verification_results if r.result == VerificationResult.ERROR)
        
        for module_type, counts in results_by_type.items():
            print(f"\n{module_type.upper()} MODULES:")
            print(f"  PASS: {counts['PASS']}")
            print(f"  FAIL: {counts['FAIL']}")
            print(f"  ERROR: {counts['ERROR']}")
            total_module_tests = sum(counts.values())
            print(f"  Success Rate: {counts['PASS']/total_module_tests*100:.1f}%" if total_module_tests > 0 else "  Success Rate: 0%")
        
        print("\n" + "="*60)
        print(f"TOTAL TESTS: {total_tests}")
        print(f"PASSED: {passed_tests}")
        print(f"FAILED: {failed_tests}")
        print(f"ERRORS: {error_tests}")
        print(f"OVERALL SUCCESS RATE: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "OVERALL SUCCESS RATE: 0%")
        print("="*60)
        
        # Print failed tests for debugging
        failed_results = [r for r in self.verification_results if r.result != VerificationResult.PASS]
        if failed_results:
            print(f"\nFAILED/ERROR TESTS ({len(failed_results)}):")
            for result in failed_results:
                print(f"  - {result.module_type}.{result.module_name}::{result.test_name} - {result.result.value}")
                if result.error:
                    print(f"    Error: {result.error}")


def main():
    """Main entry point for the verification system"""
    print("Initializing SuperShader Enhanced Module Verification System...")
    
    verifier = ModuleVerificationSystem()
    results = verifier.verify_all_modules()
    
    # Return success code based on verification results
    failed_count = sum(1 for r in results if r.result != VerificationResult.PASS)
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)