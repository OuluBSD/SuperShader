# Module Testing Framework

import unittest
import json
import os
from pathlib import Path

class ModuleTester:
    def __init__(self, registry_file="registry/modules.json"):
        with open(registry_file, 'r') as f:
            self.registry = json.load(f)
    
    def test_module_syntax(self, module_path):
        """Test if a module has valid GLSL syntax (basic check)."""
        try:
            with open(module_path, 'r') as f:
                content = f.read()
            
            # Basic checks
            if not content.strip():
                return False, "Module is empty"
            
            # Check for common GLSL syntax issues
            if 'float' in content and 'vec' in content:
                # Check that variable declarations are properly formatted
                pass  # More detailed syntax checking could go here
            
            return True, "Syntax appears valid"
        except Exception as e:
            return False, f"Could not read file: {str(e)}"
    
    def test_all_modules(self):
        """Run tests on all modules."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        for module in self.registry['modules']:
            results["total"] += 1
            module_path = Path("modules") / module['path']
            
            success, message = self.test_module_syntax(module_path)
            
            if success:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "module": module['name'],
                    "error": message
                })
        
        return results
    
    def run_compatibility_tests(self, module1_name, module2_name):
        """Test if two modules are compatible."""
        # This is a simplified test - a full implementation would be more complex
        # For now, just check if they have conflicting dependencies
        
        module1 = None
        module2 = None
        
        for m in self.registry['modules']:
            if m['name'] == module1_name:
                module1 = m
            if m['name'] == module2_name:
                module2 = m
        
        if not module1 or not module2:
            return False, "One or both modules not found"
        
        # Check for conflicts (simplified)
        conflicts = set(module1.get('conflicts', [])) & set(module2.get('tags', []))
        conflicts |= set(module2.get('conflicts', [])) & set(module1.get('tags', []))
        
        if conflicts:
            return False, f"Conflicts found: {', '.join(conflicts)}"
        
        return True, "Modules appear compatible"


def main():
    tester = ModuleTester()
    results = tester.test_all_modules()
    
    print(f"Module Testing Results:")
    print(f"Total modules tested: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    if results['errors']:
        print("\nErrors found:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  {error['module']}: {error['error']}")


if __name__ == "__main__":
    main()
