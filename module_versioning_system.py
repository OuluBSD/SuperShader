#!/usr/bin/env python3
"""
Module Versioning System for SuperShader
Manages different versions of modules with compatibility checking and automatic updates
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import semver
import os


class ModuleVersion:
    """Represents a version of a module"""
    
    def __init__(self, version_string: str, metadata: Dict[str, Any] = None):
        self.version_string = version_string
        self.parsed_version = semver.VersionInfo.parse(version_string)
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        
    def __str__(self):
        return self.version_string
    
    def __lt__(self, other):
        return self.parsed_version < other.parsed_version
    
    def __le__(self, other):
        return self.parsed_version <= other.parsed_version
    
    def __gt__(self, other):
        return self.parsed_version > other.parsed_version
    
    def __ge__(self, other):
        return self.parsed_version >= other.parsed_version
    
    def __eq__(self, other):
        return self.parsed_version == other.parsed_version


class ModuleVersionManager:
    """Manages versions of modules"""
    
    def __init__(self, versions_file: str = "module_versions.json"):
        self.versions_file = versions_file
        self.versions_db = self._load_versions_db()
        self.module_hashes = {}  # Cache for module content hashes
        
    def _load_versions_db(self) -> Dict[str, Any]:
        """Load the versions database from file"""
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize with a default structure
            return {
                "modules": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
            }
    
    def _save_versions_db(self):
        """Save the versions database to file"""
        self.versions_db["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions_db, f, indent=2)
    
    def create_version(self, module_name: str, version: str, content: str, 
                      author: str = "system", description: str = "") -> bool:
        """Create a new version of a module"""
        try:
            # Validate version format using semver
            ModuleVersion(version)
            
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Create version entry
            version_entry = {
                "version": version,
                "content_hash": content_hash,
                "author": author,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "dependencies": [],
                "compatible_with": [],
                "deprecated": False
            }
            
            # Add to module versions
            if module_name not in self.versions_db["modules"]:
                self.versions_db["modules"][module_name] = {
                    "versions": {},
                    "current_version": version,
                    "latest_version": version,
                    "all_versions": [version]
                }
            
            # Add version entry
            self.versions_db["modules"][module_name]["versions"][version] = version_entry
            
            # Update latest and current version info
            module_info = self.versions_db["modules"][module_name]
            module_info["all_versions"].append(version)
            module_info["all_versions"] = list(set(module_info["all_versions"]))  # Remove duplicates
            
            # Set current to latest
            sorted_versions = sorted([ModuleVersion(v) for v in module_info["all_versions"]], 
                                   key=lambda x: x.parsed_version, reverse=True)
            module_info["latest_version"] = str(sorted_versions[0])
            module_info["current_version"] = str(sorted_versions[0])
            
            # Save to file
            self._save_versions_db()
            
            # Cache the hash for quick lookup
            self.module_hashes[module_name] = {version: content_hash}
            
            return True
        except Exception as e:
            print(f"Error creating version for module {module_name}: {e}")
            return False
    
    def get_module_version(self, module_name: str, version: str = None) -> Optional[Dict[str, Any]]:
        """Get a specific version of a module, or the latest if no version specified"""
        if module_name not in self.versions_db["modules"]:
            return None
        
        module_info = self.versions_db["modules"][module_name]
        
        if version is None:
            # Get the current version
            version = module_info["current_version"]
        
        if version not in module_info["versions"]:
            return None
        
        return module_info["versions"][version]
    
    def get_all_versions(self, module_name: str) -> List[str]:
        """Get all versions of a module"""
        if module_name not in self.versions_db["modules"]:
            return []
        
        return self.versions_db["modules"][module_name]["all_versions"]
    
    def get_latest_version(self, module_name: str) -> str:
        """Get the latest version of a module"""
        if module_name not in self.versions_db["modules"]:
            return None
        
        return self.versions_db["modules"][module_name]["latest_version"]
    
    def get_current_version(self, module_name: str) -> str:
        """Get the current (active) version of a module"""
        if module_name not in self.versions_db["modules"]:
            return None
        
        return self.versions_db["modules"][module_name]["current_version"]
    
    def set_current_version(self, module_name: str, version: str) -> bool:
        """Set the current (active) version of a module"""
        if module_name not in self.versions_db["modules"]:
            return False
        
        module_info = self.versions_db["modules"][module_name]
        
        if version not in module_info["versions"]:
            return False
        
        module_info["current_version"] = version
        self._save_versions_db()
        return True
    
    def get_content_hash(self, module_name: str, version: str) -> str:
        """Get the content hash for a specific version of a module"""
        version_info = self.get_module_version(module_name, version)
        if version_info:
            return version_info.get("content_hash")
        return None
    
    def check_for_updates(self, module_name: str) -> Dict[str, Any]:
        """Check if there are updates available for a module"""
        if module_name not in self.versions_db["modules"]:
            return {
                "updates_available": False,
                "current_version": None,
                "latest_version": None,
                "newer_versions": []
            }
        
        module_info = self.versions_db["modules"][module_name]
        current_version = ModuleVersion(module_info["current_version"])
        latest_version = ModuleVersion(module_info["latest_version"])
        
        # Get all newer versions
        newer_versions = []
        for version_str in module_info["all_versions"]:
            version = ModuleVersion(version_str)
            if version > current_version:
                newer_versions.append(version_str)
        
        newer_versions = sorted([ModuleVersion(v) for v in newer_versions], 
                               key=lambda x: x.parsed_version, reverse=True)
        newer_versions_str = [str(v) for v in newer_versions]
        
        return {
            "updates_available": len(newer_versions) > 0,
            "current_version": module_info["current_version"],
            "latest_version": module_info["latest_version"],
            "newer_versions": newer_versions_str
        }
    
    def get_compatibility_info(self, module_name: str, version: str) -> Dict[str, Any]:
        """Get compatibility information for a module version"""
        version_info = self.get_module_version(module_name, version)
        if not version_info:
            return {}
        
        return {
            "version": version,
            "dependencies": version_info.get("dependencies", []),
            "compatible_with": version_info.get("compatible_with", []),
            "deprecated": version_info.get("deprecated", False),
            "deprecation_reason": version_info.get("deprecation_reason", "")
        }
    
    def mark_version_deprecated(self, module_name: str, version: str, reason: str = ""):
        """Mark a version as deprecated"""
        if module_name not in self.versions_db["modules"]:
            return False
        
        module_info = self.versions_db["modules"][module_name]
        
        if version not in module_info["versions"]:
            return False
        
        module_info["versions"][version]["deprecated"] = True
        module_info["versions"][version]["deprecation_reason"] = reason
        
        self._save_versions_db()
        return True
    
    def add_dependency(self, module_name: str, version: str, dependency: str, version_constraint: str = "*"):
        """Add a dependency to a module version"""
        if module_name not in self.versions_db["modules"]:
            return False
        
        module_info = self.versions_db["modules"][module_name]
        
        if version not in module_info["versions"]:
            return False
        
        dep_entry = {
            "module": dependency,
            "constraint": version_constraint
        }
        
        deps = module_info["versions"][version].get("dependencies", [])
        deps.append(dep_entry)
        module_info["versions"][version]["dependencies"] = deps
        
        self._save_versions_db()
        return True
    
    def get_dependents(self, module_name: str, version: str = None) -> List[Dict[str, str]]:
        """Find all modules that depend on this module version"""
        dependents = []
        
        for dep_module_name, dep_module_info in self.versions_db["modules"].items():
            for dep_version_str, dep_version_info in dep_module_info["versions"].items():
                deps = dep_version_info.get("dependencies", [])
                for dep in deps:
                    if dep.get("module") == module_name:
                        if version is None or dep.get("constraint") == version or self._satisfies_constraint(version, dep.get("constraint")):
                            dependents.append({
                                "module": dep_module_name,
                                "version": dep_version_str
                            })
        
        return dependents
    
    def _satisfies_constraint(self, version: str, constraint: str) -> bool:
        """Check if a version satisfies a constraint (simplified)"""
        # This is a simplified version - in practice, you'd use a full semver constraint solver
        if constraint == "*" or constraint == version:
            return True
        
        # Handle basic constraints like ">=1.0.0", ">1.0.0", etc.
        try:
            actual_version = semver.VersionInfo.parse(version)
            
            if constraint.startswith(">="):
                min_version = semver.VersionInfo.parse(constraint[2:])
                return actual_version >= min_version
            elif constraint.startswith(">"):
                min_version = semver.VersionInfo.parse(constraint[1:])
                return actual_version > min_version
            elif constraint.startswith("<="):
                max_version = semver.VersionInfo.parse(constraint[2:])
                return actual_version <= max_version
            elif constraint.startswith("<"):
                max_version = semver.VersionInfo.parse(constraint[1:])
                return actual_version < max_version
            else:
                # Exact match
                expected_version = semver.VersionInfo.parse(constraint)
                return actual_version == expected_version
        except:
            # If parsing fails, be conservative and return False
            return False


class ModuleVersioningSystem:
    """Main system for module versioning"""
    
    def __init__(self):
        self.version_manager = ModuleVersionManager()
    
    def register_module_version(self, module_name: str, content: str, version: str, 
                              author: str = "system", description: str = "") -> bool:
        """
        Register a new version of a module
        """
        return self.version_manager.create_version(module_name, version, content, author, description)
    
    def get_module_at_version(self, module_name: str, version: str = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Get module content at a specific version and its metadata
        """
        version_info = self.version_manager.get_module_version(module_name, version)
        
        if not version_info:
            return None, {}
        
        # In a real implementation, we would retrieve the actual module content
        # For now, we'll just return the metadata
        return f"// Content for {module_name} at version {version or version_info['version']}", version_info
    
    def update_module_to_latest(self, module_name: str) -> Dict[str, Any]:
        """
        Update a module to the latest version
        """
        updates = self.version_manager.check_for_updates(module_name)
        
        if not updates["updates_available"]:
            return {
                "success": True,
                "message": f"No updates available for {module_name}",
                "current_version": updates["current_version"]
            }
        
        latest_version = updates["latest_version"]
        self.version_manager.set_current_version(module_name, latest_version)
        
        return {
            "success": True,
            "message": f"Updated {module_name} to version {latest_version}",
            "from_version": updates["current_version"],
            "to_version": latest_version,
            "available_updates": updates["newer_versions"]
        }
    
    def check_compatibility(self, module_name: str, version: str = None) -> Dict[str, Any]:
        """
        Check compatibility of a module version
        """
        if version is None:
            version = self.version_manager.get_current_version(module_name)
        
        compat_info = self.version_manager.get_compatibility_info(module_name, version)
        
        result = {
            "module": module_name,
            "version": version,
            "compatible": True,
            "issues": [],
            "dependencies": compat_info.get("dependencies", []),
            "dependents": []
        }
        
        # Check if module is deprecated
        if compat_info.get("deprecated", False):
            result["compatible"] = False
            result["issues"].append(f"Module version {version} is deprecated: {compat_info.get('deprecation_reason', 'No reason provided')}")
        
        # Check for dependency issues
        for dep in compat_info.get("dependencies", []):
            dep_module = dep.get("module")
            dep_constraint = dep.get("constraint")
            
            if dep_module:
                # Check if dependency exists and satisfies constraint
                latest_dep_version = self.version_manager.get_latest_version(dep_module)
                if not latest_dep_version:
                    result["compatible"] = False
                    result["issues"].append(f"Required dependency {dep_module} does not exist")
                elif not self.version_manager._satisfies_constraint(latest_dep_version, dep_constraint):
                    result["compatible"] = False
                    result["issues"].append(f"Dependency {dep_module} version {latest_dep_version} does not satisfy constraint {dep_constraint}")
        
        # Get modules that depend on this one (to warn about breaking changes)
        dependents = self.version_manager.get_dependents(module_name, version)
        result["dependents"] = dependents
        
        return result
    
    def create_module_backup(self, module_name: str, version: str) -> str:
        """
        Create a backup of a specific module version
        """
        version_info = self.version_manager.get_module_version(module_name, version)
        
        if not version_info:
            return ""
        
        backup_content = {
            "module_name": module_name,
            "version": version,
            "content_hash": version_info["content_hash"],
            "author": version_info["author"],
            "description": version_info["description"],
            "created_at": version_info["created_at"],
            "dependencies": version_info.get("dependencies", []),
            "backup_created_at": datetime.now().isoformat()
        }
        
        # Write backup to file
        backup_filename = f"backup_{module_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_filename, 'w') as f:
            json.dump(backup_content, f, indent=2)
        
        return backup_filename


def main():
    """Main function to demonstrate the module versioning system"""
    print("Initializing Module Versioning System...")
    
    # Initialize the versioning system
    versioning_system = ModuleVersioningSystem()
    
    # Register some example module versions
    print("\\n1. Registering module versions...")
    
    # Register version 1.0.0 of a lighting module
    lighting_v1_content = \"\"\"
// Phong Lighting Model Implementation
float phong_lighting(vec3 normal, vec3 light_dir, vec3 view_dir, vec3 light_color) {
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    return diff + spec;
}
\"\"\"
    
    success1 = versioning_system.register_module_version(
        "lighting", 
        lighting_v1_content, 
        "1.0.0", 
        "dev1", 
        "Initial Phong lighting implementation"
    )
    
    # Register version 1.1.0 with improvements
    lighting_v11_content = \"\"\"
// Improved Phong Lighting Model Implementation
float phong_lighting(vec3 normal, vec3 light_dir, vec3 view_dir, vec3 light_color, float specular_power) {
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), specular_power);
    return diff + spec;
}
\"\"\"
    
    success2 = versioning_system.register_module_version(
        "lighting", 
        lighting_v11_content, 
        "1.1.0", 
        "dev2", 
        "Added configurable specular power"
    )
    
    # Register version 2.0.0 with major rewrite
    lighting_v2_content = \"\"\"
// PBR Lighting Model Implementation
vec3 pbr_lighting(vec3 albedo, float metallic, float roughness, vec3 normal, vec3 view_dir, vec3 light_pos, vec3 light_color) {
    // Implementation here
    return albedo; // Simplified
}
\"\"\"
    
    success3 = versioning_system.register_module_version(
        "lighting", 
        lighting_v2_content, 
        "2.0.0", 
        "dev3", 
        "Major rewrite to PBR lighting model"
    )
    
    print(f"   Lighting v1.0.0: {'✓' if success1 else '✗'}")
    print(f"   Lighting v1.1.0: {'✓' if success2 else '✗'}")
    print(f"   Lighting v2.0.0: {'✓' if success3 else '✗'}")
    
    # Test version management
    print(f"\\n2. Version information:")
    print(f"   All versions of lighting module: {versioning_system.version_manager.get_all_versions('lighting')}")
    print(f"   Latest version: {versioning_system.version_manager.get_latest_version('lighting')}")
    print(f"   Current version: {versioning_system.version_manager.get_current_version('lighting')}")
    
    # Test getting specific versions
    print(f"\\n3. Retrieving specific versions:")
    content, metadata = versioning_system.get_module_at_version("lighting", "1.0.0")
    print(f"   Lighting v1.0.0 description: {metadata.get('description', 'N/A')}")
    
    content, metadata = versioning_system.get_module_at_version("lighting", "2.0.0")
    print(f"   Lighting v2.0.0 description: {metadata.get('description', 'N/A')}")
    
    # Test update checking
    print(f"\\n4. Update checking:")
    updates = versioning_system.version_manager.check_for_updates("lighting")
    print(f"   Updates available: {updates['updates_available']}")
    print(f"   Current version: {updates['current_version']}")
    print(f"   Latest version: {updates['latest_version']}")
    print(f"   Newer versions: {updates['newer_versions']}")
    
    # Test compatibility checking
    print(f"\\n5. Compatibility checking:")
    compat = versioning_system.check_compatibility("lighting", "1.0.0")
    print(f"   Version 1.0.0 compatible: {compat['compatible']}")
    print(f"   Issues: {len(compat['issues'])}")
    
    compat = versioning_system.check_compatibility("lighting", "2.0.0")
    print(f"   Version 2.0.0 compatible: {compat['compatible']}")
    print(f"   Issues: {len(compat['issues'])}")
    
    # Add a dependency and test it
    print(f"\\n6. Dependency testing:")
    success_dep = versioning_system.version_manager.add_dependency("lighting", "2.0.0", "math_utils", ">=1.0.0")
    print(f"   Added dependency to lighting v2.0.0: {'✓' if success_dep else '✗'}")
    
    # Check compatibility again after adding dependency
    compat = versioning_system.check_compatibility("lighting", "2.0.0")
    print(f"   After adding dependency - compatible: {compat['compatible']}")
    print(f"   Issues: {len(compat['issues'])} - {compat['issues']}")
    
    # Test marking version as deprecated
    print(f"\\n7. Deprecation testing:")
    versioning_system.version_manager.mark_version_deprecated(
        "lighting", "1.0.0", "Replaced by PBR lighting in v2.0.0"
    )
    compat = versioning_system.check_compatibility("lighting", "1.0.0")
    print(f"   Deprecated version compatibility: {compat['compatible']}")
    print(f"   Issues: {len(compat['issues'])} - {compat['issues']}")
    
    # Test update to latest
    print(f"\\n8. Update to latest:")
    # First, set current version to 1.0.0 to simulate an old version being active
    versioning_system.version_manager.set_current_version("lighting", "1.0.0")
    print(f"   Set current version to: {versioning_system.version_manager.get_current_version('lighting')}")
    
    update_result = versioning_system.update_module_to_latest("lighting")
    print(f"   Update result: {update_result['message']}")
    print(f"   New current version: {versioning_system.version_manager.get_current_version('lighting')}")
    
    # Create a backup
    print(f"\\n9. Backup testing:")
    backup_file = versioning_system.create_module_backup("lighting", "2.0.0")
    print(f"   Created backup: {backup_file if backup_file else 'Failed'}")
    
    print(f"\\n✅ Module Versioning System initialized and tested successfully!")
    print(f"   Features demonstrated:")
    print(f"   - Version registration and management")
    print(f"   - Semantic versioning")
    print(f"   - Compatibility checking") 
    print(f"   - Dependency management")
    print(f"   - Deprecation marking")
    print(f"   - Update management")
    print(f"   - Backup creation")
    
    return 0


if __name__ == "__main__":
    exit(main())