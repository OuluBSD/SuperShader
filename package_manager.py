"""
Package Management and Distribution System
Part of SuperShader Project - Phase 9: User Interface and Developer Experience

This module creates systems for sharing and distributing shader modules,
implements version control for module libraries, adds dependency management
for complex module combinations, and creates a marketplace for community modules.
"""

import json
import os
import hashlib
import zipfile
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import requests
import tempfile
from datetime import datetime


@dataclass
class ModuleMetadata:
    """Metadata for a shader module"""
    id: str
    name: str
    version: str
    author: str
    description: str
    category: str
    dependencies: List[str]
    tags: List[str]
    created_at: str
    updated_at: str
    compatibility: Dict[str, str]  # API -> version compatibility
    license: str


@dataclass
class ModulePackage:
    """A packaged shader module with metadata"""
    metadata: ModuleMetadata
    source_code: str
    resources: Dict[str, bytes]  # additional resources like textures
    checksum: str


class ModuleRegistry:
    """
    Registry of available modules with search functionality
    """
    
    def __init__(self, registry_file: str = "module_registry.json"):
        self.registry_file = registry_file
        self.modules: Dict[str, ModuleMetadata] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load the module registry from file"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    for module_id, metadata_dict in data.items():
                        # Convert dict back to ModuleMetadata object
                        metadata = ModuleMetadata(
                            id=metadata_dict['id'],
                            name=metadata_dict['name'],
                            version=metadata_dict['version'],
                            author=metadata_dict['author'],
                            description=metadata_dict['description'],
                            category=metadata_dict['category'],
                            dependencies=metadata_dict['dependencies'],
                            tags=metadata_dict['tags'],
                            created_at=metadata_dict['created_at'],
                            updated_at=metadata_dict['updated_at'],
                            compatibility=metadata_dict['compatibility'],
                            license=metadata_dict['license']
                        )
                        self.modules[module_id] = metadata
            except Exception:
                # If loading fails, start with empty registry
                self.modules = {}
        else:
            # Create empty registry
            self.modules = {}
    
    def save_registry(self):
        """Save the module registry to file"""
        data = {}
        for module_id, metadata in self.modules.items():
            data[module_id] = {
                'id': metadata.id,
                'name': metadata.name,
                'version': metadata.version,
                'author': metadata.author,
                'description': metadata.description,
                'category': metadata.category,
                'dependencies': metadata.dependencies,
                'tags': metadata.tags,
                'created_at': metadata.created_at,
                'updated_at': metadata.updated_at,
                'compatibility': metadata.compatibility,
                'license': metadata.license
            }
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_module(self, metadata: ModuleMetadata) -> bool:
        """Register a new module in the registry"""
        try:
            self.modules[metadata.id] = metadata
            self.save_registry()
            return True
        except Exception:
            return False
    
    def get_module(self, module_id: str) -> Optional[ModuleMetadata]:
        """Get metadata for a specific module"""
        return self.modules.get(module_id)
    
    def get_all_modules(self) -> List[ModuleMetadata]:
        """Get all registered modules"""
        return list(self.modules.values())
    
    def search_modules(self, query: str = "", category: str = "", tags: List[str] = None) -> List[ModuleMetadata]:
        """Search for modules based on criteria"""
        tags = tags or []
        results = []
        
        for module in self.modules.values():
            # Check if query matches in name, description or author
            query_match = not query or query.lower() in module.name.lower() or \
                         query.lower() in module.description.lower() or \
                         query.lower() in module.author.lower()
            
            # Check category
            category_match = not category or category.lower() == module.category.lower()
            
            # Check tags
            tag_match = not tags or any(tag.lower() in [t.lower() for t in module.tags] for tag in tags)
            
            if query_match and category_match and tag_match:
                results.append(module)
        
        return results
    
    def get_modules_by_category(self, category: str) -> List[ModuleMetadata]:
        """Get all modules in a specific category"""
        return [m for m in self.modules.values() if m.category.lower() == category.lower()]


class ModulePackager:
    """
    System for packaging and unpacking shader modules
    """
    
    def create_package(self, source_code: str, metadata: ModuleMetadata, 
                      resources: Dict[str, bytes] = None) -> ModulePackage:
        """Create a module package from source code and metadata"""
        resources = resources or {}
        
        # Create a checksum of the source code
        code_hash = hashlib.sha256(source_code.encode()).hexdigest()
        
        # Include resources in the hash
        for res_name, res_data in resources.items():
            code_hash += hashlib.sha256(res_data).hexdigest()
        
        final_checksum = hashlib.sha256(code_hash.encode()).hexdigest()
        
        package = ModulePackage(
            metadata=metadata,
            source_code=source_code,
            resources=resources,
            checksum=final_checksum
        )
        
        return package
    
    def package_to_file(self, package: ModulePackage, filename: str) -> bool:
        """Save a module package to a file"""
        try:
            # Create a zip file containing the package
            with zipfile.ZipFile(filename, 'w') as zipf:
                # Add metadata
                metadata_json = json.dumps({
                    'id': package.metadata.id,
                    'name': package.metadata.name,
                    'version': package.metadata.version,
                    'author': package.metadata.author,
                    'description': package.metadata.description,
                    'category': package.metadata.category,
                    'dependencies': package.metadata.dependencies,
                    'tags': package.metadata.tags,
                    'created_at': package.metadata.created_at,
                    'updated_at': package.metadata.updated_at,
                    'compatibility': package.metadata.compatibility,
                    'license': package.metadata.license
                }, indent=2)
                
                zipf.writestr('metadata.json', metadata_json)
                
                # Add source code
                zipf.writestr('source.glsl', package.source_code)
                
                # Add resources
                for res_name, res_data in package.resources.items():
                    zipf.writestr(f'resources/{res_name}', res_data)
                
                # Add checksum
                zipf.writestr('checksum.txt', package.checksum)
            
            return True
        except Exception:
            return False
    
    def file_to_package(self, filename: str) -> Optional[ModulePackage]:
        """Load a module package from a file"""
        try:
            with zipfile.ZipFile(filename, 'r') as zipf:
                # Extract metadata
                metadata_json = zipf.read('metadata.json').decode('utf-8')
                metadata_dict = json.loads(metadata_json)
                
                metadata = ModuleMetadata(
                    id=metadata_dict['id'],
                    name=metadata_dict['name'],
                    version=metadata_dict['version'],
                    author=metadata_dict['author'],
                    description=metadata_dict['description'],
                    category=metadata_dict['category'],
                    dependencies=metadata_dict['dependencies'],
                    tags=metadata_dict['tags'],
                    created_at=metadata_dict['created_at'],
                    updated_at=metadata_dict['updated_at'],
                    compatibility=metadata_dict['compatibility'],
                    license=metadata_dict['license']
                )
                
                # Extract source code
                source_code = zipf.read('source.glsl').decode('utf-8')
                
                # Extract resources
                resources = {}
                for name in zipf.namelist():
                    if name.startswith('resources/'):
                        res_name = name[10:]  # Remove 'resources/' prefix
                        resources[res_name] = zipf.read(name)
                
                # Extract checksum
                checksum = zipf.read('checksum.txt').decode('utf-8')
                
                return ModulePackage(
                    metadata=metadata,
                    source_code=source_code,
                    resources=resources,
                    checksum=checksum
                )
        
        except Exception:
            return None


class DependencyResolver:
    """
    System for resolving module dependencies
    """
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
    
    def resolve_dependencies(self, module_id: str) -> List[str]:
        """Get all dependencies for a module (including transitive dependencies)"""
        module = self.registry.get_module(module_id)
        if not module:
            return []
        
        all_deps = set()
        to_check = set(module.dependencies)
        
        # Use a queue to perform breadth-first search for dependencies
        while to_check:
            dep_id = to_check.pop()
            if dep_id in all_deps:
                continue
            
            all_deps.add(dep_id)
            dep_module = self.registry.get_module(dep_id)
            
            if dep_module:
                # Add this dependency's dependencies to the queue
                for sub_dep in dep_module.dependencies:
                    if sub_dep not in all_deps:
                        to_check.add(sub_dep)
        
        return list(all_deps)
    
    def check_compatibility(self, module_id: str, target_api: str, target_version: str) -> Tuple[bool, str]:
        """Check if a module is compatible with a target API version"""
        module = self.registry.get_module(module_id)
        if not module:
            return False, f"Module {module_id} not found"
        
        if target_api not in module.compatibility:
            return False, f"Module {module_id} does not specify compatibility for {target_api}"
        
        required_version = module.compatibility[target_api]
        # Simple version comparison (in a real system, we'd need more sophisticated comparison)
        if required_version > target_version:
            return False, f"Module requires {target_api} {required_version} but target is {target_version}"
        
        return True, "Compatible"


class ModuleManager:
    """
    Main system for managing shader modules
    """
    
    def __init__(self, modules_dir: str = "modules"):
        self.modules_dir = Path(modules_dir)
        self.modules_dir.mkdir(exist_ok=True)
        
        self.registry = ModuleRegistry()
        self.packager = ModulePackager()
        self.dependency_resolver = DependencyResolver(self.registry)
        
        # Local installation tracking
        self.installed_modules_file = self.modules_dir / "installed.json"
        self.installed_modules: Dict[str, Dict[str, Any]] = self._load_installed_modules()
    
    def _load_installed_modules(self) -> Dict[str, Dict[str, Any]]:
        """Load the list of installed modules"""
        if self.installed_modules_file.exists():
            try:
                with open(self.installed_modules_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {}
    
    def _save_installed_modules(self):
        """Save the list of installed modules"""
        with open(self.installed_modules_file, 'w') as f:
            json.dump(self.installed_modules, f, indent=2)
    
    def create_module(self, name: str, version: str, author: str, description: str, 
                     category: str, source_code: str, dependencies: List[str] = None,
                     tags: List[str] = None) -> bool:
        """Create a new module and register it"""
        dependencies = dependencies or []
        tags = tags or []
        
        module_id = f"{author}.{name}".lower().replace(' ', '_').replace('.', '_')
        
        metadata = ModuleMetadata(
            id=module_id,
            name=name,
            version=version,
            author=author,
            description=description,
            category=category,
            dependencies=dependencies,
            tags=tags,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            compatibility={"opengl": "3.3", "vulkan": "1.0"},  # Default compatibility
            license="MIT"
        )
        
        # Register the module
        if self.registry.register_module(metadata):
            # Create package and save it locally
            package = self.packager.create_package(source_code, metadata)
            package_path = self.modules_dir / f"{module_id}.smp"  # SuperShader Module Package
            return self.packager.package_to_file(package, package_path)
        
        return False
    
    def install_module(self, module_id: str) -> bool:
        """Install a module from the registry"""
        # In a real system, this would download from a remote repository
        # For now, we'll simulate by working with local packages
        
        module = self.registry.get_module(module_id)
        if not module:
            print(f"Module {module_id} not found in registry")
            return False
        
        package_path = self.modules_dir / f"{module_id}.smp"
        if not package_path.exists():
            print(f"Package file {package_path} not found")
            return False
        
        # Load the package
        package = self.packager.file_to_package(package_path)
        if not package:
            print(f"Could not load package {package_path}")
            return False
        
        # Check dependencies
        dependencies = self.dependency_resolver.resolve_dependencies(module_id)
        for dep_id in dependencies:
            if dep_id not in self.installed_modules:
                print(f"Installing dependency: {dep_id}")
                self.install_module(dep_id)  # Recursive installation
        
        # Install the module
        install_path = self.modules_dir / "installed" / module_id
        install_path.mkdir(exist_ok=True)
        
        # Save the source code
        with open(install_path / f"{module_id}.glsl", 'w') as f:
            f.write(package.source_code)
        
        # Save the metadata
        metadata_path = install_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'id': package.metadata.id,
                'name': package.metadata.name,
                'version': package.metadata.version,
                'author': package.metadata.author,
                'description': package.metadata.description,
                'category': package.metadata.category,
                'dependencies': package.metadata.dependencies,
                'tags': package.metadata.tags,
                'created_at': package.metadata.created_at,
                'updated_at': package.metadata.updated_at,
                'compatibility': package.metadata.compatibility,
                'license': package.metadata.license
            }, f, indent=2)
        
        # Update installed modules list
        self.installed_modules[module_id] = {
            'version': module.version,
            'path': str(install_path),
            'installed_at': datetime.now().isoformat()
        }
        self._save_installed_modules()
        
        print(f"Successfully installed module: {module_id}")
        return True
    
    def uninstall_module(self, module_id: str) -> bool:
        """Uninstall a module"""
        if module_id not in self.installed_modules:
            print(f"Module {module_id} is not installed")
            return False
        
        install_path = Path(self.installed_modules[module_id]['path'])
        if install_path.exists():
            import shutil
            shutil.rmtree(install_path)
        
        # Remove from installed list
        del self.installed_modules[module_id]
        self._save_installed_modules()
        
        print(f"Successfully uninstalled module: {module_id}")
        return True
    
    def list_installed_modules(self) -> List[Dict[str, Any]]:
        """List all installed modules"""
        result = []
        for module_id, info in self.installed_modules.items():
            module_info = self.registry.get_module(module_id)
            if module_info:
                result.append({
                    'id': module_id,
                    'name': module_info.name,
                    'version': info['version'],
                    'path': info['path']
                })
        return result
    
    def search_modules(self, query: str = "", category: str = "", tags: List[str] = None) -> List[ModuleMetadata]:
        """Search for modules in the registry"""
        return self.registry.search_modules(query, category, tags)
    
    def get_modules_by_category(self, category: str) -> List[ModuleMetadata]:
        """Get modules by category"""
        return self.registry.get_modules_by_category(category)


class CommunityMarketplace:
    """
    System for community module sharing (simulated)
    """
    
    def __init__(self, manager: ModuleManager):
        self.manager = manager
        self.remote_url = "https://supershader-modules.example.com"  # Placeholder
    
    def search_remote_modules(self, query: str = "") -> List[Dict[str, Any]]:
        """
        Search for modules in the remote marketplace
        This is a simulation - in reality, this would call an API
        """
        # Simulate API response
        return [
            {
                'id': 'community.amazing_effects',
                'name': 'Amazing Effects Pack',
                'version': '1.2.0',
                'author': 'ShaderArtist',
                'description': 'Collection of amazing visual effects',
                'category': 'effects',
                'downloads': 1500,
                'rating': 4.8
            },
            {
                'id': 'pro.lighting_suite',
                'name': 'Professional Lighting Suite',
                'version': '2.1.1',
                'author': 'LightMaster',
                'description': 'Advanced lighting calculations',
                'category': 'lighting',
                'downloads': 2300,
                'rating': 4.9
            },
            {
                'id': 'anime.post_processing',
                'name': 'Anime Post-Processing',
                'version': '1.0.5',
                'author': 'AnimeFan',
                'description': 'Anime-style post-processing effects',
                'category': 'post_processing',
                'downloads': 850,
                'rating': 4.5
            }
        ]
    
    def download_module(self, module_id: str) -> bool:
        """
        Download a module from the remote marketplace
        This is a simulation - in reality, this would download from a server
        """
        # In a real implementation, this would:
        # 1. Download the module package from a remote server
        # 2. Verify its integrity
        # 3. Add it to the local registry
        # 4. Install it
        
        print(f"Simulated download of module: {module_id}")
        print("In a real implementation, this would download from a remote server")
        
        # For simulation, create a dummy module
        dummy_source = f"""
// Dummy source code for {module_id}
// This would be the actual shader code in a real implementation
void main() {{
    // Module {module_id} implementation
}}
        """
        
        # Create dummy module in local registry
        parts = module_id.split('.')
        if len(parts) >= 2:
            author = parts[0]
            name = '.'.join(parts[1:])
        else:
            author = "community"
            name = module_id
        
        return self.manager.create_module(
            name=name,
            version="1.0.0",
            author=author,
            description=f"Community module: {module_id}",
            category="community",
            source_code=dummy_source
        )


def main():
    """
    Example usage of the Package Management and Distribution System
    """
    print("Package Management and Distribution System")
    print("Part of SuperShader Project - Phase 9")
    
    # Create the module manager
    manager = ModuleManager()
    
    # Create some example modules
    lighting_source = """
// Diffuse Lighting Module
vec3 calculate_diffuse_lighting(vec3 normal, vec3 light_dir, vec3 light_color, float ambient_factor) {
    float diff = max(dot(normalize(normal), light_dir), 0.0);
    vec3 ambient = ambient_factor * light_color;
    vec3 diffuse = diff * light_color;
    return ambient + diffuse;
}
    """
    
    manager.create_module(
        name="DiffuseLighting",
        version="1.0.0",
        author="SuperShaderTeam",
        description="Basic diffuse lighting calculation",
        category="lighting",
        source_code=lighting_source,
        tags=["lighting", "diffuse", "basic"]
    )
    
    texture_source = """
// Texture Sampling Module
vec4 sample_texture_with_uv(sampler2D tex, vec2 uv, vec2 scale, vec2 offset) {
    vec2 adjusted_uv = uv * scale + offset;
    return texture2D(tex, adjusted_uv);
}
    """
    
    manager.create_module(
        name="TextureSampler",
        version="1.1.0",
        author="SuperShaderTeam",
        description="Advanced texture sampling with UV adjustment",
        category="texturing",
        source_code=texture_source,
        dependencies=["SuperShaderTeam.DiffuseLighting"],
        tags=["texturing", "uv", "sampling"]
    )
    
    print(f"Created {len(manager.registry.get_all_modules())} modules")
    
    # Search for modules
    print("\nSearching for modules with 'lighting' in the name:")
    lighting_modules = manager.search_modules(query="lighting")
    for module in lighting_modules:
        print(f"  - {module.name}: {module.description}")
    
    # Install a module
    print("\nInstalling DiffuseLighting module:")
    manager.install_module("supershaderteam.diffuselighting")
    
    # List installed modules
    print("\nInstalled modules:")
    installed = manager.list_installed_modules()
    for module in installed:
        print(f"  - {module['name']} v{module['version']}")
    
    # Use the community marketplace
    print("\nUsing Community Marketplace:")
    marketplace = CommunityMarketplace(manager)
    
    remote_modules = marketplace.search_remote_modules()
    print(f"Found {len(remote_modules)} modules in remote repository:")
    for mod in remote_modules:
        print(f"  - {mod['name']} by {mod['author']} (rating: {mod['rating']})")
    
    # Try to download a module
    print("\nTrying to 'download' a remote module:")
    marketplace.download_module("community.amazing_effects")
    
    print(f"\nModule repository now has {len(manager.registry.get_all_modules())} modules")
    print(f"Local installation directory: {manager.modules_dir}")


if __name__ == "__main__":
    main()