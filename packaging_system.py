#!/usr/bin/env python3
"""
Packaging script for SuperShader management code
Packages the management components for distribution
"""

import os
import sys
import shutil
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List


class SuperShaderPackager:
    """System for packaging SuperShader management code for distribution"""
    
    def __init__(self, project_root=".", output_dir="dist"):
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define the core management components to include
        self.management_components = [
            "management/",
            "create_pseudocode_translator.py",
            "create_module_registry.py",
            "create_performance_system.py",
            "create_pseudocode_translator.py",
            "management/module_combiner.py",
            "management/module_linker.py",
            "management/shader_assembler.py",
            "modules/"
        ]
        
        # Define documentation files
        self.documentation = [
            "README.md",
            "PLAN.md",
            "QWEN.md",
            "TASKS.md",
            "DATA_FLOW_TECHNICAL_DOCS.md"
        ]
        
        # Define configuration files
        self.config_files = [
            "config_cel_shading.json",
            "config_pbr_with_shadows.json", 
            "config_phong_lighting.json",
            "config_raymarching.json",
            "data_flow_graph.json"
        ]
    
    def create_source_distribution(self, version="1.0.0") -> str:
        """Create a source distribution archive"""
        print(f"Creating source distribution for version {version}...")
        
        dist_name = f"supershader-{version}-src"
        archive_path = self.output_dir / f"{dist_name}.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add management components
            for component in self.management_components:
                src_path = self.project_root / component
                if src_path.exists():
                    if src_path.is_dir():
                        for root, dirs, files in os.walk(src_path):
                            for file in files:
                                file_path = Path(root) / file
                                # Add to archive with relative path
                                arc_path = file_path.relative_to(self.project_root.parent)
                                zipf.write(file_path, arc_path)
                    else:
                        # Add individual file
                        arc_path = src_path.relative_to(self.project_root.parent)
                        zipf.write(src_path, arc_path)
            
            # Add documentation
            for doc in self.documentation:
                doc_path = self.project_root / doc
                if doc_path.exists():
                    arc_path = doc_path.relative_to(self.project_root.parent)
                    zipf.write(doc_path, arc_path)
            
            # Add config files
            for config in self.config_files:
                config_path = self.project_root / config
                if config_path.exists():
                    arc_path = config_path.relative_to(self.project_root.parent)
                    zipf.write(config_path, arc_path)
        
        print(f"Source distribution created: {archive_path}")
        return str(archive_path.absolute())
    
    def create_wheel_distribution(self, version="1.0.0") -> str:
        """Create a wheel distribution archive"""
        print(f"Creating wheel distribution for version {version}...")
        
        dist_name = f"supershader-{version}-py3-none-any.whl"
        archive_path = self.output_dir / dist_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as wheel:
            # Add a wheel metadata file
            wheel_metadata = f"""Wheel-Version: 1.0
Generator: SuperShader-Packager
Root-Is-Purelib: true
Tag: py3-none-any
"""
            wheel.writestr("WHEEL", wheel_metadata)
            
            # Add management components with proper package structure
            for component in self.management_components:
                src_path = self.project_root / component
                if src_path.exists():
                    if src_path.is_dir():
                        for root, dirs, files in os.walk(src_path):
                            for file in files:
                                file_path = Path(root) / file
                                # Map to Python package structure
                                rel_path = file_path.relative_to(self.project_root)
                                package_path = f"supershader/{rel_path}"
                                wheel.write(file_path, package_path)
                    else:
                        # Add individual file
                        rel_path = src_path.relative_to(self.project_root)
                        package_path = f"supershader/{rel_path}"
                        wheel.write(src_path, package_path)
            
            # Add top-level package metadata
            metadata = f"""Metadata-Version: 2.1
Name: supershader
Version: {version}
Summary: Modular shader generation system with branching for conflicting features
Author: SuperShader Team
Author-email: 
Home-page: 
License: MIT
Classifier: Development Status :: 4 - Beta
Classifier: Intended Audience :: Developers
Classifier: License :: OSI Approved :: MIT License
Classifier: Programming Language :: Python :: 3
Classifier: Topic :: Software Development :: Libraries :: Python Modules
Classifier: Topic :: Multimedia :: Graphics
Requires-Dist: numpy

SuperShader: Advanced Modular Shader Generation System
"""
            wheel.writestr("supershader-{}.dist-info/METADATA".format(version), metadata)
            wheel.writestr("supershader-{}.dist-info/top_level.txt".format(version), "supershader")
        
        print(f"Wheel distribution created: {archive_path}")
        return str(archive_path.absolute())
    
    def create_tarball_distribution(self, version="1.0.0") -> str:
        """Create a tarball distribution archive"""
        print(f"Creating tarball distribution for version {version}...")
        
        dist_name = f"supershader-{version}-src.tar.gz"
        archive_path = self.output_dir / dist_name
        
        with tarfile.open(archive_path, "w:gz") as tar:
            # Add management components
            for component in self.management_components:
                src_path = self.project_root / component
                if src_path.exists():
                    # Add to archive with relative path
                    tar.add(src_path, arcname=component)
            
            # Add documentation
            for doc in self.documentation:
                doc_path = self.project_root / doc
                if doc_path.exists():
                    tar.add(doc_path, arcname=doc)
            
            # Add config files
            for config in self.config_files:
                config_path = self.project_root / config
                if config_path.exists():
                    tar.add(config_path, arcname=config)
        
        print(f"Tarball distribution created: {archive_path}")
        return str(archive_path.absolute())
    
    def create_install_script(self) -> str:
        """Create an installation script"""
        script_content = """#!/bin/bash
# SuperShader Installation Script

set -e  # Exit on any error

echo "Installing SuperShader..."

# Detect OS
OS=$(uname -s)
ARCH=$(uname -m)

echo "Target OS: $OS, Architecture: $ARCH"

# Create installation directory
INSTALL_DIR="/usr/local/supershader"

if [ ! -d "$INSTALL_DIR" ]; then
    echo "Creating installation directory: $INSTALL_DIR"
    sudo mkdir -p "$INSTALL_DIR"
fi

# Copy files to installation directory
echo "Copying SuperShader files..."
sudo cp -r management "$INSTALL_DIR/"
sudo cp -r modules "$INSTALL_DIR/"
sudo cp -r analysis "$INSTALL_DIR/"
sudo cp create_pseudocode_translator.py "$INSTALL_DIR/"
sudo cp create_module_registry.py "$INSTALL_DIR/"
sudo cp create_performance_system.py "$INSTALL_DIR/"
sudo cp create_pseudocode_translator.py "$INSTALL_DIR/"

# Create symlinks for CLI tools
echo "Creating command-line tools..."
sudo ln -sf "$INSTALL_DIR/create_pseudocode_translator.py" /usr/local/bin/supershader-translator
sudo ln -sf "$INSTALL_DIR/management/module_combiner.py" /usr/local/bin/supershader-combine

# Make executable
sudo chmod +x "$INSTALL_DIR/create_pseudocode_translator.py"
sudo chmod +x "$INSTALL_DIR/management/module_combiner.py"

echo "SuperShader installed successfully!"
echo "Installation location: $INSTALL_DIR"
echo "CLI tools available: supershader-translator, supershader-combine"

# Verify installation
echo "Verifying installation..."
python3 -c "import sys; sys.path.insert(0, '$INSTALL_DIR'); from create_pseudocode_translator import PseudocodeTranslator; print('Installation OK')"
"""
        
        script_path = self.output_dir / "install_supershader.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        print(f"Installation script created: {script_path}")
        return str(script_path.absolute())
    
    def create_setup_py(self) -> str:
        """Create a setup.py file for pip installation"""
        setup_content = '''#!/usr/bin/env python3
"""
Setup script for SuperShader package
"""
import setuptools
from pathlib import Path

# Read the long description from README
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Find all packages
def find_packages():
    """Find all packages in the project"""
    packages = []
    for root, dirs, files in os.walk("management"):
        if "__init__.py" in files or any(f.endswith(".py") for f in files):
            pkg = root.replace("/", ".").replace("\\", ".")
            packages.append(pkg)
    
    for root, dirs, files in os.walk("modules"):
        if "__init__.py" in files or any(f.endswith(".py") for f in files):
            pkg = root.replace("/", ".").replace("\\", ".")
            packages.append(pkg)
    
    return packages

setuptools.setup(
    name="supershader",
    version="1.0.0",
    author="SuperShader Team",
    author_email="",
    description="Modular shader generation system with branching for conflicting features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Add required dependencies here
    ],
    entry_points={
        "console_scripts": [
            "supershader-translator=create_pseudocode_translator:main",
            "supershader-combine=management.module_combiner:main",
        ],
    },
)
'''
        
        setup_path = self.output_dir / "setup.py"
        
        with open(setup_path, 'w') as f:
            f.write(setup_content)
        
        print(f"Setup file created: {setup_path}")
        return str(setup_path.absolute())
    
    def create_package_metadata(self, version="1.0.0") -> str:
        """Create package metadata file"""
        metadata = {
            "name": "supershader",
            "version": version,
            "description": "Modular shader generation system with branching for conflicting features",
            "author": "SuperShader Team",
            "license": "MIT",
            "components": {
                "management": self.management_components,
                "documentation": self.documentation,
                "config_files": self.config_files
            },
            "installation_date": datetime.now().isoformat(),
            "build_info": {
                "python_version": sys.version,
                "build_timestamp": datetime.now().isoformat()
            }
        }
        
        metadata_path = self.output_dir / "package_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Package metadata created: {metadata_path}")
        return str(metadata_path.absolute())
    
    def build_all_distributions(self, version="1.0.0") -> Dict[str, str]:
        """Build all types of distributions"""
        print(f"Building SuperShader distributions (version {version})...")
        
        results = {}
        
        # Create source distribution
        results['source_zip'] = self.create_source_distribution(version)
        
        # Create wheel distribution
        results['wheel'] = self.create_wheel_distribution(version)
        
        # Create tarball distribution
        results['tarball'] = self.create_tarball_distribution(version)
        
        # Create installation script
        results['install_script'] = self.create_install_script()
        
        # Create setup.py
        results['setup_py'] = self.create_setup_py()
        
        # Create package metadata
        results['metadata'] = self.create_package_metadata(version)
        
        print("\nDistribution building completed successfully!")
        print(f"Files created in {self.output_dir}:")
        for dist_type, path in results.items():
            print(f"  {dist_type}: {path}")
        
        return results


def main():
    """Main function to build SuperShader distributions"""
    print("Initializing SuperShader Packaging System...")
    
    packager = SuperShaderPackager(".", "distributions")
    
    # Build all distributions
    results = packager.build_all_distributions("1.0.0")
    
    print(f"\n✅ SuperShader management code packaged successfully for distribution!")
    print(f"  Created {len(results)} distribution artifacts")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)