#!/usr/bin/env python3
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
            pkg = root.replace("/", ".").replace("\", ".")
            packages.append(pkg)
    
    for root, dirs, files in os.walk("modules"):
        if "__init__.py" in files or any(f.endswith(".py") for f in files):
            pkg = root.replace("/", ".").replace("\", ".")
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
