#!/bin/bash

# Build script for SuperShader Rendering Process
set -e

echo "Building SuperShader Rendering Process..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake to generate build files
echo "Configuring with CMake..."
cmake ..

# Build the project
echo "Building renderer..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Renderer executable is in: build/bin/renderer"

# Go back to original directory
cd ..