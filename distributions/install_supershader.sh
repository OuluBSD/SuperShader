#!/bin/bash
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
