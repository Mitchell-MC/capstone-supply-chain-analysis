#!/bin/bash

echo "========================================"
echo "Setting up FAF5.7 Analysis with Python 3.11"
echo "========================================"

# Remove existing environment if it exists
echo "Removing existing environment..."
conda env remove -n faf5-supply-chain -y 2>/dev/null || true

# Create new environment from environment.yml
echo "Creating new Python 3.11 environment..."
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
conda activate faf5-supply-chain

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To use this environment:"
echo "1. conda activate faf5-supply-chain"
echo "2. jupyter notebook"
echo ""
echo "Or run: python -m ipykernel install --user --name faf5-supply-chain --display-name 'Python 3.11 (FAF5)'"
echo "Then select this kernel in Jupyter" 