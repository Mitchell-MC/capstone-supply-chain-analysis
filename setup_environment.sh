#!/bin/bash

echo "========================================"
echo "FAF5.7 Supply Chain Analysis Environment Setup"
echo "========================================"

# Check if Python is installed
echo ""
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv faf5_env

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source faf5_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install packages
echo ""
echo "Installing required packages..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, xgboost; print('All packages installed successfully!')"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source faf5_env/bin/activate"
echo ""
echo "To start Jupyter Notebook, run:"
echo "  jupyter notebook"
echo "" 