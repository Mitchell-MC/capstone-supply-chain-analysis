#!/usr/bin/env python3
"""
Quick setup script for FAF5.7 Supply Chain Resilience Analysis
This script will create a virtual environment and install all dependencies
"""

import subprocess
import sys
import os
import platform

def run_command(command, shell=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=shell, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("=" * 60)
    print("FAF5.7 Supply Chain Analysis - Quick Setup")
    print("=" * 60)
    
    # Check Python version
    print(f"\n1. Python Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required")
        return False
    
    # Determine platform
    is_windows = platform.system() == "Windows"
    venv_activate = "faf5_env\\Scripts\\activate" if is_windows else "source faf5_env/bin/activate"
    
    print(f"Platform: {platform.system()}")
    
    # Create virtual environment
    print("\n2. Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv faf5_env"):
        return False
    
    # Determine pip path
    pip_path = "faf5_env\\Scripts\\pip" if is_windows else "faf5_env/bin/pip"
    python_path = "faf5_env\\Scripts\\python" if is_windows else "faf5_env/bin/python"
    
    # Upgrade pip
    print("\n3. Upgrading pip...")
    if not run_command(f"{pip_path} install --upgrade pip"):
        return False
    
    # Install core packages first
    print("\n4. Installing core packages...")
    core_packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0"
    ]
    
    for package in core_packages:
        if not run_command(f"{pip_path} install {package}"):
            print(f"Warning: Failed to install {package}")
    
    # Install ML packages
    print("\n5. Installing ML packages...")
    ml_packages = [
        "xgboost>=1.6.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0"
    ]
    
    for package in ml_packages:
        if not run_command(f"{pip_path} install {package}"):
            print(f"Warning: Failed to install {package}")
    
    # Install optional packages
    print("\n6. Installing optional packages...")
    optional_packages = [
        "openpyxl>=3.0.0",
        "plotly>=5.0.0"
    ]
    
    for package in optional_packages:
        run_command(f"{pip_path} install {package}")  # Don't fail on optional
    
    # Test imports
    print("\n7. Testing package imports...")
    test_script = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
try:
    import xgboost as xgb
    print("✓ All packages imported successfully!")
    print(f"✓ Pandas: {pd.__version__}")
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ Scikit-learn: {sklearn.__version__}")
    print(f"✓ XGBoost: {xgb.__version__}")
except ImportError as e:
    print(f"✗ Import error: {e}")
    '''
    
    if run_command(f'{python_path} -c "{test_script}"'):
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        print("\nTo activate the environment:")
        if is_windows:
            print("  faf5_env\\Scripts\\activate")
        else:
            print("  source faf5_env/bin/activate")
        
        print("\nTo start Jupyter Notebook:")
        print("  jupyter notebook")
        
        print("\nTo run the analysis:")
        print("  Open FAF5_Supply_Chain_Resilience_Analysis.ipynb")
        
        return True
    else:
        print("\n❌ Setup failed - package import test failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 