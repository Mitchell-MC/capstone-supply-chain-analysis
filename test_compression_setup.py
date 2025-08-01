#!/usr/bin/env python3
"""
Test script to validate compression setup and requirements
"""

import os
import sys

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking FAF5.7 Compression Setup")
    print("=" * 40)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 6):
        print("❌ Python 3.6+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check required packages
    required_packages = ['pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 To install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check input file
    input_file = 'FAF5.7_State.csv'
    if os.path.exists(input_file):
        file_size = os.path.getsize(input_file) / (1024 * 1024)
        print(f"✅ {input_file} found ({file_size:.1f} MB)")
    else:
        print(f"❌ {input_file} not found")
        print("   Place the FAF5.7_State.csv file in the current directory")
        return False
    
    # Check compression script
    if os.path.exists('create_faf5_compressed_dataset.py'):
        print("✅ Compression script found")
    else:
        print("❌ create_faf5_compressed_dataset.py not found")
        return False
    
    print("\n🎯 Setup Status: READY")
    print("\nTo run compression:")
    print("  python create_faf5_compressed_dataset.py")
    print("  OR")
    print("  python run_compression.py")
    
    return True

if __name__ == "__main__":
    success = check_requirements()
    if not success:
        print("\n❌ Setup incomplete. Please resolve the issues above.")
        sys.exit(1)
    else:
        print("\n🚀 All checks passed! Ready to compress dataset.")