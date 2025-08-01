#!/usr/bin/env python3
"""
Simple runner script for FAF5.7 dataset compression
"""

import subprocess
import sys
import os

def main():
    print("🚀 Running FAF5.7 Dataset Compression")
    print("=" * 40)
    
    # Check if the main script exists
    if not os.path.exists('create_faf5_compressed_dataset.py'):
        print("❌ Error: create_faf5_compressed_dataset.py not found!")
        return
    
    # Check if input file exists
    if not os.path.exists('FAF5.7_State.csv'):
        print("❌ Error: FAF5.7_State.csv not found!")
        print("   Please ensure the FAF5.7_State.csv file is in the current directory.")
        return
    
    print("✅ Files found, starting compression...")
    
    try:
        # Run the compression script
        result = subprocess.run([sys.executable, 'create_faf5_compressed_dataset.py'], 
                              capture_output=True, text=True)
        
        # Print output
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n🎉 Compression completed successfully!")
        else:
            print(f"\n❌ Compression failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running compression script: {e}")

if __name__ == "__main__":
    main()