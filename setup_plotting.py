#!/usr/bin/env python3
"""
Setup script to install plotting dependencies for BitNet analysis.
"""

import subprocess
import sys

def install_packages():
    """Install required plotting packages."""
    packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0", 
        "pandas>=1.3.0"
    ]
    
    print("📦 Installing plotting dependencies...")
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All plotting dependencies installed successfully!")
    return True

if __name__ == "__main__":
    success = install_packages()
    if success:
        print("\n🎨 You can now run: python generate_plots.py")
    else:
        print("\n❌ Installation failed. Please install dependencies manually.") 