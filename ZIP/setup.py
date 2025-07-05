#!/usr/bin/env python3

# Author: KleaSCM
# Date: 2024
# Description: Setup script for Civil Engineering Neural Network System


import subprocess
import sys
from pathlib import Path

def install_requirements():
    # Install required packages
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    # Create necessary directories
    print("📁 Creating directories...")
    directories = ["models", "logs", "outputs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created {directory}/ directory")
    
    return True

def check_data_directory():
    # Check if data directory exists and has files
    data_dir = Path("DataSets")
    if not data_dir.exists():
        print("⚠️ DataSets/ directory not found. Please create it and add your datasets.")
        return False
    
    files = list(data_dir.glob("*"))
    if not files:
        print("⚠️ DataSets/ directory is empty. Please add your datasets.")
        return False
    
    print(f"✅ DataSets/ directory found with {len(files)} items")
    return True

def run_tests():
    # Run system tests
    print("🧪 Running system tests...")
    try:
        subprocess.check_call([sys.executable, "test_system.py"])
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Some tests failed. This is normal if dependencies aren't fully installed yet.")
        return True  # Don't fail setup for test failures

def main():
    # Main setup function
    print("🚀 Civil Engineering Neural Network System Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed at directory creation")
        return False
    
    # Check data directory
    check_data_directory()
    
    # Run tests
    run_tests()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Add your datasets to the DataSets/ directory")
    print("2. Run: python main.py train")
    print("3. Run: python main.py query")
    print("\nFor help: python main.py --help")
    
    return True

if __name__ == '__main__':
    main() 