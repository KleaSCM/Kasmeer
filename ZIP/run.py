#!/usr/bin/env python3

# Author: KleaSCM
# Date: 2024
# Kasmeer Runner Script
# Description: Kasmeer Runner Script - Automatically uses virtual environment


import sys
import subprocess
from pathlib import Path

def main():
    # Run the Kasmeer system with virtual environment
    
    # Get the virtual environment Python path
    venv_python = Path(__file__).parent / "venv" / "bin" / "python"
    
    if not venv_python.exists():
        print("❌ Virtual environment not found!")
        print("Please run: python setup.py")
        sys.exit(1)
    
    # Pass all arguments to the main script
    args = sys.argv[1:] if len(sys.argv) > 1 else ["--help"]
    
    try:
        # Run the main script with virtual environment
        result = subprocess.run([str(venv_python), "main.py"] + args)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error running Kasmeer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 