#!/usr/bin/env python3

# Author: KleaSCM
# Date: 2024
# Main entry point for the CLI application
# Description: Civil Engineering Neural Network System - Main entry point

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.cli.cli_interface import cli

if __name__ == '__main__':
    cli() 