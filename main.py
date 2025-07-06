#!/usr/bin/env python3
"""
Kasmeer - Civil Engineering AI Analysis System
Main entry point for the integrated system
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.system_integration import SystemIntegration
from src.cli.cli_interface import cli
from src.utils.logging_utils import setup_logging

logger = setup_logging(__name__)

def main():
    """Main entry point for the Kasmeer system"""
    try:
        logger.info("Starting Kasmeer - Civil Engineering AI Analysis System")
        
        # Initialize the integrated system
        system = SystemIntegration()
        
        # Check system status
        status = system.get_system_status()
        logger.info(f"System Status: {status}")
        
        # Start CLI interface
        cli()
        
    except KeyboardInterrupt:
        logger.info("System shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 