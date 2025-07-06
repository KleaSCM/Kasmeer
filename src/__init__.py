# Author: KleaSCM
# Date: 2024
# Description: Kasmeer - Civil Engineering Neural Network System

# Temporarily comment out QueryEngine import due to syntax errors
# from .core import QueryEngine
from .ml import CivilEngineeringNN
from .data import DataProcessor
from .cli import cli

__all__ = ['CivilEngineeringNN', 'DataProcessor', 'cli']
__version__ = "1.0.0" 