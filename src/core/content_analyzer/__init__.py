# Author: KleaSCM
# Date: 2024
# Description: Content Analyzer - AI-powered dataset content detection and auto-tagging

from .content_analyzer import ContentAnalyzer
from .smart_tagger import SmartTagger
from .cross_dataset_intelligence import CrossDatasetIntelligence
from .content_detector import ContentDetector

__all__ = [
    'ContentAnalyzer',
    'SmartTagger', 
    'CrossDatasetIntelligence',
    'ContentDetector'
]

__version__ = "1.0.0" 