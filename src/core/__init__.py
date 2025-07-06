# Author: KleaSCM
# Date: 2024
# Description: Core modules for civil engineering neural network system

# Temporarily comment out QueryEngine import due to syntax errors
# from .query_engine import QueryEngine
from .universal_reporter import UniversalReporter
from .system_integration import SystemIntegration
from .analyzers.risk_analyzer import RiskAnalyzer
from .analyzers.survey_analyzer import SurveyAnalyzer
from .report_formatter import ReportFormatter
from .dataset_config import DatasetConfig
from .content_analyzer import ContentAnalyzer, SmartTagger, CrossDatasetIntelligence, ContentDetector

__all__ = [
    'UniversalReporter',
    'SystemIntegration', 
    'RiskAnalyzer',
    'SurveyAnalyzer',
    'ReportFormatter',
    'DatasetConfig',
    'ContentAnalyzer',
    'SmartTagger',
    'CrossDatasetIntelligence', 
    'ContentDetector'
]

__version__ = "1.0.0" 