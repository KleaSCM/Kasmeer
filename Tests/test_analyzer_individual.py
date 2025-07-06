#!/usr/bin/env python3
"""
Individual analyzer tests to ensure each analyzer works correctly
"""

import pandas as pd
import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.analyzers import *
from src.data.data_processor import DataProcessor

class TestIndividualAnalyzers(unittest.TestCase):
    """Test each analyzer individually"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'latitude': [40.7128, 40.7589, 40.6782],
            'longitude': [-74.0060, -73.9851, -73.9442],
            'Project Title': ['P.S. 126 Upgrade', 'P.S. 130 Playground', 'Stuyvesant HS'],
            'Project Description': ['Roof/parapet/window upgrades', 'Playground upgrade', 'Structural improvements'],
            'Estimated Contract Value': ['$26.9M', '$478K', '$15.2M'],
            'Construction Award': [26900000, 478000, 15200000],
            'School Name': ['P.S. 126', 'P.S. 130', 'Stuyvesant HS'],
            'Campus Name': ['Campus A', 'Campus B', 'Campus C'],
            'Project ID': ['PRJ001', 'PRJ002', 'PRJ003'],
            'Borough': ['Manhattan', 'Brooklyn', 'Manhattan'],
            'Contract Advertise Date': ['01/15/2023', '02/20/2023', '03/10/2023'],
            'Geographical District': [78, 13, 1]
        })
        
        # Load real data for comprehensive testing
        try:
            data_processor = DataProcessor('/home/klea/Documents/Dev/AI/DataSets')
            self.real_data = data_processor.discover_and_load_all_data()
        except Exception as e:
            print(f"Warning: Could not load real data: {e}")
            self.real_data = {}
    
    def test_construction_analyzer(self):
        """Test ConstructionAnalyzer"""
        analyzer = ConstructionAnalyzer()
        
        # Test with sample data
        result = analyzer.analyze(self.sample_data)
        
        # Check structure
        self.assertIn('site_materials', result)
        self.assertIn('work_history', result)
        
        # Check that it found data
        self.assertTrue(isinstance(result['site_materials'], dict))
        self.assertTrue(isinstance(result['work_history'], dict))
        
        # Test with real data if available
        if 'construction' in self.real_data:
            real_result = analyzer.analyze(self.real_data['construction'])
            self.assertIn('site_materials', real_result)
            self.assertIn('work_history', real_result)
    
    def test_infrastructure_analyzer(self):
        """Test InfrastructureAnalyzer"""
        analyzer = InfrastructureAnalyzer()
        
        # Test with sample data
        result = analyzer.analyze(self.sample_data)
        
        # Check structure
        self.assertIn('utilities_infrastructure', result)
        
        # Check that it found data
        self.assertTrue(isinstance(result['utilities_infrastructure'], dict))
        
        # Test with real data if available
        if 'infrastructure' in self.real_data:
            real_result = analyzer.analyze(self.real_data['infrastructure'])
            self.assertIn('utilities_infrastructure', real_result)
    
    def test_environmental_analyzer(self):
        """Test EnvironmentalAnalyzer"""
        analyzer = EnvironmentalAnalyzer()
        
        # Test with sample data
        result = analyzer.analyze(self.sample_data)
        
        # Check structure
        self.assertIn('environmental_context', result)
        
        # Check that it found data (may be empty for construction data)
        self.assertTrue(isinstance(result['environmental_context'], dict))
        
        # Test with real data if available
        if 'environmental' in self.real_data:
            real_result = analyzer.analyze(self.real_data['environmental'])
            self.assertIn('environmental_context', real_result)
    
    def test_financial_analyzer(self):
        """Test FinancialAnalyzer"""
        analyzer = FinancialAnalyzer()
        
        # Test with sample data
        result = analyzer.analyze(self.sample_data)
        
        # Check structure
        self.assertIn('costs_funding', result)
        
        # Check that it found data
        self.assertTrue(isinstance(result['costs_funding'], dict))
        
        # Test with real data if available
        if 'construction' in self.real_data:
            real_result = analyzer.analyze(self.real_data['construction'])
            self.assertIn('costs_funding', real_result)
    
    def test_risk_analyzer(self):
        """Test RiskAnalyzer"""
        analyzer = RiskAnalyzer()
        
        # Test with sample data
        result = analyzer.analyze(self.sample_data)
        
        # Check structure
        self.assertIn('risk_assessment', result)
        self.assertIn('summary', result)
        
        # Check that it found data
        self.assertTrue(isinstance(result['risk_assessment'], dict))
        self.assertTrue(isinstance(result['summary'], list))
        
        # Test with real data if available
        if 'construction' in self.real_data:
            real_result = analyzer.analyze(self.real_data['construction'])
            self.assertIn('risk_assessment', real_result)
            self.assertIn('summary', real_result)
    
    def test_spatial_analyzer(self):
        """Test SpatialAnalyzer"""
        analyzer = SpatialAnalyzer()
        
        # Test with sample data
        result = analyzer.analyze(self.sample_data)
        
        # Check structure
        self.assertIn('spatial_analysis', result)
        
        # Check that it found data
        self.assertTrue(isinstance(result['spatial_analysis'], dict))
        
        # Test with location context
        location = {'lat': 40.7128, 'lon': -74.0060}
        result_with_location = analyzer.analyze(self.sample_data, location=location)
        self.assertIn('spatial_analysis', result_with_location)
    
    def test_temporal_analyzer(self):
        """Test TemporalAnalyzer"""
        analyzer = TemporalAnalyzer()
        
        # Test with sample data
        result = analyzer.analyze(self.sample_data)
        
        # Check structure
        self.assertIn('temporal_analysis', result)
        
        # Check that it found data
        self.assertTrue(isinstance(result['temporal_analysis'], dict))
        
        # Test with real data if available
        if 'construction' in self.real_data:
            real_result = analyzer.analyze(self.real_data['construction'])
            self.assertIn('temporal_analysis', real_result)
    
    def test_cross_dataset_analyzer(self):
        """Test CrossDatasetAnalyzer"""
        analyzer = CrossDatasetAnalyzer()
        
        # Test with multiple datasets
        datasets = {
            'construction': self.sample_data,
            'infrastructure': self.sample_data.copy()
        }
        
        result = analyzer.analyze(datasets)
        
        # Check structure
        self.assertIn('cross_dataset_analysis', result)
        self.assertIn('spatial_relationships', result)
        self.assertIn('temporal_relationships', result)
        self.assertIn('data_quality_comparison', result)
        self.assertIn('correlations', result)
        self.assertIn('anomalies', result)
        self.assertIn('summary', result)
        
        # Check that it found data
        self.assertTrue(isinstance(result['cross_dataset_analysis'], dict))
        self.assertTrue(isinstance(result['summary'], list))
        
        # Test with real data if available
        if len(self.real_data) > 1:
            real_result = analyzer.analyze(self.real_data)
            self.assertIn('cross_dataset_analysis', real_result)
            self.assertIn('summary', real_result)
    
    def test_survey_analyzer(self):
        """Test SurveyAnalyzer"""
        analyzer = SurveyAnalyzer()
        
        # Test with sample data
        result = analyzer.analyze(self.sample_data)
        
        # Check structure
        self.assertIn('survey_analysis', result)
        
        # Check that it found data
        self.assertTrue(isinstance(result['survey_analysis'], dict))
        
        # Test with real data if available
        if 'survey' in self.real_data:
            real_result = analyzer.analyze(self.real_data['survey'])
            self.assertIn('survey_analysis', real_result)

class TestAnalyzerIntegration(unittest.TestCase):
    """Test analyzer integration with Universal Reporter"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'latitude': [40.7128, 40.7589, 40.6782],
            'longitude': [-74.0060, -73.9851, -73.9442],
            'Project Title': ['P.S. 126 Upgrade', 'P.S. 130 Playground', 'Stuyvesant HS'],
            'Project Description': ['Roof/parapet/window upgrades', 'Playground upgrade', 'Structural improvements'],
            'Estimated Contract Value': ['$26.9M', '$478K', '$15.2M'],
            'Construction Award': [26900000, 478000, 15200000],
            'School Name': ['P.S. 126', 'P.S. 130', 'Stuyvesant HS'],
            'Campus Name': ['Campus A', 'Campus B', 'Campus C'],
            'Project ID': ['PRJ001', 'PRJ002', 'PRJ003'],
            'Borough': ['Manhattan', 'Brooklyn', 'Manhattan'],
            'Contract Advertise Date': ['01/15/2023', '02/20/2023', '03/10/2023'],
            'Geographical District': [78, 13, 1]
        })
    
    def test_universal_reporter_integration(self):
        """Test that Universal Reporter properly integrates all analyzers"""
        from src.core.universal_reporter import UniversalReporter
        
        reporter = UniversalReporter()
        
        # Test with sample data
        result = reporter.analyze_dataset(self.sample_data)
        
        # Check that all analyzer results are present
        expected_keys = [
            'executive_summary',
            'site_materials',
            'work_history',
            'utilities_infrastructure',
            'environmental_context',
            'costs_funding',
            'risks_hazards',
            'missing_data',
            'recommendations',
            'nn_insights',
            'survey_analysis',
            'spatial_analysis',
            'temporal_analysis',
            'cross_dataset_analysis'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")
    
    def test_analyzer_error_handling(self):
        """Test that analyzers handle errors gracefully"""
        # Create problematic data
        problematic_data = pd.DataFrame({
            'invalid_column': ['not_a_number', 'also_not_a_number'],
            'empty_column': [None, None],
            'mixed_types': [1, 'string']
        })
        
        analyzers = [
            ConstructionAnalyzer(),
            InfrastructureAnalyzer(),
            EnvironmentalAnalyzer(),
            FinancialAnalyzer(),
            RiskAnalyzer(),
            SpatialAnalyzer(),
            TemporalAnalyzer(),
            CrossDatasetAnalyzer(),
            SurveyAnalyzer()
        ]
        
        for analyzer in analyzers:
            try:
                result = analyzer.analyze(problematic_data)
                # Should not raise an exception
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"Analyzer {analyzer.__class__.__name__} failed with error: {e}")

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 