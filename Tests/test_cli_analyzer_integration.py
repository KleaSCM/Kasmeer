#!/usr/bin/env python3
"""
Test CLI integration with analyzers
"""

import pandas as pd
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.cli.cli_interface import analyze
from src.core.universal_reporter import UniversalReporter
from src.core.analyzers import *

class TestCLIAnalyzerIntegration(unittest.TestCase):
    """Test CLI integration with analyzers"""
    
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
        
        # Create temporary data directory
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data.to_csv(os.path.join(self.temp_dir, 'test_construction.csv'), index=False)
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.cli.cli_interface.DataProcessor')
    @patch('src.cli.cli_interface.UniversalReporter')
    @patch('src.cli.cli_interface.Console')
    def test_analyze_command_with_analyzers(self, mock_console, mock_reporter, mock_data_processor):
        """Test that analyze command properly uses all analyzers"""
        # Mock the data processor
        mock_processor_instance = Mock()
        mock_processor_instance.discover_and_load_all_data.return_value = {
            'construction': self.sample_data
        }
        mock_data_processor.return_value = mock_processor_instance
        
        # Mock the universal reporter
        mock_reporter_instance = Mock()
        mock_reporter_instance.analyze_dataset.return_value = {
            'executive_summary': {'site_overview': 'Test site'},
            'site_materials': {'summary': ['Material 1', 'Material 2']},
            'work_history': {'summary': ['Project 1', 'Project 2']},
            'utilities_infrastructure': {'summary': ['Utility 1']},
            'environmental_context': {'summary': []},
            'costs_funding': {'summary': ['Cost 1']},
            'risks_hazards': {'summary': ['Risk 1']},
            'missing_data': {'critical_missing': []},
            'recommendations': {'immediate_actions': ['Action 1']},
            'nn_insights': {'nn_status': 'Working'},
            'survey_analysis': {'survey_data': 'Test'},
            'spatial_analysis': {'coordinate_analysis': {}},
            'temporal_analysis': {'time_series_analysis': {}},
            'cross_dataset_analysis': {'summary': ['Cross analysis']}
        }
        mock_reporter.return_value = mock_reporter_instance
        
        # Mock console
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        
        # Test the analyze function
        with patch('src.cli.cli_interface.find_coordinate_columns') as mock_find_coords:
            mock_find_coords.return_value = ('latitude', 'longitude')
            
            # Call the analyze function
            analyze('40.7128,-74.0060', self.temp_dir, None)
            
            # Verify that the universal reporter was called
            mock_reporter_instance.analyze_dataset.assert_called_once()
            
            # Verify that all analyzer results are present in the output
            call_args = mock_reporter_instance.analyze_dataset.call_args
            self.assertIsNotNone(call_args)
            
            # Verify that console.print was called multiple times (indicating output)
            self.assertGreater(mock_console_instance.print.call_count, 0)
    
    def test_analyzer_output_structure(self):
        """Test that all analyzers produce the expected output structure"""
        reporter = UniversalReporter()
        
        # Test with sample data
        result = reporter.analyze_dataset(self.sample_data)
        
        # Check that all expected keys are present
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
            self.assertIsInstance(result[key], (dict, list), f"Key {key} should be dict or list")
    
    def test_analyzer_data_extraction(self):
        """Test that analyzers extract meaningful data"""
        reporter = UniversalReporter()
        
        # Test with sample data
        result = reporter.analyze_dataset(self.sample_data)
        
        # Check that construction analyzer found project data
        self.assertIn('site_materials', result)
        self.assertIn('work_history', result)
        
        # Check that financial analyzer found cost data
        self.assertIn('costs_funding', result)
        
        # Check that spatial analyzer found coordinate data
        self.assertIn('spatial_analysis', result)
        
        # Check that risk analyzer found risk data
        self.assertIn('risks_hazards', result)
    
    def test_analyzer_error_recovery(self):
        """Test that analyzers recover from errors gracefully"""
        # Create problematic data that might cause issues
        problematic_data = pd.DataFrame({
            'latitude': ['invalid', 'also_invalid', 40.7128],
            'longitude': ['invalid', 'also_invalid', -74.0060],
            'Project Title': [None, '', 'Valid Project'],
            'Construction Award': ['not_a_number', 'also_not_a_number', 1000000]
        })
        
        reporter = UniversalReporter()
        
        # Should not raise an exception
        try:
            result = reporter.analyze_dataset(problematic_data)
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"Universal Reporter failed with error: {e}")
    
    def test_analyzer_performance(self):
        """Test that analyzers perform reasonably fast"""
        import time
        
        # Create larger dataset
        large_data = pd.concat([self.sample_data] * 100, ignore_index=True)
        
        reporter = UniversalReporter()
        
        # Time the analysis
        start_time = time.time()
        result = reporter.analyze_dataset(large_data)
        end_time = time.time()
        
        # Should complete within reasonable time (5 seconds)
        analysis_time = end_time - start_time
        self.assertLess(analysis_time, 5.0, f"Analysis took too long: {analysis_time:.2f} seconds")
        
        # Should still produce valid results
        self.assertIsInstance(result, dict)
        self.assertIn('executive_summary', result)

class TestAnalyzerDataQuality(unittest.TestCase):
    """Test analyzer data quality and validation"""
    
    def setUp(self):
        """Set up test data"""
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
    
    def test_analyzer_data_consistency(self):
        """Test that analyzers produce consistent results"""
        reporter = UniversalReporter()
        
        # Run analysis multiple times
        results = []
        for _ in range(3):
            result = reporter.analyze_dataset(self.sample_data)
            results.append(result)
        
        # Check that results are consistent
        for i in range(1, len(results)):
            # Compare key structures
            self.assertEqual(set(results[0].keys()), set(results[i].keys()))
            
            # Compare specific values that should be consistent
            self.assertEqual(
                results[0]['executive_summary']['site_overview'],
                results[i]['executive_summary']['site_overview']
            )
    
    def test_analyzer_data_completeness(self):
        """Test that analyzers handle missing data appropriately"""
        # Create data with missing values
        incomplete_data = pd.DataFrame({
            'latitude': [40.7128, None, 40.6782],
            'longitude': [-74.0060, None, -73.9442],
            'Project Title': ['P.S. 126 Upgrade', None, 'Stuyvesant HS'],
            'Construction Award': [26900000, None, 15200000],
            'School Name': ['P.S. 126', '', 'Stuyvesant HS']
        })
        
        reporter = UniversalReporter()
        
        # Should handle missing data gracefully
        result = reporter.analyze_dataset(incomplete_data)
        
        # Should still produce valid results
        self.assertIsInstance(result, dict)
        self.assertIn('executive_summary', result)
        self.assertIn('missing_data', result)
        
        # Should identify missing data
        missing_data = result['missing_data']
        self.assertIsInstance(missing_data, dict)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 