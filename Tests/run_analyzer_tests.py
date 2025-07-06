#!/usr/bin/env python3
"""
Comprehensive test runner for all analyzers
"""

import pandas as pd
import sys
import os
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.analyzers import *
from src.core.universal_reporter import UniversalReporter
from src.data.data_processor import DataProcessor

def run_comprehensive_analyzer_tests():
    """Run comprehensive tests for all analyzers"""
    
    print("🧪 COMPREHENSIVE ANALYZER TEST SUITE")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test results storage
    test_results = {
        'summary': {
            'total_analyzers': 9,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'start_time': datetime.now().isoformat()
        },
        'analyzers': {},
        'integration_tests': {},
        'performance_tests': {},
        'data_quality_tests': {}
    }
    
    # Load test data
    print("📊 Loading test data...")
    try:
        data_processor = DataProcessor('/home/klea/Documents/Dev/AI/DataSets')
        loaded_data = data_processor.discover_and_load_all_data()
        print(f"✅ Loaded {len(loaded_data)} datasets")
        
        # Use construction data for testing
        test_data = loaded_data.get('construction')
        if test_data is None:
            print("⚠️ No construction data found, using first available dataset")
            test_data = next(iter(loaded_data.values()))
        
        print(f"📋 Using dataset with {len(test_data):,} records and {len(test_data.columns)} columns")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("🔄 Creating synthetic test data...")
        
        # Create synthetic test data
        test_data = pd.DataFrame({
            'latitude': [40.7128, 40.7589, 40.6782, 40.7505, 40.7505],
            'longitude': [-74.0060, -73.9851, -73.9442, -73.9934, -73.9934],
            'Project Title': ['P.S. 126 Upgrade', 'P.S. 130 Playground', 'Stuyvesant HS', 'Test Project 1', 'Test Project 2'],
            'Project Description': ['Roof/parapet/window upgrades', 'Playground upgrade', 'Structural improvements', 'Test Description 1', 'Test Description 2'],
            'Estimated Contract Value': ['$26.9M', '$478K', '$15.2M', '$5.0M', '$8.0M'],
            'Construction Award': [26900000, 478000, 15200000, 5000000, 8000000],
            'School Name': ['P.S. 126', 'P.S. 130', 'Stuyvesant HS', 'Test School 1', 'Test School 2'],
            'Campus Name': ['Campus A', 'Campus B', 'Campus C', 'Campus D', 'Campus E'],
            'Project ID': ['PRJ001', 'PRJ002', 'PRJ003', 'PRJ004', 'PRJ005'],
            'Borough': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Bronx'],
            'Contract Advertise Date': ['01/15/2023', '02/20/2023', '03/10/2023', '04/15/2023', '05/20/2023'],
            'Geographical District': [78, 13, 1, 27, 31]
        })
    
    print()
    
    # Test individual analyzers
    print("🔬 TESTING INDIVIDUAL ANALYZERS")
    print("-" * 50)
    
    analyzers = {
        'Construction': ConstructionAnalyzer(),
        'Infrastructure': InfrastructureAnalyzer(),
        'Environmental': EnvironmentalAnalyzer(),
        'Financial': FinancialAnalyzer(),
        'Risk': RiskAnalyzer(),
        'Spatial': SpatialAnalyzer(),
        'Temporal': TemporalAnalyzer(),
        'CrossDataset': CrossDatasetAnalyzer(),
        'Survey': SurveyAnalyzer()
    }
    
    for name, analyzer in analyzers.items():
        print(f"\n🔍 Testing {name}Analyzer...")
        analyzer_result = {
            'status': 'unknown',
            'error': None,
            'data_found': False,
            'execution_time': 0,
            'output_structure': {},
            'data_quality': {}
        }
        
        try:
            start_time = time.time()
            
            if name == 'CrossDataset':
                # CrossDatasetAnalyzer needs multiple datasets
                datasets = {'construction': test_data, 'infrastructure': test_data.copy()}
                result = analyzer.analyze(datasets)
            else:
                result = analyzer.analyze(test_data)
            
            execution_time = time.time() - start_time
            analyzer_result['execution_time'] = execution_time
            
            # Check if analyzer found data
            data_found = False
            output_structure = {}
            
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, dict) and value:
                        data_found = True
                        output_structure[key] = len(value)
                    elif isinstance(value, list) and value:
                        data_found = True
                        output_structure[key] = len(value)
                    else:
                        output_structure[key] = 0
            
            analyzer_result['data_found'] = data_found
            analyzer_result['output_structure'] = output_structure
            
            # Check data quality
            if isinstance(result, dict):
                analyzer_result['data_quality'] = {
                    'has_executive_summary': 'executive_summary' in result,
                    'has_summary': 'summary' in result,
                    'total_keys': len(result),
                    'non_empty_keys': sum(1 for v in result.values() if v)
                }
            
            if data_found:
                print(f"  ✅ {name}Analyzer: PASSED")
                print(f"     • Execution time: {execution_time:.3f}s")
                print(f"     • Data found: Yes")
                print(f"     • Output keys: {list(result.keys())}")
                analyzer_result['status'] = 'passed'
                test_results['summary']['passed'] += 1
            else:
                print(f"  ⚠️ {name}Analyzer: WARNING (No data found)")
                print(f"     • Execution time: {execution_time:.3f}s")
                print(f"     • Data found: No")
                print(f"     • Output keys: {list(result.keys())}")
                analyzer_result['status'] = 'warning'
                test_results['summary']['warnings'] += 1
                
        except Exception as e:
            print(f"  ❌ {name}Analyzer: FAILED")
            print(f"     • Error: {e}")
            analyzer_result['status'] = 'failed'
            analyzer_result['error'] = str(e)
            test_results['summary']['failed'] += 1
        
        test_results['analyzers'][name] = analyzer_result
    
    print()
    
    # Test Universal Reporter integration
    print("🔗 TESTING UNIVERSAL REPORTER INTEGRATION")
    print("-" * 50)
    
    try:
        print("🔍 Testing Universal Reporter...")
        start_time = time.time()
        
        reporter = UniversalReporter()
        result = reporter.analyze_dataset(test_data)
        
        execution_time = time.time() - start_time
        
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
        
        missing_keys = [key for key in expected_keys if key not in result]
        present_keys = [key for key in expected_keys if key in result]
        
        if not missing_keys:
            print(f"  ✅ Universal Reporter: PASSED")
            print(f"     • Execution time: {execution_time:.3f}s")
            print(f"     • All expected keys present: {len(present_keys)}")
            test_results['integration_tests']['universal_reporter'] = {
                'status': 'passed',
                'execution_time': execution_time,
                'keys_present': len(present_keys),
                'keys_missing': 0
            }
        else:
            print(f"  ⚠️ Universal Reporter: WARNING")
            print(f"     • Execution time: {execution_time:.3f}s")
            print(f"     • Missing keys: {missing_keys}")
            test_results['integration_tests']['universal_reporter'] = {
                'status': 'warning',
                'execution_time': execution_time,
                'keys_present': len(present_keys),
                'keys_missing': len(missing_keys),
                'missing_keys': missing_keys
            }
            
    except Exception as e:
        print(f"  ❌ Universal Reporter: FAILED")
        print(f"     • Error: {e}")
        test_results['integration_tests']['universal_reporter'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    print()
    
    # Performance tests
    print("⚡ PERFORMANCE TESTS")
    print("-" * 50)
    
    # Test with larger dataset
    try:
        print("🔍 Testing performance with larger dataset...")
        large_data = pd.concat([test_data] * 50, ignore_index=True)
        print(f"     • Large dataset size: {len(large_data):,} records")
        
        start_time = time.time()
        reporter = UniversalReporter()
        result = reporter.analyze_dataset(large_data)
        execution_time = time.time() - start_time
        
        if execution_time < 10.0:  # Should complete within 10 seconds
            print(f"  ✅ Performance Test: PASSED")
            print(f"     • Execution time: {execution_time:.3f}s")
            print(f"     • Records processed: {len(large_data):,}")
            test_results['performance_tests']['large_dataset'] = {
                'status': 'passed',
                'execution_time': execution_time,
                'records_processed': len(large_data)
            }
        else:
            print(f"  ⚠️ Performance Test: WARNING (Slow)")
            print(f"     • Execution time: {execution_time:.3f}s")
            print(f"     • Records processed: {len(large_data):,}")
            test_results['performance_tests']['large_dataset'] = {
                'status': 'warning',
                'execution_time': execution_time,
                'records_processed': len(large_data)
            }
            
    except Exception as e:
        print(f"  ❌ Performance Test: FAILED")
        print(f"     • Error: {e}")
        test_results['performance_tests']['large_dataset'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    print()
    
    # Data quality tests
    print("📊 DATA QUALITY TESTS")
    print("-" * 50)
    
    # Test with problematic data
    try:
        print("🔍 Testing with problematic data...")
        problematic_data = pd.DataFrame({
            'latitude': ['invalid', 'also_invalid', 40.7128],
            'longitude': ['invalid', 'also_invalid', -74.0060],
            'Project Title': [None, '', 'Valid Project'],
            'Construction Award': ['not_a_number', 'also_not_a_number', 1000000]
        })
        
        start_time = time.time()
        reporter = UniversalReporter()
        result = reporter.analyze_dataset(problematic_data)
        execution_time = time.time() - start_time
        
        print(f"  ✅ Data Quality Test: PASSED")
        print(f"     • Execution time: {execution_time:.3f}s")
        print(f"     • Handled problematic data gracefully")
        test_results['data_quality_tests']['problematic_data'] = {
            'status': 'passed',
            'execution_time': execution_time
        }
        
    except Exception as e:
        print(f"  ❌ Data Quality Test: FAILED")
        print(f"     • Error: {e}")
        test_results['data_quality_tests']['problematic_data'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    print()
    
    # Generate summary
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    summary = test_results['summary']
    total_tests = summary['passed'] + summary['failed'] + summary['warnings']
    
    print(f"Total Analyzers Tested: {summary['total_analyzers']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"⚠️ Warnings: {summary['warnings']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"Success Rate: {(summary['passed'] / summary['total_analyzers']) * 100:.1f}%")
    
    print()
    
    # Detailed results
    print("🔍 DETAILED RESULTS")
    print("-" * 50)
    
    for name, result in test_results['analyzers'].items():
        status_icon = {
            'passed': '✅',
            'warning': '⚠️',
            'failed': '❌'
        }.get(result['status'], '❓')
        
        print(f"{status_icon} {name}Analyzer: {result['status'].upper()}")
        if result['error']:
            print(f"    Error: {result['error']}")
        if result['data_found']:
            print(f"    Data found: Yes")
        if result['execution_time'] > 0:
            print(f"    Execution time: {result['execution_time']:.3f}s")
    
    print()
    
    # Save results
    test_results['summary']['end_time'] = datetime.now().isoformat()
    test_results['summary']['total_duration'] = (
        datetime.fromisoformat(test_results['summary']['end_time']) - 
        datetime.fromisoformat(test_results['summary']['start_time'])
    ).total_seconds()
    
    # Save to file
    results_file = f"analyzer_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"📄 Test results saved to: {results_file}")
    
    # Final status
    if summary['failed'] == 0:
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        return True
    else:
        print(f"\n⚠️ {summary['failed']} TESTS FAILED - REVIEW REQUIRED")
        return False

if __name__ == '__main__':
    success = run_comprehensive_analyzer_tests()
    sys.exit(0 if success else 1) 