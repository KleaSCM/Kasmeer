#!/usr/bin/env python3
"""
Test script to check which analyzers are working and finding data
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.analyzers import *
from src.data.data_processor import DataProcessor

def test_analyzers():
    """Test all analyzers with real data"""
    
    print("🔍 TESTING ALL ANALYZERS")
    print("=" * 60)
    
    # Load data
    data_processor = DataProcessor('/home/klea/Documents/Dev/AI/DataSets')
    loaded_data = data_processor.discover_and_load_all_data()
    
    if not loaded_data:
        print("❌ No data loaded!")
        return
    
    print(f"📊 Loaded {len(loaded_data)} datasets:")
    for name, data in loaded_data.items():
        if hasattr(data, '__len__'):
            print(f"  • {name}: {len(data):,} records, {len(data.columns)} columns")
        else:
            print(f"  • {name}: raster/geospatial data")
    
    print("\n" + "=" * 60)
    print("🧪 TESTING INDIVIDUAL ANALYZERS")
    print("=" * 60)
    
    # Test each analyzer
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
    
    # Test with construction data (most comprehensive)
    construction_data = loaded_data.get('construction')
    if construction_data is None:
        print("❌ No construction data available for testing")
        return
    
    print(f"\n📋 Testing with construction dataset: {len(construction_data):,} records")
    print(f"Columns: {list(construction_data.columns)}")
    
    results = {}
    
    for name, analyzer in analyzers.items():
        print(f"\n🔬 Testing {name}Analyzer...")
        try:
            if name == 'CrossDataset':
                # CrossDatasetAnalyzer needs multiple datasets
                result = analyzer.analyze(loaded_data)
            else:
                result = analyzer.analyze(construction_data)
            
            results[name] = result
            
            # Check if analyzer found any data
            data_found = False
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, dict) and value:
                        data_found = True
                        break
                    elif isinstance(value, list) and value:
                        data_found = True
                        break
            
            if data_found:
                print(f"  ✅ {name}Analyzer: Found data")
                # Show what was found
                for key, value in result.items():
                    if isinstance(value, dict) and value:
                        print(f"    • {key}: {len(value)} items")
                    elif isinstance(value, list) and value:
                        print(f"    • {key}: {len(value)} items")
            else:
                print(f"  ⚠️ {name}Analyzer: No data found")
                
        except Exception as e:
            print(f"  ❌ {name}Analyzer: Error - {e}")
            results[name] = {'error': str(e)}
    
    print("\n" + "=" * 60)
    print("📊 ANALYZER SUMMARY")
    print("=" * 60)
    
    working_analyzers = []
    missing_data_analyzers = []
    error_analyzers = []
    
    for name, result in results.items():
        if 'error' in result:
            error_analyzers.append(name)
        elif isinstance(result, dict):
            has_data = False
            for key, value in result.items():
                if isinstance(value, dict) and value:
                    has_data = True
                    break
                elif isinstance(value, list) and value:
                    has_data = True
                    break
            
            if has_data:
                working_analyzers.append(name)
            else:
                missing_data_analyzers.append(name)
    
    print(f"✅ Working analyzers ({len(working_analyzers)}): {', '.join(working_analyzers)}")
    print(f"⚠️ Missing data analyzers ({len(missing_data_analyzers)}): {', '.join(missing_data_analyzers)}")
    print(f"❌ Error analyzers ({len(error_analyzers)}): {', '.join(error_analyzers)}")
    
    # Detailed analysis of what each analyzer found
    print("\n" + "=" * 60)
    print("🔍 DETAILED ANALYSIS")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"\n📋 {name}Analyzer Results:")
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            for key, value in result.items():
                if isinstance(value, dict):
                    if value:
                        print(f"  ✅ {key}: {len(value)} items")
                        # Show first few items
                        for subkey, subvalue in list(value.items())[:3]:
                            if isinstance(subvalue, dict):
                                print(f"    • {subkey}: {len(subvalue)} sub-items")
                            else:
                                print(f"    • {subkey}: {subvalue}")
                    else:
                        print(f"  ⚠️ {key}: Empty")
                elif isinstance(value, list):
                    if value:
                        print(f"  ✅ {key}: {len(value)} items")
                        # Show first few items
                        for item in value[:3]:
                            print(f"    • {item}")
                    else:
                        print(f"  ⚠️ {key}: Empty")
                else:
                    print(f"  ℹ️ {key}: {value}")

if __name__ == "__main__":
    test_analyzers() 