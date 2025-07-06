#!/usr/bin/env python3
"""
Test script for Universal Reporter integration
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.universal_reporter import UniversalReporter
from src.core.system_integration import SystemIntegration

def create_test_dataset():
    """Create a test dataset for Universal Reporter analysis"""
    # Create sample civil engineering data
    np.random.seed(42)
    n_records = 100
    
    data = {
        'id': range(1, n_records + 1),
        'latitude': np.random.uniform(-37.9, -37.7, n_records),
        'longitude': np.random.uniform(144.9, 145.1, n_records),
        'pipe_diameter': np.random.uniform(100, 500, n_records),
        'pipe_material': np.random.choice(['PVC', 'Steel', 'Concrete', 'HDPE'], n_records),
        'installation_date': pd.date_range('2020-01-01', periods=n_records, freq='D'),
        'soil_type': np.random.choice(['Clay', 'Sand', 'Silt', 'Rock'], n_records),
        'bearing_capacity': np.random.uniform(50, 200, n_records),
        'flood_risk': np.random.uniform(0, 1, n_records),
        'maintenance_cost': np.random.uniform(1000, 10000, n_records),
        'structural_condition': np.random.choice(['Good', 'Fair', 'Poor', 'Critical'], n_records),
        'temperature': np.random.uniform(10, 30, n_records),
        'rainfall': np.random.uniform(0, 100, n_records)
    }
    
    return pd.DataFrame(data)

def test_universal_reporter():
    """Test Universal Reporter functionality"""
    print("🧪 Testing Universal Reporter Integration")
    print("=" * 50)
    
    try:
        # Create test dataset
        print("📊 Creating test dataset...")
        test_data = create_test_dataset()
        print(f"✅ Created dataset with {len(test_data)} records and {len(test_data.columns)} columns")
        
        # Test Universal Reporter
        print("\n🔍 Testing Universal Reporter...")
        universal_reporter = UniversalReporter()
        
        # Analyze dataset
        analysis_result = universal_reporter.analyze_dataset(
            test_data, 
            dataset_type="infrastructure",
            location={'lat': -37.8136, 'lon': 144.9631}
        )
        
        print("✅ Universal Reporter analysis completed")
        
        # Display results
        print("\n📋 Analysis Results:")
        print(f"  • Dataset Overview: {analysis_result.get('dataset_overview', {}).get('total_records', 0)} records")
        print(f"  • Data Quality: {analysis_result.get('data_quality', {}).get('completeness_score', 0):.1f}% complete")
        print(f"  • Infrastructure Insights: {len(analysis_result.get('infrastructure_insights', {}))} categories")
        print(f"  • Environmental Insights: {len(analysis_result.get('environmental_insights', {}))} categories")
        print(f"  • Risk Assessment: {len(analysis_result.get('risk_assessment', {}))} categories")
        print(f"  • Spatial Analysis: {'Available' if analysis_result.get('spatial_analysis', {}).get('coordinate_analysis') else 'Not available'}")
        print(f"  • Temporal Analysis: {'Available' if analysis_result.get('temporal_analysis', {}).get('time_series_analysis') else 'Not available'}")
        print(f"  • Recommendations: {len(analysis_result.get('recommendations', []))} items")
        print(f"  • Action Items: {len(analysis_result.get('action_items', []))} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Universal Reporter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_integration():
    """Test System Integration with Universal Reporter"""
    print("\n🔧 Testing System Integration...")
    print("=" * 50)
    
    try:
        # Create test dataset
        test_data = create_test_dataset()
        
        # Test System Integration
        system = SystemIntegration()
        
        # Comprehensive analysis
        comprehensive_result = system.analyze_dataset_comprehensive(
            test_data,
            dataset_type="infrastructure",
            location={'lat': -37.8136, 'lon': 144.9631}
        )
        
        print("✅ System Integration analysis completed")
        
        # Display results
        print("\n📋 System Integration Results:")
        system_summary = comprehensive_result.get('system_summary', {})
        components = system_summary.get('analysis_components', {})
        
        for component, status in components.items():
            print(f"  • {component.replace('_', ' ').title()}: {status}")
        
        print(f"  • Key Insights: {len(system_summary.get('key_insights', []))} items")
        print(f"  • Critical Findings: {len(system_summary.get('critical_findings', []))} items")
        print(f"  • Next Steps: {len(system_summary.get('next_steps', []))} items")
        
        return True
        
    except Exception as e:
        print(f"❌ System Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Kasmeer Universal Reporter Integration Test")
    print("=" * 60)
    
    # Test Universal Reporter
    universal_success = test_universal_reporter()
    
    # Test System Integration
    integration_success = test_system_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"  • Universal Reporter: {'✅ PASSED' if universal_success else '❌ FAILED'}")
    print(f"  • System Integration: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    if universal_success and integration_success:
        print("\n🎉 All tests passed! Universal Reporter is properly integrated.")
        print("\nNext steps:")
        print("  • Run: python main.py universal-analyze")
        print("  • Run: python main.py analyze --comprehensive")
        print("  • Run: python main.py query")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return universal_success and integration_success

if __name__ == "__main__":
    main() 