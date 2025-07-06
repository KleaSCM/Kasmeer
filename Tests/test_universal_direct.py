#!/usr/bin/env python3
"""
Direct test script for Universal Reporter - minimal imports
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

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

def test_universal_reporter_direct():
    """Test Universal Reporter functionality with direct import"""
    print("🧪 Testing Universal Reporter Direct Integration")
    print("=" * 50)
    
    try:
        # Create test dataset
        print("📊 Creating test dataset...")
        test_data = create_test_dataset()
        print(f"✅ Created dataset with {len(test_data)} records and {len(test_data.columns)} columns")
        
        # Test Universal Reporter with direct import
        print("\n🔍 Testing Universal Reporter...")
        
        # Import Universal Reporter directly
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "src" / "core"))
        from universal_reporter import UniversalReporter
        
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
        
        # Show some detailed results
        print("\n🔍 Detailed Analysis:")
        
        # Infrastructure insights
        infra_insights = analysis_result.get('infrastructure_insights', {})
        if infra_insights:
            print("  🏗️ Infrastructure Analysis:")
            for insight_type, insight_data in infra_insights.items():
                if insight_data:
                    print(f"    - {insight_type.replace('_', ' ').title()}: Available")
        
        # Environmental insights
        env_insights = analysis_result.get('environmental_insights', {})
        if env_insights:
            print("  🌱 Environmental Analysis:")
            for insight_type, insight_data in env_insights.items():
                if insight_data:
                    print(f"    - {insight_type.replace('_', ' ').title()}: Available")
        
        # Risk assessment
        risk_assessment = analysis_result.get('risk_assessment', {})
        if risk_assessment:
            print("  ⚠️ Risk Assessment:")
            for risk_type, risk_data in risk_assessment.items():
                if risk_data:
                    print(f"    - {risk_type.replace('_', ' ').title()}: Available")
        
        # Recommendations
        recommendations = analysis_result.get('recommendations', [])
        if recommendations:
            print("  💡 Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"    {i}. {rec}")
        
        # Action items
        action_items = analysis_result.get('action_items', [])
        if action_items:
            print("  🎯 Action Items:")
            for i, action in enumerate(action_items[:3], 1):
                print(f"    {i}. {action.get('action', 'Unknown action')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Universal Reporter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Kasmeer Universal Reporter Direct Test")
    print("=" * 60)
    
    # Test Universal Reporter
    success = test_universal_reporter_direct()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"  • Universal Reporter: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print("\n🎉 Universal Reporter test passed!")
        print("\nNext steps:")
        print("  • Install missing dependencies: pip install rasterio")
        print("  • Fix query_engine.py syntax errors")
        print("  • Run: python main.py universal-analyze")
        print("  • Run: python main.py analyze --comprehensive")
    else:
        print("\n⚠️ Test failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main() 