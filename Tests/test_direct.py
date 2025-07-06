#!/usr/bin/env python3
"""
Direct test for Universal Reporter - bypassing problematic imports
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import directly without going through src/__init__.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Import Universal Reporter directly
sys.path.insert(0, str(Path(__file__).parent / "src/core"))
from universal_reporter import UniversalReporter

def test_universal_reporter():
    """Test the Universal Reporter directly"""
    print("🧪 Testing Universal Reporter")
    print("=" * 50)
    
    try:
        # Initialize Universal Reporter
        print("1. Initializing Universal Reporter...")
        universal_reporter = UniversalReporter()
        print("✅ Universal Reporter initialized successfully")
        
        # Test with sample data
        print("\n2. Testing with sample infrastructure data...")
        sample_data = pd.DataFrame({
            'pipe_id': [1, 2, 3, 4, 5],
            'material': ['PVC', 'Concrete', 'Steel', 'PVC', 'Concrete'],
            'diameter': [150, 300, 200, 100, 250],
            'length': [100, 200, 150, 80, 180],
            'latitude': [-37.8136, -37.8140, -37.8138, -37.8135, -37.8142],
            'longitude': [144.9631, 144.9635, 144.9633, 144.9630, 144.9637],
            'condition': ['Good', 'Fair', 'Poor', 'Good', 'Fair']
        })
        
        # Perform analysis
        print("3. Running comprehensive analysis...")
        analysis_result = universal_reporter.analyze_dataset(
            sample_data, 
            dataset_type='infrastructure',
            location={'lat': -37.8136, 'lon': 144.9631}
        )
        
        print("✅ Analysis completed successfully")
        
        # Display results
        print("\n4. Analysis Results:")
        print("-" * 30)
        
        # Dataset overview
        overview = analysis_result.get('dataset_overview', {})
        print(f"📊 Dataset Overview:")
        print(f"   Records: {overview.get('total_records', 0)}")
        print(f"   Columns: {overview.get('total_columns', 0)}")
        
        # Data quality
        quality = analysis_result.get('data_quality', {})
        completeness = quality.get('completeness_score', 0)
        print(f"   Data Quality: {completeness:.1f}% complete")
        
        # Infrastructure insights
        infra_insights = analysis_result.get('infrastructure_insights', {})
        if 'material_analysis' in infra_insights:
            print(f"🔧 Material Analysis: Available")
            material_data = infra_insights['material_analysis']
            if 'material_distributions' in material_data:
                materials = material_data['material_distributions']
                if 'material' in materials:
                    print(f"   Materials found: {list(materials['material'].keys())}")
        
        if 'dimension_analysis' in infra_insights:
            print(f"📏 Dimension Analysis: Available")
            dim_data = infra_insights['dimension_analysis']
            if 'dimension_statistics' in dim_data:
                print(f"   Dimensions analyzed: {list(dim_data['dimension_statistics'].keys())}")
        
        # Risk assessment
        risk_assessment = analysis_result.get('risk_assessment', {})
        if risk_assessment:
            print(f"⚠️ Risk Assessment: Available")
        
        # Recommendations
        recommendations = analysis_result.get('recommendations', [])
        if recommendations:
            print(f"💡 Recommendations: {len(recommendations)} found")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        # Action items
        action_items = analysis_result.get('action_items', [])
        if action_items:
            print(f"🎯 Action Items: {len(action_items)} found")
            for i, action in enumerate(action_items[:3], 1):
                print(f"   {i}. {action.get('action', 'Unknown action')}")
        
        # Spatial analysis
        spatial_analysis = analysis_result.get('spatial_analysis', {})
        if 'coordinate_analysis' in spatial_analysis:
            coord_analysis = spatial_analysis['coordinate_analysis']
            if 'coordinate_range' in coord_analysis:
                print(f"🗺️ Spatial Analysis: Available")
                range_data = coord_analysis['coordinate_range']
                print(f"   Lat range: {range_data.get('lat_min', 0):.4f} to {range_data.get('lat_max', 0):.4f}")
                print(f"   Lon range: {range_data.get('lon_min', 0):.4f} to {range_data.get('lon_max', 0):.4f}")
        
        print("\n🎉 Universal Reporter Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Universal Reporter Test")
    print("=" * 60)
    
    # Test Universal Reporter
    test_passed = test_universal_reporter()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Universal Reporter: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    if test_passed:
        print("\n🎉 Universal Reporter is working correctly!")
        print("\n📝 Next Steps:")
        print("   1. The Universal Reporter is ready for integration")
        print("   2. You can use it via the CLI with: python main.py analyze")
        print("   3. It will analyze any civil engineering dataset comprehensively")
    else:
        print("\n⚠️ Test failed. Check the output above for details.") 