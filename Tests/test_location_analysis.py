#!/usr/bin/env python3
"""
Test script to run Universal Reporter analysis on infrastructure data
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.universal_reporter import UniversalReporter

def main():
    """Run Universal Reporter analysis on infrastructure data"""
    print("🚀 Kasmeer Universal Reporter - Location Analysis Test")
    print("=" * 60)
    
    try:
        # Initialize Universal Reporter
        print("1. Initializing Universal Reporter...")
        universal_reporter = UniversalReporter()
        print("✅ Universal Reporter initialized successfully")
        
        # Load infrastructure data
        print("\n2. Loading infrastructure data...")
        infrastructure_file = "/home/klea/Documents/Dev/AI/DataSets/INF_DRN_PIPES__PV_-8971823211995978582.csv"
        
        if not os.path.exists(infrastructure_file):
            print(f"❌ Infrastructure file not found: {infrastructure_file}")
            return False
        
        # Load the data
        infrastructure_data = pd.read_csv(infrastructure_file)
        print(f"✅ Loaded infrastructure data: {len(infrastructure_data)} records, {len(infrastructure_data.columns)} columns")
        
        # Display sample data
        print(f"\n3. Sample data preview:")
        print(f"   Columns: {list(infrastructure_data.columns)}")
        print(f"   First few records:")
        print(infrastructure_data.head(3).to_string())
        
        # Melbourne location context
        melbourne_location = {
            'lat': -37.8136,
            'lon': 144.9631,
            'city': 'Melbourne',
            'state': 'Victoria',
            'country': 'Australia'
        }
        
        print(f"\n4. Running comprehensive analysis for Melbourne location...")
        print(f"   Location: {melbourne_location['city']}, {melbourne_location['state']}")
        print(f"   Coordinates: {melbourne_location['lat']}, {melbourne_location['lon']}")
        
        # Perform comprehensive analysis
        analysis_result = universal_reporter.analyze_dataset(
            infrastructure_data,
            dataset_type='infrastructure',
            location=melbourne_location
        )
        
        print("✅ Analysis completed successfully!")
        
        # Display comprehensive results
        print(f"\n5. Analysis Results Summary:")
        print("=" * 50)
        
        # Dataset overview
        overview = analysis_result.get('dataset_overview', {})
        print(f"📊 Dataset Overview:")
        print(f"   • Total Records: {overview.get('total_records', 0):,}")
        print(f"   • Total Columns: {overview.get('total_columns', 0)}")
        print(f"   • Memory Usage: {overview.get('memory_usage', 0):,} bytes")
        
        # Data quality
        quality = analysis_result.get('data_quality', {})
        completeness = quality.get('completeness_score', 0)
        print(f"\n✅ Data Quality:")
        print(f"   • Completeness Score: {completeness:.1f}%")
        print(f"   • Missing Data: {len(quality.get('missing_data', {}).get('columns_with_missing', []))} columns")
        print(f"   • Duplicates: {quality.get('duplicates', 0)} records")
        
        # Infrastructure insights
        infra_insights = analysis_result.get('infrastructure_insights', {})
        print(f"\n🏗️ Infrastructure Insights:")
        
        if 'pipe_analysis' in infra_insights:
            print(f"   • Pipe Analysis: {len(infrastructure_data)} pipes detected")
        
        if 'material_analysis' in infra_insights:
            material_data = infra_insights['material_analysis']
            if 'material_distributions' in material_data:
                print(f"   • Materials: {len(material_data['material_distributions'])} types found")
        
        if 'dimension_analysis' in infra_insights:
            dim_data = infra_insights['dimension_analysis']
            if 'dimension_statistics' in dim_data:
                print(f"   • Dimensions: {len(dim_data['dimension_statistics'])} analyzed")
        
        # Risk assessment
        risks = analysis_result.get('risk_assessment', {})
        print(f"\n⚠️ Risk Assessment:")
        if risks:
            for risk_type, risk_data in risks.items():
                if risk_data:
                    print(f"   • {risk_type.replace('_', ' ').title()}: {len(risk_data)} risk factors")
        
        # Spatial analysis
        spatial = analysis_result.get('spatial_analysis', {})
        print(f"\n🗺️ Spatial Analysis:")
        if 'coordinate_analysis' in spatial:
            coord_data = spatial['coordinate_analysis']
            if 'coordinate_range' in coord_data:
                range_data = coord_data['coordinate_range']
                print(f"   • Geographic Range: {range_data.get('lat_min', 0):.4f} to {range_data.get('lat_max', 0):.4f} lat")
                print(f"   • Coordinate Count: {coord_data.get('coordinate_count', 0)}")
        
        # Proximity analysis
        if 'proximity_analysis' in spatial:
            prox_data = spatial['proximity_analysis']
            if 'nearest_distance' in prox_data:
                print(f"   • Nearest to Melbourne: {prox_data['nearest_distance']:.4f} degrees")
                print(f"   • Average Distance: {prox_data.get('average_distance', 0):.4f} degrees")
        
        # Correlations
        correlations = analysis_result.get('correlations', {})
        print(f"\n🔗 Correlations:")
        if 'numeric_correlations' in correlations:
            corr_data = correlations['numeric_correlations']
            if 'strong_correlations' in corr_data:
                strong_corrs = corr_data['strong_correlations']
                print(f"   • Strong Correlations: {len(strong_corrs)} found")
                for i, corr in enumerate(strong_corrs[:3], 1):
                    print(f"     {i}. {corr['variable1']} ↔ {corr['variable2']} (r={corr['correlation']:.3f})")
        
        # Anomalies
        anomalies = analysis_result.get('anomalies', {})
        print(f"\n🔍 Anomalies:")
        if 'outliers' in anomalies:
            outlier_count = sum(len(data) for data in anomalies['outliers'].values())
            print(f"   • Outliers Detected: {outlier_count} across {len(anomalies['outliers'])} variables")
        
        # Recommendations
        recommendations = analysis_result.get('recommendations', [])
        print(f"\n💡 Key Recommendations:")
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")
        else:
            print("   • No specific recommendations generated")
        
        # Action items
        action_items = analysis_result.get('action_items', [])
        print(f"\n🎯 Action Items:")
        if action_items:
            for i, action in enumerate(action_items[:5], 1):
                print(f"   {i}. {action.get('action', 'Unknown action')} (Priority: {action.get('priority', 'Unknown')})")
        else:
            print("   • No specific action items identified")
        
        # Save detailed results
        print(f"\n6. Saving detailed results...")
        import json
        
        # Convert pandas dtypes to strings for JSON serialization
        def convert_for_json(obj):
            if hasattr(obj, 'dtype'):
                return str(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Convert the analysis result
        json_safe_result = convert_for_json(analysis_result)
        
        output_file = "melbourne_infrastructure_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(json_safe_result, f, indent=2, default=str)
        print(f"✅ Detailed results saved to: {output_file}")
        
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Analyzed {len(infrastructure_data):,} infrastructure records")
        print(f"📍 Location: Melbourne, Victoria, Australia")
        print(f"📋 Generated {len(recommendations)} recommendations and {len(action_items)} action items")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Universal Reporter analysis completed successfully!")
    else:
        print("\n❌ Analysis failed. Check the output above for details.") 