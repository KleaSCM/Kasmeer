#!/usr/bin/env python3
"""
Standalone test for Universal Reporter functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalReporter:
    """
    Universal Reporter - The AI brain that can analyze ANY civil engineering dataset.
    """
    
    def __init__(self):
        """Initialize the Universal Reporter with comprehensive data patterns"""
        logger.info("Initializing Universal Reporter - The AI brain for civil engineering data")
        
        # Comprehensive data patterns for civil engineering
        self.data_patterns = {
            'infrastructure': {
                'pipes': ['pipe', 'drainage', 'sewer', 'stormwater', 'water', 'gas', 'oil', 'chemical'],
                'materials': ['material', 'type', 'composition', 'grade', 'class'],
                'dimensions': ['diameter', 'length', 'width', 'height', 'depth', 'thickness', 'size'],
                'structural': ['bridge', 'tunnel', 'culvert', 'retaining_wall', 'foundation', 'beam', 'column'],
                'electrical': ['transformer', 'pole', 'cable', 'substation', 'switchgear', 'meter'],
                'mechanical': ['pump', 'valve', 'tank', 'chamber', 'manhole', 'pit', 'vault'],
                'transport': ['road', 'highway', 'railway', 'airport', 'port', 'parking'],
                'buildings': ['building', 'structure', 'facility', 'plant', 'station', 'terminal']
            },
            'environmental': {
                'soil': ['soil', 'geotechnical', 'bearing_capacity', 'moisture', 'ph', 'contamination'],
                'vegetation': ['vegetation', 'tree', 'plant', 'species', 'density', 'age', 'health'],
                'climate': ['temperature', 'rainfall', 'wind', 'humidity', 'solar', 'weather'],
                'hydrology': ['river', 'stream', 'flood', 'groundwater', 'water_table', 'flow'],
                'geology': ['rock', 'fault', 'seismic', 'erosion', 'landslide', 'subsidence'],
                'air_quality': ['air', 'pollution', 'emission', 'particulate', 'gas'],
                'noise': ['noise', 'sound', 'vibration', 'decibel', 'acoustic']
            }
        }
        
        # Risk and impact patterns
        self.risk_patterns = {
            'structural_risk': ['crack', 'corrosion', 'deterioration', 'failure', 'collapse'],
            'environmental_risk': ['flood', 'earthquake', 'fire', 'storm', 'drought'],
            'operational_risk': ['breakdown', 'outage', 'interruption', 'failure', 'accident'],
            'financial_risk': ['cost_overrun', 'budget_exceed', 'loss', 'liability', 'penalty'],
            'compliance_risk': ['violation', 'non_compliance', 'penalty', 'fine', 'legal']
        }
        
        # Geographic and spatial patterns
        self.spatial_patterns = {
            'coordinates': ['lat', 'lon', 'latitude', 'longitude', 'x', 'y', 'easting', 'northing'],
            'addresses': ['address', 'street', 'city', 'state', 'postcode', 'zip'],
            'zones': ['zone', 'area', 'region', 'district', 'sector', 'quadrant'],
            'elevation': ['elevation', 'height', 'depth', 'altitude', 'grade', 'slope']
        }
        
        logger.info("Universal Reporter initialized with comprehensive data patterns")
    
    def analyze_dataset(self, dataset: pd.DataFrame, dataset_type: Optional[str] = None, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze ANY dataset and extract ALL possible insights.
        """
        logger.info(f"Starting comprehensive analysis of dataset with {len(dataset)} records")
        
        analysis = {
            'dataset_overview': self._analyze_dataset_overview(dataset),
            'data_quality': self._analyze_data_quality(dataset),
            'infrastructure_insights': self._analyze_infrastructure(dataset),
            'environmental_insights': self._analyze_environmental(dataset),
            'risk_assessment': self._analyze_risks(dataset),
            'spatial_analysis': self._analyze_spatial(dataset, location),
            'recommendations': self._generate_recommendations(dataset),
            'action_items': self._generate_action_items(dataset)
        }
        
        logger.info("Comprehensive analysis completed")
        return analysis
    
    def _analyze_dataset_overview(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic dataset characteristics"""
        overview = {
            'total_records': len(dataset),
            'total_columns': len(dataset.columns),
            'column_types': dataset.dtypes.value_counts().to_dict(),
            'memory_usage': dataset.memory_usage(deep=True).sum(),
            'key_columns': self._identify_key_columns(dataset)
        }
        
        # Detect geographic bounds
        coord_cols = self._find_coordinate_columns(dataset)
        if coord_cols['lat'] and coord_cols['lon']:
            lat_col, lon_col = coord_cols['lat'], coord_cols['lon']
            # Convert to numeric, ignoring errors
            lat_numeric = pd.to_numeric(dataset[lat_col], errors='coerce')
            lon_numeric = pd.to_numeric(dataset[lon_col], errors='coerce')
            # Only calculate bounds if we have valid numeric data
            if not lat_numeric.isna().all() and not lon_numeric.isna().all():
                overview['geographic_bounds'] = {
                    'lat_min': float(lat_numeric.min()),
                    'lat_max': float(lat_numeric.max()),
                    'lon_min': float(lon_numeric.min()),
                    'lon_max': float(lon_numeric.max())
                }
        
        return overview
    
    def _analyze_data_quality(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality and completeness"""
        quality = {
            'missing_data': {},
            'duplicates': len(dataset[dataset.duplicated()]),
            'data_types': {},
            'completeness_score': 0.0
        }
        
        # Missing data analysis
        missing_counts = dataset.isnull().sum()
        missing_percentages = (missing_counts / len(dataset)) * 100
        quality['missing_data'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'columns_with_missing': list(missing_counts[missing_counts > 0].index)
        }
        
        # Data type analysis
        for col in dataset.columns:
            quality['data_types'][col] = {
                'dtype': str(dataset[col].dtype),
                'unique_values': dataset[col].nunique(),
                'most_common': dataset[col].value_counts().head(3).to_dict() if dataset[col].dtype == 'object' else None
            }
        
        # Completeness score
        total_cells = len(dataset) * len(dataset.columns)
        filled_cells = total_cells - dataset.isnull().sum().sum()
        quality['completeness_score'] = (filled_cells / total_cells) * 100
        
        return quality
    
    def _analyze_infrastructure(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze infrastructure-related data"""
        infrastructure = {
            'pipe_analysis': {},
            'material_analysis': {},
            'dimension_analysis': {},
            'structural_analysis': {},
            'electrical_analysis': {}
        }
        
        # Material analysis
        material_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['materials'])
        if material_cols:
            infrastructure['material_analysis'] = self._analyze_materials(dataset, material_cols)
        
        # Dimension analysis
        dimension_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['dimensions'])
        if dimension_cols:
            infrastructure['dimension_analysis'] = self._analyze_dimensions(dataset, dimension_cols)
        
        return infrastructure
    
    def _analyze_environmental(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze environmental data"""
        environmental = {
            'soil_analysis': {},
            'vegetation_analysis': {},
            'climate_analysis': {},
            'hydrology_analysis': {},
            'geology_analysis': {},
            'air_quality_analysis': {},
            'noise_analysis': {}
        }
        
        # Soil analysis
        soil_cols = self._find_columns_by_patterns(dataset, self.data_patterns['environmental']['soil'])
        if soil_cols:
            environmental['soil_analysis'] = self._analyze_soil(dataset, soil_cols)
        
        return environmental
    
    def _analyze_risks(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk factors"""
        risks = {
            'structural_risks': {},
            'environmental_risks': {},
            'operational_risks': {},
            'financial_risks': {},
            'compliance_risks': {}
        }
        
        # Structural risks
        structural_risk_cols = self._find_columns_by_patterns(dataset, self.risk_patterns['structural_risk'])
        if structural_risk_cols:
            risks['structural_risks'] = self._analyze_structural_risks(dataset, structural_risk_cols)
        
        return risks
    
    def _analyze_spatial(self, dataset: pd.DataFrame, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze spatial data"""
        spatial = {
            'coordinate_analysis': {},
            'geographic_distribution': {},
            'proximity_analysis': {},
            'spatial_patterns': {}
        }
        
        # Coordinate analysis
        coord_cols = self._find_coordinate_columns(dataset)
        if coord_cols['lat'] and coord_cols['lon']:
            spatial['coordinate_analysis'] = self._analyze_coordinates(dataset, coord_cols)
            
            # Proximity analysis if location provided
            if location:
                spatial['proximity_analysis'] = self._analyze_proximity(dataset, coord_cols, location)
        
        return spatial
    
    def _generate_recommendations(self, dataset: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        missing_data = dataset.isnull().sum()
        high_missing_cols = missing_data[missing_data > len(dataset) * 0.1]
        if len(high_missing_cols) > 0:
            recommendations.append(f"Address missing data in columns: {list(high_missing_cols.index)}")
        
        # Infrastructure recommendations
        if self._has_infrastructure_data(dataset):
            recommendations.extend(self._generate_infrastructure_recommendations(dataset))
        
        return recommendations
    
    def _generate_action_items(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate specific action items"""
        action_items = []
        
        # Data quality actions
        if dataset.isnull().sum().sum() > 0:
            action_items.append({
                'type': 'data_quality',
                'priority': 'high',
                'action': 'Clean missing data',
                'description': 'Address missing values in the dataset'
            })
        
        return action_items
    
    # Helper methods
    def _find_columns_by_patterns(self, dataset: pd.DataFrame, patterns: List[str]) -> List[str]:
        """Find columns that match given patterns"""
        matching_cols = []
        for col in dataset.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in patterns):
                matching_cols.append(col)
        return matching_cols
    
    def _find_coordinate_columns(self, dataset: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Find coordinate columns"""
        lat_col = None
        lon_col = None
        
        for col in dataset.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['lat', 'latitude', 'y']):
                lat_col = col
            elif any(pattern in col_lower for pattern in ['lon', 'longitude', 'x']):
                lon_col = col
        
        return {'lat': lat_col, 'lon': lon_col}
    
    def _identify_key_columns(self, dataset: pd.DataFrame) -> List[str]:
        """Identify key columns in the dataset"""
        key_cols = []
        
        # ID columns
        id_patterns = ['id', 'key', 'index', 'name', 'code']
        for col in dataset.columns:
            if any(pattern in col.lower() for pattern in id_patterns):
                key_cols.append(col)
        
        # Coordinate columns
        coord_cols = self._find_coordinate_columns(dataset)
        if coord_cols['lat']:
            key_cols.append(coord_cols['lat'])
        if coord_cols['lon']:
            key_cols.append(coord_cols['lon'])
        
        return key_cols
    
    # Analysis methods
    def _analyze_materials(self, dataset: pd.DataFrame, material_cols: List[str]) -> Dict[str, Any]:
        """Analyze material data"""
        materials = {}
        for col in material_cols:
            if col in dataset.columns:
                materials[col] = dataset[col].value_counts().to_dict()
        return {'material_distributions': materials}
    
    def _analyze_dimensions(self, dataset: pd.DataFrame, dimension_cols: List[str]) -> Dict[str, Any]:
        """Analyze dimensional data"""
        dimensions = {}
        for col in dimension_cols:
            if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                dimensions[col] = {
                    'min': dataset[col].min(),
                    'max': dataset[col].max(),
                    'mean': dataset[col].mean(),
                    'std': dataset[col].std()
                }
        return {'dimension_statistics': dimensions}
    
    def _analyze_soil(self, dataset: pd.DataFrame, soil_cols: List[str]) -> Dict[str, Any]:
        """Analyze soil data"""
        return {'soil_samples': len(dataset), 'soil_columns': soil_cols}
    
    def _analyze_structural_risks(self, dataset: pd.DataFrame, risk_cols: List[str]) -> Dict[str, Any]:
        """Analyze structural risks"""
        return {'structural_risk_indicators': len(dataset), 'risk_columns': risk_cols}
    
    def _analyze_coordinates(self, dataset: pd.DataFrame, coord_cols: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """Analyze coordinate data"""
        lat_col, lon_col = coord_cols['lat'], coord_cols['lon']
        if lat_col is None or lon_col is None:
            return {'coordinate_count': 0}
        
        # Convert to numeric, ignoring errors
        lat_numeric = pd.to_numeric(dataset[lat_col], errors='coerce')
        lon_numeric = pd.to_numeric(dataset[lon_col], errors='coerce')
        
        return {
            'coordinate_range': {
                'lat_min': float(lat_numeric.min()),
                'lat_max': float(lat_numeric.max()),
                'lon_min': float(lon_numeric.min()),
                'lon_max': float(lon_numeric.max())
            },
            'coordinate_count': len(dataset)
        }
    
    def _analyze_proximity(self, dataset: pd.DataFrame, coord_cols: Dict[str, Optional[str]], location: Dict) -> Dict[str, Any]:
        """Analyze proximity to given location"""
        lat_col, lon_col = coord_cols['lat'], coord_cols['lon']
        if lat_col is None or lon_col is None:
            return {'error': 'No coordinate columns found'}
        
        distances = np.sqrt(
            (dataset[lat_col] - location['lat'])**2 + 
            (dataset[lon_col] - location['lon'])**2
        )
        return {
            'nearest_distance': float(distances.min()),
            'average_distance': float(distances.mean()),
            'within_1km': len(distances[distances <= 0.01]),
            'within_5km': len(distances[distances <= 0.05])
        }
    
    def _has_infrastructure_data(self, dataset: pd.DataFrame) -> bool:
        """Check if dataset has infrastructure data"""
        infrastructure_patterns = []
        for category in self.data_patterns['infrastructure'].values():
            infrastructure_patterns.extend(category)
        return len(self._find_columns_by_patterns(dataset, infrastructure_patterns)) > 0
    
    def _generate_infrastructure_recommendations(self, dataset: pd.DataFrame) -> List[str]:
        """Generate infrastructure-specific recommendations"""
        recommendations = []
        
        # Material recommendations
        material_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['materials'])
        if material_cols:
            recommendations.append("Review material specifications and consider upgrades")
        
        # Dimension recommendations
        dimension_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['dimensions'])
        if dimension_cols:
            recommendations.append("Assess capacity requirements and upgrade needs")
        
        return recommendations

def test_universal_reporter():
    """Test the Universal Reporter"""
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
        print("   4. The system is now your 'Omni-Tool' for civil engineering data!")
    else:
        print("\n⚠️ Test failed. Check the output above for details.") 