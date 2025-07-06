# Author: KleaSCM
# Date: 2024
# Description: Universal Reporter - The AI brain that can analyze ANY civil engineering dataset
# This module can detect, analyze, and report on literally everything in civil engineering data

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings
from ..utils.logging_utils import setup_logging, log_performance
warnings.filterwarnings('ignore')

logger = setup_logging(__name__)

class UniversalReporter:
    """
    Universal Reporter - The AI brain that can analyze ANY civil engineering dataset.
    
    This module can:
    - Detect ANY type of civil engineering data
    - Extract ALL possible insights and patterns
    - Generate comprehensive reports for ANY dataset
    - Provide actionable engineering intelligence
    - Handle infrastructure, environmental, construction, financial, and more
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
            },
            'construction': {
                'project': ['project', 'phase', 'stage', 'milestone', 'schedule', 'timeline'],
                'resources': ['equipment', 'material', 'labor', 'contractor', 'subcontractor'],
                'quality': ['inspection', 'test', 'quality', 'compliance', 'certification'],
                'safety': ['safety', 'incident', 'accident', 'hazard', 'risk', 'compliance'],
                'permits': ['permit', 'license', 'approval', 'authorization', 'clearance']
            },
            'financial': {
                'costs': ['cost', 'budget', 'estimate', 'actual', 'variance', 'expense'],
                'assets': ['asset', 'value', 'depreciation', 'replacement', 'insurance'],
                'contracts': ['contract', 'agreement', 'warranty', 'bond', 'guarantee'],
                'billing': ['billing', 'invoice', 'payment', 'revenue', 'income']
            },
            'operational': {
                'maintenance': ['maintenance', 'repair', 'replacement', 'upgrade', 'refurbishment'],
                'performance': ['performance', 'efficiency', 'capacity', 'throughput', 'utilization'],
                'monitoring': ['monitor', 'sensor', 'measurement', 'reading', 'data'],
                'control': ['control', 'automation', 'system', 'operation', 'management']
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
    
    @log_performance(logger)
    def analyze_dataset(self, dataset: pd.DataFrame, dataset_type: Optional[str] = None, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze ANY dataset and extract ALL possible insights.
        
        Args:
            dataset: The dataset to analyze
            dataset_type: Optional dataset type hint
            location: Optional location context
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Starting comprehensive analysis of dataset with {len(dataset)} records")
        
        analysis = {
            'dataset_overview': self._analyze_dataset_overview(dataset),
            'data_quality': self._analyze_data_quality(dataset),
            'infrastructure_insights': self._analyze_infrastructure(dataset),
            'environmental_insights': self._analyze_environmental(dataset),
            'construction_insights': self._analyze_construction(dataset),
            'financial_insights': self._analyze_financial(dataset),
            'operational_insights': self._analyze_operational(dataset),
            'risk_assessment': self._analyze_risks(dataset),
            'spatial_analysis': self._analyze_spatial(dataset, location),
            'temporal_analysis': self._analyze_temporal(dataset),
            'correlations': self._find_correlations(dataset),
            'anomalies': self._detect_anomalies(dataset),
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
            'date_range': None,
            'geographic_bounds': None,
            'key_columns': self._identify_key_columns(dataset)
        }
        
        # Detect date range
        date_cols = dataset.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            overview['date_range'] = {
                'columns': list(date_cols),
                'earliest': dataset[date_cols].min().to_dict(),
                'latest': dataset[date_cols].max().to_dict()
            }
        
        # Detect geographic bounds
        coord_cols = self._find_coordinate_columns(dataset)
        if coord_cols['lat'] and coord_cols['lon']:
            lat_col, lon_col = coord_cols['lat'], coord_cols['lon']
            try:
                # Convert to numeric, ignoring errors
                lat_numeric = pd.to_numeric(dataset[lat_col], errors='coerce')  # type: ignore
                lon_numeric = pd.to_numeric(dataset[lon_col], errors='coerce')  # type: ignore
                # Only calculate bounds if we have valid numeric data
                if not lat_numeric.isna().all() and not lon_numeric.isna().all():  # type: ignore
                    overview['geographic_bounds'] = {
                        'lat_min': float(lat_numeric.min()),  # type: ignore
                        'lat_max': float(lat_numeric.max()),  # type: ignore
                        'lon_min': float(lon_numeric.min()),  # type: ignore
                        'lon_max': float(lon_numeric.max())   # type: ignore
                    }
            except Exception as e:
                logger.warning(f"Geographic bounds calculation failed: {e}")
        
        return overview
    
    def _analyze_data_quality(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality and completeness"""
        quality = {
            'missing_data': {},
            'duplicates': len(dataset[dataset.duplicated()]),
            'data_types': {},
            'outliers': {},
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
            'structural_analysis': {},
            'electrical_analysis': {},
            'mechanical_analysis': {},
            'transport_analysis': {},
            'building_analysis': {}
        }
        
        # Pipe analysis
        pipe_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['pipes'])
        if pipe_cols:
            infrastructure['pipe_analysis'] = self._analyze_pipes(dataset, pipe_cols)
        
        # Material analysis
        material_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['materials'])
        if material_cols:
            infrastructure['material_analysis'] = self._analyze_materials(dataset, material_cols)
        
        # Dimension analysis
        dimension_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['dimensions'])
        if dimension_cols:
            infrastructure['dimension_analysis'] = self._analyze_dimensions(dataset, dimension_cols)
        
        # Structural analysis
        structural_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['structural'])
        if structural_cols:
            infrastructure['structural_analysis'] = self._analyze_structural(dataset, structural_cols)
        
        # Electrical analysis
        electrical_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['electrical'])
        if electrical_cols:
            infrastructure['electrical_analysis'] = self._analyze_electrical(dataset, electrical_cols)
        
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
        
        # Vegetation analysis
        vegetation_cols = self._find_columns_by_patterns(dataset, self.data_patterns['environmental']['vegetation'])
        if vegetation_cols:
            environmental['vegetation_analysis'] = self._analyze_vegetation(dataset, vegetation_cols)
        
        # Climate analysis
        climate_cols = self._find_columns_by_patterns(dataset, self.data_patterns['environmental']['climate'])
        if climate_cols:
            environmental['climate_analysis'] = self._analyze_climate(dataset, climate_cols)
        
        return environmental
    
    def _analyze_construction(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze construction-related data"""
        construction = {
            'project_analysis': {},
            'resource_analysis': {},
            'quality_analysis': {},
            'safety_analysis': {},
            'permit_analysis': {}
        }
        
        # Project analysis
        project_cols = self._find_columns_by_patterns(dataset, self.data_patterns['construction']['project'])
        if project_cols:
            construction['project_analysis'] = self._analyze_projects(dataset, project_cols)
        
        # Resource analysis
        resource_cols = self._find_columns_by_patterns(dataset, self.data_patterns['construction']['resources'])
        if resource_cols:
            construction['resource_analysis'] = self._analyze_resources(dataset, resource_cols)
        
        return construction
    
    def _analyze_financial(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze financial data"""
        financial = {
            'cost_analysis': {},
            'asset_analysis': {},
            'contract_analysis': {},
            'billing_analysis': {}
        }
        
        # Cost analysis
        cost_cols = self._find_columns_by_patterns(dataset, self.data_patterns['financial']['costs'])
        if cost_cols:
            financial['cost_analysis'] = self._analyze_costs(dataset, cost_cols)
        
        # Asset analysis
        asset_cols = self._find_columns_by_patterns(dataset, self.data_patterns['financial']['assets'])
        if asset_cols:
            financial['asset_analysis'] = self._analyze_assets(dataset, asset_cols)
        
        return financial
    
    def _analyze_operational(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operational data"""
        operational = {
            'maintenance_analysis': {},
            'performance_analysis': {},
            'monitoring_analysis': {},
            'control_analysis': {}
        }
        
        # Maintenance analysis
        maintenance_cols = self._find_columns_by_patterns(dataset, self.data_patterns['operational']['maintenance'])
        if maintenance_cols:
            operational['maintenance_analysis'] = self._analyze_maintenance(dataset, maintenance_cols)
        
        # Performance analysis
        performance_cols = self._find_columns_by_patterns(dataset, self.data_patterns['operational']['performance'])
        if performance_cols:
            operational['performance_analysis'] = self._analyze_performance(dataset, performance_cols)
        
        return operational
    
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
        
        # Environmental risks
        environmental_risk_cols = self._find_columns_by_patterns(dataset, self.risk_patterns['environmental_risk'])
        if environmental_risk_cols:
            risks['environmental_risks'] = self._analyze_environmental_risks(dataset, environmental_risk_cols)
        
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
    
    def _analyze_temporal(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal data"""
        temporal = {
            'time_series_analysis': {},
            'seasonal_patterns': {},
            'trend_analysis': {},
            'temporal_distribution': {}
        }
        
        # Find date/time columns
        date_cols = list(dataset.select_dtypes(include=['datetime64']).columns)
        if len(date_cols) > 0:
            temporal['time_series_analysis'] = self._analyze_time_series(dataset, date_cols)
        
        return temporal
    
    def _find_correlations(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Find correlations between variables"""
        correlations = {
            'numeric_correlations': {},
            'categorical_associations': {},
            'significant_relationships': []
        }
        
        # Numeric correlations
        numeric_cols = list(dataset.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) > 1:
            try:
                corr_matrix = dataset[numeric_cols].corr()  # type: ignore
                correlations['numeric_correlations'] = {
                    'matrix': corr_matrix.to_dict(),
                    'strong_correlations': self._find_strong_correlations(corr_matrix)
                }
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
                correlations['numeric_correlations'] = {'error': str(e)}
        
        return correlations
    
    def _detect_anomalies(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        anomalies = {
            'outliers': {},
            'missing_patterns': {},
            'data_inconsistencies': {},
            'unusual_patterns': {}
        }
        
        # Outlier detection for numeric columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                Q1 = dataset[col].quantile(0.25)
                Q3 = dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = dataset[(dataset[col] < Q1 - 1.5 * IQR) | (dataset[col] > Q3 + 1.5 * IQR)]
                if len(outliers) > 0:
                    anomalies['outliers'][col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(dataset)) * 100,
                        'values': outliers[col].tolist()
                    }
            except Exception as e:
                logger.warning(f"Outlier detection failed for column {col}: {e}")
        
        return anomalies
    
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
        
        # Risk recommendations
        if self._has_risk_indicators(dataset):
            recommendations.extend(self._generate_risk_recommendations(dataset))
        
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
        
        # Infrastructure actions
        if self._has_infrastructure_data(dataset):
            action_items.extend(self._generate_infrastructure_actions(dataset))
        
        return action_items
    
    # Helper methods for specific analyses
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
        coord_patterns = self.spatial_patterns['coordinates']
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
        
        # Date columns
        date_cols = dataset.select_dtypes(include=['datetime64']).columns
        key_cols.extend(list(date_cols))
        
        # Coordinate columns
        coord_cols = self._find_coordinate_columns(dataset)
        if coord_cols['lat']:
            key_cols.append(coord_cols['lat'])
        if coord_cols['lon']:
            key_cols.append(coord_cols['lon'])
        
        return key_cols
    
    # Placeholder methods for specific analyses (to be implemented)
    def _analyze_pipes(self, dataset: pd.DataFrame, pipe_cols: List[str]) -> Dict[str, Any]:
        """Analyze pipe-related data"""
        return {'pipe_count': len(dataset), 'pipe_columns': pipe_cols}
    
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
                try:
                    dimensions[col] = {
                        'min': float(dataset[col].min()),
                        'max': float(dataset[col].max()),
                        'mean': float(dataset[col].mean()),
                        'std': float(dataset[col].std())
                    }
                except Exception as e:
                    logger.warning(f"Dimension analysis failed for column {col}: {e}")
        return {'dimension_statistics': dimensions}
    
    def _analyze_structural(self, dataset: pd.DataFrame, structural_cols: List[str]) -> Dict[str, Any]:
        """Analyze structural data"""
        return {'structural_elements': len(dataset), 'structural_columns': structural_cols}
    
    def _analyze_electrical(self, dataset: pd.DataFrame, electrical_cols: List[str]) -> Dict[str, Any]:
        """Analyze electrical data"""
        return {'electrical_components': len(dataset), 'electrical_columns': electrical_cols}
    
    def _analyze_soil(self, dataset: pd.DataFrame, soil_cols: List[str]) -> Dict[str, Any]:
        """Analyze soil data"""
        return {'soil_samples': len(dataset), 'soil_columns': soil_cols}
    
    def _analyze_vegetation(self, dataset: pd.DataFrame, vegetation_cols: List[str]) -> Dict[str, Any]:
        """Analyze vegetation data"""
        return {'vegetation_zones': len(dataset), 'vegetation_columns': vegetation_cols}
    
    def _analyze_climate(self, dataset: pd.DataFrame, climate_cols: List[str]) -> Dict[str, Any]:
        """Analyze climate data"""
        return {'climate_stations': len(dataset), 'climate_columns': climate_cols}
    
    def _analyze_projects(self, dataset: pd.DataFrame, project_cols: List[str]) -> Dict[str, Any]:
        """Analyze project data"""
        return {'projects': len(dataset), 'project_columns': project_cols}
    
    def _analyze_resources(self, dataset: pd.DataFrame, resource_cols: List[str]) -> Dict[str, Any]:
        """Analyze resource data"""
        return {'resources': len(dataset), 'resource_columns': resource_cols}
    
    def _analyze_costs(self, dataset: pd.DataFrame, cost_cols: List[str]) -> Dict[str, Any]:
        """Analyze cost data"""
        costs = {}
        for col in cost_cols:
            if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                try:
                    costs[col] = {
                        'total': float(dataset[col].sum()),
                        'mean': float(dataset[col].mean()),
                        'min': float(dataset[col].min()),
                        'max': float(dataset[col].max())
                    }
                except Exception as e:
                    logger.warning(f"Cost analysis failed for column {col}: {e}")
        return {'cost_statistics': costs}
    
    def _analyze_assets(self, dataset: pd.DataFrame, asset_cols: List[str]) -> Dict[str, Any]:
        """Analyze asset data"""
        return {'assets': len(dataset), 'asset_columns': asset_cols}
    
    def _analyze_maintenance(self, dataset: pd.DataFrame, maintenance_cols: List[str]) -> Dict[str, Any]:
        """Analyze maintenance data"""
        return {'maintenance_records': len(dataset), 'maintenance_columns': maintenance_cols}
    
    def _analyze_performance(self, dataset: pd.DataFrame, performance_cols: List[str]) -> Dict[str, Any]:
        """Analyze performance data"""
        return {'performance_metrics': len(dataset), 'performance_columns': performance_cols}
    
    def _analyze_structural_risks(self, dataset: pd.DataFrame, risk_cols: List[str]) -> Dict[str, Any]:
        """Analyze structural risks"""
        return {'structural_risk_indicators': len(dataset), 'risk_columns': risk_cols}
    
    def _analyze_environmental_risks(self, dataset: pd.DataFrame, risk_cols: List[str]) -> Dict[str, Any]:
        """Analyze environmental risks"""
        return {'environmental_risk_indicators': len(dataset), 'risk_columns': risk_cols}
    
    def _analyze_coordinates(self, dataset: pd.DataFrame, coord_cols: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """Analyze coordinate data"""
        lat_col, lon_col = coord_cols['lat'], coord_cols['lon']
        if lat_col is None or lon_col is None:
            return {'coordinate_count': 0}
        
        try:
            # Convert to numeric, ignoring errors
            lat_numeric = pd.to_numeric(dataset[lat_col], errors='coerce')  # type: ignore
            lon_numeric = pd.to_numeric(dataset[lon_col], errors='coerce')  # type: ignore
            
            return {
                'coordinate_range': {
                    'lat_min': float(lat_numeric.min()),  # type: ignore
                    'lat_max': float(lat_numeric.max()),  # type: ignore
                    'lon_min': float(lon_numeric.min()),  # type: ignore
                    'lon_max': float(lon_numeric.max())   # type: ignore
                },
                'coordinate_count': len(dataset)
            }
        except Exception as e:
            logger.warning(f"Coordinate analysis failed: {e}")
            return {'coordinate_count': 0, 'error': str(e)}
    
    def _analyze_proximity(self, dataset: pd.DataFrame, coord_cols: Dict[str, Optional[str]], location: Dict) -> Dict[str, Any]:
        """Analyze proximity to given location"""
        lat_col, lon_col = coord_cols['lat'], coord_cols['lon']
        if lat_col is None or lon_col is None:
            return {'error': 'No coordinate columns found'}
        
        try:
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
        except Exception as e:
            logger.warning(f"Proximity analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_time_series(self, dataset: pd.DataFrame, date_cols: List[str]) -> Dict[str, Any]:
        """Analyze time series data"""
        time_analysis = {}
        for col in date_cols:
            try:
                time_analysis[col] = {
                    'earliest': dataset[col].min(),
                    'latest': dataset[col].max(),
                    'duration_days': (dataset[col].max() - dataset[col].min()).days,
                    'record_count': len(dataset)
                }
            except Exception as e:
                logger.warning(f"Time series analysis failed for column {col}: {e}")
                time_analysis[col] = {'error': str(e)}
        return time_analysis
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlations in correlation matrix"""
        strong_correlations = []
        try:
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })
        except Exception as e:
            logger.warning(f"Strong correlation analysis failed: {e}")
        return strong_correlations
    
    def _has_infrastructure_data(self, dataset: pd.DataFrame) -> bool:
        """Check if dataset has infrastructure data"""
        infrastructure_patterns = []
        for category in self.data_patterns['infrastructure'].values():
            infrastructure_patterns.extend(category)
        return len(self._find_columns_by_patterns(dataset, infrastructure_patterns)) > 0
    
    def _has_risk_indicators(self, dataset: pd.DataFrame) -> bool:
        """Check if dataset has risk indicators"""
        risk_patterns = []
        for category in self.risk_patterns.values():
            risk_patterns.extend(category)
        return len(self._find_columns_by_patterns(dataset, risk_patterns)) > 0
    
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
    
    def _generate_risk_recommendations(self, dataset: pd.DataFrame) -> List[str]:
        """Generate risk-specific recommendations"""
        recommendations = []
        
        # Structural risk recommendations
        structural_risk_cols = self._find_columns_by_patterns(dataset, self.risk_patterns['structural_risk'])
        if structural_risk_cols:
            recommendations.append("Conduct structural integrity assessments")
        
        # Environmental risk recommendations
        environmental_risk_cols = self._find_columns_by_patterns(dataset, self.risk_patterns['environmental_risk'])
        if environmental_risk_cols:
            recommendations.append("Implement environmental monitoring systems")
        
        return recommendations
    
    def _generate_infrastructure_actions(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate infrastructure-specific action items"""
        actions = []
        
        # Material actions
        material_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['materials'])
        if material_cols:
            actions.append({
                'type': 'infrastructure',
                'priority': 'medium',
                'action': 'Material assessment',
                'description': 'Review and assess material conditions'
            })
        
        return actions 