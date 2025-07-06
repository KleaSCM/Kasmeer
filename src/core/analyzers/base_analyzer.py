# Author: KleaSCM
# Date: 2024
# Description: Base analyzer class for civil engineering data

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from ...utils.logging_utils import setup_logging

logger = setup_logging(__name__)

class BaseAnalyzer(ABC):
    """Base class for all data analyzers"""
    
    def __init__(self):
        self.logger = logger
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
        
        self.risk_patterns = {
            'structural_risk': ['crack', 'corrosion', 'deterioration', 'failure', 'collapse'],
            'environmental_risk': ['flood', 'earthquake', 'fire', 'storm', 'drought'],
            'operational_risk': ['breakdown', 'outage', 'interruption', 'failure', 'accident'],
            'financial_risk': ['cost_overrun', 'budget_exceed', 'loss', 'liability', 'penalty'],
            'compliance_risk': ['violation', 'non_compliance', 'penalty', 'fine', 'legal']
        }
        
        self.spatial_patterns = {
            'coordinates': ['lat', 'lon', 'latitude', 'longitude', 'x', 'y', 'easting', 'northing'],
            'addresses': ['address', 'street', 'city', 'state', 'postcode', 'zip'],
            'zones': ['zone', 'area', 'region', 'district', 'sector', 'quadrant'],
            'elevation': ['elevation', 'height', 'depth', 'altitude', 'grade', 'slope']
        }
    
    @abstractmethod
    def analyze(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Main analysis method to be implemented by subclasses"""
        pass
    
    def _find_columns_by_patterns(self, dataset: pd.DataFrame, patterns: List[str]) -> List[str]:
        """Find columns that match given patterns"""
        matching_cols = []
        for col in dataset.columns:
            col_lower = col.lower()
            # Check for exact matches and partial matches
            if any(pattern.lower() in col_lower for pattern in patterns):
                matching_cols.append(col)
            # Also check for common variations
            elif any(pattern.lower().replace('_', '') in col_lower.replace('_', '') for pattern in patterns):
                matching_cols.append(col)
        return matching_cols
    
    def _find_coordinate_columns(self, dataset: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Find coordinate columns with improved detection"""
        coord_patterns = self.spatial_patterns['coordinates']
        lat_col = None
        lon_col = None
        
        for col in dataset.columns:
            col_lower = col.lower()
            # Check for latitude patterns - be more precise to avoid false matches
            if any(pattern in col_lower for pattern in ['latitude', 'lat']):
                # Avoid false matches like "lat" in "Tabulation"
                if not any(false_match in col_lower for false_match in ['tabulation', 'calculation', 'relation']):
                    lat_col = col
            # Check for longitude patterns - be more precise
            elif any(pattern in col_lower for pattern in ['longitude', 'lon', 'lng']):
                lon_col = col
            # Only use x/y/northing/easting if no lat/lon found
            elif lat_col is None and any(pattern in col_lower for pattern in ['y', 'northing']):
                lat_col = col
            elif lon_col is None and any(pattern in col_lower for pattern in ['x', 'easting']):
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