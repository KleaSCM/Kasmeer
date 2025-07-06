# Author: KleaSCM
# Date: 2024
# Description: Spatial data analyzer

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_analyzer import BaseAnalyzer

class SpatialAnalyzer(BaseAnalyzer):
    """Analyzes spatial and geographic data"""
    
    def analyze(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze spatial data"""
        self.logger.info(f"Analyzing spatial data with {len(dataset)} records")
        
        location = kwargs.get('location', None)
        
        return {
            'spatial_analysis': self._analyze_spatial(dataset, location),
            'summary': self._generate_summary(dataset)
        }
    
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
            self.logger.warning(f"Coordinate analysis failed: {e}")
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
            self.logger.warning(f"Proximity analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_summary(self, dataset: pd.DataFrame) -> List[str]:
        """Generate spatial summary"""
        summary = []
        summary.append(f"Spatial dataset: {len(dataset)} records")
        
        # Check for coordinate data
        coord_cols = self._find_coordinate_columns(dataset)
        if coord_cols['lat'] and coord_cols['lon']:
            summary.append(f"Coordinate data: {coord_cols['lat']}, {coord_cols['lon']}")
        else:
            summary.append("No coordinate data found")
        
        return summary 