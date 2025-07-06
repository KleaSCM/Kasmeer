# Author: KleaSCM
# Date: 2024
# Description: Environmental data analyzer

import pandas as pd
from typing import Dict, List, Any
from .base_analyzer import BaseAnalyzer

class EnvironmentalAnalyzer(BaseAnalyzer):
    """Analyzes environmental data"""
    
    def analyze(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze environmental data"""
        self.logger.info(f"Analyzing environmental data with {len(dataset)} records")
        
        return {
            'environmental_context': self._analyze_environmental_context(dataset),
            'summary': self._generate_summary(dataset)
        }
    
    def _analyze_environmental_context(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze environmental context of the site"""
        environmental = {
            'soil_conditions': {},
            'climate_data': {},
            'vegetation': {},
            'water_resources': {},
            'air_quality': {},
            'protected_areas': {},
            'summary': []
        }
        
        # Dynamic environmental discovery
        soil_patterns = ['soil', 'geotechnical', 'bearing_capacity', 'moisture', 'ph']
        soil_cols = self._find_columns_by_patterns(dataset, soil_patterns)
        
        climate_patterns = ['temperature', 'rainfall', 'wind', 'humidity', 'climate']
        climate_cols = self._find_columns_by_patterns(dataset, climate_patterns)
        
        vegetation_patterns = ['vegetation', 'tree', 'plant', 'species', 'density']
        vegetation_cols = self._find_columns_by_patterns(dataset, vegetation_patterns)
        
        water_patterns = ['water', 'river', 'stream', 'flood', 'groundwater', 'wetland']
        water_cols = self._find_columns_by_patterns(dataset, water_patterns)
        
        air_patterns = ['air', 'pollution', 'emission', 'particulate', 'gas']
        air_cols = self._find_columns_by_patterns(dataset, air_patterns)
        
        # Analyze soil conditions
        for col in soil_cols:
            if col in dataset.columns:
                try:
                    soil_data = dataset[col].value_counts()
                    environmental['soil_conditions'][col] = soil_data.to_dict()
                    environmental['summary'].append(f"Soil: {len(soil_data)} different conditions found")
                except Exception as e:
                    self.logger.warning(f"Soil analysis failed for column {col}: {e}")
        
        # Analyze climate data
        for col in climate_cols:
            if col in dataset.columns:
                try:
                    if dataset[col].dtype in ['int64', 'float64']:
                        climate_stats = {
                            'min': float(dataset[col].min()),
                            'max': float(dataset[col].max()),
                            'mean': float(dataset[col].mean()),
                            'std': float(dataset[col].std())
                        }
                        environmental['climate_data'][col] = climate_stats
                        environmental['summary'].append(f"Climate {col}: {climate_stats['mean']:.1f} avg (range: {climate_stats['min']:.1f}-{climate_stats['max']:.1f})")
                    else:
                        climate_data = dataset[col].value_counts()
                        environmental['climate_data'][col] = climate_data.to_dict()
                        environmental['summary'].append(f"Climate: {len(climate_data)} different values found")
                except Exception as e:
                    self.logger.warning(f"Climate analysis failed for column {col}: {e}")
        
        # Analyze vegetation
        for col in vegetation_cols:
            if col in dataset.columns:
                try:
                    vegetation_data = dataset[col].value_counts()
                    environmental['vegetation'][col] = vegetation_data.to_dict()
                    environmental['summary'].append(f"Vegetation: {len(vegetation_data)} different types found")
                except Exception as e:
                    self.logger.warning(f"Vegetation analysis failed for column {col}: {e}")
        
        return environmental
    
    def _generate_summary(self, dataset: pd.DataFrame) -> List[str]:
        """Generate environmental summary"""
        summary = []
        summary.append(f"Environmental dataset: {len(dataset)} records")
        
        # Count environmental types found
        env_patterns = []
        for category in self.data_patterns['environmental'].values():
            env_patterns.extend(category)
        
        env_cols = self._find_columns_by_patterns(dataset, env_patterns)
        if env_cols:
            summary.append(f"Environmental factors: {len(env_cols)} columns identified")
        
        return summary 