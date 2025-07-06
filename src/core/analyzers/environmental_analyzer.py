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
        soil_patterns = ['soil', 'geotechnical', 'bearing_capacity', 'moisture', 'ph', 'contamination']
        soil_cols = self._find_columns_by_patterns(dataset, soil_patterns)
        
        # Filter out false positives for soil columns
        filtered_soil_cols = []
        for col in soil_cols:
            col_lower = col.lower()
            # Exclude columns that are clearly not soil-related
            if any(exclude_pattern in col_lower for exclude_pattern in ['geographical', 'district', 'zone', 'area', 'region']):
                continue
            # Only include columns that are actually soil-related
            if any(soil_pattern in col_lower for soil_pattern in ['soil', 'geotechnical', 'bearing', 'moisture', 'ph', 'contamination']):
                filtered_soil_cols.append(col)
        
        soil_cols = filtered_soil_cols
        
        climate_patterns = ['temperature', 'temp', 'rainfall', 'wind', 'humidity', 'rh', 'climate']
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
        
        # Analyze climate data with enhanced processing
        climate_analysis = self._analyze_climate_data(dataset, climate_cols)
        environmental['climate_data'] = climate_analysis['climate_data']
        environmental['summary'].extend(climate_analysis['summary'])
        
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
    
    def _analyze_climate_data(self, dataset: pd.DataFrame, climate_cols: List[str]) -> Dict[str, Any]:
        """Enhanced climate data analysis with monthly averages"""
        climate_data = {}
        summary = []
        
        # Check for temperature data
        temp_cols = [col for col in climate_cols if 'temp' in col.lower()]
        if temp_cols:
            for col in temp_cols:
                if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                    try:
                        # Basic stats
                        temp_stats = {
                            'min': float(dataset[col].min()),
                            'max': float(dataset[col].max()),
                            'mean': float(dataset[col].mean()),
                            'std': float(dataset[col].std())
                        }
                        climate_data[f'{col}_stats'] = temp_stats
                        
                        # Monthly averages if month column exists
                        if 'month' in dataset.columns:
                            monthly_avg = dataset.groupby('month')[col].mean()
                            climate_data[f'{col}_monthly_averages'] = monthly_avg.to_dict()
                            summary.append(f"Temperature: {temp_stats['mean']:.1f}°C avg with monthly data")
                        else:
                            summary.append(f"Temperature: {temp_stats['mean']:.1f}°C avg (range: {temp_stats['min']:.1f}-{temp_stats['max']:.1f}°C)")
                    except Exception as e:
                        self.logger.warning(f"Temperature analysis failed for column {col}: {e}")
        
        # Check for humidity data
        hum_cols = [col for col in climate_cols if 'rh' in col.lower() or 'humidity' in col.lower()]
        if hum_cols:
            for col in hum_cols:
                if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                    try:
                        # Basic stats
                        hum_stats = {
                            'min': float(dataset[col].min()),
                            'max': float(dataset[col].max()),
                            'mean': float(dataset[col].mean()),
                            'std': float(dataset[col].std())
                        }
                        climate_data[f'{col}_stats'] = hum_stats
                        
                        # Monthly averages if month column exists
                        if 'month' in dataset.columns:
                            monthly_avg = dataset.groupby('month')[col].mean()
                            climate_data[f'{col}_monthly_averages'] = monthly_avg.to_dict()
                            summary.append(f"Humidity: {hum_stats['mean']:.1f}% avg with monthly data")
                        else:
                            summary.append(f"Humidity: {hum_stats['mean']:.1f}% avg (range: {hum_stats['min']:.1f}-{hum_stats['max']:.1f}%)")
                    except Exception as e:
                        self.logger.warning(f"Humidity analysis failed for column {col}: {e}")
        
        # Process other climate columns
        other_climate_cols = [col for col in climate_cols if col not in temp_cols and col not in hum_cols]
        for col in other_climate_cols:
            if col in dataset.columns:
                try:
                    if dataset[col].dtype in ['int64', 'float64']:
                        climate_stats = {
                            'min': float(dataset[col].min()),
                            'max': float(dataset[col].max()),
                            'mean': float(dataset[col].mean()),
                            'std': float(dataset[col].std())
                        }
                        climate_data[f'{col}_stats'] = climate_stats
                        summary.append(f"Climate {col}: {climate_stats['mean']:.1f} avg (range: {climate_stats['min']:.1f}-{climate_stats['max']:.1f})")
                    else:
                        climate_values = dataset[col].value_counts()
                        climate_data[f'{col}_values'] = climate_values.to_dict()
                        summary.append(f"Climate {col}: {len(climate_values)} different values found")
                except Exception as e:
                    self.logger.warning(f"Climate analysis failed for column {col}: {e}")
        
        # Add overall climate summary
        if climate_data:
            climate_data['monthly_averages'] = {}
            if 'month' in dataset.columns:
                # Create combined monthly summary
                monthly_summary = {}
                for col in climate_cols:
                    if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                        monthly_avg = dataset.groupby('month')[col].mean()
                        monthly_summary[col] = monthly_avg.to_dict()
                climate_data['monthly_averages'] = monthly_summary
                summary.append(f"Climate data: Monthly averages available for {len(monthly_summary)} parameters")
        
        return {
            'climate_data': climate_data,
            'summary': summary
        }
    
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