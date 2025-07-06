# Author: KleaSCM
# Date: 2024
# Description: Temporal data analyzer

import pandas as pd
from typing import Dict, List, Any
from .base_analyzer import BaseAnalyzer

class TemporalAnalyzer(BaseAnalyzer):
    """Analyzes temporal and time-series data"""
    
    def analyze(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze temporal data"""
        self.logger.info(f"Analyzing temporal data with {len(dataset)} records")
        
        return {
            'temporal_analysis': self._analyze_temporal(dataset),
            'summary': self._generate_summary(dataset)
        }
    
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
                self.logger.warning(f"Time series analysis failed for column {col}: {e}")
                time_analysis[col] = {'error': str(e)}
        return time_analysis
    
    def _generate_summary(self, dataset: pd.DataFrame) -> List[str]:
        """Generate temporal summary"""
        summary = []
        summary.append(f"Temporal dataset: {len(dataset)} records")
        
        # Check for date columns
        date_cols = list(dataset.select_dtypes(include=['datetime64']).columns)
        if date_cols:
            summary.append(f"Date columns: {', '.join(date_cols)}")
        else:
            summary.append("No date columns found")
        
        return summary 