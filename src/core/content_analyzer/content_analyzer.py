# Author: KleaSCM
# Date: 2024
# Description: AI-powered content analyzer for automatic dataset classification

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from ..analyzers.base_analyzer import BaseAnalyzer
import logging

logger = logging.getLogger(__name__)

class ContentAnalyzer(BaseAnalyzer):
    """
    AI-powered content analyzer that automatically detects what's in any dataset.
    
    This analyzer can identify:
    - Traffic data (speed, congestion, flow)
    - Weather data (temperature, precipitation, humidity)
    - Construction data (permits, projects, costs)
    - Infrastructure data (pipes, electrical, roads)
    - Environmental data (soil, vegetation, pollution)
    - And much more!
    """
    
    def __init__(self):
        """Initialize the content analyzer"""
        super().__init__()
        self.logger.info("Initialized ContentAnalyzer - AI-powered dataset content detection")
        
        # Content detection patterns
        self.content_patterns = {
            'traffic': {
                'columns': ['speed', 'congestion', 'flow', 'volume', 'delay', 'jam', 'traffic', 'vehicle', 'car', 'truck'],
                'values': ['slow', 'fast', 'heavy', 'light', 'free_flow', 'stop_and_go'],
                'indicators': ['traffic_signal', 'intersection', 'highway', 'road', 'street']
            },
            'weather': {
                'columns': ['temperature', 'temp', 'humidity', 'precipitation', 'rain', 'snow', 'wind', 'pressure', 'weather', 'climate'],
                'values': ['sunny', 'rainy', 'cloudy', 'storm', 'clear', 'foggy'],
                'indicators': ['weather_station', 'forecast', 'meteorological', 'atmospheric']
            },
            'construction': {
                'columns': ['permit', 'project', 'construction', 'building', 'contract', 'cost', 'budget', 'contractor', 'work', 'site'],
                'values': ['active', 'completed', 'pending', 'approved', 'rejected', 'ongoing'],
                'indicators': ['construction_site', 'building_permit', 'project_phase', 'completion_date']
            },
            'infrastructure': {
                'columns': ['pipe', 'electrical', 'water', 'gas', 'sewer', 'utility', 'infrastructure', 'system', 'network', 'grid'],
                'values': ['functional', 'broken', 'maintenance', 'upgrade', 'repair'],
                'indicators': ['utility_pole', 'manhole', 'valve', 'transformer', 'substation']
            },
            'environmental': {
                'columns': ['soil', 'vegetation', 'pollution', 'air_quality', 'noise', 'contamination', 'environmental', 'ecology', 'green'],
                'values': ['clean', 'contaminated', 'healthy', 'damaged', 'protected'],
                'indicators': ['soil_sample', 'air_monitor', 'water_quality', 'vegetation_survey']
            },
            'transportation': {
                'columns': ['transit', 'bus', 'train', 'subway', 'metro', 'transport', 'public_transport', 'route', 'line', 'station'],
                'values': ['on_time', 'delayed', 'cancelled', 'crowded', 'empty'],
                'indicators': ['bus_stop', 'train_station', 'subway_line', 'route_number']
            },
            'safety': {
                'columns': ['safety', 'inspection', 'violation', 'hazard', 'risk', 'emergency', 'fire', 'security', 'compliance'],
                'values': ['safe', 'unsafe', 'violation', 'compliant', 'hazardous'],
                'indicators': ['safety_inspection', 'fire_department', 'building_code', 'violation_notice']
            },
            'financial': {
                'columns': ['cost', 'budget', 'expense', 'revenue', 'funding', 'grant', 'financial', 'money', 'dollar', 'fund'],
                'values': ['over_budget', 'under_budget', 'funded', 'unfunded', 'approved'],
                'indicators': ['budget_allocation', 'funding_source', 'cost_overrun', 'financial_report']
            }
        }
    
    def analyze_content(self, dataset: pd.DataFrame, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze dataset content and determine what type of data it contains.
        
        Args:
            dataset: The dataset to analyze
            filename: Optional filename for additional context
            
        Returns:
            Dictionary with content analysis results
        """
        self.logger.info(f"Analyzing content of dataset with {len(dataset)} records and {len(dataset.columns)} columns")
        
        analysis = {
            'content_type': self._detect_content_type(dataset),
            'confidence_scores': self._calculate_confidence_scores(dataset),
            'key_indicators': self._find_key_indicators(dataset),
            'data_characteristics': self._analyze_data_characteristics(dataset),
            'suggested_tags': self._generate_suggested_tags(dataset, filename),
            'content_summary': self._generate_content_summary(dataset),
            'cross_references': self._identify_cross_references(dataset)
        }
        
        self.logger.info(f"Content analysis completed. Detected type: {analysis['content_type']}")
        return analysis
    
    def analyze(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze dataset using the content analyzer"""
        return self.analyze_content(dataset, kwargs.get('filename'))
    
    def _detect_content_type(self, dataset: pd.DataFrame) -> str:
        """Detect the primary content type of the dataset"""
        scores = {}
        
        for content_type, patterns in self.content_patterns.items():
            score = 0
            
            # Check column names
            for col in dataset.columns:
                col_lower = col.lower()
                for pattern in patterns['columns']:
                    if pattern in col_lower:
                        score += 2  # Column match is strong indicator
            
            # Check sample values
            for col in dataset.columns:
                if dataset[col].dtype == 'object':
                    sample_values = dataset[col].dropna().astype(str).str.lower()
                    for pattern in patterns['values']:
                        if sample_values.str.contains(pattern).any():
                            score += 1  # Value match is good indicator
            
            scores[content_type] = score
        
        # Find the content type with highest score
        if scores:
            best_type = max(scores, key=lambda x: scores[x])
            if scores[best_type] > 0:
                return best_type
        
        return 'unknown'
    
    def _calculate_confidence_scores(self, dataset: pd.DataFrame) -> Dict[str, float]:
        """Calculate confidence scores for each content type"""
        confidence_scores = {}
        
        for content_type, patterns in self.content_patterns.items():
            score = 0
            max_possible = len(patterns['columns']) * 2 + len(patterns['values'])
            
            # Check column matches
            for col in dataset.columns:
                col_lower = col.lower()
                for pattern in patterns['columns']:
                    if pattern in col_lower:
                        score += 2
            
            # Check value matches
            for col in dataset.columns:
                if dataset[col].dtype == 'object':
                    sample_values = dataset[col].dropna().astype(str).str.lower()
                    for pattern in patterns['values']:
                        if sample_values.str.contains(pattern).any():
                            score += 1
            
            # Calculate confidence as percentage
            if max_possible > 0:
                confidence = min(score / max_possible, 1.0)
            else:
                confidence = 0.0
            
            confidence_scores[content_type] = confidence
        
        return confidence_scores
    
    def _find_key_indicators(self, dataset: pd.DataFrame) -> Dict[str, List[str]]:
        """Find key indicators that suggest content type"""
        indicators = {}
        
        for content_type, patterns in self.content_patterns.items():
            found_indicators = []
            
            # Check column names for indicators
            for col in dataset.columns:
                col_lower = col.lower()
                for indicator in patterns['indicators']:
                    if indicator in col_lower:
                        found_indicators.append(f"Column: {col}")
            
            # Check sample values for indicators
            for col in dataset.columns:
                if dataset[col].dtype == 'object':
                    sample_values = dataset[col].dropna().astype(str).str.lower()
                    for indicator in patterns['indicators']:
                        if sample_values.str.contains(indicator).any():
                            found_indicators.append(f"Value in {col}: {indicator}")
            
            if found_indicators:
                indicators[content_type] = found_indicators
        
        return indicators
    
    def _analyze_data_characteristics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze general data characteristics"""
        characteristics = {
            'total_records': len(dataset),
            'total_columns': len(dataset.columns),
            'numeric_columns': len(dataset.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(dataset.select_dtypes(include=['object']).columns),
            'date_columns': len(dataset.select_dtypes(include=['datetime64']).columns),
            'missing_data_percentage': (dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))) * 100,
            'has_coordinates': self._has_coordinate_data(dataset),
            'has_temporal_data': self._has_temporal_data(dataset),
            'data_completeness': self._calculate_completeness(dataset)
        }
        
        return characteristics
    
    def _generate_suggested_tags(self, dataset: pd.DataFrame, filename: Optional[str] = None) -> List[str]:
        """Generate suggested tags for the dataset"""
        tags = []
        
        # Content type tag
        content_type = self._detect_content_type(dataset)
        if content_type != 'unknown':
            tags.append(content_type)
        
        # Geographic tag
        if self._has_coordinate_data(dataset):
            tags.append('geospatial')
        
        # Temporal tag
        if self._has_temporal_data(dataset):
            tags.append('temporal')
        
        # Size tag
        if len(dataset) > 10000:
            tags.append('large_dataset')
        elif len(dataset) > 1000:
            tags.append('medium_dataset')
        else:
            tags.append('small_dataset')
        
        # Quality tag
        completeness = self._calculate_completeness(dataset)
        if completeness > 90:
            tags.append('high_quality')
        elif completeness > 70:
            tags.append('medium_quality')
        else:
            tags.append('low_quality')
        
        # Filename-based tags
        if filename:
            filename_lower = filename.lower()
            if '2024' in filename_lower:
                tags.append('2024')
            if '2023' in filename_lower:
                tags.append('2023')
            if 'nyc' in filename_lower or 'new_york' in filename_lower:
                tags.append('nyc')
        
        return tags
    
    def _generate_content_summary(self, dataset: pd.DataFrame) -> str:
        """Generate a human-readable content summary"""
        content_type = self._detect_content_type(dataset)
        record_count = len(dataset)
        column_count = len(dataset.columns)
        
        summary = f"This appears to be {content_type} data with {record_count:,} records and {column_count} columns."
        
        if self._has_coordinate_data(dataset):
            summary += " The dataset contains geographic coordinates."
        
        if self._has_temporal_data(dataset):
            summary += " The dataset includes temporal information."
        
        completeness = self._calculate_completeness(dataset)
        summary += f" Data completeness is {completeness:.1f}%."
        
        return summary
    
    def _identify_cross_references(self, dataset: pd.DataFrame) -> List[str]:
        """Identify potential cross-references with other datasets"""
        cross_refs = []
        
        # Check for common ID patterns that might link to other datasets
        id_columns = [col for col in dataset.columns if 'id' in col.lower()]
        if id_columns:
            cross_refs.append(f"Contains ID columns: {', '.join(id_columns)} - may link to related datasets")
        
        # Check for coordinate data that could be spatially related
        if self._has_coordinate_data(dataset):
            cross_refs.append("Contains coordinates - can be spatially joined with other geospatial datasets")
        
        # Check for temporal data that could be temporally related
        if self._has_temporal_data(dataset):
            cross_refs.append("Contains temporal data - can be temporally joined with other time-series datasets")
        
        return cross_refs
    
    def _has_coordinate_data(self, dataset: pd.DataFrame) -> bool:
        """Check if dataset has coordinate data"""
        coord_cols = self._find_coordinate_columns(dataset)
        return coord_cols['lat'] is not None and coord_cols['lon'] is not None
    
    def _has_temporal_data(self, dataset: pd.DataFrame) -> bool:
        """Check if dataset has temporal data"""
        date_cols = dataset.select_dtypes(include=['datetime64']).columns
        return len(date_cols) > 0
    
    def _calculate_completeness(self, dataset: pd.DataFrame) -> float:
        """Calculate data completeness percentage"""
        total_cells = len(dataset) * len(dataset.columns)
        filled_cells = total_cells - dataset.isnull().sum().sum()
        return (filled_cells / total_cells) * 100 if total_cells > 0 else 0.0 