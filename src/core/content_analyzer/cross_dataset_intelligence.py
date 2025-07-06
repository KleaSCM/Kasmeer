# Author: KleaSCM
# Date: 2024
# Description: Cross Dataset Intelligence - Find relationships between datasets

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from ..analyzers.base_analyzer import BaseAnalyzer
from .content_analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)

class CrossDatasetIntelligence:
    """
    Cross Dataset Intelligence - Find relationships and correlations between datasets.
    
    This system:
    - Identifies spatial relationships between datasets
    - Finds temporal correlations
    - Discovers data quality patterns
    - Suggests dataset combinations for analysis
    """
    
    def __init__(self):
        """Initialize the cross dataset intelligence system"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized CrossDatasetIntelligence - Finding dataset relationships")
        
        # Initialize content analyzer
        self.content_analyzer = ContentAnalyzer()
        
        # Relationship types
        self.relationship_types = {
            'spatial': 'Datasets share geographic areas',
            'temporal': 'Datasets cover similar time periods',
            'thematic': 'Datasets cover related topics',
            'structural': 'Datasets have similar data structures',
            'quality': 'Datasets have similar data quality characteristics'
        }
    
    def analyze_dataset_relationships(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze relationships between multiple datasets.
        
        Args:
            datasets: Dictionary of dataset names to DataFrames
            
        Returns:
            Dictionary with relationship analysis results
        """
        self.logger.info(f"Analyzing relationships between {len(datasets)} datasets")
        
        # Step 1: Analyze each dataset individually
        dataset_analyses = {}
        for name, dataset in datasets.items():
            dataset_analyses[name] = self.content_analyzer.analyze_content(dataset, name)
        
        # Step 2: Find relationships between datasets
        relationships = self._find_relationships(datasets, dataset_analyses)
        
        # Step 3: Identify potential combinations
        combinations = self._identify_combinations(datasets, dataset_analyses, relationships)
        
        # Step 4: Generate insights
        insights = self._generate_insights(datasets, dataset_analyses, relationships)
        
        return {
            'dataset_analyses': dataset_analyses,
            'relationships': relationships,
            'combinations': combinations,
            'insights': insights,
            'summary': self._generate_summary(datasets, relationships, combinations)
        }
    
    def _find_relationships(self, datasets: Dict[str, pd.DataFrame], 
                          dataset_analyses: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Find relationships between datasets"""
        relationships = {
            'spatial': [],
            'temporal': [],
            'thematic': [],
            'structural': [],
            'quality': []
        }
        
        dataset_names = list(datasets.keys())
        
        # Compare each pair of datasets
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names[i+1:], i+1):
                dataset1 = datasets[name1]
                dataset2 = datasets[name2]
                analysis1 = dataset_analyses[name1]
                analysis2 = dataset_analyses[name2]
                
                # Check spatial relationships
                spatial_rel = self._check_spatial_relationship(dataset1, dataset2, name1, name2)
                if spatial_rel:
                    relationships['spatial'].append(spatial_rel)
                
                # Check temporal relationships
                temporal_rel = self._check_temporal_relationship(dataset1, dataset2, name1, name2)
                if temporal_rel:
                    relationships['temporal'].append(temporal_rel)
                
                # Check thematic relationships
                thematic_rel = self._check_thematic_relationship(analysis1, analysis2, name1, name2)
                if thematic_rel:
                    relationships['thematic'].append(thematic_rel)
                
                # Check structural relationships
                structural_rel = self._check_structural_relationship(dataset1, dataset2, name1, name2)
                if structural_rel:
                    relationships['structural'].append(structural_rel)
                
                # Check quality relationships
                quality_rel = self._check_quality_relationship(analysis1, analysis2, name1, name2)
                if quality_rel:
                    relationships['quality'].append(quality_rel)
        
        return relationships
    
    def _check_spatial_relationship(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame, 
                                  name1: str, name2: str) -> Optional[Dict[str, Any]]:
        """Check if two datasets have spatial relationships"""
        # Get coordinate columns for both datasets
        coord1 = self._find_coordinate_columns(dataset1)
        coord2 = self._find_coordinate_columns(dataset2)
        
        if not (coord1['lat'] and coord1['lon'] and coord2['lat'] and coord2['lon']):
            return None
        
        try:
            # Convert coordinates to numeric and ensure they're Series
            lat1 = pd.to_numeric(dataset1[coord1['lat']], errors='coerce')
            lon1 = pd.to_numeric(dataset1[coord1['lon']], errors='coerce')
            lat2 = pd.to_numeric(dataset2[coord2['lat']], errors='coerce')
            lon2 = pd.to_numeric(dataset2[coord2['lon']], errors='coerce')
            
            # Calculate spatial overlap with error handling
            try:
                # Check if we have valid numeric data
                if not isinstance(lat1, pd.Series) or not isinstance(lon1, pd.Series) or \
                   not isinstance(lat2, pd.Series) or not isinstance(lon2, pd.Series):
                    return None
                
                lat1_range = (float(lat1.min()), float(lat1.max()))
                lon1_range = (float(lon1.min()), float(lon1.max()))
                lat2_range = (float(lat2.min()), float(lat2.max()))
                lon2_range = (float(lon2.min()), float(lon2.max()))
            except (ValueError, TypeError):
                return None
            
            # Check for overlap
            lat_overlap = max(0, min(lat1_range[1], lat2_range[1]) - max(lat1_range[0], lat2_range[0]))
            lon_overlap = max(0, min(lon1_range[1], lon2_range[1]) - max(lon1_range[0], lon2_range[0]))
            
            if lat_overlap > 0 and lon_overlap > 0:
                overlap_area = lat_overlap * lon_overlap
                area1 = (lat1_range[1] - lat1_range[0]) * (lon1_range[1] - lon1_range[0])
                area2 = (lat2_range[1] - lat2_range[0]) * (lon2_range[1] - lon2_range[0])
                
                overlap_percentage = min(overlap_area / area1, overlap_area / area2) * 100
                
                return {
                    'dataset1': name1,
                    'dataset2': name2,
                    'relationship_type': 'spatial_overlap',
                    'overlap_percentage': overlap_percentage,
                    'strength': 'strong' if overlap_percentage > 50 else 'moderate' if overlap_percentage > 20 else 'weak',
                    'description': f"{overlap_percentage:.1f}% spatial overlap between {name1} and {name2}"
                }
        
        except Exception as e:
            self.logger.warning(f"Error checking spatial relationship: {e}")
        
        return None
    
    def _check_temporal_relationship(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame,
                                   name1: str, name2: str) -> Optional[Dict[str, Any]]:
        """Check if two datasets have temporal relationships"""
        # Find date columns
        date_cols1 = dataset1.select_dtypes(include=['datetime64']).columns
        date_cols2 = dataset2.select_dtypes(include=['datetime64']).columns
        
        if len(date_cols1) == 0 or len(date_cols2) == 0:
            return None
        
        try:
            # Get date ranges
            date_range1 = (dataset1[date_cols1[0]].min(), dataset1[date_cols1[0]].max())
            date_range2 = (dataset2[date_cols2[0]].min(), dataset2[date_cols2[0]].max())
            
            # Check for overlap
            overlap_start = max(date_range1[0], date_range2[0])
            overlap_end = min(date_range1[1], date_range2[1])
            
            if overlap_start <= overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                total_range1 = (date_range1[1] - date_range1[0]).days
                total_range2 = (date_range2[1] - date_range2[0]).days
                
                overlap_percentage = min(overlap_days / total_range1, overlap_days / total_range2) * 100
                
                return {
                    'dataset1': name1,
                    'dataset2': name2,
                    'relationship_type': 'temporal_overlap',
                    'overlap_percentage': overlap_percentage,
                    'overlap_days': overlap_days,
                    'strength': 'strong' if overlap_percentage > 50 else 'moderate' if overlap_percentage > 20 else 'weak',
                    'description': f"{overlap_percentage:.1f}% temporal overlap ({overlap_days} days) between {name1} and {name2}"
                }
        
        except Exception as e:
            self.logger.warning(f"Error checking temporal relationship: {e}")
        
        return None
    
    def _check_thematic_relationship(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any],
                                   name1: str, name2: str) -> Optional[Dict[str, Any]]:
        """Check if two datasets have thematic relationships"""
        content_type1 = analysis1.get('content_type', 'unknown')
        content_type2 = analysis2.get('content_type', 'unknown')
        
        if content_type1 == content_type2 and content_type1 != 'unknown':
            return {
                'dataset1': name1,
                'dataset2': name2,
                'relationship_type': 'thematic_same',
                'content_type': content_type1,
                'strength': 'strong',
                'description': f"Both {name1} and {name2} contain {content_type1} data"
            }
        
        # Check for related content types
        related_types = {
            'traffic': ['transportation', 'infrastructure'],
            'weather': ['environmental'],
            'construction': ['infrastructure', 'financial'],
            'infrastructure': ['construction', 'safety'],
            'environmental': ['weather', 'safety'],
            'transportation': ['traffic', 'infrastructure'],
            'safety': ['construction', 'infrastructure', 'environmental'],
            'financial': ['construction']
        }
        
        if content_type1 in related_types and content_type2 in related_types[content_type1]:
            return {
                'dataset1': name1,
                'dataset2': name2,
                'relationship_type': 'thematic_related',
                'content_type1': content_type1,
                'content_type2': content_type2,
                'strength': 'moderate',
                'description': f"{name1} ({content_type1}) and {name2} ({content_type2}) are thematically related"
            }
        
        return None
    
    def _check_structural_relationship(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame,
                                     name1: str, name2: str) -> Optional[Dict[str, Any]]:
        """Check if two datasets have structural relationships"""
        # Compare column structures
        cols1 = set(dataset1.columns)
        cols2 = set(dataset2.columns)
        
        common_cols = cols1.intersection(cols2)
        if len(common_cols) > 0:
            similarity = len(common_cols) / max(len(cols1), len(cols2))
            
            return {
                'dataset1': name1,
                'dataset2': name2,
                'relationship_type': 'structural_similarity',
                'common_columns': list(common_cols),
                'similarity_percentage': similarity * 100,
                'strength': 'strong' if similarity > 0.5 else 'moderate' if similarity > 0.2 else 'weak',
                'description': f"{similarity*100:.1f}% structural similarity between {name1} and {name2}"
            }
        
        return None
    
    def _check_quality_relationship(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any],
                                  name1: str, name2: str) -> Optional[Dict[str, Any]]:
        """Check if two datasets have similar quality characteristics"""
        quality1 = analysis1.get('data_characteristics', {})
        quality2 = analysis2.get('data_characteristics', {})
        
        completeness1 = quality1.get('data_completeness', 0)
        completeness2 = quality2.get('data_completeness', 0)
        
        # Check if both datasets have similar completeness
        if abs(completeness1 - completeness2) < 20:  # Within 20% of each other
            return {
                'dataset1': name1,
                'dataset2': name2,
                'relationship_type': 'quality_similarity',
                'completeness1': completeness1,
                'completeness2': completeness2,
                'difference': abs(completeness1 - completeness2),
                'strength': 'moderate',
                'description': f"Similar data quality: {completeness1:.1f}% vs {completeness2:.1f}% completeness"
            }
        
        return None
    
    def _identify_combinations(self, datasets: Dict[str, pd.DataFrame], 
                             dataset_analyses: Dict[str, Any],
                             relationships: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify potential dataset combinations for analysis"""
        combinations = []
        
        # Find datasets with strong relationships
        strong_relationships = []
        for rel_type, rels in relationships.items():
            for rel in rels:
                if rel.get('strength') == 'strong':
                    strong_relationships.append(rel)
        
        # Group related datasets
        dataset_groups = {}
        for rel in strong_relationships:
            ds1, ds2 = rel['dataset1'], rel['dataset2']
            
            # Find or create group for dataset1
            group1 = None
            for group_id, group in dataset_groups.items():
                if ds1 in group:
                    group1 = group_id
                    break
            
            # Find or create group for dataset2
            group2 = None
            for group_id, group in dataset_groups.items():
                if ds2 in group:
                    group2 = group_id
                    break
            
            # Merge groups if needed
            if group1 and group2 and group1 != group2:
                dataset_groups[group1].extend(dataset_groups[group2])
                del dataset_groups[group2]
            elif group1 and not group2:
                dataset_groups[group1].append(ds2)
            elif group2 and not group1:
                dataset_groups[group2].append(ds1)
            else:
                # Create new group
                new_group_id = len(dataset_groups)
                dataset_groups[new_group_id] = [ds1, ds2]
        
        # Create combination suggestions
        for group_id, group_datasets in dataset_groups.items():
            if len(group_datasets) >= 2:
                combinations.append({
                    'combination_id': f"group_{group_id}",
                    'datasets': group_datasets,
                    'relationship_types': [rel['relationship_type'] for rel in strong_relationships 
                                         if rel['dataset1'] in group_datasets and rel['dataset2'] in group_datasets],
                    'suggested_analysis': self._suggest_analysis_for_combination(group_datasets, dataset_analyses),
                    'strength': 'strong' if len(group_datasets) > 2 else 'moderate'
                })
        
        return combinations
    
    def _suggest_analysis_for_combination(self, datasets: List[str], 
                                        dataset_analyses: Dict[str, Any]) -> List[str]:
        """Suggest analysis types for a dataset combination"""
        suggestions = []
        
        # Analyze content types in the combination
        content_types = [dataset_analyses[ds].get('content_type', 'unknown') for ds in datasets]
        
        # Suggest based on content types
        if 'traffic' in content_types and 'weather' in content_types:
            suggestions.append("Traffic-weather correlation analysis")
        
        if 'construction' in content_types and 'financial' in content_types:
            suggestions.append("Construction cost analysis")
        
        if 'infrastructure' in content_types and 'safety' in content_types:
            suggestions.append("Infrastructure safety assessment")
        
        if 'environmental' in content_types and 'weather' in content_types:
            suggestions.append("Environmental impact analysis")
        
        # General suggestions
        if len(datasets) >= 3:
            suggestions.append("Multi-dataset correlation analysis")
        
        if any('geospatial' in dataset_analyses[ds].get('suggested_tags', []) for ds in datasets):
            suggestions.append("Spatial analysis and mapping")
        
        if any('temporal' in dataset_analyses[ds].get('suggested_tags', []) for ds in datasets):
            suggestions.append("Time series analysis")
        
        return suggestions
    
    def _generate_insights(self, datasets: Dict[str, pd.DataFrame], 
                          dataset_analyses: Dict[str, Any],
                          relationships: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate insights about the dataset collection"""
        insights = []
        
        # Count relationship types
        rel_counts = {rel_type: len(rels) for rel_type, rels in relationships.items()}
        
        if rel_counts.get('spatial', 0) > 0:
            insights.append(f"Found {rel_counts['spatial']} spatial relationships - datasets can be geographically joined")
        
        if rel_counts.get('temporal', 0) > 0:
            insights.append(f"Found {rel_counts['temporal']} temporal relationships - datasets can be temporally correlated")
        
        if rel_counts.get('thematic', 0) > 0:
            insights.append(f"Found {rel_counts['thematic']} thematic relationships - datasets cover related topics")
        
        # Content type distribution
        content_types = [analysis.get('content_type', 'unknown') for analysis in dataset_analyses.values()]
        type_counts = {}
        for ct in content_types:
            type_counts[ct] = type_counts.get(ct, 0) + 1
        
        most_common = max(type_counts.items(), key=lambda x: x[1]) if type_counts else ('unknown', 0)
        insights.append(f"Most common content type: {most_common[0]} ({most_common[1]} datasets)")
        
        # Quality insights
        completeness_scores = [analysis.get('data_characteristics', {}).get('data_completeness', 0) 
                             for analysis in dataset_analyses.values()]
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        insights.append(f"Average data completeness: {avg_completeness:.1f}%")
        
        return insights
    
    def _generate_summary(self, datasets: Dict[str, pd.DataFrame], 
                         relationships: Dict[str, List[Dict[str, Any]]],
                         combinations: List[Dict[str, Any]]) -> str:
        """Generate a summary of the cross-dataset analysis"""
        total_datasets = len(datasets)
        total_relationships = sum(len(rels) for rels in relationships.values())
        total_combinations = len(combinations)
        
        summary = f"Cross-dataset analysis of {total_datasets} datasets"
        summary += f"\nFound {total_relationships} relationships across {len(relationships)} categories"
        summary += f"\nIdentified {total_combinations} potential dataset combinations"
        
        if total_combinations > 0:
            strong_combinations = [c for c in combinations if c.get('strength') == 'strong']
            summary += f"\n{len(strong_combinations)} strong combinations recommended for analysis"
        
        return summary
    
    def _find_coordinate_columns(self, dataset: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Find coordinate columns in a dataset"""
        lat_col, lon_col = None, None
        
        for col in dataset.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['latitude', 'lat']):
                if not any(false_match in col_lower for false_match in ['tabulation', 'calculation', 'relation']):
                    lat_col = col
            elif any(pattern in col_lower for pattern in ['longitude', 'lon', 'lng']):
                lon_col = col
            elif lat_col is None and 'y' in col_lower:
                lat_col = col
            elif lon_col is None and 'x' in col_lower:
                lon_col = col
        
        return {'lat': lat_col, 'lon': lon_col} 