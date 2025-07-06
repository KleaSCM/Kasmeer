# Author: KleaSCM
# Date: 2024
# Description: Smart Tagger - Automatic dataset tagging and classification

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set
import logging
from pathlib import Path
from ..analyzers.base_analyzer import BaseAnalyzer
from .content_analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)

class SmartTagger:
    """
    Smart Tagger - Automatically tags datasets based on content analysis.
    
    This system:
    - Analyzes dataset content automatically
    - Suggests relevant tags
    - Learns from user feedback
    - Maintains tag consistency across datasets
    """
    
    def __init__(self):
        """Initialize the smart tagger"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized SmartTagger - Automatic dataset tagging")
        
        # Initialize content analyzer
        self.content_analyzer = ContentAnalyzer()
        
        # Tag categories and their relationships
        self.tag_categories = {
            'data_type': ['traffic', 'weather', 'construction', 'infrastructure', 'environmental', 'transportation', 'safety', 'financial'],
            'geographic': ['nyc', 'los_angeles', 'chicago', 'miami', 'geospatial', 'local', 'regional', 'national'],
            'temporal': ['2024', '2023', '2022', 'historical', 'real_time', 'daily', 'monthly', 'annual'],
            'quality': ['high_quality', 'medium_quality', 'low_quality', 'verified', 'raw', 'processed'],
            'size': ['small_dataset', 'medium_dataset', 'large_dataset', 'massive_dataset'],
            'source': ['government', 'public', 'private', 'academic', 'research', 'commercial']
        }
        
        # Tag relationships and dependencies
        self.tag_relationships = {
            'traffic': ['geospatial', 'temporal', 'transportation'],
            'weather': ['temporal', 'environmental'],
            'construction': ['geospatial', 'financial', 'safety'],
            'infrastructure': ['geospatial', 'safety'],
            'environmental': ['geospatial', 'temporal'],
            'transportation': ['geospatial', 'temporal'],
            'safety': ['geospatial', 'temporal'],
            'financial': ['temporal']
        }
    
    def auto_tag_dataset(self, dataset: pd.DataFrame, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Automatically tag a dataset based on its content.
        
        Args:
            dataset: The dataset to tag
            filename: Optional filename for additional context
            
        Returns:
            Dictionary with tagging results
        """
        self.logger.info(f"Auto-tagging dataset with {len(dataset)} records")
        
        # Step 1: Analyze content
        content_analysis = self.content_analyzer.analyze_content(dataset, filename)
        
        # Step 2: Generate base tags
        base_tags = self._generate_base_tags(content_analysis, dataset, filename)
        
        # Step 3: Add related tags
        related_tags = self._add_related_tags(base_tags)
        
        # Step 4: Validate and clean tags
        final_tags = self._validate_tags(related_tags)
        
        # Step 5: Generate tagging summary
        tagging_summary = self._generate_tagging_summary(content_analysis, final_tags)
        
        return {
            'tags': final_tags,
            'content_analysis': content_analysis,
            'tagging_summary': tagging_summary,
            'confidence': self._calculate_tagging_confidence(content_analysis, final_tags),
            'suggestions': self._generate_tagging_suggestions(content_analysis, final_tags)
        }
    
    def _generate_base_tags(self, content_analysis: Dict[str, Any], dataset: pd.DataFrame, filename: Optional[str] = None) -> Set[str]:
        """Generate base tags from content analysis"""
        tags = set()
        
        # Content type tag
        content_type = content_analysis.get('content_type', 'unknown')
        if content_type != 'unknown':
            tags.add(content_type)
        
        # Data characteristics tags
        characteristics = content_analysis.get('data_characteristics', {})
        
        # Size tag
        record_count = characteristics.get('total_records', 0)
        if record_count > 100000:
            tags.add('massive_dataset')
        elif record_count > 10000:
            tags.add('large_dataset')
        elif record_count > 1000:
            tags.add('medium_dataset')
        else:
            tags.add('small_dataset')
        
        # Quality tag
        completeness = characteristics.get('data_completeness', 0)
        if completeness > 90:
            tags.add('high_quality')
        elif completeness > 70:
            tags.add('medium_quality')
        else:
            tags.add('low_quality')
        
        # Geographic tag
        if characteristics.get('has_coordinates', False):
            tags.add('geospatial')
        
        # Temporal tag
        if characteristics.get('has_temporal_data', False):
            tags.add('temporal')
        
        # Filename-based tags
        if filename:
            filename_lower = filename.lower()
            
            # Year tags
            for year in ['2024', '2023', '2022', '2021', '2020']:
                if year in filename_lower:
                    tags.add(year)
            
            # Location tags
            if 'nyc' in filename_lower or 'new_york' in filename_lower:
                tags.add('nyc')
            elif 'la' in filename_lower or 'los_angeles' in filename_lower:
                tags.add('los_angeles')
            elif 'chicago' in filename_lower:
                tags.add('chicago')
            elif 'miami' in filename_lower:
                tags.add('miami')
            
            # Source tags
            if 'gov' in filename_lower or 'government' in filename_lower:
                tags.add('government')
            elif 'public' in filename_lower:
                tags.add('public')
            elif 'private' in filename_lower:
                tags.add('private')
        
        return tags
    
    def _add_related_tags(self, base_tags: Set[str]) -> Set[str]:
        """Add related tags based on tag relationships"""
        related_tags = base_tags.copy()
        
        for tag in base_tags:
            if tag in self.tag_relationships:
                related_tags.update(self.tag_relationships[tag])
        
        return related_tags
    
    def _validate_tags(self, tags: Set[str]) -> List[str]:
        """Validate and clean tags"""
        valid_tags = []
        
        # Get all valid tags from categories
        all_valid_tags = set()
        for category_tags in self.tag_categories.values():
            all_valid_tags.update(category_tags)
        
        # Filter to only valid tags
        for tag in tags:
            if tag in all_valid_tags:
                valid_tags.append(tag)
        
        # Remove duplicates and sort
        return sorted(list(set(valid_tags)))
    
    def _generate_tagging_summary(self, content_analysis: Dict[str, Any], tags: List[str]) -> str:
        """Generate a human-readable tagging summary"""
        content_type = content_analysis.get('content_type', 'unknown')
        record_count = content_analysis.get('data_characteristics', {}).get('total_records', 0)
        
        summary = f"Dataset tagged as {content_type} with {len(tags)} tags: {', '.join(tags)}"
        summary += f"\nContains {record_count:,} records with {content_analysis.get('data_characteristics', {}).get('total_columns', 0)} columns"
        
        # Add quality info
        completeness = content_analysis.get('data_characteristics', {}).get('data_completeness', 0)
        summary += f"\nData completeness: {completeness:.1f}%"
        
        return summary
    
    def _calculate_tagging_confidence(self, content_analysis: Dict[str, Any], tags: List[str]) -> float:
        """Calculate confidence in the tagging"""
        confidence_scores = content_analysis.get('confidence_scores', {})
        
        if not confidence_scores:
            return 0.5  # Default confidence
        
        # Get confidence for the detected content type
        content_type = content_analysis.get('content_type', 'unknown')
        if content_type in confidence_scores:
            return confidence_scores[content_type]
        
        # Return average confidence
        return sum(confidence_scores.values()) / len(confidence_scores)
    
    def _generate_tagging_suggestions(self, content_analysis: Dict[str, Any], tags: List[str]) -> List[str]:
        """Generate suggestions for improving tags"""
        suggestions = []
        
        # Check for missing important tags
        content_type = content_analysis.get('content_type', 'unknown')
        if content_type == 'unknown':
            suggestions.append("Content type unclear - consider manual review")
        
        # Check for low confidence
        confidence = self._calculate_tagging_confidence(content_analysis, tags)
        if confidence < 0.5:
            suggestions.append("Low confidence in tagging - consider manual review")
        
        # Check for missing geographic tags
        if 'geospatial' not in tags and content_analysis.get('data_characteristics', {}).get('has_coordinates', False):
            suggestions.append("Dataset has coordinates but no geographic tag")
        
        # Check for missing temporal tags
        if 'temporal' not in tags and content_analysis.get('data_characteristics', {}).get('has_temporal_data', False):
            suggestions.append("Dataset has temporal data but no temporal tag")
        
        return suggestions
    
    def suggest_tags_for_query(self, query: str) -> List[str]:
        """Suggest tags based on a search query"""
        query_lower = query.lower()
        suggested_tags = []
        
        # Check query against tag categories
        for category, tags in self.tag_categories.items():
            for tag in tags:
                if tag in query_lower:
                    suggested_tags.append(tag)
        
        # Check for related tags
        for tag in suggested_tags:
            if tag in self.tag_relationships:
                suggested_tags.extend(self.tag_relationships[tag])
        
        return sorted(list(set(suggested_tags)))
    
    def get_tag_statistics(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get statistics about tag usage across datasets"""
        tag_counts = {}
        category_counts = {}
        
        for dataset_name, dataset in datasets.items():
            # Auto-tag each dataset
            tagging_result = self.auto_tag_dataset(dataset, dataset_name)
            tags = tagging_result.get('tags', [])
            
            # Count individual tags
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Count by category
            for category, category_tags in self.tag_categories.items():
                category_matches = [tag for tag in tags if tag in category_tags]
                if category_matches:
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'tag_counts': tag_counts,
            'category_counts': category_counts,
            'total_datasets': len(datasets),
            'most_common_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        } 