# Author: KleaSCM
# Date: 2024
# Description: Content Detector - Real-time content detection and classification

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from ..analyzers.base_analyzer import BaseAnalyzer
from .content_analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)

class ContentDetector:
    """
    Content Detector - Real-time content detection and classification.
    
    This system provides:
    - Real-time content type detection
    - Streaming data analysis
    - Content change detection
    - Anomaly detection in data streams
    """
    
    def __init__(self):
        """Initialize the content detector"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized ContentDetector - Real-time content detection")
        
        # Initialize content analyzer
        self.content_analyzer = ContentAnalyzer()
        
        # Detection thresholds
        self.detection_thresholds = {
            'confidence_minimum': 0.3,
            'change_sensitivity': 0.1,
            'anomaly_threshold': 0.05
        }
        
        # Detection history for change tracking
        self.detection_history = {}
        
        # Real-time patterns
        self.realtime_patterns = {
            'traffic_indicators': ['speed', 'congestion', 'flow', 'volume', 'delay'],
            'weather_indicators': ['temperature', 'humidity', 'precipitation', 'wind', 'pressure'],
            'construction_indicators': ['permit', 'project', 'construction', 'building', 'contract'],
            'infrastructure_indicators': ['pipe', 'electrical', 'water', 'gas', 'utility'],
            'environmental_indicators': ['soil', 'vegetation', 'pollution', 'air_quality', 'noise'],
            'safety_indicators': ['safety', 'inspection', 'violation', 'hazard', 'risk'],
            'financial_indicators': ['cost', 'budget', 'expense', 'revenue', 'funding']
        }
    
    def detect_content_realtime(self, data_chunk: pd.DataFrame, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect content type in real-time for a data chunk.
        
        Args:
            data_chunk: Small chunk of data to analyze
            dataset_id: Optional identifier for the dataset
            
        Returns:
            Dictionary with real-time detection results
        """
        self.logger.info(f"Real-time content detection for chunk with {len(data_chunk)} records")
        
        # Quick content detection
        quick_analysis = self._quick_content_detection(data_chunk)
        
        # Check for content changes if we have history
        change_detection = self._detect_content_changes(data_chunk, dataset_id, quick_analysis)
        
        # Anomaly detection
        anomaly_detection = self._detect_anomalies(data_chunk, quick_analysis)
        
        # Update detection history
        if dataset_id:
            self._update_detection_history(dataset_id, quick_analysis)
        
        return {
            'content_type': quick_analysis['content_type'],
            'confidence': quick_analysis['confidence'],
            'indicators': quick_analysis['indicators'],
            'change_detected': change_detection['change_detected'],
            'change_details': change_detection['change_details'],
            'anomalies': anomaly_detection['anomalies'],
            'anomaly_score': anomaly_detection['anomaly_score'],
            'timestamp': pd.Timestamp.now(),
            'recommendations': self._generate_realtime_recommendations(quick_analysis, change_detection, anomaly_detection)
        }
    
    def _quick_content_detection(self, data_chunk: pd.DataFrame) -> Dict[str, Any]:
        """Perform quick content detection on a data chunk"""
        # Analyze column names
        column_scores = {}
        for content_type, indicators in self.realtime_patterns.items():
            score = 0
            for col in data_chunk.columns:
                col_lower = col.lower()
                for indicator in indicators:
                    if indicator in col_lower:
                        score += 1
            
            column_scores[content_type] = score
        
        # Analyze sample values
        value_scores = {}
        for content_type, indicators in self.realtime_patterns.items():
            score = 0
            for col in data_chunk.columns:
                if data_chunk[col].dtype == 'object':
                    sample_values = data_chunk[col].dropna().astype(str).str.lower()
                    for indicator in indicators:
                        if sample_values.str.contains(indicator).any():
                            score += 0.5  # Lower weight for value matches
            
            value_scores[content_type] = score
        
        # Combine scores
        total_scores = {}
        for content_type in self.realtime_patterns.keys():
            total_scores[content_type] = column_scores.get(content_type, 0) + value_scores.get(content_type, 0)
        
        # Find best match
        if total_scores:
            best_type = max(total_scores, key=lambda x: total_scores[x])
            best_score = total_scores[best_type]
            
            # Calculate confidence
            max_possible = len(self.realtime_patterns[best_type]) * 1.5  # Column + value matches
            confidence = min(best_score / max_possible, 1.0) if max_possible > 0 else 0.0
            
            return {
                'content_type': best_type if confidence > self.detection_thresholds['confidence_minimum'] else 'unknown',
                'confidence': confidence,
                'indicators': [ind for ind in self.realtime_patterns[best_type] if any(ind in col.lower() for col in data_chunk.columns)],
                'scores': total_scores
            }
        
        return {
            'content_type': 'unknown',
            'confidence': 0.0,
            'indicators': [],
            'scores': {}
        }
    
    def _detect_content_changes(self, data_chunk: pd.DataFrame, dataset_id: Optional[str], 
                              current_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in content type over time"""
        if not dataset_id or dataset_id not in self.detection_history:
            return {
                'change_detected': False,
                'change_details': 'No previous detection history'
            }
        
        history = self.detection_history[dataset_id]
        if not history:
            return {
                'change_detected': False,
                'change_details': 'No previous detection history'
            }
        
        # Get previous detection
        previous_analysis = history[-1]
        previous_type = previous_analysis.get('content_type', 'unknown')
        current_type = current_analysis.get('content_type', 'unknown')
        
        # Check for content type change
        if previous_type != current_type:
            return {
                'change_detected': True,
                'change_details': f"Content type changed from {previous_type} to {current_type}",
                'previous_type': previous_type,
                'current_type': current_type,
                'change_magnitude': 'major'
            }
        
        # Check for confidence change
        previous_confidence = previous_analysis.get('confidence', 0.0)
        current_confidence = current_analysis.get('confidence', 0.0)
        confidence_change = abs(current_confidence - previous_confidence)
        
        if confidence_change > self.detection_thresholds['change_sensitivity']:
            return {
                'change_detected': True,
                'change_details': f"Confidence changed by {confidence_change:.3f}",
                'previous_confidence': previous_confidence,
                'current_confidence': current_confidence,
                'change_magnitude': 'moderate'
            }
        
        return {
            'change_detected': False,
            'change_details': 'No significant changes detected'
        }
    
    def _detect_anomalies(self, data_chunk: pd.DataFrame, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in the data chunk"""
        anomalies = []
        anomaly_score = 0.0
        
        # Check for unusual data patterns
        for col in data_chunk.columns:
            if data_chunk[col].dtype in ['int64', 'float64']:
                # Check for outliers
                Q1 = data_chunk[col].quantile(0.25)
                Q3 = data_chunk[col].quantile(0.75)
                IQR = Q3 - Q1
                
                outliers = data_chunk[(data_chunk[col] < Q1 - 1.5 * IQR) | (data_chunk[col] > Q3 + 1.5 * IQR)]
                outlier_percentage = len(outliers) / len(data_chunk) * 100
                
                if outlier_percentage > 10:  # More than 10% outliers
                    anomalies.append(f"High outlier percentage in {col}: {outlier_percentage:.1f}%")
                    anomaly_score += 0.2
        
        # Check for missing data patterns
        missing_percentage = (data_chunk.isnull().sum().sum() / (len(data_chunk) * len(data_chunk.columns))) * 100
        if missing_percentage > 50:
            anomalies.append(f"High missing data percentage: {missing_percentage:.1f}%")
            anomaly_score += 0.3
        
        # Check for content type anomalies
        content_type = content_analysis.get('content_type', 'unknown')
        confidence = content_analysis.get('confidence', 0.0)
        
        if content_type == 'unknown' and len(data_chunk) > 0:
            anomalies.append("Unable to determine content type")
            anomaly_score += 0.4
        
        if confidence < self.detection_thresholds['confidence_minimum']:
            anomalies.append(f"Low confidence in content detection: {confidence:.3f}")
            anomaly_score += 0.2
        
        return {
            'anomalies': anomalies,
            'anomaly_score': min(anomaly_score, 1.0),
            'anomaly_level': 'high' if anomaly_score > 0.7 else 'medium' if anomaly_score > 0.3 else 'low'
        }
    
    def _update_detection_history(self, dataset_id: str, analysis: Dict[str, Any]):
        """Update detection history for a dataset"""
        if dataset_id not in self.detection_history:
            self.detection_history[dataset_id] = []
        
        # Add timestamp to analysis
        analysis_with_timestamp = analysis.copy()
        analysis_with_timestamp['timestamp'] = pd.Timestamp.now()
        
        # Keep only last 10 detections
        self.detection_history[dataset_id].append(analysis_with_timestamp)
        if len(self.detection_history[dataset_id]) > 10:
            self.detection_history[dataset_id] = self.detection_history[dataset_id][-10:]
    
    def _generate_realtime_recommendations(self, content_analysis: Dict[str, Any], 
                                         change_detection: Dict[str, Any],
                                         anomaly_detection: Dict[str, Any]) -> List[str]:
        """Generate real-time recommendations based on detection results"""
        recommendations = []
        
        # Content type recommendations
        content_type = content_analysis.get('content_type', 'unknown')
        confidence = content_analysis.get('confidence', 0.0)
        
        if content_type != 'unknown':
            recommendations.append(f"Data appears to be {content_type} related")
        
        if confidence < 0.5:
            recommendations.append("Consider manual review of content classification")
        
        # Change detection recommendations
        if change_detection.get('change_detected', False):
            recommendations.append("Content change detected - review data source")
            
            if change_detection.get('change_magnitude') == 'major':
                recommendations.append("Major content change - update analysis parameters")
        
        # Anomaly recommendations
        anomaly_level = anomaly_detection.get('anomaly_level', 'low')
        if anomaly_level == 'high':
            recommendations.append("High anomaly level - investigate data quality")
        elif anomaly_level == 'medium':
            recommendations.append("Moderate anomalies detected - monitor data stream")
        
        if anomaly_detection.get('anomalies'):
            recommendations.append("Review detected anomalies for data quality issues")
        
        return recommendations
    
    def get_detection_summary(self, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of detection history"""
        if dataset_id and dataset_id in self.detection_history:
            history = self.detection_history[dataset_id]
            
            if not history:
                return {'error': 'No detection history available'}
            
            # Analyze history
            content_types = [detection.get('content_type', 'unknown') for detection in history]
            confidences = [detection.get('confidence', 0.0) for detection in history]
            
            return {
                'total_detections': len(history),
                'most_common_type': max(set(content_types), key=content_types.count) if content_types else 'unknown',
                'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
                'confidence_trend': 'increasing' if len(confidences) > 1 and confidences[-1] > confidences[0] else 'decreasing',
                'last_detection': history[-1] if history else None,
                'detection_history': history
            }
        
        # Overall summary
        total_datasets = len(self.detection_history)
        total_detections = sum(len(history) for history in self.detection_history.values())
        
        return {
            'total_datasets': total_datasets,
            'total_detections': total_detections,
            'average_detections_per_dataset': total_detections / total_datasets if total_datasets > 0 else 0,
            'active_datasets': list(self.detection_history.keys())
        }
    
    def reset_detection_history(self, dataset_id: Optional[str] = None):
        """Reset detection history for a dataset or all datasets"""
        if dataset_id:
            if dataset_id in self.detection_history:
                del self.detection_history[dataset_id]
                self.logger.info(f"Reset detection history for dataset: {dataset_id}")
        else:
            self.detection_history.clear()
            self.logger.info("Reset all detection history")
    
    def set_detection_thresholds(self, thresholds: Dict[str, float]):
        """Update detection thresholds"""
        for key, value in thresholds.items():
            if key in self.detection_thresholds:
                self.detection_thresholds[key] = value
                self.logger.info(f"Updated detection threshold {key}: {value}")
    
    def get_detection_thresholds(self) -> Dict[str, float]:
        """Get current detection thresholds"""
        return self.detection_thresholds.copy() 