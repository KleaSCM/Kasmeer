# Author: KleaSCM
# Date: 2024
# Description: Cross-dataset analyzer

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BaseAnalyzer

class CrossDatasetAnalyzer(BaseAnalyzer):
    """Analyzes relationships and patterns across multiple datasets"""
    
    def analyze(self, datasets: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """Analyze relationships across multiple datasets"""
        self.logger.info(f"Analyzing cross-dataset relationships with {len(datasets)} datasets")
        
        return {
            'cross_dataset_analysis': self._analyze_cross_dataset(datasets),
            'correlations': self._find_correlations(datasets),
            'anomalies': self._detect_anomalies(datasets),
            'summary': self._generate_summary(datasets)
        }
    
    def _analyze_cross_dataset(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze relationships across datasets"""
        cross_analysis = {
            'dataset_overlap': {},
            'shared_columns': {},
            'data_consistency': {},
            'relationships': {}
        }
        
        # Find shared columns across datasets
        all_columns = {}
        for name, dataset in datasets.items():
            all_columns[name] = set(dataset.columns)
        
        # Find common columns
        if len(all_columns) > 1:
            dataset_names = list(all_columns.keys())
            common_columns = set.intersection(*all_columns.values())
            cross_analysis['shared_columns'] = {'columns': list(common_columns)}
        
        return cross_analysis
    
    def _find_correlations(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Find correlations between variables across datasets"""
        correlations = {
            'numeric_correlations': {},
            'categorical_associations': {},
            'significant_relationships': []
        }
        
        # For now, just analyze individual datasets
        for name, dataset in datasets.items():
            # Numeric correlations
            numeric_cols = list(dataset.select_dtypes(include=[np.number]).columns)
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = dataset[numeric_cols].corr()  # type: ignore
                    correlations['numeric_correlations'][name] = {
                        'matrix': corr_matrix.to_dict(),
                        'strong_correlations': self._find_strong_correlations(corr_matrix)
                    }
                except Exception as e:
                    self.logger.warning(f"Correlation analysis failed for {name}: {e}")
                    correlations['numeric_correlations'][name] = {'error': str(e)}
        
        return correlations
    
    def _detect_anomalies(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect anomalies across datasets"""
        anomalies = {
            'outliers': {},
            'missing_patterns': {},
            'data_inconsistencies': {},
            'unusual_patterns': {}
        }
        
        # Analyze each dataset for anomalies
        for name, dataset in datasets.items():
            # Outlier detection for numeric columns
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    Q1 = dataset[col].quantile(0.25)
                    Q3 = dataset[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = dataset[(dataset[col] < Q1 - 1.5 * IQR) | (dataset[col] > Q3 + 1.5 * IQR)]
                    if len(outliers) > 0:
                        if name not in anomalies['outliers']:
                            anomalies['outliers'][name] = {}
                        anomalies['outliers'][name][col] = {
                            'count': len(outliers),
                            'percentage': (len(outliers) / len(dataset)) * 100,
                            'values': outliers[col].tolist()
                        }
                except Exception as e:
                    self.logger.warning(f"Outlier detection failed for {name}.{col}: {e}")
        
        return anomalies
    
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
            self.logger.warning(f"Strong correlation analysis failed: {e}")
        return strong_correlations
    
    def _generate_summary(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate cross-dataset summary"""
        summary = []
        summary.append(f"Cross-dataset analysis: {len(datasets)} datasets")
        
        # Dataset sizes
        for name, dataset in datasets.items():
            summary.append(f"{name}: {len(dataset)} records, {len(dataset.columns)} columns")
        
        # Shared columns
        all_columns = {}
        for name, dataset in datasets.items():
            all_columns[name] = set(dataset.columns)
        
        if len(all_columns) > 1:
            common_columns = set.intersection(*all_columns.values())
            if common_columns:
                summary.append(f"Shared columns: {len(common_columns)} common columns")
        
        return summary 