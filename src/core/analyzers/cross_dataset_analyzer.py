# Author: KleaSCM
# Date: 2024
# Description: Cross-dataset analyzer for comprehensive multi-dataset analysis

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .base_analyzer import BaseAnalyzer

class CrossDatasetAnalyzer(BaseAnalyzer):
    """Analyzes relationships and patterns across multiple datasets"""
    
    def analyze(self, datasets: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """Analyze relationships across multiple datasets"""
        self.logger.info(f"Analyzing cross-dataset relationships with {len(datasets)} datasets")
        
        # Extract location context if provided
        location = kwargs.get('location')
        
        return {
            'cross_dataset_analysis': self._analyze_cross_dataset(datasets),
            'spatial_relationships': self._analyze_spatial_relationships(datasets, location),
            'temporal_relationships': self._analyze_temporal_relationships(datasets),
            'data_quality_comparison': self._compare_data_quality(datasets),
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
            'relationships': {},
            'complementary_data': {}
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
            
            # Analyze data consistency for shared columns
            for col in common_columns:
                consistency_analysis = self._analyze_column_consistency(datasets, col)
                if consistency_analysis:
                    cross_analysis['data_consistency'][col] = consistency_analysis
        
        # Find complementary data (datasets that could be joined)
        cross_analysis['complementary_data'] = self._find_complementary_datasets(datasets)
        
        return cross_analysis
    
    def _analyze_spatial_relationships(self, datasets: Dict[str, pd.DataFrame], 
                                     location: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Analyze spatial relationships between datasets"""
        spatial_analysis = {
            'coordinate_overlap': {},
            'spatial_coverage': {},
            'proximity_analysis': {},
            'geographic_patterns': {}
        }
        
        # Find datasets with coordinates
        datasets_with_coords = {}
        for name, dataset in datasets.items():
            coord_cols = self._find_coordinate_columns(dataset)
            if coord_cols['lat'] and coord_cols['lon']:
                datasets_with_coords[name] = {
                    'dataset': dataset,
                    'lat_col': coord_cols['lat'],
                    'lon_col': coord_cols['lon']
                }
        
        if len(datasets_with_coords) > 1:
            # Analyze spatial overlap
            spatial_analysis['coordinate_overlap'] = self._analyze_coordinate_overlap(datasets_with_coords)
            
            # Analyze spatial coverage
            spatial_analysis['spatial_coverage'] = self._analyze_spatial_coverage(datasets_with_coords)
            
            # Analyze proximity if location provided
            if location:
                spatial_analysis['proximity_analysis'] = self._analyze_proximity_to_location(
                    datasets_with_coords, location
                )
        
        return spatial_analysis
    
    def _analyze_temporal_relationships(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze temporal relationships between datasets"""
        temporal_analysis = {
            'time_overlap': {},
            'temporal_patterns': {},
            'seasonal_analysis': {},
            'trend_analysis': {}
        }
        
        # Find datasets with temporal data
        datasets_with_time = {}
        for name, dataset in datasets.items():
            time_cols = self._find_temporal_columns(dataset)
            if time_cols:
                datasets_with_time[name] = {
                    'dataset': dataset,
                    'time_columns': time_cols
                }
        
        if len(datasets_with_time) > 1:
            # Analyze temporal overlap
            temporal_analysis['time_overlap'] = self._analyze_temporal_overlap(datasets_with_time)
            
            # Analyze temporal patterns
            temporal_analysis['temporal_patterns'] = self._analyze_temporal_patterns(datasets_with_time)
        
        return temporal_analysis
    
    def _compare_data_quality(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compare data quality across datasets"""
        quality_comparison = {
            'completeness_scores': {},
            'consistency_scores': {},
            'quality_rankings': {},
            'quality_issues': {}
        }
        
        for name, dataset in datasets.items():
            # Calculate completeness
            total_cells = len(dataset) * len(dataset.columns)
            filled_cells = total_cells - dataset.isnull().sum().sum()
            completeness = (filled_cells / total_cells) * 100 if total_cells > 0 else 0
            
            quality_comparison['completeness_scores'][name] = completeness
            
            # Calculate consistency (check for duplicate rows)
            duplicates = dataset.duplicated().sum()
            consistency = ((len(dataset) - duplicates) / len(dataset)) * 100 if len(dataset) > 0 else 0
            quality_comparison['consistency_scores'][name] = consistency
            
            # Identify quality issues
            issues = []
            if completeness < 50:
                issues.append("Low data completeness")
            if consistency < 90:
                issues.append("High duplicate rate")
            if len(dataset) == 0:
                issues.append("Empty dataset")
            
            quality_comparison['quality_issues'][name] = issues
        
        # Create quality rankings
        avg_completeness = np.mean(list(quality_comparison['completeness_scores'].values()))
        avg_consistency = np.mean(list(quality_comparison['consistency_scores'].values()))
        
        quality_comparison['quality_rankings'] = {
            'best_completeness': max(quality_comparison['completeness_scores'].items(), key=lambda x: x[1])[0],
            'best_consistency': max(quality_comparison['consistency_scores'].items(), key=lambda x: x[1])[0],
            'average_completeness': avg_completeness,
            'average_consistency': avg_consistency
        }
        
        return quality_comparison
    
    def _find_correlations(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Find correlations between variables across datasets"""
        correlations = {
            'numeric_correlations': {},
            'categorical_associations': {},
            'cross_dataset_correlations': {},
            'significant_relationships': []
        }
        
        # Analyze individual datasets
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
        
        # Find cross-dataset correlations for shared numeric columns
        shared_numeric_cols = self._find_shared_numeric_columns(datasets)
        if shared_numeric_cols:
            correlations['cross_dataset_correlations'] = self._analyze_cross_dataset_correlations(
                datasets, shared_numeric_cols
            )
        
        return correlations
    
    def _detect_anomalies(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect anomalies across datasets"""
        anomalies = {
            'outliers': {},
            'missing_patterns': {},
            'data_inconsistencies': {},
            'unusual_patterns': {},
            'cross_dataset_anomalies': {}
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
        
        # Detect cross-dataset anomalies
        anomalies['cross_dataset_anomalies'] = self._detect_cross_dataset_anomalies(datasets)
        
        return anomalies
    
    def _analyze_column_consistency(self, datasets: Dict[str, pd.DataFrame], column: str) -> Optional[Dict[str, Any]]:
        """Analyze consistency of a specific column across datasets"""
        try:
            values_by_dataset = {}
            for name, dataset in datasets.items():
                if column in dataset.columns:
                    values_by_dataset[name] = dataset[column].dropna().tolist()
            
            if len(values_by_dataset) < 2:
                return None
            
            # Check for value overlap
            all_values = set()
            for values in values_by_dataset.values():
                all_values.update(values)
            
            unique_values_by_dataset = {name: set(values) for name, values in values_by_dataset.items()}
            
            # Calculate overlap metrics
            overlaps = {}
            for name1 in unique_values_by_dataset:
                for name2 in unique_values_by_dataset:
                    if name1 != name2:
                        overlap = len(unique_values_by_dataset[name1] & unique_values_by_dataset[name2])
                        total_unique = len(unique_values_by_dataset[name1] | unique_values_by_dataset[name2])
                        overlap_ratio = overlap / total_unique if total_unique > 0 else 0
                        overlaps[f"{name1}_vs_{name2}"] = {
                            'overlap_count': overlap,
                            'overlap_ratio': overlap_ratio
                        }
            
            return {
                'total_unique_values': len(all_values),
                'values_by_dataset': {name: len(values) for name, values in values_by_dataset.items()},
                'overlaps': overlaps
            }
        except Exception as e:
            self.logger.warning(f"Column consistency analysis failed for {column}: {e}")
            return None
    
    def _find_complementary_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Find datasets that could be complementary (joined together)"""
        complementary = {
            'potential_joins': [],
            'shared_keys': {},
            'join_recommendations': []
        }
        
        # Look for potential join keys (ID columns, names, etc.)
        for name1, dataset1 in datasets.items():
            for name2, dataset2 in datasets.items():
                if name1 != name2:
                    # Find potential join columns
                    potential_keys = self._find_potential_join_keys(dataset1, dataset2)
                    if potential_keys:
                        complementary['potential_joins'].append({
                            'dataset1': name1,
                            'dataset2': name2,
                            'join_keys': potential_keys
                        })
        
        return complementary
    
    def _find_potential_join_keys(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> List[str]:
        """Find potential columns that could be used to join two datasets"""
        potential_keys = []
        
        # Look for common column names
        common_cols = set(dataset1.columns) & set(dataset2.columns)
        
        for col in common_cols:
            # Check if it looks like an ID or key column
            if any(keyword in col.lower() for keyword in ['id', 'key', 'name', 'code', 'identifier']):
                # Check if values overlap
                values1 = set(dataset1[col].dropna())
                values2 = set(dataset2[col].dropna())
                overlap = len(values1 & values2)
                
                if overlap > 0:
                    potential_keys.append(col)
        
        return potential_keys
    
    def _analyze_coordinate_overlap(self, datasets_with_coords: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze overlap of coordinate data between datasets"""
        overlap_analysis = {}
        
        dataset_names = list(datasets_with_coords.keys())
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                overlap = self._calculate_coordinate_overlap(
                    datasets_with_coords[name1],
                    datasets_with_coords[name2]
                )
                if overlap:
                    overlap_analysis[f"{name1}_vs_{name2}"] = overlap
        
        return overlap_analysis
    
    def _calculate_coordinate_overlap(self, dataset1_info: Dict, dataset2_info: Dict) -> Optional[Dict[str, Any]]:
        """Calculate coordinate overlap between two datasets"""
        try:
            dataset1 = dataset1_info['dataset']
            dataset2 = dataset2_info['dataset']
            lat1, lon1 = dataset1_info['lat_col'], dataset1_info['lon_col']
            lat2, lon2 = dataset2_info['lat_col'], dataset2_info['lon_col']
            
            # Convert to numeric
            lat1_vals = pd.to_numeric(dataset1[lat1], errors='coerce')
            lon1_vals = pd.to_numeric(dataset1[lon1], errors='coerce')
            lat2_vals = pd.to_numeric(dataset2[lat2], errors='coerce')
            lon2_vals = pd.to_numeric(dataset2[lon2], errors='coerce')
            
            # Get valid coordinates using proper pandas operations
            valid1 = pd.notna(lat1_vals) & pd.notna(lon1_vals)
            valid2 = pd.notna(lat2_vals) & pd.notna(lon2_vals)
            
            # Filter and convert to coordinates
            if valid1.any():
                lat1_filtered = lat1_vals[valid1].astype(float)
                lon1_filtered = lon1_vals[valid1].astype(float)
            else:
                lat1_filtered = pd.Series(dtype=float)
                lon1_filtered = pd.Series(dtype=float)
                
            if valid2.any():
                lat2_filtered = lat2_vals[valid2].astype(float)
                lon2_filtered = lon2_vals[valid2].astype(float)
            else:
                lat2_filtered = pd.Series(dtype=float)
                lon2_filtered = pd.Series(dtype=float)
            
            coords1 = set(zip(lat1_filtered.round(4), lon1_filtered.round(4)))
            coords2 = set(zip(lat2_filtered.round(4), lon2_filtered.round(4)))
            
            overlap = len(coords1 & coords2)
            total_unique = len(coords1 | coords2)
            
            return {
                'overlap_count': overlap,
                'overlap_ratio': overlap / total_unique if total_unique > 0 else 0,
                'dataset1_coords': len(coords1),
                'dataset2_coords': len(coords2)
            }
        except Exception as e:
            self.logger.warning(f"Coordinate overlap calculation failed: {e}")
            return None
    
    def _analyze_spatial_coverage(self, datasets_with_coords: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze spatial coverage of datasets"""
        coverage_analysis = {}
        
        for name, info in datasets_with_coords.items():
            dataset = info['dataset']
            lat_col, lon_col = info['lat_col'], info['lon_col']
            
            try:
                lat_vals = pd.to_numeric(dataset[lat_col], errors='coerce')
                lon_vals = pd.to_numeric(dataset[lon_col], errors='coerce')
                
                valid_coords = pd.notna(lat_vals) & pd.notna(lon_vals)
                if valid_coords.any():
                    lat_filtered = lat_vals[valid_coords].astype(float)
                    lon_filtered = lon_vals[valid_coords].astype(float)
                    coverage_analysis[name] = {
                        'lat_min': float(lat_filtered.min()) if len(lat_filtered) > 0 else 0.0,
                        'lat_max': float(lat_filtered.max()) if len(lat_filtered) > 0 else 0.0,
                        'lon_min': float(lon_filtered.min()) if len(lon_filtered) > 0 else 0.0,
                        'lon_max': float(lon_filtered.max()) if len(lon_filtered) > 0 else 0.0,
                        'coordinate_count': int(valid_coords.sum()),
                        'coverage_area_km2': self._estimate_coverage_area(
                            lat_filtered, lon_filtered
                        )
                    }
            except Exception as e:
                self.logger.warning(f"Spatial coverage analysis failed for {name}: {e}")
        
        return coverage_analysis
    
    def _estimate_coverage_area(self, lat_vals: pd.Series, lon_vals: pd.Series) -> float:
        """Estimate coverage area in square kilometers"""
        try:
            # Simple approximation: bounding box area
            lat_range = lat_vals.max() - lat_vals.min()
            lon_range = lon_vals.max() - lon_vals.min()
            
            # Convert to km (rough approximation: 1 degree ≈ 111 km)
            area_km2 = lat_range * 111 * lon_range * 111
            return float(area_km2)
        except:
            return 0.0
    
    def _analyze_proximity_to_location(self, datasets_with_coords: Dict[str, Dict], 
                                     location: Dict[str, float]) -> Dict[str, Any]:
        """Analyze proximity of datasets to a specific location"""
        proximity_analysis = {}
        
        for name, info in datasets_with_coords.items():
            dataset = info['dataset']
            lat_col, lon_col = info['lat_col'], info['lon_col']
            
            try:
                lat_vals = pd.to_numeric(dataset[lat_col], errors='coerce')
                lon_vals = pd.to_numeric(dataset[lon_col], errors='coerce')
                
                valid_coords = pd.notna(lat_vals) & pd.notna(lon_vals)
                if valid_coords.any():
                    # Calculate distances to location
                    lat_filtered = lat_vals[valid_coords].astype(float)
                    lon_filtered = lon_vals[valid_coords].astype(float)
                    distances = np.sqrt(
                        (lat_filtered - location['lat'])**2 + 
                        (lon_filtered - location['lon'])**2
                    )
                    
                    proximity_analysis[name] = {
                        'min_distance_km': float(distances.min() * 111),  # Convert to km
                        'max_distance_km': float(distances.max() * 111),
                        'avg_distance_km': float(distances.mean() * 111),
                        'records_within_1km': int((distances * 111 <= 1).sum()),
                        'records_within_5km': int((distances * 111 <= 5).sum())
                    }
            except Exception as e:
                self.logger.warning(f"Proximity analysis failed for {name}: {e}")
        
        return proximity_analysis
    
    def _find_temporal_columns(self, dataset: pd.DataFrame) -> List[str]:
        """Find temporal columns in a dataset"""
        temporal_cols = []
        
        for col in dataset.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'year', 'month', 'day']):
                temporal_cols.append(col)
        
        return temporal_cols
    
    def _analyze_temporal_overlap(self, datasets_with_time: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze temporal overlap between datasets"""
        # This would require parsing dates and finding overlaps
        # For now, return basic structure
        return {
            'temporal_datasets': list(datasets_with_time.keys()),
            'analysis_note': 'Temporal overlap analysis requires date parsing implementation'
        }
    
    def _analyze_temporal_patterns(self, datasets_with_time: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns across datasets"""
        # This would analyze seasonal patterns, trends, etc.
        # For now, return basic structure
        return {
            'temporal_datasets': list(datasets_with_time.keys()),
            'analysis_note': 'Temporal pattern analysis requires date parsing implementation'
        }
    
    def _find_shared_numeric_columns(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Find numeric columns that are shared across datasets"""
        shared_cols = set()
        first_dataset = True
        
        for dataset in datasets.values():
            numeric_cols = set(dataset.select_dtypes(include=[np.number]).columns)
            if first_dataset:
                shared_cols = numeric_cols
                first_dataset = False
            else:
                shared_cols &= numeric_cols
        
        return list(shared_cols)
    
    def _analyze_cross_dataset_correlations(self, datasets: Dict[str, pd.DataFrame], 
                                          shared_cols: List[str]) -> Dict[str, Any]:
        """Analyze correlations for shared numeric columns across datasets"""
        cross_correlations = {}
        
        for col in shared_cols:
            col_data = {}
            for name, dataset in datasets.items():
                if col in dataset.columns:
                    col_data[name] = dataset[col].dropna()
            
            if len(col_data) > 1:
                # Calculate correlation between datasets for this column
                cross_correlations[col] = self._calculate_cross_dataset_correlation(col_data)
        
        return cross_correlations
    
    def _calculate_cross_dataset_correlation(self, col_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate correlation between datasets for a specific column"""
        try:
            # For now, return basic statistics
            stats = {}
            for name, data in col_data.items():
                stats[name] = {
                    'count': len(data),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max())
                }
            
            return {
                'statistics': stats,
                'note': 'Cross-dataset correlation requires aligned data points'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_cross_dataset_anomalies(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect anomalies that span across datasets"""
        cross_anomalies = {
            'inconsistent_values': {},
            'missing_patterns': {},
            'data_gaps': {}
        }
        
        # Check for inconsistent values in shared columns
        all_columns = {}
        for name, dataset in datasets.items():
            all_columns[name] = set(dataset.columns)
        
        if len(all_columns) > 1:
            common_columns = set.intersection(*all_columns.values())
            for col in common_columns:
                inconsistency = self._check_value_inconsistency(datasets, col)
                if inconsistency:
                    cross_anomalies['inconsistent_values'][col] = inconsistency
        
        return cross_anomalies
    
    def _check_value_inconsistency(self, datasets: Dict[str, pd.DataFrame], column: str) -> Optional[Dict[str, Any]]:
        """Check for inconsistent values in a column across datasets"""
        try:
            values_by_dataset = {}
            for name, dataset in datasets.items():
                if column in dataset.columns:
                    values_by_dataset[name] = dataset[column].dropna()
            
            if len(values_by_dataset) < 2:
                return None
            
            # Check for value range inconsistencies
            ranges = {}
            for name, values in values_by_dataset.items():
                if len(values) > 0:
                    ranges[name] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean())
                    }
            
            # Check for significant differences
            inconsistencies = []
            dataset_names = list(ranges.keys())
            for i, name1 in enumerate(dataset_names):
                for name2 in dataset_names[i+1:]:
                    range1, range2 = ranges[name1], ranges[name2]
                    
                    # Check if ranges are significantly different
                    mean_diff = abs(range1['mean'] - range2['mean'])
                    if mean_diff > (range1['std'] + range2['std']) / 2:
                        inconsistencies.append({
                            'datasets': [name1, name2],
                            'mean_difference': mean_diff,
                            'range1': range1,
                            'range2': range2
                        })
            
            return {
                'inconsistencies': inconsistencies,
                'ranges_by_dataset': ranges
            } if inconsistencies else None
            
        except Exception as e:
            self.logger.warning(f"Value inconsistency check failed for {column}: {e}")
            return None
    
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
        
        # Data quality summary
        total_records = sum(len(dataset) for dataset in datasets.values())
        total_columns = sum(len(dataset.columns) for dataset in datasets.values())
        summary.append(f"Total data: {total_records:,} records, {total_columns} columns")
        
        return summary

    def _analyze_dimensions(self, dataset: pd.DataFrame, dimension_cols: List[str]) -> Dict[str, Any]:
        """Analyze dimensional data"""
        dimensions = {}
        for col in dimension_cols:
            if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                try:
                    # Convert to numeric and get actual statistics
                    numeric_data = pd.to_numeric(dataset[col], errors='coerce')
                    import numpy as np
                    arr = np.asarray(numeric_data)
                    is_valid = not np.isnan(arr).all()
                    if is_valid:
                        dimensions[col] = {
                            'min': float(np.nanmin(arr)),
                            'max': float(np.nanmax(arr)),
                            'mean': float(np.nanmean(arr)),
                            'std': float(np.nanstd(arr)),
                            'median': float(np.nanmedian(arr))
                        }
                except Exception as e:
                    self.logger.warning(f"Dimension analysis failed for column {col}: {e}")
        return {'dimension_statistics': dimensions} 