# Author: KleaSCM
# Date: 2024
# Description: Flexible dataset configuration and discovery system

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from ..utils.logging_utils import setup_logging, log_performance
import glob
import re

logger = setup_logging(__name__)

class DatasetConfig:
    # Flexible dataset configuration system for different companies
    # This class handles loading, validating, and managing dataset configurations
    # so companies can use their own datasets without hardcoding anything
    
    @log_performance(logger)
    def __init__(self, config_path: str = "config.yaml"):
        # Initialize the dataset configuration system
        # Args:
        #   config_path: Path to the YAML configuration file
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.data_dir = Path(self.config.get('directories', {}).get('data_dir', 'DataSets'))
        self.dataset_configs = self.config.get('datasets', {})
        self.company_config = self.config.get('company', {})
        logger.info(f"Initialized DatasetConfig with config_path={config_path}")
        
    def _load_config(self) -> Dict[str, Any]:
        # Load configuration from YAML file
        # Returns: Dictionary containing the configuration
        # TODO: Add configuration validation and schema checking
        # TODO: Support environment-specific configs (dev, prod, test)
        logger.debug(f"Loading configuration from: {self.config_path}")
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using default configuration")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        # Get default configuration when no config file exists
        # This provides sensible defaults for new installations
        # TODO: Add more comprehensive default configurations
        # TODO: Include region-specific defaults
        logger.debug("Creating default configuration")
        return {
            'directories': {
                'data_dir': 'DataSets',
                'model_dir': 'models',
                'logs_dir': 'Logs',
                'visuals_dir': 'visuals'
            },
            'datasets': {
                'infrastructure': {
                    'enabled': True,
                    'file_patterns': ['*infrastructure*', '*pipes*', '*drainage*'],
                    'required_columns': [],  # Dataset agnostic - no hardcoded requirements
                    'optional_columns': []
                },
                'vegetation': {
                    'enabled': True,
                    'file_patterns': ['*vegetation*', '*zones*'],
                    'required_columns': [],  # Dataset agnostic - no hardcoded requirements
                    'optional_columns': []
                },
                'climate': {
                    'enabled': True,
                    'file_patterns': ['*climate*', '*weather*'],
                    'required_columns': [],
                    'optional_columns': []
                }
            },
            'company': {
                'name': 'Default Company',
                'region': 'Default Region',
                'data_mappings': {
                    'column_mappings': {},
                    'coordinate_mappings': {
                        'lat': ['latitude', 'lat', 'y', 'Y'],
                        'lon': ['longitude', 'lon', 'x', 'X']
                    }
                }
            }
        }
    
    @log_performance(logger)
    def discover_datasets(self) -> Dict[str, Dict]:
        # Discover available datasets based on configuration patterns
        # This method scans the data directory and matches files against configured patterns
        # Returns: Dictionary of discovered datasets with their metadata
        # TODO: Add caching for discovery results to improve performance
        # TODO: Add file modification time checking for incremental discovery
        logger.info("Discovering datasets in data directory")
        discovered_datasets = {}
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return discovered_datasets
        
        # Discover each dataset type based on configuration
        for dataset_type, config in self.dataset_configs.items():
            if not config.get('enabled', True):
                logger.debug(f"Dataset type {dataset_type} is disabled")
                continue
                
            discovered = self._discover_dataset_type(dataset_type, config)
            if discovered:
                discovered_datasets[dataset_type] = discovered
                logger.info(f"Discovered {dataset_type}: {len(discovered['files'])} files")
        
        # Discover custom datasets (user-defined dataset types)
        custom_datasets = self.dataset_configs.get('custom_datasets', {})
        for dataset_name, config in custom_datasets.items():
            if config.get('enabled', False):
                discovered = self._discover_dataset_type(dataset_name, config)
                if discovered:
                    discovered_datasets[dataset_name] = discovered
                    logger.info(f"Discovered custom dataset {dataset_name}: {len(discovered['files'])} files")
        
        logger.info(f"Total datasets discovered: {len(discovered_datasets)}")
        return discovered_datasets
    
    def _discover_dataset_type(self, dataset_type: str, config: Dict) -> Optional[Dict]:
        # Discover files for a specific dataset type
        # This method uses file patterns to find matching files in the data directory
        # Args:
        #   dataset_type: Name of the dataset type (e.g., 'infrastructure', 'vegetation')
        #   config: Configuration dictionary for this dataset type
        # Returns: Dictionary with discovered files and metadata, or None if no files found
        # TODO: Add support for more complex file pattern matching
        # TODO: Add file size and modification time filtering
        logger.debug(f"Discovering files for dataset type: {dataset_type}")
        
        files = []
        file_patterns = config.get('file_patterns', [])
        
        # Search for files matching each pattern
        for pattern in file_patterns:
            # Create search pattern by combining data directory with file pattern
            search_pattern = str(self.data_dir / pattern)
            matching_files = glob.glob(search_pattern, recursive=True)
            
            # Analyze each matching file
            for file_path in matching_files:
                file_info = self._analyze_file(file_path, config)
                if file_info:
                    files.append(file_info)
        
        if not files:
            logger.debug(f"No files found for dataset type: {dataset_type}")
            return None
        
        # Return discovery results with metadata
        return {
            'type': dataset_type,
            'config': config,
            'files': files,
            'total_files': len(files),
            'enabled': config.get('enabled', True),
            'priority': config.get('priority', 999)
        }
    
    def _analyze_file(self, file_path: str, config: Dict) -> Optional[Dict]:
        """Analyze a file to determine if it matches the dataset configuration"""
        logger.debug(f"Analyzing file: {file_path}")
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return None
            
            file_info = {
                'path': str(file_path_obj),
                'name': file_path_obj.name,
                'size': file_path_obj.stat().st_size,
                'extension': file_path_obj.suffix.lower(),
                'modified': file_path_obj.stat().st_mtime
            }
            
            # Analyze file content based on extension
            if file_info['extension'] in ['.csv', '.xlsx', '.xls']:
                content_info = self._analyze_tabular_file(file_path, config)
                if content_info:
                    file_info.update(content_info)
                    return file_info
            elif file_info['extension'] in ['.tif', '.tiff']:
                content_info = self._analyze_raster_file(file_path, config)
                if content_info:
                    file_info.update(content_info)
                    return file_info
            elif file_info['extension'] in ['.shp', '.geojson', '.json']:
                content_info = self._analyze_geospatial_file(file_path, config)
                if content_info:
                    file_info.update(content_info)
                    return file_info
            
            logger.debug(f"File {file_path} does not match dataset requirements")
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return None
    
    def _analyze_tabular_file(self, file_path: str, config: Dict) -> Optional[Dict]:
        """Analyze a tabular file (CSV, Excel)"""
        try:
            # Try to read the file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=100)  # Read first 100 rows for analysis
            else:
                df = pd.read_excel(file_path, nrows=100)
            
            # Check required columns
            required_columns = config.get('required_columns', [])
            optional_columns = config.get('optional_columns', [])
            
            # Apply column mappings
            column_mappings = self.company_config.get('data_mappings', {}).get('column_mappings', {})
            mapped_columns = []
            for col in df.columns:
                mapped_col = column_mappings.get(col, col)
                mapped_columns.append(mapped_col)
            
            # Check if required columns are present (after mapping) - dataset agnostic
            missing_required = []
            for req_col in required_columns:
                if req_col not in mapped_columns and req_col not in df.columns:
                    missing_required.append(req_col)
            
            # Only fail if there are hardcoded required columns (for backward compatibility)
            # In dataset agnostic mode, we accept any columns
            if missing_required and required_columns:
                logger.debug(f"Missing required columns: {missing_required}")
                return None
            
            # Check for coordinate columns
            coordinate_mappings = self.company_config.get('data_mappings', {}).get('coordinate_mappings', {})
            lat_columns = coordinate_mappings.get('lat', ['latitude', 'lat', 'y', 'Y'])
            lon_columns = coordinate_mappings.get('lon', ['longitude', 'lon', 'x', 'X'])
            
            has_coordinates = any(col in df.columns for col in lat_columns + lon_columns)
            
            return {
                'file_type': 'tabular',
                'columns': list(df.columns),
                'mapped_columns': mapped_columns,
                'rows': len(df),
                'has_coordinates': has_coordinates,
                'missing_required': missing_required,
                'present_optional': [col for col in optional_columns if col in df.columns],
                'data_types': df.dtypes.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tabular file {file_path}: {e}")
            return None
    
    def _analyze_raster_file(self, file_path: str, config: Dict) -> Optional[Dict]:
        """Analyze a raster file (GeoTIFF)"""
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                return {
                    'file_type': 'raster',
                    'bands': src.count,
                    'width': src.width,
                    'height': src.height,
                    'crs': str(src.crs),
                    'bounds': src.bounds,
                    'transform': src.transform,
                    'dtype': src.dtypes[0]
                }
        except ImportError:
            logger.warning("rasterio not available, skipping raster analysis")
            return None
        except Exception as e:
            logger.error(f"Error analyzing raster file {file_path}: {e}")
            return None
    
    def _analyze_geospatial_file(self, file_path: str, config: Dict) -> Optional[Dict]:
        """Analyze a geospatial file (Shapefile, GeoJSON)"""
        try:
            import geopandas as gpd
            gdf = gpd.read_file(file_path)
            
            return {
                'file_type': 'geospatial',
                'columns': list(gdf.columns),
                'rows': len(gdf),
                'geometry_type': str(gdf.geometry.geom_type.iloc[0]) if len(gdf) > 0 else 'unknown',
                'crs': str(gdf.crs),
                'bounds': gdf.total_bounds.tolist()
            }
        except ImportError:
            logger.warning("geopandas not available, skipping geospatial analysis")
            return None
        except Exception as e:
            logger.error(f"Error analyzing geospatial file {file_path}: {e}")
            return None
    
    @log_performance(logger)
    def get_dataset_loader(self, dataset_type: str) -> Optional[Dict]:
        """Get loader configuration for a specific dataset type"""
        logger.debug(f"Getting loader configuration for dataset type: {dataset_type}")
        
        if dataset_type not in self.dataset_configs:
            logger.warning(f"Unknown dataset type: {dataset_type}")
            return None
        
        config = self.dataset_configs[dataset_type]
        if not config.get('enabled', True):
            logger.debug(f"Dataset type {dataset_type} is disabled")
            return None
        
        return {
            'type': dataset_type,
            'config': config,
            'file_patterns': config.get('file_patterns', []),
            'required_columns': config.get('required_columns', []),
            'optional_columns': config.get('optional_columns', []),
            'coordinate_columns': config.get('coordinate_columns', {}),
            'data_cleaning': config.get('data_cleaning', {}),
            'coordinate_generation': config.get('coordinate_generation', {})
        }
    
    @log_performance(logger)
    def validate_dataset_configuration(self) -> Dict[str, Any]:
        """Validate the dataset configuration"""
        logger.info("Validating dataset configuration")
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'datasets': {}
        }
        
        # Validate each dataset configuration
        for dataset_type, config in self.dataset_configs.items():
            if dataset_type == 'custom_datasets':
                continue  # Handle custom datasets separately
                
            dataset_validation = self._validate_dataset_config(dataset_type, config)
            validation_results['datasets'][dataset_type] = dataset_validation
            
            if not dataset_validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(dataset_validation['errors'])
            
            validation_results['warnings'].extend(dataset_validation['warnings'])
        
        # Validate custom datasets
        custom_datasets = self.dataset_configs.get('custom_datasets', {})
        for dataset_name, config in custom_datasets.items():
            if config.get('enabled', False):
                dataset_validation = self._validate_dataset_config(dataset_name, config)
                validation_results['datasets'][dataset_name] = dataset_validation
                
                if not dataset_validation['valid']:
                    validation_results['valid'] = False
                    validation_results['errors'].extend(dataset_validation['errors'])
                
                validation_results['warnings'].extend(dataset_validation['warnings'])
        
        logger.info(f"Configuration validation: {validation_results['valid']}, {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings")
        return validation_results
    
    def _validate_dataset_config(self, dataset_type: str, config: Dict) -> Dict[str, Any]:
        """Validate a specific dataset configuration"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        if 'file_patterns' not in config:
            validation['errors'].append(f"Missing file_patterns for {dataset_type}")
            validation['valid'] = False
        
        if not config.get('file_patterns'):
            validation['warnings'].append(f"No file patterns defined for {dataset_type}")
        
        # Check coordinate generation configuration
        coord_gen = config.get('coordinate_generation', {})
        if coord_gen.get('enabled', False):
            bounds = coord_gen.get('bounds', {})
            required_bounds = ['lat_min', 'lat_max', 'lon_min', 'lon_max']
            missing_bounds = [b for b in required_bounds if b not in bounds]
            if missing_bounds:
                validation['errors'].append(f"Missing coordinate bounds for {dataset_type}: {missing_bounds}")
                validation['valid'] = False
        
        return validation
    
    @log_performance(logger)
    def create_dataset_template(self, dataset_type: str) -> Dict[str, Any]:
        """Create a template configuration for a new dataset type"""
        logger.info(f"Creating template for dataset type: {dataset_type}")
        
        template = {
            'enabled': True,
            'priority': 999,
            'file_patterns': [f"*{dataset_type.lower()}*"],
            'required_columns': [],
            'optional_columns': [],
            'coordinate_columns': {
                'lat': 'latitude',
                'lon': 'longitude'
            },
            'data_cleaning': {
                'remove_duplicates': True,
                'handle_missing': 'drop',
                'numeric_conversion': False
            },
            'coordinate_generation': {
                'enabled': False,
                'method': 'random_distribution',
                'bounds': {
                    'lat_min': -90,
                    'lat_max': 90,
                    'lon_min': -180,
                    'lon_max': 180
                }
            }
        }
        
        return template
    
    @log_performance(logger)
    def save_configuration(self, config_path: Optional[str] = None) -> bool:
        # Save the current configuration to file
        # Args:
        #   config_path: Optional path to save config, uses default if None
        # Returns: True if successful, False otherwise
        # TODO: Add configuration backup before saving
        # TODO: Add configuration validation before saving
        if config_path is None:
            config_path = str(self.config_path)
        
        logger.info(f"Saving configuration to: {config_path}")
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved successfully to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    @log_performance(logger)
    def get_company_info(self) -> Dict[str, Any]:
        """Get company-specific information"""
        return {
            'name': self.company_config.get('name', 'Unknown Company'),
            'region': self.company_config.get('region', 'Unknown Region'),
            'coordinate_bounds': self.company_config.get('coordinate_bounds', {}),
            'data_mappings': self.company_config.get('data_mappings', {})
        }
    
    @log_performance(logger)
    def map_column_names(self, original_columns: List[str]) -> List[str]:
        """Map original column names to standard names"""
        column_mappings = self.company_config.get('data_mappings', {}).get('column_mappings', {})
        mapped_columns = []
        
        for col in original_columns:
            mapped_col = column_mappings.get(col, col)
            mapped_columns.append(mapped_col)
        
        return mapped_columns
    
    @log_performance(logger)
    def find_coordinate_columns(self, columns: List[str]) -> Dict[str, Optional[str]]:
        """Find coordinate columns in a dataset"""
        coordinate_mappings = self.company_config.get('data_mappings', {}).get('coordinate_mappings', {})
        
        lat_columns = coordinate_mappings.get('lat', ['latitude', 'lat', 'y', 'Y'])
        lon_columns = coordinate_mappings.get('lon', ['longitude', 'lon', 'x', 'X'])
        
        lat_col = None
        lon_col = None
        
        for col in columns:
            if col in lat_columns:
                lat_col = col
            elif col in lon_columns:
                lon_col = col
        
        return {
            'lat': lat_col,
            'lon': lon_col
        } 