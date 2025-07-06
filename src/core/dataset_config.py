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
    def __init__(self, config_path: str = "config.yaml", data_dir: str = "DataSets"):
        # Initialize the dataset configuration system
        # Args:
        #   config_path: Path to the YAML configuration file
        #   data_dir: Base directory for dataset discovery and loading

        # Robustly handle both config file and directory cases
        config_path_obj = Path(config_path)
        if config_path_obj.is_dir():
            self.data_dir = config_path_obj.resolve()
            self.config_path = self.data_dir / "config.yaml"
            if self.config_path.exists():
                self.config = self._load_config()
            else:
                self.config = self._get_default_config()
        elif config_path_obj.name == "config.yaml" and config_path_obj.parent.exists():
            self.config_path = config_path_obj.resolve()
            self.data_dir = Path(os.path.abspath(data_dir))
            if self.config_path.exists():
                self.config = self._load_config()
            else:
                self.config = self._get_default_config()
        else:
            # If a file or non-existent path is given, treat as data_dir
            self.data_dir = Path(os.path.abspath(config_path))
            self.config_path = self.data_dir / "config.yaml"
            if self.config_path.exists():
                self.config = self._load_config()
            else:
                self.config = self._get_default_config()
        self.dataset_configs = self.config.get('datasets', {})
        self.company_config = self.config.get('company', {})
        logger.info(f"Initialized DatasetConfig with config_path={self.config_path} and data_dir={self.data_dir}")
        
    def _load_config(self) -> Dict[str, Any]:
        # Load configuration from a YAML file

        # TODO: Add configuration validation and schema checking
        # TODO: Support environment-specific configs (dev, prod, test)

        logger.debug(f"Loading configuration from: {self.config_path}")
        try:
            # Attempt to read and parse the YAML config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config

        except FileNotFoundError:
            # If file not found, fall back to default configuration
            logger.warning(f"Config file {self.config_path} not found, using default configuration")
            return self._get_default_config()

        except Exception as e:
            # Handle other I/O or parsing errors
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()


    def _get_default_config(self) -> Dict[str, Any]:
        # Return a default configuration dictionary
        # Used when no config file is found or valid

        # TODO: Add more comprehensive default configurations
        # TODO: Include region-specific defaults

        logger.debug("Creating default configuration")
        return {
            'directories': {
                'data_dir': 'DataSets',        # Default folder for raw datasets
                'model_dir': 'models',         # Directory for model outputs or checkpoints
                'logs_dir': 'Logs',            # Logging output directory
                'visuals_dir': 'visuals'       # Path for charts, images, visual outputs
            },
            'datasets': {
                'infrastructure': {
                    'enabled': True,
                    'file_patterns': ['*infrastructure*', '*pipes*', '*drainage*', '*INF_DRN*', '*INF*'],
                    'required_columns': [],       # Dataset agnostic - no fixed schema
                    'optional_columns': []
                },
                'vegetation': {
                    'enabled': True,
                    'file_patterns': ['*vegetation*', '*zones*', '*VegetationZones*'],
                    'required_columns': [],
                    'optional_columns': []
                },
                'climate': {
                    'enabled': True,
                    'file_patterns': ['*climate*', '*weather*', '*tas*', '*hurs*', '*pan-evap*', '*prec*', '*srad*'],
                    'required_columns': [],
                    'optional_columns': []
                },
                'wind': {
                    'enabled': True,
                    'file_patterns': ['*wind*', '*wind-observations*'],
                    'required_columns': [],
                    'optional_columns': []
                }
            },
            'company': {
                'name': 'Default Company',         # Placeholder company name
                'region': 'Default Region',        # Default operational region
                'data_mappings': {
                    'column_mappings': {},         # No remapping defined initially
                    'coordinate_mappings': {
                        'lat': ['latitude', 'lat', 'y', 'Y'],     # Accepted lat column names
                        'lon': ['longitude', 'lon', 'x', 'X']     # Accepted lon column names
                    }
                }
            }
        }

    
    @log_performance(logger)
    def discover_datasets(self) -> Dict[str, Dict]:
        # Discover available datasets based on configuration patterns
        # This method scans the data directory and matches files against configured patterns

        # TODO: Add caching for discovery results to improve performance
        # TODO: Add file modification time checking for incremental discovery

        logger.info("Discovering datasets in data directory")
        discovered_datasets = {}

        # Check if the data directory exists before scanning
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return discovered_datasets

        # Discover each configured dataset type
        for dataset_type, config in self.dataset_configs.items():
            if not config.get('enabled', True):
                logger.debug(f"Dataset type {dataset_type} is disabled")
                continue

            discovered = self._discover_dataset_type(dataset_type, config)
            if discovered:
                discovered_datasets[dataset_type] = discovered
                logger.info(f"Discovered {dataset_type}: {len(discovered['files'])} files")

        # Discover user-defined custom datasets (if any)
        custom_datasets = self.dataset_configs.get('custom_datasets', {})
        for dataset_name, config in custom_datasets.items():
            if config.get('enabled', False):
                discovered = self._discover_dataset_type(dataset_name, config)
                if discovered:
                    discovered_datasets[dataset_name] = discovered
                    logger.info(f"Discovered custom dataset {dataset_name}: {len(discovered['files'])} files")

        # Final summary log
        logger.info(f"Total datasets discovered: {len(discovered_datasets)}")
        return discovered_datasets

    
    def _discover_dataset_type(self, dataset_type: str, config: Dict) -> Optional[Dict]:
        # Discover files for a specific dataset type
        # This method uses file patterns to find matching files in the data directory

        # TODO: Add support for more complex file pattern matching
        # TODO: Add file size and modification time filtering

        logger.debug(f"Discovering files for dataset type: {dataset_type}")
        
        files = []  # List to hold metadata for all matched and valid files
        file_patterns = config.get('file_patterns', [])  # Patterns to search for

        # Search for files matching each pattern
        for pattern in file_patterns:
            # Construct full search path using pattern
            search_pattern = str(self.data_dir / pattern)
            matching_files = glob.glob(search_pattern, recursive=True)

            # Analyze each matching file and collect valid results
            for file_path in matching_files:
                file_info = self._analyze_file(file_path, config)
                if file_info:
                    files.append(file_info)

        # If no valid files found, return None
        if not files:
            logger.debug(f"No files found for dataset type: {dataset_type}")
            return None

        # Return structured metadata about the discovered dataset
        return {
            'type': dataset_type,                          # Dataset identifier
            'config': config,                              # Configuration used for discovery
            'files': files,                                # List of file metadata dictionaries
            'total_files': len(files),                     # Count of successfully analyzed files
            'enabled': config.get('enabled', True),        # Whether dataset is enabled
            'priority': config.get('priority', 999)        # Optional processing priority
        }

    
    def _analyze_file(self, file_path: str, config: Dict) -> Optional[Dict]:
        # Analyze a file to determine if it matches the dataset configuration
        logger.debug(f"Analyzing file: {file_path}")
        
        try:
            file_path_obj = Path(file_path)
            
            # Check if the file exists
            if not file_path_obj.exists():
                return None

            # Collect basic file metadata
            file_info = {
                'path': str(file_path_obj),                  # Full path to file
                'name': file_path_obj.name,                  # Filename only
                'size': file_path_obj.stat().st_size,        # Size in bytes
                'extension': file_path_obj.suffix.lower(),   # Lowercase file extension
                'modified': file_path_obj.stat().st_mtime    # Last modified timestamp
            }

            # ────── Analyze file content based on extension ──────

            if file_info['extension'] in ['.csv', '.xlsx', '.xls']:
                # Try analyzing as a tabular file
                content_info = self._analyze_tabular_file(file_path, config)
                if content_info:
                    file_info.update(content_info)
                    return file_info

            elif file_info['extension'] in ['.tif', '.tiff']:
                # Try analyzing as a raster file
                content_info = self._analyze_raster_file(file_path, config)
                if content_info:
                    file_info.update(content_info)
                    return file_info

            elif file_info['extension'] in ['.shp', '.geojson', '.json']:
                # Try analyzing as a geospatial vector file
                content_info = self._analyze_geospatial_file(file_path, config)
                if content_info:
                    file_info.update(content_info)
                    return file_info

            # If file type is unsupported or analysis failed
            logger.debug(f"File {file_path} does not match dataset requirements")
            return None

        except Exception as e:
            # Log unexpected errors during file analysis
            logger.error(f"Error analyzing file {file_path}: {e}")
            return None

    
    def _analyze_tabular_file(self, file_path: str, config: Dict) -> Optional[Dict]:
        # Analyze a tabular data file (CSV or Excel)
        try:
            # Try to read the file (only first 100 rows to reduce load)
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=100)
            else:
                df = pd.read_excel(file_path, nrows=100)
            
            # Extract required and optional columns from config
            required_columns = config.get('required_columns', [])
            optional_columns = config.get('optional_columns', [])

            # Map column names using company config (if mapping is defined)
            column_mappings = self.company_config.get('data_mappings', {}).get('column_mappings', {})
            mapped_columns = []
            for col in df.columns:
                mapped_col = column_mappings.get(col, col)
                mapped_columns.append(mapped_col)

            # Check for required columns after mapping
            missing_required = []
            for req_col in required_columns:
                if req_col not in mapped_columns and req_col not in df.columns:
                    missing_required.append(req_col)

            # If there are required columns and some are missing, reject the file
            if missing_required and required_columns:
                logger.debug(f"Missing required columns: {missing_required}")
                return None

            # Detect coordinate columns using company mapping or default names
            coordinate_mappings = self.company_config.get('data_mappings', {}).get('coordinate_mappings', {})
            lat_columns = coordinate_mappings.get('lat', ['latitude', 'lat', 'y', 'Y'])
            lon_columns = coordinate_mappings.get('lon', ['longitude', 'lon', 'x', 'X'])

            has_coordinates = any(col in df.columns for col in lat_columns + lon_columns)

            # Return analysis result as a structured dictionary
            return {
                'file_type': 'tabular',                                     # File classification
                'columns': list(df.columns),                                # Original columns
                'mapped_columns': mapped_columns,                           # Mapped column names
                'rows': len(df),                                            # Number of sampled rows
                'has_coordinates': has_coordinates,                         # Whether coordinate columns were found
                'missing_required': missing_required,                       # List of required columns not found
                'present_optional': [col for col in optional_columns if col in df.columns],  # Optional columns present
                'data_types': df.dtypes.to_dict()                           # Data types of each column
            }

        except Exception as e:
            # Handle errors during tabular file analysis
            logger.error(f"Error analyzing tabular file {file_path}: {e}")
            return None



    def _analyze_raster_file(self, file_path: str, config: Dict) -> Optional[Dict]:
        # Analyze a raster file (e.g., GeoTIFF) using rasterio
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                return {
                    'file_type': 'raster',               # Identifies file as raster
                    'bands': src.count,                  # Number of bands in the raster
                    'width': src.width,                  # Width in pixels
                    'height': src.height,                # Height in pixels
                    'crs': str(src.crs),                 # Coordinate reference system
                    'bounds': src.bounds,                # Spatial bounds of the raster
                    'transform': src.transform,          # Affine transform matrix
                    'dtype': src.dtypes[0]               # Data type of first band
                }
        except ImportError:
            # Handle missing rasterio dependency
            logger.warning("rasterio not available, skipping raster analysis")
            return None
        except Exception as e:
            # Handle any other errors during raster analysis
            logger.error(f"Error analyzing raster file {file_path}: {e}")
            return None


    def _analyze_geospatial_file(self, file_path: str, config: Dict) -> Optional[Dict]:
        # Analyze a geospatial vector file (e.g., Shapefile, GeoJSON) using geopandas
        try:
            import geopandas as gpd
            gdf = gpd.read_file(file_path)

            return {
                'file_type': 'geospatial',                           # Identifies file as geospatial vector
                'columns': list(gdf.columns),                        # List of attribute columns
                'rows': len(gdf),                                    # Total number of features/records
                'geometry_type': str(gdf.geometry.geom_type.iloc[0]) if len(gdf) > 0 else 'unknown',
                'crs': str(gdf.crs),                                 # Coordinate reference system
                'bounds': gdf.total_bounds.tolist()                  # Spatial extent [minx, miny, maxx, maxy]
            }
        except ImportError:
            # Handle missing geopandas dependency
            logger.warning("geopandas not available, skipping geospatial analysis")
            return None
        except Exception as e:
            # Handle any other errors during geospatial file parsing
            logger.error(f"Error analyzing geospatial file {file_path}: {e}")
            return None

    
    @log_performance(logger)
    def get_dataset_loader(self, dataset_type: str) -> Optional[Dict]:
        # Retrieve loader configuration for the given dataset type
        logger.debug(f"Getting loader configuration for dataset type: {dataset_type}")
        
        # Return None if dataset type is not recognized
        if dataset_type not in self.dataset_configs:
            logger.warning(f"Unknown dataset type: {dataset_type}")
            return None
        
        config = self.dataset_configs[dataset_type]
        
        # Return None if dataset is explicitly disabled
        if not config.get('enabled', True):
            logger.debug(f"Dataset type {dataset_type} is disabled")
            return None
        
        # Return structured loader configuration dictionary
        return {
            'type': dataset_type,                                       # Dataset identifier
            'config': config,                                           # Full raw config
            'file_patterns': config.get('file_patterns', []),           # File matching patterns
            'required_columns': config.get('required_columns', []),     # Mandatory columns
            'optional_columns': config.get('optional_columns', []),     # Optional columns
            'coordinate_columns': config.get('coordinate_columns', {}), # Latitude/longitude columns
            'data_cleaning': config.get('data_cleaning', {}),           # Data cleaning rules
            'coordinate_generation': config.get('coordinate_generation', {})  # Coordinate generation config
        }

    
    @log_performance(logger)
    def validate_dataset_configuration(self) -> Dict[str, Any]:
        """Validate the full dataset configuration, including built-in and custom datasets"""

        logger.info("Validating dataset configuration")

        validation_results = {
            'valid': True,         # Overall config validity (set False if any dataset is invalid)
            'errors': [],          # Global list of all blocking errors from all datasets
            'warnings': [],        # Global list of all non-blocking issues
            'datasets': {}         # Per-dataset validation results
        }

        # ────── Validate predefined dataset configurations ──────
        for dataset_type, config in self.dataset_configs.items():
            if dataset_type == 'custom_datasets':
                continue  # Custom datasets handled separately

            dataset_validation = self._validate_dataset_config(dataset_type, config)
            validation_results['datasets'][dataset_type] = dataset_validation

            # Mark entire config as invalid if any dataset fails validation
            if not dataset_validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(dataset_validation['errors'])

            validation_results['warnings'].extend(dataset_validation['warnings'])

        # ────── Validate custom dataset configurations ──────
        custom_datasets = self.dataset_configs.get('custom_datasets', {})
        for dataset_name, config in custom_datasets.items():
            if config.get('enabled', False):
                dataset_validation = self._validate_dataset_config(dataset_name, config)
                validation_results['datasets'][dataset_name] = dataset_validation

                if not dataset_validation['valid']:
                    validation_results['valid'] = False
                    validation_results['errors'].extend(dataset_validation['errors'])

                validation_results['warnings'].extend(dataset_validation['warnings'])

        # ────── Final summary log ──────
        logger.info(
            f"Configuration validation: {validation_results['valid']}, "
            f"{len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings"
        )

        return validation_results

    
    def _validate_dataset_config(self, dataset_type: str, config: Dict) -> Dict[str, Any]:
        """Validate a specific dataset configuration for required structure and coordinate bounds"""

        validation = {
            'valid': True,        # Flag set to False if any critical error is found
            'errors': [],         # List of blocking issues
            'warnings': []        # List of non-blocking issues or incomplete settings
        }

        # ────── Check for required fields ──────

        if 'file_patterns' not in config:
            validation['errors'].append(f"Missing file_patterns for {dataset_type}")
            validation['valid'] = False  # Critical: file_patterns is required for dataset discovery

        # file_patterns is present but empty
        if not config.get('file_patterns'):
            validation['warnings'].append(f"No file patterns defined for {dataset_type}")

        # ────── Coordinate generation validation ──────

        coord_gen = config.get('coordinate_generation', {})
        if coord_gen.get('enabled', False):
            # If coordinate generation is enabled, check that all required bounds are present
            bounds = coord_gen.get('bounds', {})
            required_bounds = ['lat_min', 'lat_max', 'lon_min', 'lon_max']
            missing_bounds = [b for b in required_bounds if b not in bounds]

            if missing_bounds:
                validation['errors'].append(f"Missing coordinate bounds for {dataset_type}: {missing_bounds}")
                validation['valid'] = False  # Invalid if any required bounds are missing

        return validation

    
    @log_performance(logger)
    def create_dataset_template(self, dataset_type: str) -> Dict[str, Any]:
        """Create a default configuration template for a new dataset type"""
        
        logger.info(f"Creating template for dataset type: {dataset_type}")

        template = {
            'enabled': True,  # Whether the dataset is active for processing
            'priority': 999,  # Default low-priority placeholder (higher numbers = lower priority)

            # File discovery pattern, based on dataset type keyword
            'file_patterns': [f"*{dataset_type.lower()}*"],

            # Define required and optional columns for the dataset
            'required_columns': [],
            'optional_columns': [],

            # Mapping of coordinate fields (can be overridden per dataset)
            'coordinate_columns': {
                'lat': 'latitude',
                'lon': 'longitude'
            },

            # Basic data cleaning configuration
            'data_cleaning': {
                'remove_duplicates': True,          # Whether to drop duplicate rows
                'handle_missing': 'drop',           # Strategy for missing values: 'drop' or 'fill'
                'numeric_conversion': False         # Whether to force conversion of columns to numeric
            },

            # Coordinate generation configuration (e.g. for synthetic or missing coordinates)
            'coordinate_generation': {
                'enabled': False,                   # Whether to auto-generate coordinates
                'method': 'random_distribution',    # Default generation method
                'bounds': {                         # Bounding box to constrain generated coordinates
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
        # Save the current configuration dictionary to a YAML file.
        # Args:
        #   config_path: Optional custom path to save the config; defaults to internal config_path.
        # Returns:
        #   True if saving was successful, False otherwise.

        # TODO: Add backup mechanism before overwriting existing config
        # TODO: Validate configuration structure before writing

        if config_path is None:
            config_path = str(self.config_path)

        logger.info(f"Saving configuration to: {config_path}")
        try:
            # Open file for writing and dump config as YAML
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved successfully to {config_path}")
            return True
        except Exception as e:
            # Log any exceptions that occur during save operation
            logger.error(f"Error saving configuration: {e}")
            return False


    @log_performance(logger)
    def get_company_info(self) -> Dict[str, Any]:
        """Get company-specific metadata from configuration"""
        return {
            'name': self.company_config.get('name', 'Unknown Company'),
            'region': self.company_config.get('region', 'Unknown Region'),
            'coordinate_bounds': self.company_config.get('coordinate_bounds', {}),
            'data_mappings': self.company_config.get('data_mappings', {})
        }


    @log_performance(logger)
    def map_column_names(self, original_columns: List[str]) -> List[str]:
        """Map original column names to standardized names based on company-defined mappings"""
        column_mappings = self.company_config.get('data_mappings', {}).get('column_mappings', {})
        mapped_columns = []

        for col in original_columns:
            # Use mapped name if available, fallback to original
            mapped_col = column_mappings.get(col, col)
            mapped_columns.append(mapped_col)

        return mapped_columns


    @log_performance(logger)
    def find_coordinate_columns(self, columns: List[str]) -> Dict[str, Optional[str]]:
        """Identify coordinate column names (latitude/longitude) from a list of input columns"""
        
        coordinate_mappings = self.company_config.get('data_mappings', {}).get('coordinate_mappings', {})

        # Fallback lists for possible lat/lon column names if not explicitly configured
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
            'lat': lat_col,  # Best-guess latitude column, or None if not found
            'lon': lon_col   # Best-guess longitude column, or None if not found
        }
