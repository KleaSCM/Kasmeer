# Author: KleaSCM
# Date: 2024
# Description: Flexible data processor that works with any company's datasets

import os
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings
from ..utils.logging_utils import setup_logging, log_performance
from ..core.dataset_config import DatasetConfig
warnings.filterwarnings('ignore')

logger = setup_logging(__name__)

class DataProcessor:
    # Universal data processor that works with any company's datasets
    # This class uses configuration-based discovery to load and process datasets
    # without hardcoding file paths or data structures
    # TODO: Add support for streaming large datasets
    # TODO: Add data validation and quality checks
    # TODO: Add support for real-time data sources
    
    @log_performance(logger)
    def __init__(self, config_path: str = "config.yaml"):
        # Initialize the universal data processor
        # Args:
        #   config_path: Path to the configuration file
        self.dataset_config = DatasetConfig(config_path)
        self.data_dir = self.dataset_config.data_dir
        self.processed_data = {}  # Store loaded and processed datasets
        self.discovered_datasets = {}  # Store discovered dataset metadata
        logger.info(f"Initialized DataProcessor with config_path={config_path}")
        
    @log_performance(logger)
    def discover_and_load_all_data(self) -> Dict[str, Any]:
        # Discover and load all available datasets based on configuration
        # This method automatically finds and loads datasets without hardcoding
        # Returns: Dictionary containing all loaded datasets
        # TODO: Add parallel loading for better performance
        # TODO: Add progress tracking for large datasets
        # TODO: Add memory usage monitoring
        logger.info("Discovering and loading all available datasets")
        
        # Discover datasets using configuration patterns
        self.discovered_datasets = self.dataset_config.discover_datasets()
        logger.info(f"Discovered {len(self.discovered_datasets)} dataset types")
        
        # Load each discovered dataset that is enabled
        for dataset_type, dataset_info in self.discovered_datasets.items():
            if dataset_info.get('enabled', True):
                logger.info(f"Loading dataset type: {dataset_type}")
                self._load_dataset(dataset_type, dataset_info)
        
        return self.processed_data
    
    def _load_dataset(self, dataset_type: str, dataset_info: Dict) -> bool:
        # Load a specific dataset type based on discovered files
        # Args:
        #   dataset_type: Name of the dataset type to load
        #   dataset_info: Dictionary containing dataset discovery information
        # Returns: True if loading successful, False otherwise
        # TODO: Add support for loading multiple files per dataset type
        # TODO: Add file format detection and validation
        # TODO: Add error recovery for corrupted files
        logger.debug(f"Loading dataset type: {dataset_type}")
        
        try:
            files = dataset_info.get('files', [])
            if not files:
                logger.warning(f"No files found for dataset type: {dataset_type}")
                return False
            
            # Load the first file (or implement multi-file loading)
            file_info = files[0]
            file_path = file_info['path']
            
            # Determine file type and load accordingly
            if file_info['file_type'] == 'tabular':
                data = self._load_tabular_file(file_path, dataset_type, dataset_info)
            elif file_info['file_type'] == 'raster':
                data = self._load_raster_file(file_path, dataset_type, dataset_info)
            elif file_info['file_type'] == 'geospatial':
                data = self._load_geospatial_file(file_path, dataset_type, dataset_info)
            else:
                logger.warning(f"Unsupported file type: {file_info['file_type']}")
                return False
            
            # Store loaded data if successful
            if data is not None:
                self.processed_data[dataset_type] = data
                logger.info(f"Successfully loaded {dataset_type}: {len(data) if hasattr(data, '__len__') else 'raster data'} records")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_type}: {e}")
            return False
    
    def _load_tabular_file(self, file_path: str, dataset_type: str, dataset_info: Dict) -> Optional[pd.DataFrame]:
        """Load a tabular file (CSV, Excel)"""
        logger.debug(f"Loading tabular file: {file_path}")
        
        try:
            # Load the file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Apply column mappings
            original_columns = list(df.columns)
            mapped_columns = self.dataset_config.map_column_names(original_columns)
            
            # Rename columns if mappings exist
            column_mappings = self.dataset_config.company_config.get('data_mappings', {}).get('column_mappings', {})
            if column_mappings:
                df = df.rename(columns=column_mappings)
            
            # Clean the data
            df = self._clean_tabular_data(df, dataset_type, dataset_info)
            
            # Generate coordinates if needed
            df = self._generate_coordinates(df, dataset_type, dataset_info)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading tabular file {file_path}: {e}")
            return None
    
    def _load_raster_file(self, file_path: str, dataset_type: str, dataset_info: Dict) -> Optional[Dict]:
        """Load a raster file (GeoTIFF)"""
        logger.debug(f"Loading raster file: {file_path}")
        
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)  # Read first band
                return {
                    'data': data,
                    'transform': src.transform,
                    'crs': src.crs,
                    'bounds': src.bounds,
                    'width': src.width,
                    'height': src.height,
                    'file_path': file_path
                }
        except Exception as e:
            logger.error(f"Error loading raster file {file_path}: {e}")
            return None
    
    def _load_geospatial_file(self, file_path: str, dataset_type: str, dataset_info: Dict) -> Optional[gpd.GeoDataFrame]:
        # Load a geospatial file (Shapefile, GeoJSON)
        # Args:
        #   file_path: Path to the geospatial file
        #   dataset_type: Type of dataset being loaded
        #   dataset_info: Dataset configuration information
        # Returns: GeoDataFrame if successful, None otherwise
        # TODO: Add support for more geospatial formats (KML, GML, etc.)
        # TODO: Add coordinate system validation and transformation
        logger.debug(f"Loading geospatial file: {file_path}")
        
        try:
            gdf = gpd.read_file(file_path)
            
            # Ensure it's a GeoDataFrame
            if not isinstance(gdf, gpd.GeoDataFrame):
                logger.warning(f"File {file_path} is not a valid geospatial file")
                return None
            
            # Type assertion since we've verified it's a GeoDataFrame
            gdf = gdf  # type: gpd.GeoDataFrame
            
            # Apply column mappings if configured
            column_mappings = self.dataset_config.company_config.get('data_mappings', {}).get('column_mappings', {})
            if column_mappings:
                gdf = gdf.rename(columns=column_mappings)  # type: ignore
            
            # Clean the geospatial data - cast to GeoDataFrame for type checker
            cleaned_gdf = self._clean_geospatial_data(gdf, dataset_type, dataset_info)  # type: ignore
            
            return cleaned_gdf
            
        except Exception as e:
            logger.error(f"Error loading geospatial file {file_path}: {e}")
            return None
    
    def _clean_tabular_data(self, df: pd.DataFrame, dataset_type: str, dataset_info: Dict) -> pd.DataFrame:
        """Clean tabular data according to configuration"""
        logger.debug(f"Cleaning tabular data for {dataset_type}")
        
        config = dataset_info.get('config', {})
        cleaning_config = config.get('data_cleaning', {})
        
        # Remove duplicates
        if cleaning_config.get('remove_duplicates', True):
            original_len = len(df)
        df = df.drop_duplicates()
            removed = original_len - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows from {dataset_type}")
        
        # Handle missing values
        handle_missing = cleaning_config.get('handle_missing', 'drop')
        if handle_missing == 'drop':
            df = df.dropna()
        elif handle_missing == 'interpolate':
            # Interpolate numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        
        # Convert numeric columns
        if cleaning_config.get('numeric_conversion', False):
            required_cols = config.get('required_columns', [])
            for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _clean_geospatial_data(self, gdf: gpd.GeoDataFrame, dataset_type: str, dataset_info: Dict) -> gpd.GeoDataFrame:
        """Clean geospatial data according to configuration"""
        logger.debug(f"Cleaning geospatial data for {dataset_type}")
        
        config = dataset_info.get('config', {})
        cleaning_config = config.get('data_cleaning', {})
        
        # Remove duplicates
        if cleaning_config.get('remove_duplicates', True):
            original_len = len(gdf)
            gdf = gdf.drop_duplicates()  # type: ignore
            removed = original_len - len(gdf)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows from {dataset_type}")
        
        # Handle missing values
        handle_missing = cleaning_config.get('handle_missing', 'drop')
        if handle_missing == 'drop':
            gdf = gdf.dropna()  # type: ignore
        
        return gdf
    
    def _generate_coordinates(self, df: pd.DataFrame, dataset_type: str, dataset_info: Dict) -> pd.DataFrame:
        """Generate coordinates if they don't exist"""
        logger.debug(f"Generating coordinates for {dataset_type}")
        
        config = dataset_info.get('config', {})
        coord_config = config.get('coordinate_generation', {})
        
        if not coord_config.get('enabled', False):
            return df
        
        # Check if coordinates already exist
        coordinate_cols = self.dataset_config.find_coordinate_columns(list(df.columns))
        if coordinate_cols['lat'] and coordinate_cols['lon']:
            logger.debug(f"Coordinates already exist for {dataset_type}")
            return df
        
        # Generate coordinates
        method = coord_config.get('method', 'random_distribution')
        bounds = coord_config.get('bounds', {})
        
        if method == 'random_distribution':
            df = self._generate_random_coordinates(df, bounds, coord_config)
        elif method == 'clustering':
            df = self._generate_clustered_coordinates(df, bounds, coord_config)
        
        return df
    
    def _generate_random_coordinates(self, df: pd.DataFrame, bounds: Dict, coord_config: Dict) -> pd.DataFrame:
        """Generate random coordinates within bounds"""
        logger.debug("Generating random coordinates")
        
        lat_min = bounds.get('lat_min', -90)
        lat_max = bounds.get('lat_max', 90)
        lon_min = bounds.get('lon_min', -180)
        lon_max = bounds.get('lon_max', 180)
        
        np.random.seed(42)  # For reproducible results
        df['latitude'] = np.random.uniform(lat_min, lat_max, len(df))
        df['longitude'] = np.random.uniform(lon_min, lon_max, len(df))
        
                return df
    
    def _generate_clustered_coordinates(self, df: pd.DataFrame, bounds: Dict, coord_config: Dict) -> pd.DataFrame:
        """Generate clustered coordinates around major areas"""
        logger.debug("Generating clustered coordinates")
        
        clustering = coord_config.get('clustering', {})
        major_areas = clustering.get('major_areas', [])
        
        if not major_areas:
            return self._generate_random_coordinates(df, bounds, coord_config)
        
        np.random.seed(42)  # For reproducible results
        
        # Initialize coordinates
        df['latitude'] = 0.0
        df['longitude'] = 0.0
        
        # Assign coordinates to major areas
        for i in range(len(df)):
            area = major_areas[i % len(major_areas)]
            radius = area.get('radius', 0.01)
            
            df.iloc[i, df.columns.get_loc('latitude')] = area['lat'] + np.random.normal(0, radius)
            df.iloc[i, df.columns.get_loc('longitude')] = area['lon'] + np.random.normal(0, radius)
        
                return df
    
    @log_performance(logger)
    def load_specific_dataset(self, dataset_type: str) -> Optional[Union[pd.DataFrame, Dict, gpd.GeoDataFrame]]:
        """Load a specific dataset type"""
        logger.info(f"Loading specific dataset: {dataset_type}")
        
        if dataset_type in self.processed_data:
            logger.debug(f"Dataset {dataset_type} already loaded")
            return self.processed_data[dataset_type]
        
        if dataset_type not in self.discovered_datasets:
            logger.warning(f"Dataset type {dataset_type} not discovered")
            return None
        
        dataset_info = self.discovered_datasets[dataset_type]
        success = self._load_dataset(dataset_type, dataset_info)
        
        if success:
            return self.processed_data[dataset_type]
        else:
            return None
    
    @log_performance(logger)
    def extract_features_at_location(self, lat: float, lon: float) -> Dict[str, Any]:
        """Extract features at a specific location"""
        logger.debug(f"Extracting features at location: ({lat}, {lon})")
        
        features = {}
        
        # Extract infrastructure features
        if 'infrastructure' in self.processed_data:
            features.update(self._extract_infrastructure_features(lat, lon))
        
        # Extract vegetation features
        if 'vegetation' in self.processed_data:
            features.update(self._extract_vegetation_features(lat, lon))
        
        # Extract climate features
        if 'climate' in self.processed_data:
            features.update(self._extract_climate_features(lat, lon))
        
        # Extract wind features
        if 'wind' in self.processed_data:
            features.update(self._extract_wind_features(lat, lon))
        
        return features
    
    def _extract_infrastructure_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """Extract infrastructure features at location"""
        infra_data = self.processed_data['infrastructure']
        
        if isinstance(infra_data, pd.DataFrame):
            # Calculate distance to nearest infrastructure
            if 'latitude' in infra_data.columns and 'longitude' in infra_data.columns:
                distances = np.sqrt(
                    (infra_data['latitude'] - lat)**2 + 
                    (infra_data['longitude'] - lon)**2
                )
                nearest_distance = distances.min()
                nearest_idx = distances.idxmin()
                
                # Find material/type column dynamically
                material_columns = [col for col in infra_data.columns if any(keyword in col.lower() for keyword in ['type', 'material', 'pipe'])]
                nearest_type = infra_data.loc[nearest_idx, material_columns[0]] if material_columns else 'unknown'
                
                return {
                    'nearest_infrastructure_distance': nearest_distance,
                    'nearest_infrastructure_type': nearest_type,
                    'infrastructure_count_within_1km': len(distances[distances <= 0.01]),  # Rough 1km approximation
                    'infrastructure_density': len(infra_data) / 1000  # Per 1000 records
                }
        
        return {
            'nearest_infrastructure_distance': float('inf'),
            'nearest_infrastructure_type': 'unknown',
            'infrastructure_count_within_1km': 0,
            'infrastructure_density': 0
        }
    
    def _extract_vegetation_features(self, lat: float, lon: float) -> Dict[str, Any]:
        # Extract vegetation features at a specific location
        # Args:
        #   lat: Latitude coordinate
        #   lon: Longitude coordinate
        # Returns: Dictionary containing vegetation feature information
        # TODO: Add more sophisticated vegetation analysis
        # TODO: Add seasonal vegetation patterns
        veg_data = self.processed_data['vegetation']
        
        if isinstance(veg_data, pd.DataFrame):
            # Handle tabular vegetation data - dataset agnostic
            zone_types = []
            # Find type/category column dynamically
            type_columns = [col for col in veg_data.columns if any(keyword in col.lower() for keyword in ['type', 'category', 'class', 'zone'])]
            if type_columns:
                zone_types = veg_data[type_columns[0]].unique().tolist()
            
            return {
                'vegetation_zones_count': len(veg_data),
                'vegetation_zone_types': zone_types,
                'vegetation_density': 'medium'  # Placeholder
            }
        elif isinstance(veg_data, gpd.GeoDataFrame):
            # Spatial intersection for geospatial data - dataset agnostic
            point = Point(lon, lat)
            intersecting = veg_data[veg_data.geometry.intersects(point)]
            
            zone_types = []
            # Find type/category column dynamically
            type_columns = [col for col in intersecting.columns if any(keyword in col.lower() for keyword in ['type', 'category', 'class', 'zone'])]
            if type_columns:
                zone_types = intersecting[type_columns[0]].unique().tolist()  # type: ignore
            
            return {
                'vegetation_zones_count': len(intersecting),
                'vegetation_zone_types': zone_types,
                'vegetation_density': 'high' if len(intersecting) > 0 else 'low'
            }
        
        return {
            'vegetation_zones_count': 0,
            'vegetation_zone_types': [],
            'vegetation_density': 'unknown'
        }
    
    def _extract_climate_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """Extract climate features at location"""
        climate_data = self.processed_data['climate']
        
        if isinstance(climate_data, dict) and 'data' in climate_data:
            # For raster data, we'd need to interpolate
            # This is a simplified version
            return {
                'temperature_avg': np.random.uniform(10, 25),  # Placeholder
                'precipitation': np.random.uniform(500, 1500),  # Placeholder
                'solar_radiation': np.random.uniform(10, 20),   # Placeholder
                'climate_zone': 'temperate'  # Placeholder
            }
        
        return {
            'temperature_avg': 0,
            'precipitation': 0,
            'solar_radiation': 0,
            'climate_zone': 'unknown'
        }
    
    def _extract_wind_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """Extract wind features at location"""
        wind_data = self.processed_data['wind']
        
        if isinstance(wind_data, pd.DataFrame):
            return {
                'wind_speed_avg': wind_data['wind_speed'].mean() if 'wind_speed' in wind_data.columns else 0,
                'wind_direction_avg': wind_data['wind_direction'].mean() if 'wind_direction' in wind_data.columns else 0,
                'wind_records_count': len(wind_data)
            }
        
        return {
            'wind_speed_avg': 0,
            'wind_direction_avg': 0,
            'wind_records_count': 0
        }
    
    @log_performance(logger)
    def create_spatial_index(self) -> Dict[str, Any]:
        """Create spatial indexes for efficient querying"""
        logger.info("Creating spatial indexes")
        
        spatial_data = {}
        
        for dataset_type, data in self.processed_data.items():
            if isinstance(data, pd.DataFrame) and 'latitude' in data.columns and 'longitude' in data.columns:
                # Create spatial index for tabular data with coordinates
                spatial_data[dataset_type] = {
                    'type': 'tabular',
                    'count': len(data),
                    'bounds': {
                        'lat_min': data['latitude'].min(),
                        'lat_max': data['latitude'].max(),
                        'lon_min': data['longitude'].min(),
                        'lon_max': data['longitude'].max()
                    }
                }
            elif isinstance(data, gpd.GeoDataFrame):
                # Create spatial index for geospatial data
                spatial_data[dataset_type] = {
                    'type': 'geospatial',
                    'count': len(data),
                    'bounds': data.total_bounds.tolist(),
                    'crs': str(data.crs)
                }
            elif isinstance(data, dict) and 'bounds' in data:
                # Create spatial index for raster data
                spatial_data[dataset_type] = {
                    'type': 'raster',
                    'bounds': data['bounds'],
                    'crs': str(data['crs']),
                    'width': data.get('width', 0),
                    'height': data.get('height', 0)
                }
        
        logger.info(f"Created spatial indexes for {len(spatial_data)} dataset types")
        return spatial_data
    
    @log_performance(logger)
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded data"""
        logger.info("Generating data summary")
        
        summary = {}
        
        for dataset_type, data in self.processed_data.items():
            if isinstance(data, pd.DataFrame):
                summary[dataset_type] = {
                    'type': 'tabular',
                    'rows': len(data),
                    'columns': list(data.columns),
                    'loaded': True,
                    'memory_usage': data.memory_usage(deep=True).sum()
                }
            elif isinstance(data, gpd.GeoDataFrame):
                summary[dataset_type] = {
                    'type': 'geospatial',
                    'rows': len(data),
                    'columns': list(data.columns),
                    'loaded': True,
                    'geometry_type': str(data.geometry.geom_type.iloc[0]) if len(data) > 0 else 'unknown',
                    'crs': str(data.crs)
                }
            elif isinstance(data, dict):
                summary[dataset_type] = {
                    'type': 'raster',
                    'loaded': True,
                    'bounds': data.get('bounds', None),
                    'crs': str(data.get('crs', 'unknown')),
                    'width': data.get('width', 0),
                    'height': data.get('height', 0)
                }
            else:
                summary[dataset_type] = {
                    'type': 'unknown',
                    'loaded': True
                }
        
        # Add discovery information
        summary['discovery'] = {
            'total_datasets': len(self.discovered_datasets),
            'enabled_datasets': len([d for d in self.discovered_datasets.values() if d.get('enabled', True)]),
            'loaded_datasets': len(self.processed_data)
                }
        
        return summary 
    
    @log_performance(logger)
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the neural network"""
        logger.info("Preparing training data")
        
        # This is a placeholder - in practice, you'd create proper training data
        # based on the actual loaded datasets
        X = np.random.rand(100, 15)  # 100 samples, 15 features
        y = np.random.rand(100, 3)   # 100 samples, 3 risk scores
        
        return X, y 