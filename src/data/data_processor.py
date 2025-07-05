
# Author: KleaSCM
# Date: 2024
# Data Processor Module
# Description: Handles loading, cleaning, and preprocessing of civil engineering datasets

import os
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
from utils.logging_utils import setup_logging, log_performance
warnings.filterwarnings('ignore')

logger = setup_logging(__name__)

class DataProcessor:
    # Main data processor for civil engineering datasets
    
    @log_performance(logger)
    def __init__(self, data_dir: str = "DataSets"):
        self.data_dir = Path(data_dir)
        self.processed_data = {}
        logger.info(f"Initialized DataProcessor with data_dir={data_dir}")
        
    def load_infrastructure_data(self) -> pd.DataFrame:
        # Load and process infrastructure data (drainage pipes, etc.)
        try:
            # Load the main infrastructure CSV
            infra_file = self.data_dir / "INF_DRN_PIPES__PV_-8971823211995978582.csv"
            if infra_file.exists():
                df = pd.read_csv(infra_file)
                
                # Clean and process the data
                df = self._clean_infrastructure_data(df)
                self.processed_data['infrastructure'] = df
                logger.info(f"Loaded infrastructure data: {len(df)} records")
                return df
            else:
                logger.warning("Infrastructure file not found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading infrastructure data: {e}")
            return pd.DataFrame()
    
    def _clean_infrastructure_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize infrastructure data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Convert numeric columns
        numeric_cols = ['Diameter', 'Pipe Length', 'Pipe Height', 'Average Depth', 
                       'Up Invert Elevation', 'Down Invert Elevation', 'Grade']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create a unique identifier for each pipe
        df['pipe_id'] = df['Pipe Unit ID'].fillna('UNKNOWN')
        
        # Since we don't have actual coordinates, we'll create a more realistic distribution
        # based on pipe IDs to simulate a real network
        np.random.seed(42)  # For reproducible results
        df['estimated_lat'] = np.random.uniform(-38.5, -37.5, len(df))  # Victoria, Australia
        df['estimated_lon'] = np.random.uniform(144.5, 145.5, len(df))
        
        # Add some clustering around major areas
        major_areas = [
            (-37.8136, 144.9631),  # Melbourne CBD
            (-37.8500, 145.0000),  # Eastern suburbs
            (-37.7500, 144.9000),  # Western suburbs
        ]
        
        # Assign some pipes to major areas
        for i in range(min(100, len(df))):
            area = major_areas[i % len(major_areas)]
            df.iloc[i, df.columns.get_loc('estimated_lat')] = area[0] + np.random.normal(0, 0.01)
            df.iloc[i, df.columns.get_loc('estimated_lon')] = area[1] + np.random.normal(0, 0.01)
        
        return df
    
    def load_vegetation_data(self) -> pd.DataFrame:
        """Load vegetation zones data"""
        try:
            veg_file = self.data_dir / "VegetationZones_718376949849166399.csv"
            if veg_file.exists():
                df = pd.read_csv(veg_file)
                self.processed_data['vegetation'] = df
                logger.info(f"Loaded vegetation data: {len(df)} zones")
                return df
            else:
                logger.warning("Vegetation file not found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading vegetation data: {e}")
            return pd.DataFrame()
    
    def load_climate_data(self) -> Dict[str, np.ndarray]:
        """Load WorldClim climate data"""
        climate_data = {}
        
        # Define climate variables and their directories
        climate_vars = {
            'precipitation': 'wc2.1_10m_prec',
            'temperature_avg': 'wc2.1_10m_tavg',
            'temperature_max': 'wc2.1_10m_tmax',
            'temperature_min': 'wc2.1_10m_tmin',
            'solar_radiation': 'wc2.1_10m_srad'
        }
        
        for var_name, dir_name in climate_vars.items():
            var_dir = self.data_dir / dir_name
            if var_dir.exists():
                try:
                    # Load the first month as a sample (we'll need to implement full loading)
                    # TODO: Implement full climate data loading for all months
                    # TODO: Add climate data interpolation for specific coordinates
                    sample_file = list(var_dir.glob("*.tif"))[0]
                    with rasterio.open(sample_file) as src:
                        climate_data[var_name] = {
                            'data': src.read(1),
                            'transform': src.transform,
                            'crs': src.crs,
                            'bounds': src.bounds
                        }
                    logger.info(f"Loaded {var_name} climate data")
                except Exception as e:
                    logger.error(f"Error loading {var_name}: {e}")
        
        self.processed_data['climate'] = climate_data
        return climate_data
    
    def load_wind_data(self) -> pd.DataFrame:
        """Load wind observations data"""
        try:
            wind_file = self.data_dir / "wind-observations.csv"
            if wind_file.exists():
                # Read only first few rows to understand structure
                # TODO: Implement full wind data loading
                # TODO: Add wind data analysis and processing
                df = pd.read_csv(wind_file, nrows=1000)
                self.processed_data['wind'] = df
                logger.info(f"Loaded wind data sample: {len(df)} records")
                return df
            else:
                logger.warning("Wind data file not found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading wind data: {e}")
            return pd.DataFrame()
    
    def create_spatial_index(self) -> Dict[str, gpd.GeoDataFrame]:
        """Create spatial indexes for all data"""
        spatial_data = {}
        
        # Create spatial index for infrastructure
        if 'infrastructure' in self.processed_data:
            infra_df = self.processed_data['infrastructure']
            if not infra_df.empty and 'estimated_lat' in infra_df.columns:
                geometry = [Point(xy) for xy in zip(infra_df['estimated_lon'], infra_df['estimated_lat'])]
                spatial_data['infrastructure'] = gpd.GeoDataFrame(infra_df, geometry=geometry, crs="EPSG:4326")
        
        # Create spatial index for vegetation zones
        if 'vegetation' in self.processed_data:
            veg_df = self.processed_data['vegetation']
            if not veg_df.empty:
                # Only create GeoDataFrame if geometry column exists
                if 'geometry' in veg_df.columns:
                    spatial_data['vegetation'] = gpd.GeoDataFrame(veg_df, geometry='geometry', crs="EPSG:4326")
                else:
                    spatial_data['vegetation'] = veg_df  # Store as DataFrame if no geometry
        
        self.processed_data['spatial'] = spatial_data
        return spatial_data
    
    def extract_features_at_location(self, lat: float, lon: float, radius_km: float = 1.0) -> Dict:
        """Extract all features at a specific location"""
        features = {}
        
        # Extract infrastructure features
        if 'spatial' in self.processed_data and 'infrastructure' in self.processed_data['spatial']:
            infra_gdf = self.processed_data['spatial']['infrastructure']
            point = Point(lon, lat)
            buffer = point.buffer(radius_km / 111.0)  # Approximate km to degrees
            
            nearby_infra = infra_gdf[infra_gdf.geometry.within(buffer)]
            features['infrastructure'] = {
                'count': len(nearby_infra),
                'materials': nearby_infra['Material'].value_counts().to_dict() if 'Material' in nearby_infra.columns else {},
                'diameters': nearby_infra['Diameter'].describe().to_dict() if 'Diameter' in nearby_infra.columns else {},
                'total_length': nearby_infra['Pipe Length'].sum() if 'Pipe Length' in nearby_infra.columns else 0
            }
        
        # Extract climate features
        if 'climate' in self.processed_data:
            features['climate'] = self._extract_climate_features(lat, lon)
        
        # Extract vegetation features
        if 'vegetation' in self.processed_data:
            features['vegetation'] = self._extract_vegetation_features(lat, lon)
        
        return features
    
    def _extract_climate_features(self, lat: float, lon: float) -> Dict:
        """Extract climate features at specific location"""
        # TODO: Implement proper climate data extraction from raster files
        # TODO: Add interpolation for coordinates not exactly on grid
        # TODO: Include seasonal climate patterns
        
        climate_features = {}
        if 'climate' in self.processed_data:
            for var_name, var_data in self.processed_data['climate'].items():
                # For now, return sample values
                # In practice, you'd interpolate from the raster data
                climate_features[var_name] = np.random.uniform(0, 100)  # Placeholder
        
        return climate_features
    
    def _extract_vegetation_features(self, lat: float, lon: float) -> Dict:
        """Extract vegetation features at specific location"""
        # TODO: Implement proper vegetation zone intersection
        # TODO: Add vegetation density analysis
        # TODO: Include protected species information
        
        veg_features = {
            'zones_count': 0,
            'zone_types': [],
            'vegetation_density': 'unknown'
        }
        
        if 'vegetation' in self.processed_data:
            veg_df = self.processed_data['vegetation']
            if not veg_df.empty:
                # Simple count for now
                veg_features['zones_count'] = len(veg_df)
                if 'Zone_Type' in veg_df.columns:
                    veg_features['zone_types'] = veg_df['Zone_Type'].unique().tolist()
        
        return veg_features
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the neural network"""
        # TODO: Implement comprehensive training data preparation
        # TODO: Add feature engineering
        # TODO: Include data validation and quality checks
        
        # This is a placeholder - in practice, you'd create proper training data
        X = np.random.rand(100, 15)  # 100 samples, 15 features
        y = np.random.rand(100, 3)   # 100 samples, 3 risk scores
        
        return X, y
    
    def get_data_summary(self) -> Dict:
        """Get summary of all loaded data"""
        summary = {}
        
        # Infrastructure data summary
        if 'infrastructure' in self.processed_data:
            infra_df = self.processed_data['infrastructure']
            if not infra_df.empty:
                summary['infrastructure'] = {
                    'rows': len(infra_df),
                    'columns': list(infra_df.columns),
                    'loaded': True,
                    'memory_usage': infra_df.memory_usage(deep=True).sum()
                }
            else:
                summary['infrastructure'] = {'rows': 0, 'loaded': False}
        else:
            summary['infrastructure'] = {'rows': 0, 'loaded': False}
        
        # Vegetation data summary
        if 'vegetation' in self.processed_data:
            veg_df = self.processed_data['vegetation']
            if not veg_df.empty:
                summary['vegetation'] = {
                    'rows': len(veg_df),
                    'columns': list(veg_df.columns),
                    'loaded': True,
                    'memory_usage': veg_df.memory_usage(deep=True).sum()
                }
            else:
                summary['vegetation'] = {'rows': 0, 'loaded': False}
        else:
            summary['vegetation'] = {'rows': 0, 'loaded': False}
        
        # Climate data summary
        if 'climate' in self.processed_data:
            climate_data = self.processed_data['climate']
            summary['climate'] = {
                'variables': list(climate_data.keys()),
                'loaded': len(climate_data) > 0,
                'memory_usage': 0  # TODO: Calculate actual memory usage
            }
        else:
            summary['climate'] = {'variables': [], 'loaded': False}
        
        # Wind data summary
        if 'wind' in self.processed_data:
            wind_df = self.processed_data['wind']
            if not wind_df.empty:
                summary['wind'] = {
                    'rows': len(wind_df),
                    'columns': list(wind_df.columns),
                    'loaded': True,
                    'memory_usage': wind_df.memory_usage(deep=True).sum()
                }
            else:
                summary['wind'] = {'rows': 0, 'loaded': False}
        else:
            summary['wind'] = {'rows': 0, 'loaded': False}
        
        return summary 