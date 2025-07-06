# Author: KleaSCM
# Date: 2024
# Description: Helper utilities for the Kasmeer civil engineering neural network system

import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from .logging_utils import setup_logging, log_performance

logger = setup_logging(__name__)

@log_performance(logger)
def setup_logging_legacy(log_file: str = "Logs/kasmeer.log", level: str = "INFO") -> logging.Logger:
    # Setup logging configuration
    logger.debug(f"Setting up legacy logging: {log_file}, level={level}")
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Legacy logging setup completed: {log_file}")
    return logging.getLogger(__name__)

@log_performance(logger)
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    # Load configuration from YAML file
    logger.debug(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        # Return default config
        # TODO: Implement more comprehensive default configuration
        # TODO: Add environment-specific configs
        logger.warning(f"Config file {config_path} not found, using default configuration")
        return {
            'data_dir': 'DataSets',
            'model_dir': 'models',
            'logs_dir': 'Logs',
            'visuals_dir': 'visuals',
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validation_split': 0.2
            },
            'model': {
                'input_dim': 15,
                'output_dim': 3,
                'hidden_layers': [128, 256, 128, 64]
            }
        }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

@log_performance(logger)
def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    # Save configuration to YAML file
    logger.debug(f"Saving configuration to: {config_path}")
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

@log_performance(logger)
def validate_coordinates(lat: float, lon: float) -> bool:
    # Validate latitude and longitude coordinates
    logger.debug(f"Validating coordinates: lat={lat}, lon={lon}")
    is_valid = -90 <= lat <= 90 and -180 <= lon <= 180
    if not is_valid:
        logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
    return is_valid

@log_performance(logger)
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Calculate distance between two points using Haversine formula
    logger.debug(f"Calculating distance between ({lat1}, {lon1}) and ({lat2}, {lon2})")
    import math
    
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c
    logger.debug(f"Calculated distance: {distance:.2f} km")
    return distance

@log_performance(logger)
def format_risk_score(score: float) -> str:
    # Format risk score into human-readable string
    logger.debug(f"Formatting risk score: {score}")
    if score >= 0.8:
        return "Very High"
    elif score >= 0.6:
        return "High"
    elif score >= 0.4:
        return "Medium"
    elif score >= 0.2:
        return "Low"
    else:
        return "Very Low"

@log_performance(logger)
def get_risk_color(score: float) -> str:
    # Get color for risk score
    logger.debug(f"Getting risk color for score: {score}")
    if score >= 0.8:
        return "red"
    elif score >= 0.6:
        return "orange"
    elif score >= 0.4:
        return "yellow"
    elif score >= 0.2:
        return "lightgreen"
    else:
        return "green"

@log_performance(logger)
def sanitize_filename(filename: str) -> str:
    # Sanitize filename for safe file operations
    logger.debug(f"Sanitizing filename: {filename}")
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    logger.debug(f"Sanitized filename: {sanitized}")
    return sanitized

@log_performance(logger)
def create_backup(file_path: str, backup_dir: str = "backups") -> str:
    # Create a backup of a file
    logger.info(f"Creating backup of: {file_path}")
    try:
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file {file_path} not found")
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"{source_path.stem}_{timestamp}{source_path.suffix}"
        
        import shutil
        shutil.copy2(source_path, backup_file)
        
        logger.info(f"Backup created: {backup_file}")
        return str(backup_file)
        
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return ""

@log_performance(logger)
def load_json_safe(file_path: str) -> Optional[Dict[str, Any]]:
    # Safely load JSON file with error handling
    logger.debug(f"Loading JSON file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON file loaded successfully: {file_path}")
        return data
    except FileNotFoundError:
        logger.warning(f"JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None

@log_performance(logger)
def save_json_safe(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
    # Safely save data to JSON file with error handling
    logger.debug(f"Saving JSON data to: {file_path}")
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        
        logger.info(f"Data saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False

@log_performance(logger)
def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    # Get comprehensive information about a DataFrame
    logger.debug(f"Getting data info for DataFrame: {len(df)} rows, {len(df.columns)} columns")
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'column_names': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info['numeric_stats'] = df[numeric_cols].describe().to_dict()
        logger.debug(f"Added numeric stats for {len(numeric_cols)} columns")
    
    logger.info(f"Data info generated: {info['rows']} rows, {info['columns']} columns, {info['completeness']:.1f}% complete")
    return info

@log_performance(logger)
def format_file_size(size_bytes: int) -> str:
    # Format file size in human-readable format
    logger.debug(f"Formatting file size: {size_bytes} bytes")
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size_float = float(size_bytes)
    while size_float >= 1024 and i < len(size_names) - 1:
        size_float /= 1024.0
        i += 1
    
    formatted_size = f"{size_float:.1f}{size_names[i]}"
    logger.debug(f"Formatted file size: {formatted_size}")
    return formatted_size

@log_performance(logger)
def check_dependencies() -> Dict[str, bool]:
    # Check if all required dependencies are available
    # TODO: Implement comprehensive dependency checking
    # TODO: Add version compatibility checks
    # TODO: Include optional dependency detection
    logger.info("Checking system dependencies")
    
    dependencies = {
        'pandas': False,
        'numpy': False,
        'torch': False,
        'sklearn': False,
        'rasterio': False,
        'geopandas': False,
        'matplotlib': False,
        'seaborn': False,
        'click': False,
        'rich': False
    }
    
    try:
        import pandas
        dependencies['pandas'] = True
        logger.debug("pandas: ✓")
    except ImportError:
        logger.warning("pandas: ✗")
    
    try:
        import numpy
        dependencies['numpy'] = True
        logger.debug("numpy: ✓")
    except ImportError:
        logger.warning("numpy: ✗")
    
    try:
        import torch
        dependencies['torch'] = True
        logger.debug("torch: ✓")
    except ImportError:
        logger.warning("torch: ✗")
    
    try:
        import sklearn
        dependencies['sklearn'] = True
        logger.debug("sklearn: ✓")
    except ImportError:
        logger.warning("sklearn: ✗")
    
    try:
        import rasterio
        dependencies['rasterio'] = True
        logger.debug("rasterio: ✓")
    except ImportError:
        logger.warning("rasterio: ✗")
    
    try:
        import geopandas
        dependencies['geopandas'] = True
        logger.debug("geopandas: ✓")
    except ImportError:
        logger.warning("geopandas: ✗")
    
    try:
        import matplotlib
        dependencies['matplotlib'] = True
        logger.debug("matplotlib: ✓")
    except ImportError:
        logger.warning("matplotlib: ✗")
    
    try:
        import seaborn
        dependencies['seaborn'] = True
        logger.debug("seaborn: ✓")
    except ImportError:
        logger.warning("seaborn: ✗")
    
    try:
        import click
        dependencies['click'] = True
        logger.debug("click: ✓")
    except ImportError:
        logger.warning("click: ✗")
    
    try:
        import rich
        dependencies['rich'] = True
        logger.debug("rich: ✓")
    except ImportError:
        logger.warning("rich: ✗")
    
    available_count = sum(dependencies.values())
    total_count = len(dependencies)
    logger.info(f"Dependency check completed: {available_count}/{total_count} available")
    
    return dependencies

@log_performance(logger)
def get_system_info() -> Dict[str, Any]:
    # Get system information for debugging and logging
    # TODO: Add more comprehensive system information
    # TODO: Include hardware specifications
    # TODO: Add performance metrics
    logger.debug("Getting system information")
    
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'cpu_count': psutil.cpu_count()
    }
    
    logger.info(f"System info: {info['platform']}, Python {info['python_version']}, {info['cpu_count']} CPUs")
    return info

@log_performance(logger)
def validate_data_quality(df: pd.DataFrame, required_columns: Optional[list] = None) -> Dict[str, Any]:
    # Validate data quality and completeness
    # TODO: Implement comprehensive data quality validation
    # TODO: Add statistical outlier detection
    # TODO: Include data consistency checks
    logger.debug(f"Validating data quality for DataFrame: {len(df)} rows, {len(df.columns)} columns")
    
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_dataframe': df.empty,
        'duplicate_rows': df.duplicated().sum(),
        'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_cols
        validation_results['is_valid'] = len(missing_cols) == 0
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        logger.warning("DataFrame is empty")
    
    if validation_results['duplicate_rows'] > 0:
        logger.warning(f"Found {validation_results['duplicate_rows']} duplicate rows")
    
    if validation_results['null_percentage'] > 50:
        logger.warning(f"High null percentage: {validation_results['null_percentage']:.1f}%")
    
    logger.info(f"Data quality validation: valid={validation_results['is_valid']}, null_percentage={validation_results['null_percentage']:.1f}%")
    return validation_results

def find_coordinate_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detect latitude and longitude columns in a DataFrame."""
    lat_col, lon_col = None, None
    
    # Case-insensitive search for coordinate columns
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['lat', 'latitude', 'y']):
            lat_col = col
        if any(x in col_lower for x in ['lon', 'lng', 'longitude', 'x']):
            lon_col = col
    
    return lat_col, lon_col

def filter_by_location(df: pd.DataFrame, lat: float, lon: float, radius: float = 0.01, exact_match: bool = False) -> pd.DataFrame:
    """
    Filter DataFrame to rows within a given radius of the specified lat/lon.
    
    Args:
        df: DataFrame to filter
        lat: Latitude of target location
        lon: Longitude of target location
        radius: Radius in degrees (default 0.01 ~1km)
        exact_match: If True, only return exact coordinate matches (for precise locations like houses)
    
    Returns:
        Filtered DataFrame
    """
    import numpy as np
    lat_col, lon_col = find_coordinate_columns(df)
    if lat_col is None or lon_col is None:
        logger.warning(f"No coordinate columns found in dataset. Available columns: {list(df.columns)}")
        return pd.DataFrame()  # No spatial data
    
    try:
        # Convert to numeric, ignoring errors
        lat_vals = pd.to_numeric(df[lat_col], errors='coerce')  # type: ignore
        lon_vals = pd.to_numeric(df[lon_col], errors='coerce')  # type: ignore
        
        # Remove rows with invalid coordinates
        valid_coords = ~(lat_vals.isna() | lon_vals.isna())  # type: ignore
        df_valid = df[valid_coords].copy()
        lat_vals = lat_vals[valid_coords]  # type: ignore
        lon_vals = lon_vals[valid_coords]  # type: ignore
        
        if df_valid.empty:
            logger.warning("No valid coordinate data found after filtering")
            return pd.DataFrame()
        
        atol = 1e-4 if exact_match else radius
        if exact_match:
            # Use np.isclose for robust float comparison
            mask = np.isclose(lat_vals, lat, atol=atol) & np.isclose(lon_vals, lon, atol=atol)  # type: ignore
            logger.info(f"Exact coordinate matching (np.isclose): looking for ({lat}, {lon}) with atol={atol}")
        else:
            # Approximate matching with radius
            mask = np.isclose(lat_vals, lat, atol=radius) & np.isclose(lon_vals, lon, atol=radius)  # type: ignore
            logger.info(f"Approximate matching (np.isclose): looking within {radius} degrees of ({lat}, {lon})")
        
        filtered_df = df_valid[mask].copy()
        logger.info(f"Found {len(filtered_df)} records matching location criteria")
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error in location filtering: {e}")
        return pd.DataFrame()

def filter_by_exact_location(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """
    Filter DataFrame to exact coordinate matches (for precise locations like houses).
    
    Args:
        df: DataFrame to filter
        lat: Exact latitude
        lon: Exact longitude
    
    Returns:
        Filtered DataFrame with exact matches only
    """
    return filter_by_location(df, lat, lon, radius=1e-4, exact_match=True)

def filter_by_approximate_location(df: pd.DataFrame, lat: float, lon: float, radius_km: float = 1.0) -> pd.DataFrame:
    """
    Filter DataFrame to approximate location matches with configurable radius in kilometers.
    
    Args:
        df: DataFrame to filter
        lat: Latitude of target location
        lon: Longitude of target location
        radius_km: Radius in kilometers (default 1.0 km)
    
    Returns:
        Filtered DataFrame
    """
    # Convert km to degrees (approximate: 1 degree ≈ 111 km)
    radius_degrees = radius_km / 111.0
    return filter_by_location(df, lat, lon, radius=radius_degrees, exact_match=False) 