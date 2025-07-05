# Author: KleaSCM
# Date: 2024
# Description: Dataset validation module for Kasmeer civil engineering system

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from ..utils.logging_utils import setup_logging, log_performance

logger = setup_logging(__name__)

class DatasetValidator:
    # Dataset validation for civil engineering data
    
    @log_performance(logger)
    def __init__(self, data_dir: str = "DataSets"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        logger.info(f"Initialized DatasetValidator with data_dir={data_dir}")
        
    @log_performance(logger)
    def validate_all_datasets(self) -> Dict[str, Dict]:
        # Validate all datasets in the data directory
        logger.info("Starting validation of all datasets")
        validation_results = {}
        
        # Validate infrastructure data
        logger.debug("Validating infrastructure data")
        validation_results['infrastructure'] = self.validate_infrastructure_data()
        
        # Validate vegetation data
        logger.debug("Validating vegetation data")
        validation_results['vegetation'] = self.validate_vegetation_data()
        
        # Validate climate data
        logger.debug("Validating climate data")
        validation_results['climate'] = self.validate_climate_data()
        
        # Validate wind data
        logger.debug("Validating wind data")
        validation_results['wind'] = self.validate_wind_data()
        
        self.validation_results = validation_results
        
        # Log summary
        valid_count = sum(1 for result in validation_results.values() if result['valid'])
        total_errors = sum(len(result['errors']) for result in validation_results.values())
        total_warnings = sum(len(result['warnings']) for result in validation_results.values())
        
        logger.info(f"Validation completed: {valid_count}/{len(validation_results)} datasets valid, {total_errors} errors, {total_warnings} warnings")
        
        return validation_results
    
    @log_performance(logger)
    def validate_infrastructure_data(self) -> Dict:
        # Validate infrastructure pipeline data
        logger.debug("Validating infrastructure data")
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_path': None,
            'record_count': 0
        }
        
        try:
            # Look for infrastructure files
            infra_files = list(self.data_dir.glob("*INF_DRN_PIPES*"))
            logger.debug(f"Found {len(infra_files)} infrastructure files")
            
            if not infra_files:
                result['errors'].append("No infrastructure data files found")
                logger.warning("No infrastructure data files found")
                return result
            
            file_path = infra_files[0]
            result['file_path'] = str(file_path)
            logger.debug(f"Using infrastructure file: {file_path}")
            
            # Load and validate data
            df = pd.read_csv(file_path)
            result['record_count'] = len(df)
            logger.debug(f"Loaded {len(df)} infrastructure records")
            
            # Check required columns for infrastructure data
            required_cols = ['Pipe Type', 'Diameter', 'Pipe Length', 'Material']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                result['errors'].append(f"Missing required columns: {missing_cols}")
                logger.error(f"Missing required columns in infrastructure data: {missing_cols}")
            else:
                # Check for missing values
                missing_counts = df[required_cols].isnull().sum()
                if missing_counts.sum() > 0:
                    result['warnings'].append(f"Missing values found: {missing_counts.to_dict()}")
                    logger.warning(f"Missing values in infrastructure data: {missing_counts.to_dict()}")
                
                # Basic validation - check for non-empty values
                if 'Diameter' in df.columns:
                    empty_diameters = df['Diameter'].isna().sum()
                    if empty_diameters > 0:
                        result['warnings'].append(f"Found {empty_diameters} records with missing diameters")
                        logger.warning(f"Found {empty_diameters} records with missing diameters")
                
                if 'Pipe Length' in df.columns:
                    empty_lengths = df['Pipe Length'].isna().sum()
                    if empty_lengths > 0:
                        result['warnings'].append(f"Found {empty_lengths} records with missing pipe lengths")
                        logger.warning(f"Found {empty_lengths} records with missing pipe lengths")
                
                result['valid'] = True
                logger.info(f"Infrastructure data validation passed: {len(df)} records")
                
        except Exception as e:
            result['errors'].append(f"Error validating infrastructure data: {e}")
            logger.error(f"Infrastructure validation error: {e}")
        
        return result
    
    @log_performance(logger)
    def validate_vegetation_data(self) -> Dict:
        # Validate vegetation zone data
        logger.debug("Validating vegetation data")
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_path': None,
            'record_count': 0
        }
        
        try:
            # Look for vegetation files
            veg_files = list(self.data_dir.glob("*VegetationZones*"))
            logger.debug(f"Found {len(veg_files)} vegetation files")
            
            if not veg_files:
                result['errors'].append("No vegetation data files found")
                logger.warning("No vegetation data files found")
                return result
            
            file_path = veg_files[0]
            result['file_path'] = str(file_path)
            logger.debug(f"Using vegetation file: {file_path}")
            
            # Load and validate data
            df = pd.read_csv(file_path)
            result['record_count'] = len(df)
            logger.debug(f"Loaded {len(df)} vegetation records")
            
            # Check required columns for vegetation data
            required_cols = ['Zone', 'Type', 'SHAPE_area']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                result['errors'].append(f"Missing required columns: {missing_cols}")
                logger.error(f"Missing required columns in vegetation data: {missing_cols}")
            else:
                # Check for missing values
                missing_counts = df[required_cols].isnull().sum()
                if missing_counts.sum() > 0:
                    result['warnings'].append(f"Missing values found: {missing_counts.to_dict()}")
                    logger.warning(f"Missing values in vegetation data: {missing_counts.to_dict()}")
                
                # Validate shape areas (should be positive numbers)
                if 'SHAPE_area' in df.columns:
                    invalid_areas = df[df['SHAPE_area'] <= 0]
                    if len(invalid_areas) > 0:
                        result['warnings'].append(f"Found {len(invalid_areas)} records with invalid areas (≤0)")
                        logger.warning(f"Found {len(invalid_areas)} records with invalid areas (≤0)")
                
                result['valid'] = True
                logger.info(f"Vegetation data validation passed: {len(df)} records")
                
        except Exception as e:
            result['errors'].append(f"Error validating vegetation data: {e}")
            logger.error(f"Vegetation validation error: {e}")
        
        return result
    
    @log_performance(logger)
    def validate_climate_data(self) -> Dict:
        # Validate climate data files
        logger.debug("Validating climate data")
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_path': None,  # Add this for consistency
            'record_count': 0,  # Add this for consistency
            'files_found': [],
            'total_records': 0
        }
        
        try:
            # Look for climate data files
            climate_patterns = [
                "*tas*", "*prec*", "*srad*", "*hurs*", "*pan-evap*"
            ]
            
            climate_files = []
            for pattern in climate_patterns:
                files = list(self.data_dir.glob(pattern))
                climate_files.extend(files)
                logger.debug(f"Found {len(files)} files matching pattern: {pattern}")
            
            if not climate_files:
                result['errors'].append("No climate data files found")
                logger.warning("No climate data files found")
                return result
            
            result['files_found'] = [str(f) for f in climate_files]
            logger.debug(f"Found {len(climate_files)} climate files: {[f.name for f in climate_files]}")
            
            # Validate each climate file
            for file_path in climate_files[:3]:  # Check first 3 files
                try:
                    df = pd.read_csv(file_path)
                    result['total_records'] += len(df)
                    logger.debug(f"Loaded {len(df)} records from {file_path.name}")
                    
                    # Basic validation
                    if len(df.columns) < 3:
                        result['warnings'].append(f"File {file_path.name} has few columns")
                        logger.warning(f"File {file_path.name} has few columns: {len(df.columns)}")
                        
                except Exception as e:
                    result['warnings'].append(f"Could not read {file_path.name}: {e}")
                    logger.warning(f"Could not read {file_path.name}: {e}")
            
            result['valid'] = True
            result['record_count'] = result['total_records']  # Set record_count for consistency
            logger.info(f"Climate data validation passed: {result['total_records']} total records from {len(climate_files)} files")
                
        except Exception as e:
            result['errors'].append(f"Error validating climate data: {e}")
            logger.error(f"Climate validation error: {e}")
        
        return result
    
    @log_performance(logger)
    def validate_wind_data(self) -> Dict:
        # Validate wind observation data
        logger.debug("Validating wind data")
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_path': None,
            'record_count': 0
        }
        
        try:
            # Look for wind data files
            wind_files = list(self.data_dir.glob("*wind*"))
            logger.debug(f"Found {len(wind_files)} wind files")
            
            if not wind_files:
                result['warnings'].append("No wind data files found (optional)")
                result['valid'] = True  # Wind data is optional
                logger.info("No wind data files found (optional)")
                return result
            
            file_path = wind_files[0]
            result['file_path'] = str(file_path)
            logger.debug(f"Using wind file: {file_path}")
            
            # Load and validate data
            df = pd.read_csv(file_path)
            result['record_count'] = len(df)
            logger.debug(f"Loaded {len(df)} wind records")
            
            result['valid'] = True
            logger.info(f"Wind data validation passed: {len(df)} records")
                
        except Exception as e:
            result['errors'].append(f"Error validating wind data: {e}")
            logger.error(f"Wind validation error: {e}")
        
        return result
    
    @log_performance(logger)
    def get_validation_summary(self) -> Dict:
        # Get summary of validation results
        logger.debug("Generating validation summary")
        if not self.validation_results:
            logger.debug("No validation results found, running validation")
            self.validate_all_datasets()
        
        summary = {
            'overall_valid': True,
            'total_datasets': len(self.validation_results),
            'valid_datasets': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'datasets': {}
        }
        
        for dataset_type, result in self.validation_results.items():
            summary['datasets'][dataset_type] = {
                'valid': result['valid'],
                'errors': len(result['errors']),
                'warnings': len(result['warnings'])
            }
            
            if result['valid']:
                summary['valid_datasets'] += 1
            else:
                summary['overall_valid'] = False
            
            summary['total_errors'] += len(result['errors'])
            summary['total_warnings'] += len(result['warnings'])
        
        logger.info(f"Validation summary: {summary['valid_datasets']}/{summary['total_datasets']} datasets valid, {summary['total_errors']} errors, {summary['total_warnings']} warnings")
        return summary
    
    @log_performance(logger)
    def check_for_new_data(self) -> Dict:
        # Check if there are new datasets that haven't been processed
        # TODO: Implement file modification time checking
        # TODO: Add checksum validation for data integrity
        logger.debug("Checking for new data")
        
        return {
            'new_data_found': False,
            'new_files': [],
            'last_processed': None
        } 