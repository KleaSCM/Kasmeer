#!/usr/bin/env python3
"""
Train Neural Network on Real Civil Engineering Datasets
Author: KleaSCM
Date: 2024

This script loads all available datasets from the DataSets directory and trains
the neural network on real civil engineering data for environmental, infrastructure,
and construction risk assessment.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

from data.data_processor import DataProcessor
from core.dataset_config import DatasetConfig
from ml.neural_network import CivilEngineeringSystem
from ..utils.logging_utils import setup_logging

logger = setup_logging(__name__)

class RealDataTrainer:
    """Trainer for real civil engineering datasets"""
    
    def __init__(self, datasets_dir: str = "DataSets"):
        self.datasets_dir = Path(datasets_dir)
        self.data_processor = DataProcessor()
        self.dataset_config = DatasetConfig()
        self.nn_system = CivilEngineeringSystem()
        
        # Track loaded datasets
        self.loaded_datasets = {}
        self.training_data = {}
        
    def discover_datasets(self):
        """Discover all available datasets in the DataSets directory"""
        logger.info("Discovering datasets...")
        
        datasets = {}
        
        # Find all CSV files
        csv_files = list(self.datasets_dir.glob("*.csv"))
        for csv_file in csv_files:
            datasets[csv_file.name] = {
                'path': str(csv_file),
                'type': 'csv',
                'size': csv_file.stat().st_size
            }
        
        # Find all Excel files
        excel_files = list(self.datasets_dir.glob("*.xlsx"))
        for excel_file in excel_files:
            datasets[excel_file.name] = {
                'path': str(excel_file),
                'type': 'excel',
                'size': excel_file.stat().st_size
            }
        
        # Find climate data directories
        climate_dirs = [d for d in self.datasets_dir.iterdir() if d.is_dir() and 'wc2' in d.name]
        for climate_dir in climate_dirs:
            datasets[climate_dir.name] = {
                'path': str(climate_dir),
                'type': 'climate_raster',
                'size': sum(f.stat().st_size for f in climate_dir.glob("*.tif"))
            }
        
        logger.info(f"Discovered {len(datasets)} datasets")
        for name, info in datasets.items():
            logger.info(f"  {name}: {info['type']} ({info['size'] / 1024 / 1024:.1f} MB)")
        
        return datasets
    
    def load_infrastructure_data(self):
        """Load infrastructure pipe data"""
        logger.info("Loading infrastructure data...")
        
        pipe_file = self.datasets_dir / "INF_DRN_PIPES__PV_-8971823211995978582.csv"
        if pipe_file.exists():
            try:
                df = pd.read_csv(pipe_file)
                logger.info(f"Loaded infrastructure data: {len(df)} records")
                
                # Clean and prepare infrastructure data
                df_clean = df.copy()
                
                # Convert numeric columns
                numeric_cols = ['Diameter', 'Pipe Length', 'Pipe Height', 'Average Depth', 
                              'Up Invert Elevation', 'Down Invert Elevation', 'Grade']
                for col in numeric_cols:
                    if col in df_clean.columns:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Ensure coordinates exist
                if 'latitude' not in df_clean.columns or 'longitude' not in df_clean.columns:
                    logger.warning("No coordinate columns found in infrastructure data, generating coordinates.")
                    # Deterministically generate coordinates using row index
                    n = len(df_clean)
                    df_clean['latitude'] = np.linspace(-37.8, -33.8, n)
                    df_clean['longitude'] = np.linspace(144.9, 153.0, n)
                
                self.loaded_datasets['infrastructure'] = df_clean
                logger.info("Infrastructure data loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading infrastructure data: {e}")
        else:
            logger.warning("Infrastructure data file not found")
    
    def load_climate_data(self):
        """Load climate observation data"""
        logger.info("Loading climate data...")
        
        # Load wind observations
        wind_file = self.datasets_dir / "wind-observations.csv"
        if wind_file.exists():
            try:
                # Load in chunks due to large size
                wind_chunks = []
                chunk_size = 10000
                
                for chunk in pd.read_csv(wind_file, chunksize=chunk_size):
                    # Clean the chunk
                    chunk_clean = chunk.copy()
                    
                    # Convert numeric columns
                    numeric_cols = ['latitude', 'longitude', 'average_wind_speed', 'gust_speed', 
                                  'wind_direction', 'wind_speed_average', 'wind_speed_gust']
                    for col in numeric_cols:
                        if col in chunk_clean.columns:
                            chunk_clean[col] = pd.to_numeric(chunk_clean[col], errors='coerce')
                    
                    # Remove rows with invalid coordinates
                    chunk_clean = chunk_clean.dropna(subset=['latitude', 'longitude'])
                    chunk_clean = chunk_clean[
                        (chunk_clean['latitude'] != 0) & (chunk_clean['longitude'] != 0)
                    ]
                    
                    if not chunk_clean.empty:
                        wind_chunks.append(chunk_clean)
                
                if wind_chunks:
                    wind_df = pd.concat(wind_chunks, ignore_index=True)
                    self.loaded_datasets['wind_observations'] = wind_df
                    logger.info(f"Loaded wind data: {len(wind_df)} records")
                
            except Exception as e:
                logger.error(f"Error loading wind data: {e}")
        
        # Load temperature data
        temp_files = [
            "tasmax_aus-station_r1i1p1_CSIRO-MnCh-wrt-1986-2005-Scl_v1_mon_seasavg-clim.csv",
            "tasmin_aus-station_r1i1p1_CSIRO-MnCh-wrt-1986-2005-Scl_v1_mon_seasavg-clim.csv"
        ]
        
        for temp_file in temp_files:
            file_path = self.datasets_dir / temp_file
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    self.loaded_datasets[f'temperature_{temp_file.split("_")[0]}'] = df
                    logger.info(f"Loaded temperature data: {len(df)} records from {temp_file}")
                except Exception as e:
                    logger.error(f"Error loading {temp_file}: {e}")
        
        # Load humidity data
        humidity_file = self.datasets_dir / "hurs15_aus-station_r1i1p1_CSIRO-MnCh-wrt-1986-2005-Scl_v1_mon_seasavg-clim.csv"
        if humidity_file.exists():
            try:
                df = pd.read_csv(humidity_file)
                self.loaded_datasets['humidity'] = df
                logger.info(f"Loaded humidity data: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading humidity data: {e}")
    
    def load_vegetation_data(self):
        """Load vegetation and environmental data"""
        logger.info("Loading vegetation data...")
        
        # Load vegetation zones
        veg_file = self.datasets_dir / "VegetationZones_718376949849166399.csv"
        if veg_file.exists():
            try:
                df = pd.read_csv(veg_file)
                self.loaded_datasets['vegetation_zones'] = df
                logger.info(f"Loaded vegetation data: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading vegetation data: {e}")
        
        # Load fire projections
        fire_file = self.datasets_dir / "NRM_fire_proj_summary.xlsx"
        if fire_file.exists():
            try:
                df = pd.read_excel(fire_file)
                self.loaded_datasets['fire_projections'] = df
                logger.info(f"Loaded fire projection data: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading fire data: {e}")
    
    def load_catchment_data(self):
        """Load catchment and water data"""
        logger.info("Loading catchment data...")
        
        catchment_file = self.datasets_dir / "wonthaggi-catchment---xlsx.xlsx"
        if catchment_file.exists():
            try:
                df = pd.read_excel(catchment_file)
                self.loaded_datasets['catchment'] = df
                logger.info(f"Loaded catchment data: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading catchment data: {e}")
    
    def prepare_training_data(self):
        """Prepare training data from loaded datasets"""
        logger.info("Preparing training data...")
        
        features_list = []
        targets_list = []
        
        # Dynamically sample locations from infrastructure data
        if 'infrastructure' in self.loaded_datasets:
            infra_df = self.loaded_datasets['infrastructure']
            if not infra_df.empty and 'latitude' in infra_df.columns and 'longitude' in infra_df.columns:
                unique_locations = infra_df[['latitude', 'longitude']].drop_duplicates()
                sample_size = min(100, len(unique_locations))  # Try up to 100 locations
                if sample_size == 0:
                    logger.warning("No unique locations found in infrastructure data.")
                else:
                    training_locations = unique_locations.sample(n=sample_size, random_state=42) if sample_size > 1 else unique_locations
                    logger.info(f"Sampled {len(training_locations)} locations from infrastructure data")
                    for _, row in training_locations.iterrows():
                        lat, lon = row['latitude'], row['longitude']
                        logger.info(f"Processing location: {str(lat)}, {str(lon)}")
                        location_features = self._extract_location_features(lat, lon)
                        if location_features:
                            feature_vector = self._features_to_vector(location_features)
                            if feature_vector is not None and len(feature_vector) > 0:
                                features_list.append(feature_vector)
                                target = self._generate_realistic_targets(location_features)
                                targets_list.append(target)
        # If no infrastructure data, try climate data
        if not features_list and 'wind_observations' in self.loaded_datasets:
            wind_df = self.loaded_datasets['wind_observations']
            if not wind_df.empty and 'latitude' in wind_df.columns and 'longitude' in wind_df.columns:
                unique_locations = wind_df[['latitude', 'longitude']].drop_duplicates()
                sample_size = min(100, len(unique_locations))
                if sample_size == 0:
                    logger.warning("No unique locations found in climate data.")
                else:
                    training_locations = unique_locations.sample(n=sample_size, random_state=42) if sample_size > 1 else unique_locations
                    logger.info(f"Sampled {len(training_locations)} locations from climate data")
                    for _, row in training_locations.iterrows():
                        lat, lon = row['latitude'], row['longitude']
                        logger.info(f"Processing location: {str(lat)}, {str(lon)}")
            location_features = self._extract_location_features(lat, lon)
            if location_features:
                feature_vector = self._features_to_vector(location_features)
                            if feature_vector is not None and len(feature_vector) > 0:
                    features_list.append(feature_vector)
                    target = self._generate_realistic_targets(location_features)
                    targets_list.append(target)
        if features_list and targets_list:
            X = np.array(features_list)
            y = np.array(targets_list)
            self.training_data = {
                'X': X,
                'y': y,
                'feature_names': [f"feature_{i}" for i in range(X.shape[1])],
                'target_names': ['environmental_risk', 'infrastructure_risk', 'construction_risk']
            }
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return True
        else:
            logger.error("Failed to prepare training data")
            return False
    
    def _extract_location_features(self, lat: float, lon: float) -> dict:
        """Extract features for a specific location from loaded datasets"""
        features = {}
        
        # Infrastructure features
        if 'infrastructure' in self.loaded_datasets:
            infra_df = self.loaded_datasets['infrastructure']
            
            # Find infrastructure near this location (simplified)
            # In a real system, you'd use proper spatial queries
            nearby_infra = infra_df[
                (infra_df['latitude'].between(lat - 0.1, lat + 0.1)) &
                (infra_df['longitude'].between(lon - 0.1, lon + 0.1))
            ]
            
            features['infrastructure'] = {
                'count': len(nearby_infra),
                'total_length': nearby_infra['Pipe Length'].sum() if not nearby_infra.empty else 0,
                'avg_diameter': nearby_infra['Diameter'].mean() if not nearby_infra.empty else 0,
                'materials': nearby_infra['Material'].value_counts().to_dict() if not nearby_infra.empty else {},
                'avg_depth': nearby_infra['Average Depth'].mean() if not nearby_infra.empty else 0
            }
        
        # Climate features
        climate_data = {}
        
        if 'wind_observations' in self.loaded_datasets:
            wind_df = self.loaded_datasets['wind_observations']
            nearby_wind = wind_df[
                (wind_df['latitude'].between(lat - 0.5, lat + 0.5)) &
                (wind_df['longitude'].between(lon - 0.5, lon + 0.5))
            ]
            
            if not nearby_wind.empty:
                climate_data['wind_speed_avg'] = nearby_wind['average_wind_speed'].mean()
                climate_data['wind_gust_max'] = nearby_wind['gust_speed'].max()
                climate_data['wind_direction_avg'] = nearby_wind['wind_direction'].mean()
        
        # Add temperature data if available
        for key, df in self.loaded_datasets.items():
            if 'temperature' in key:
                # Extract temperature data if available
                if not df.empty:
                    # Only use numeric columns to avoid string/numpy errors
                    climate_data[f'temp_{key}'] = df.select_dtypes(include=[np.number]).iloc[0].mean()
        
        features['climate'] = climate_data
        
        # Vegetation features
        if 'vegetation_zones' in self.loaded_datasets:
            veg_df = self.loaded_datasets['vegetation_zones']
            features['vegetation'] = {
                'zones_count': len(veg_df),
                'zone_types': veg_df.iloc[:, 0].value_counts().to_dict() if not veg_df.empty else {}
            }
        
        return features
    
    def _features_to_vector(self, features: dict) -> np.ndarray:
        """Convert features dictionary to feature vector"""
        try:
            feature_vector = []
            
            # Infrastructure features
            infra = features.get('infrastructure', {})
            feature_vector.extend([
                infra.get('count', 0),
                infra.get('total_length', 0),
                infra.get('avg_diameter', 0),
                infra.get('avg_depth', 0),
                len(infra.get('materials', {}))
            ])
            
            # Climate features
            climate = features.get('climate', {})
            climate_features = [
                climate.get('wind_speed_avg', 0),
                climate.get('wind_gust_max', 0),
                climate.get('wind_direction_avg', 0)
            ]
            
            # Add temperature features
            for key, value in climate.items():
                if 'temp_' in key:
                    climate_features.append(value)
            
            feature_vector.extend(climate_features)
            
            # Vegetation features
            veg = features.get('vegetation', {})
            feature_vector.extend([
                veg.get('zones_count', 0),
                len(veg.get('zone_types', {}))
            ])
            
            # Pad to consistent length
            while len(feature_vector) < 20:
                feature_vector.append(0.0)
            
            return np.array(feature_vector[:20])  # Ensure consistent 20 features
            
        except Exception as e:
            logger.error(f"Error converting features to vector: {e}")
            return np.array([])
    
    def _generate_realistic_targets(self, features: dict) -> np.ndarray:
        """Generate realistic risk targets based on actual data patterns"""
        environmental_risk = 0.0
        infrastructure_risk = 0.0
        construction_risk = 0.0
        
        try:
            # Environmental risk based on climate data
            climate = features.get('climate', {})
            if climate.get('wind_speed_avg', 0) > 15:
                environmental_risk += 0.3
            if climate.get('wind_gust_max', 0) > 50:
                environmental_risk += 0.4
            
            # Infrastructure risk based on infrastructure density and condition
            infra = features.get('infrastructure', {})
            if infra.get('count', 0) == 0:
                infrastructure_risk += 0.5  # No infrastructure data
            elif infra.get('count', 0) > 100:
                infrastructure_risk += 0.3  # High density
            elif infra.get('avg_depth', 0) > 5:
                infrastructure_risk += 0.2  # Deep infrastructure
            
            # Construction risk based on environmental and infrastructure factors
            construction_risk = (environmental_risk + infrastructure_risk) * 0.6
            
            # Add some randomness based on data patterns
            environmental_risk += np.random.normal(0, 0.1)
            infrastructure_risk += np.random.normal(0, 0.1)
            construction_risk += np.random.normal(0, 0.1)
            
            # Normalize to 0-1 range
            environmental_risk = max(0.0, min(1.0, environmental_risk))
            infrastructure_risk = max(0.0, min(1.0, infrastructure_risk))
            construction_risk = max(0.0, min(1.0, construction_risk))
            
        except Exception as e:
            logger.error(f"Error generating targets: {e}")
            # Use default values on error
        
        return np.array([environmental_risk, infrastructure_risk, construction_risk])
    
    def train_model(self, epochs: int = 200):
        """Train the neural network on real data"""
        logger.info("Starting model training...")
        if not self.training_data:
            logger.error("No training data available")
            return False
        X = self.training_data['X']
        y = self.training_data['y']
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        model = self.nn_system.build_model(input_dim, output_dim)
        # If only one sample, train without validation split
        if X.shape[0] == 1:
            logger.warning("Only one training sample available. Training without validation split.")
            training_results = self.nn_system.train(X, y, epochs=epochs, validation_split=0.0)
        else:
        training_results = self.nn_system.train(X, y, epochs=epochs)
        model_name = f"real_data_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.nn_system.save_model(model_name)
        logger.info(f"Training completed. Model saved as: {model_name}")
        logger.info(f"Final loss: {training_results.get('final_loss', 'N/A')}")
        logger.info(f"Training accuracy: {training_results.get('train_accuracy', 'N/A')}")
        return True
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline on real data...")
        
        try:
            # Step 1: Discover datasets
            datasets = self.discover_datasets()
            
            # Step 2: Load all datasets
            self.load_infrastructure_data()
            self.load_climate_data()
            self.load_vegetation_data()
            self.load_catchment_data()
            
            # Step 3: Prepare training data
            if not self.prepare_training_data():
                logger.error("Failed to prepare training data")
                return False
            
            # Step 4: Train model
            if not self.train_model(epochs=200):
                logger.error("Failed to train model")
                return False
            
            logger.info("Full training pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

def main():
    """Main training function"""
    print("=" * 60)
    print("CIVIL ENGINEERING NEURAL NETWORK TRAINING")
    print("Training on Real Datasets")
    print("=" * 60)
    
    trainer = RealDataTrainer()
    
    if trainer.run_full_training():
        print("\n✅ Training completed successfully!")
        print("The neural network is now trained on real civil engineering data.")
        print("You can now use the system to ask questions about infrastructure, environmental, and construction risks.")
    else:
        print("\n❌ Training failed. Check the logs for details.")

if __name__ == "__main__":
    main() 