# Author: KleaSCM
# Date: 2024
# Neural Network Module
# Description: Handles the core machine learning functionality for civil engineering predictions

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from utils.logging_utils import setup_logging, log_performance
import os

logger = setup_logging(__name__)

class CivilEngineeringNN(nn.Module):
    # Neural Network for Civil Engineering Predictions
    
    def __init__(self, input_dim: int, output_dim: int = 3):
        super(CivilEngineeringNN, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden layers
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(64, output_dim),
            nn.Sigmoid()  # Risk scores between 0-1
        )
        
        self.output_dim = output_dim
        self.input_dim = input_dim

class CivilEngineeringSystem:
    # System wrapper for the neural network
    
    @log_performance(logger)
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.output_names = ['environmental_risk', 'infrastructure_risk', 'construction_risk']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initialized CivilEngineeringSystem with model_dir={model_dir}, device={self.device}")
        
    @log_performance(logger)
    def build_model(self, input_dim: int, output_dim: int = 3) -> CivilEngineeringNN:
        # Build the neural network architecture
        model = CivilEngineeringNN(input_dim, output_dim)
        self.model = model.to(self.device)
        logger.info(f"Built model: input_dim={input_dim}, output_dim={output_dim}")
        return self.model
    
    def prepare_features(self, data_processor) -> Tuple[np.ndarray, np.ndarray]:
        # Prepare features from the data processor
        
        # Get infrastructure data
        infra_df = data_processor.processed_data.get('infrastructure', pd.DataFrame())
        
        if infra_df.empty:
            logger.warning("No infrastructure data available")
            return np.array([]), np.array([])
        
        # Create features
        features = []
        targets = []
        
        # Sample locations for training
        sample_locations = [
            (-37.8136, 144.9631),  # Melbourne
            (-33.8688, 151.2093),  # Sydney
            (-27.4698, 153.0251),  # Brisbane
        ]
        
        for lat, lon in sample_locations:
            # Extract features at this location
            location_features = data_processor.extract_features_at_location(lat, lon)
            
            # Convert to feature vector
            feature_vector = self._features_to_vector(location_features)
            
            if feature_vector is not None:
                features.append(feature_vector)
                
                # Create synthetic targets (in practice, these would come from historical data)
                # TODO: Implement real historical risk data integration
                # TODO: Add risk assessment from engineering standards
                target = np.random.rand(3)  # 3 risk scores
                targets.append(target)
        
        if not features:
            logger.warning("No valid features extracted")
            return np.array([]), np.array([])
        
        X = np.array(features)
        y = np.array(targets)
        
        # Store feature names for later use
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        return X, y
    
    def _features_to_vector(self, features: Dict) -> Optional[np.ndarray]:
        """Convert features dictionary to feature vector"""
        try:
            feature_vector = []
            
            # Infrastructure features
            infra = features.get('infrastructure', {})
            feature_vector.extend([
                infra.get('count', 0),
                infra.get('total_length', 0),
                len(infra.get('materials', {})),
                np.mean(list(infra.get('diameters', {}).values())) if infra.get('diameters') else 0
            ])
            
            # Climate features
            climate = features.get('climate', {})
            for var in ['precipitation', 'temperature_avg', 'temperature_max', 'temperature_min', 'solar_radiation']:
                feature_vector.append(climate.get(var, 0) or 0)
            
            # Vegetation features
            veg = features.get('vegetation', {})
            feature_vector.extend([
                veg.get('zones_count', 0),
                len(veg.get('zone_types', []))
            ])
            
            # Pad or truncate to ensure consistent length
            target_length = 15  # Adjust based on your needs
            if len(feature_vector) < target_length:
                feature_vector.extend([0] * (target_length - len(feature_vector)))
            else:
                feature_vector = feature_vector[:target_length]
            
            return np.array(feature_vector)
            
        except Exception as e:
            logger.error(f"Error converting features to vector: {e}")
            return None
    
    @log_performance(logger)
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 100) -> Dict:
        """Train the neural network"""
        
        if X.size == 0 or y.size == 0:
            logger.error("No training data available")
            return {}
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        
        # Scale the targets
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.model = self.build_model(X_train.shape[1], y_train.shape[1])
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            final_outputs = self.model(X_val_tensor)
            
            # Convert back to original scale for metrics
            y_pred_original = self.scaler_y.inverse_transform(final_outputs.cpu().numpy())
            y_val_original = self.scaler_y.inverse_transform(y_val_scaled)
            
            metrics = {
                'mse': mean_squared_error(y_val_original, y_pred_original),
                'mae': mean_absolute_error(y_val_original, y_pred_original),
                'r2': r2_score(y_val_original, y_pred_original),
                'epochs_trained': epoch + 1,
                'final_val_loss': val_loss.item()
            }
        
        logger.info(f"Training completed. Final metrics: {metrics}")
        return metrics
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Scale features
        features_scaled = self.scaler_X.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(features_tensor)
        
        # Convert back to original scale
        predictions = self.scaler_y.inverse_transform(predictions_scaled.cpu().numpy())
        
        return predictions
    
    def predict_at_location(self, lat: float, lon: float, data_processor) -> Dict:
        """Predict risks at a specific location"""
        try:
            # Extract features at location
            features = data_processor.extract_features_at_location(lat, lon)
            
            # Convert to feature vector
            feature_vector = self._features_to_vector(features)
            
            if feature_vector is None:
                return {
                    'error': 'Could not extract features for location',
                    'environmental_risk': 0.0,
                    'infrastructure_risk': 0.0,
                    'construction_risk': 0.0,
                    'overall_risk': 0.0
                }
            
            # Make prediction
            if self.model is not None:
                prediction = self.predict(feature_vector.reshape(1, -1))[0]
                
                return {
                    'environmental_risk': float(prediction[0]),
                    'infrastructure_risk': float(prediction[1]),
                    'construction_risk': float(prediction[2]),
                    'overall_risk': float(np.mean(prediction)),
                    'risk_factors': self._identify_risk_factors(features, prediction)
                }
            else:
                # TODO: Implement fallback risk assessment when model not available
                # TODO: Add rule-based risk calculation
                return {
                    'environmental_risk': 0.5,
                    'infrastructure_risk': 0.5,
                    'construction_risk': 0.5,
                    'overall_risk': 0.5,
                    'risk_factors': ['Model not trained - using default values']
                }
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'error': f'Prediction error: {str(e)}',
                'environmental_risk': 0.0,
                'infrastructure_risk': 0.0,
                'construction_risk': 0.0,
                'overall_risk': 0.0
            }
    
    def _identify_risk_factors(self, features: Dict, prediction: np.ndarray) -> List[str]:
        """Identify key risk factors based on features and prediction"""
        # TODO: Implement sophisticated risk factor identification
        # TODO: Add industry-specific risk assessment criteria
        # TODO: Include regulatory compliance factors
        
        risk_factors = []
        
        # Infrastructure-based factors
        infra = features.get('infrastructure', {})
        if infra.get('count', 0) == 0:
            risk_factors.append("No existing infrastructure data")
        elif infra.get('count', 0) > 50:
            risk_factors.append("High infrastructure density")
        
        # Climate-based factors
        climate = features.get('climate', {})
        if climate.get('precipitation', 0) > 80:
            risk_factors.append("High precipitation area")
        
        # Vegetation-based factors
        veg = features.get('vegetation', {})
        if veg.get('zones_count', 0) > 5:
            risk_factors.append("Multiple vegetation zones")
        
        return risk_factors
    
    @log_performance(logger)
    def save_model(self, model_name: str = "civil_engineering_nn"):
        """Save the trained model and scalers"""
        try:
            # Save PyTorch model
            model_path = self.model_dir / f"{model_name}.pth"
            if self.model is not None:
                torch.save(self.model.state_dict(), model_path)
            
            # Save scalers
            scaler_path = self.model_dir / f"{model_name}_scalers.joblib"
            joblib.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'feature_names': self.feature_names,
                'output_names': self.output_names
            }, scaler_path)
            
            logger.info(f"Model saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    @log_performance(logger)
    def load_model(self, model_name: str = "civil_engineering_nn"):
        """Load a trained model and scalers"""
        try:
            # Load scalers first
            scaler_path = self.model_dir / f"{model_name}_scalers.joblib"
            if scaler_path.exists():
                scalers = joblib.load(scaler_path)
                self.scaler_X = scalers['scaler_X']
                self.scaler_y = scalers['scaler_y']
                self.feature_names = scalers['feature_names']
                self.output_names = scalers['output_names']
            
            # Load PyTorch model
            model_path = self.model_dir / f"{model_name}.pth"
            if model_path.exists():
                # TODO: Implement proper model architecture loading
                # TODO: Add model version compatibility checks
                input_dim = len(self.feature_names) if self.feature_names else 15
                output_dim = len(self.output_names) if self.output_names else 3
                
                self.model = self.build_model(input_dim, output_dim)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                
                logger.info(f"Model loaded from {self.model_dir}")
                return True
            else:
                logger.warning("Model file not found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    @log_performance(logger)
    def get_model_summary(self) -> Dict:
        """Get summary of the model"""
        summary = {
            'model_loaded': self.model is not None,
            'device': str(self.device),
            'feature_count': len(self.feature_names),
            'output_count': len(self.output_names),
            'feature_names': self.feature_names,
            'output_names': self.output_names
        }
        
        if self.model is not None:
            # TODO: Add model architecture summary
            # TODO: Include model performance metrics
            summary['model_architecture'] = str(self.model)
        
        return summary 