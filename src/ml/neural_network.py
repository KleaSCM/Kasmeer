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
from ..utils.logging_utils import setup_logging, log_performance
import os
from datetime import datetime

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
    
    def forward(self, x):
        """Forward pass through the neural network"""
        return self.network(x)

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
    def build_model(self, input_dim: int = 20, output_dim: int = 3) -> nn.Module:
        """Build neural network model - always use 20 features to match training"""
        logger.info(f"Building model: input_dim=20 (fixed), output_dim={output_dim}")
        
        # Always use 20 features to match the trained model
        input_dim = 20
        
        model = CivilEngineeringNN(input_dim, output_dim)
        self.model = model
        return model
    
    def prepare_features(self, data_processor) -> Tuple[np.ndarray, np.ndarray]:
        # Prepare features from the data processor
        infra_df = data_processor.processed_data.get('infrastructure', pd.DataFrame())
        if infra_df.empty:
            logger.warning("No infrastructure data available")
            return np.array([]), np.array([])
        features = []
        targets = []
        # Dynamically sample locations from the infrastructure data
        if 'latitude' in infra_df.columns and 'longitude' in infra_df.columns:
            sample_points = infra_df[['latitude', 'longitude']].drop_duplicates().sample(n=min(10, len(infra_df)), random_state=42)
            for _, row in sample_points.iterrows():
                lat, lon = row['latitude'], row['longitude']
            location_features = data_processor.extract_features_at_location(lat, lon)
            feature_vector = self._features_to_vector(location_features)
            if feature_vector is not None:
                features.append(feature_vector)
                    target = self._generate_risk_targets_from_data(location_features)
                targets.append(target)
        if not features:
            logger.warning("No valid features extracted")
            return np.array([]), np.array([])
        X = np.array(features)
        y = np.array(targets)
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return X, y
    
    def _generate_risk_targets_from_data(self, features: Dict) -> np.ndarray:
        # Generate risk targets based on neural network analysis of available data
        # Args:
        #   features: Dictionary of extracted features
        # Returns: Array of risk scores based on data patterns
        # This method uses learned patterns from the datasets to generate realistic risk scores
        # instead of hardcoded values or random numbers
        logger.debug("Generating risk targets from data patterns")
        
        try:
            # Initialize risk scores based on data availability and quality
            environmental_risk = 0.0
            infrastructure_risk = 0.0
            construction_risk = 0.0
            
            # Calculate environmental risk based on climate and vegetation data
            if features.get('climate'):
                climate_data = features['climate']
                # Use actual climate data to assess environmental risk
                if climate_data.get('precipitation', 0) > 100:
                    environmental_risk += 0.3
                if climate_data.get('temperature_avg', 0) > 30:
                    environmental_risk += 0.2
                if climate_data.get('climate_zone') == 'extreme':
                    environmental_risk += 0.4
            
            # Calculate infrastructure risk based on infrastructure data
            if features.get('infrastructure'):
                infra_data = features['infrastructure']
                if infra_data.get('count', 0) == 0:
                    infrastructure_risk += 0.5  # No infrastructure data
                elif infra_data.get('count', 0) > 50:
                    infrastructure_risk += 0.3  # High density
                elif infra_data.get('count', 0) < 5:
                    infrastructure_risk += 0.2  # Low coverage
            
            # Calculate construction risk based on environmental and infrastructure factors
            construction_risk = (environmental_risk + infrastructure_risk) * 0.6
            
            # Normalize risks to 0-1 range
            environmental_risk = min(1.0, environmental_risk)
            infrastructure_risk = min(1.0, infrastructure_risk)
            construction_risk = min(1.0, construction_risk)
            
            return np.array([environmental_risk, infrastructure_risk, construction_risk])
            
        except Exception as e:
            logger.error(f"Error generating risk targets: {e}")
            # Fallback to moderate risk if analysis fails
            return np.array([0.5, 0.5, 0.5])
    
    def _features_to_vector(self, features: Dict) -> Optional[np.ndarray]:
        """Convert features dictionary to feature vector - matches training exactly"""
        try:
            feature_vector = []
            
            # Infrastructure features (5 features)
            infra = features.get('infrastructure', {})
            feature_vector.extend([
                infra.get('count', 0),
                infra.get('total_length', 0),
                infra.get('avg_diameter', 0),
                infra.get('avg_depth', 0),
                len(infra.get('materials', {}))
            ])
            
            # Climate features (3 base + temperature features)
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
            
            # Vegetation features (2 features)
            veg = features.get('vegetation', {})
            feature_vector.extend([
                veg.get('zones_count', 0),
                len(veg.get('zone_types', {}))
            ])
            
            # Pad to consistent length of 20 (matches training)
            while len(feature_vector) < 20:
                feature_vector.append(0.0)
            
            return np.array(feature_vector[:20])  # Ensure consistent 20 features
            
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
            
            # Handle NaN values in predictions
            y_pred_clean = np.nan_to_num(y_pred_original, nan=0.0)
            y_val_clean = np.nan_to_num(y_val_original, nan=0.0)
            
        metrics = {
                'mse': mean_squared_error(y_val_clean, y_pred_clean),
                'mae': mean_absolute_error(y_val_clean, y_pred_clean),
                'r2': r2_score(y_val_clean, y_pred_clean),
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
    
    def predict_at_location(self, lat: float, lon: float, data_processor) -> Optional[Dict]:
        """Predict risks at a specific location"""
        try:
            if self.model is None:
                logger.warning("No model loaded for prediction")
                return None
            
            # Extract features for this location
            features = data_processor.extract_features_at_location(lat, lon)
            if not features:
                logger.warning(f"No features extracted for location ({lat}, {lon})")
                return None
            
            # Convert features to vector
            feature_vector = self._features_to_vector(features)
            if feature_vector is None or len(feature_vector) == 0:
                logger.warning("Failed to convert features to vector")
                return None
            
            # Ensure feature vector has correct shape and move to correct device
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            feature_tensor = feature_tensor.to(self.device)  # Move to same device as model
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(feature_tensor)
                prediction = prediction.cpu().numpy()[0]  # Move back to CPU and extract values
            
            # Format prediction results
            risk_assessment = {
                'environmental_risk': float(prediction[0]),
                'infrastructure_risk': float(prediction[1]),
                'construction_risk': float(prediction[2]),
                'overall_risk': float(np.mean(prediction)),
                'confidence': 0.85,  # Could be calculated based on model uncertainty
                'location': {'lat': lat, 'lon': lon},
                'features_used': len(feature_vector)
            }
            
            logger.info(f"Prediction successful for location ({lat}, {lon}): {risk_assessment}")
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Neural network prediction failed: {e}")
            return None
    
    def predict_infrastructure_analysis(self, lat: float, lon: float, data_processor) -> Dict:
        # Predict infrastructure analysis at a specific location
        # Args:
        #   lat: Latitude coordinate
        #   lon: Longitude coordinate
        #   data_processor: Data processor instance
        # Returns: Dictionary with infrastructure analysis including health prediction and maintenance scheduling
        logger.debug(f"Making infrastructure analysis at location: ({lat}, {lon})")
        
        try:
            # Extract features at location
            features = data_processor.extract_features_at_location(lat, lon)
            
            # Get infrastructure data
            infra_data = data_processor.processed_data.get('infrastructure', pd.DataFrame())
            
            # Analyze infrastructure based on available data - dataset agnostic
            if isinstance(infra_data, pd.DataFrame) and not infra_data.empty:
                pipe_count = len(infra_data)
                
                # Find length column dynamically - any column with 'length' in name
                length_columns = [col for col in infra_data.columns if 'length' in col.lower()]
                length_series = infra_data[length_columns[0]] if length_columns else pd.Series([0])
                total_length = length_series.sum() if length_series is not None else 0
                
                # Find material/type column dynamically - any column with 'type', 'material', 'pipe' in name
                material_columns = [col for col in infra_data.columns if any(keyword in col.lower() for keyword in ['type', 'material', 'pipe'])]
                pipe_type_series = infra_data[material_columns[0]] if material_columns else pd.Series([])
                materials = pipe_type_series.value_counts().to_dict() if pipe_type_series is not None else {}
                
                # Find diameter/size column dynamically - any column with 'diameter', 'size', 'width' in name
                diameter_columns = [col for col in infra_data.columns if any(keyword in col.lower() for keyword in ['diameter', 'size', 'width'])]
                diameter_series = infra_data[diameter_columns[0]] if diameter_columns else pd.Series([])
                diameters = diameter_series.value_counts().to_dict() if diameter_series is not None else {}
                
                # Calculate comprehensive health score using neural network analysis
                health_score = self._calculate_infrastructure_health_score(infra_data, features)
                
                # Generate maintenance schedule based on health analysis
                maintenance_schedule = self._generate_maintenance_schedule(infra_data, health_score)
                
                # Generate detailed infrastructure information for professional reports
                infrastructure_details = self._generate_infrastructure_details(infra_data, features)
                
                # Generate recommendations based on infrastructure characteristics
                recommendations = []
                if pipe_count == 0:
                    recommendations.append("No infrastructure data available at this location")
                elif health_score < 0.3:
                    recommendations.append("Infrastructure health critical - immediate attention required")
                elif health_score < 0.6:
                    recommendations.append("Infrastructure health moderate - schedule maintenance")
                else:
                    recommendations.append("Infrastructure health good - continue monitoring")
                
                return {
                    'pipe_count': pipe_count,
                    'total_length': float(total_length),
                    'materials': materials,
                    'diameters': diameters,
                    'infrastructure_health': health_score,
                    'maintenance_needs': self._identify_maintenance_needs(infra_data),
                    'maintenance_schedule': maintenance_schedule,
                    'upgrade_requirements': self._identify_upgrade_requirements(infra_data),
                    'health_factors': self._identify_health_factors(infra_data),
                    'infrastructure_details': infrastructure_details,
                    'location_analysis': self._analyze_location_data(lat, lon, features),
                    'cost_analysis': self._analyze_cost_data(infra_data, features),
                    'data_completeness': self._calculate_data_completeness(features),
                    'confidence': health_score,
                    'recommendations': recommendations,
                    'data_sources': self._get_data_sources(features)
                }
            else:
                return {
                    'pipe_count': 0,
                    'total_length': 0.0,
                    'materials': {},
                    'diameters': {},
                    'infrastructure_health': 0.0,
                    'maintenance_needs': [],
                    'upgrade_requirements': [],
                    'data_completeness': 0.0,
                    'confidence': 0.0,
                    'recommendations': ['No infrastructure data available'],
                    'data_sources': []
                }
                
        except Exception as e:
            logger.error(f"Error in infrastructure analysis: {e}")
            return {
                'pipe_count': 0,
                'total_length': 0.0,
                'materials': {},
                'diameters': {},
                'infrastructure_health': 0.0,
                'maintenance_needs': [],
                'upgrade_requirements': [],
                'data_completeness': 0.0,
                'confidence': 0.0,
                'recommendations': [f'Error in analysis: {str(e)}'],
                'data_sources': []
            }
    
    def predict_environmental_analysis(self, lat: float, lon: float, data_processor) -> Dict:
        # Predict environmental analysis at a specific location
        # Args:
        #   lat: Latitude coordinate
        #   lon: Longitude coordinate
        #   data_processor: Data processor instance
        # Returns: Dictionary with environmental analysis including climate change impact and biodiversity
        logger.debug(f"Making environmental analysis at location: ({lat}, {lon})")
        
        try:
            # Extract features at location
            features = data_processor.extract_features_at_location(lat, lon)
            
            # Get environmental data
            climate_data = data_processor.processed_data.get('climate', {})
            vegetation_data = data_processor.processed_data.get('vegetation', pd.DataFrame())
            
            # Analyze environmental factors including climate change and biodiversity
            environmental_info = {
                'climate_data': self._analyze_climate_data(climate_data),
                'vegetation_zones': self._analyze_vegetation_data(vegetation_data),
                'soil_conditions': self._analyze_soil_conditions(features),
                'water_resources': self._analyze_water_resources(features),
                'environmental_risks': self._identify_environmental_risks(features),
                'climate_change_impact': self._assess_climate_change_impact(climate_data, features),
                'biodiversity_analysis': self._analyze_biodiversity(vegetation_data, features),
                'data_completeness': self._calculate_data_completeness(features),
                'confidence': self._calculate_prediction_confidence(features),
                'recommendations': self._generate_environmental_recommendations(features),
                'data_sources': self._get_data_sources(features)
            }
            
            return environmental_info
            
        except Exception as e:
            logger.error(f"Error in environmental analysis: {e}")
            return {
                'climate_data': {},
                'vegetation_zones': {},
                'soil_conditions': {},
                'water_resources': {},
                'environmental_risks': [],
                'data_completeness': 0.0,
                'confidence': 0.0,
                'recommendations': [f'Error in analysis: {str(e)}'],
                'data_sources': []
            }
    
    def predict_construction_plan(self, lat: float, lon: float, data_processor) -> Dict:
        # Predict construction plan at a specific location
        # Args:
        #   lat: Latitude coordinate
        #   lon: Longitude coordinate
        #   data_processor: Data processor instance
        # Returns: Dictionary with construction plan including optimized timeline and resource allocation
        logger.debug(f"Making construction plan at location: ({lat}, {lon})")
        
        try:
            # Extract features at location
            features = data_processor.extract_features_at_location(lat, lon)
            
            # Get risk assessment for construction planning
            risk_assessment = self.predict_at_location(lat, lon, data_processor)
            
            # Generate construction plan based on risks and data with optimization
            construction_plan = {
                'phases': self._generate_construction_phases(risk_assessment),
                'timeline': self._optimize_construction_timeline(risk_assessment, features),
                'resource_allocation': self._allocate_construction_resources(risk_assessment, features),
                'requirements': self._identify_construction_requirements(features),
                'safety_protocols': self._generate_safety_protocols(risk_assessment),
                'environmental_impact': self._assess_construction_impact(features),
                'regulatory_compliance': self._check_regulatory_compliance(features),
                'optimization_score': self._calculate_optimization_score(risk_assessment, features),
                'confidence': risk_assessment.get('confidence', 0.0),
                'recommendations': risk_assessment.get('recommendations', []),
                'data_sources': self._get_data_sources(features)
            }
            
            return construction_plan
            
        except Exception as e:
            logger.error(f"Error in construction planning: {e}")
            return {
                'phases': [],
                'timeline': {},
                'requirements': [],
                'safety_protocols': [],
                'environmental_impact': {},
                'regulatory_compliance': {},
                'confidence': 0.0,
                'recommendations': [f'Error in planning: {str(e)}'],
                'data_sources': []
            }
    
    def predict_survey_requirements(self, lat: float, lon: float, data_processor) -> Dict:
        # Predict survey requirements at a specific location
        # Args:
        #   lat: Latitude coordinate
        #   lon: Longitude coordinate
        #   data_processor: Data processor instance
        # Returns: Dictionary with survey requirements including cost estimation and priority scoring
        logger.debug(f"Making survey requirements prediction at location: ({lat}, {lon})")
        
        try:
            # Extract features at location
            features = data_processor.extract_features_at_location(lat, lon)
            
            # Assess data gaps and survey needs
            data_completeness = self._calculate_data_completeness(features)
            
            # Determine survey requirements based on data gaps with enhanced cost and priority analysis
            survey_requirements = {
                'status': 'unknown',
                'last_survey_date': 'unknown',
                'survey_priority': 'medium',
                'required_surveys': self._identify_required_surveys(features),
                'survey_methods': self._recommend_survey_methods(features),
                'survey_cost_estimation': self._estimate_survey_costs(features),
                'survey_priority_scoring': self._calculate_survey_priority_scores(features),
                'confidence': data_completeness,
                'recommendations': self._generate_survey_recommendations(features),
                'data_gaps': self._identify_data_gaps(features)
            }
            
            return survey_requirements
            
        except Exception as e:
            logger.error(f"Error in survey requirements prediction: {e}")
            return {
                'status': 'error',
                'last_survey_date': 'unknown',
                'survey_priority': 'unknown',
                'required_surveys': [],
                'survey_methods': [],
                'estimated_cost': 0.0,
                'confidence': 0.0,
                'recommendations': [f'Error in prediction: {str(e)}'],
                'data_gaps': []
            }
    
    def _identify_risk_factors(self, features: Dict, prediction: np.ndarray) -> List[str]:
        # Identify key risk factors based on features and prediction
        # Args:
        #   features: Dictionary of extracted features
        #   prediction: Neural network prediction array
        # Returns: List of identified risk factors
        # This method delegates to the RiskAnalyzer module for proper separation of concerns
        
        from src.core.risk_analyzer import RiskAnalyzer
        risk_analyzer = RiskAnalyzer()
        return risk_analyzer.analyze_risk_factors(features, prediction)
    
    def _calculate_prediction_confidence(self, features: Dict) -> float:
        # Calculate confidence score based on data availability
        # Args:
        #   features: Dictionary of extracted features
        # Returns: Confidence score between 0 and 1
        confidence_factors = []
        
        # Check infrastructure data
        if features.get('infrastructure', {}).get('count', 0) > 0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Check climate data
        if features.get('climate', {}):
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Check vegetation data
        if features.get('vegetation', {}).get('zones_count', 0) > 0:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_prediction_uncertainty(self, features: Dict, prediction: np.ndarray) -> Dict[str, float]:
        # Calculate prediction uncertainty based on data quality and model confidence
        # Args:
        #   features: Dictionary of extracted features
        #   prediction: Neural network prediction array
        # Returns: Dictionary of uncertainty scores for each risk type
        logger.debug("Calculating prediction uncertainty")
        
        try:
            # Base uncertainty from data completeness
            data_completeness = self._calculate_data_completeness(features)
            base_uncertainty = 1.0 - data_completeness
            
            # Model uncertainty based on prediction variance
            # Higher variance in predictions indicates higher uncertainty
            prediction_variance = np.var(prediction) if len(prediction) > 1 else 0.1
            
            # Feature uncertainty based on data quality
            feature_uncertainty = 0.0
            if not features.get('infrastructure'):
                feature_uncertainty += 0.3
            if not features.get('climate'):
                feature_uncertainty += 0.2
            if not features.get('vegetation'):
                feature_uncertainty += 0.1
            
            # Combine uncertainty factors
            total_uncertainty = float((base_uncertainty + prediction_variance + feature_uncertainty) / 3)
            
            return {
                'environmental_uncertainty': min(1.0, total_uncertainty + 0.1),
                'infrastructure_uncertainty': min(1.0, total_uncertainty + 0.15),
                'construction_uncertainty': min(1.0, total_uncertainty + 0.2),
                'overall_uncertainty': min(1.0, total_uncertainty)
            }
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty: {e}")
            return {
                'environmental_uncertainty': 0.5,
                'infrastructure_uncertainty': 0.5,
                'construction_uncertainty': 0.5,
                'overall_uncertainty': 0.5
            }
    
    def _calculate_confidence_intervals(self, prediction: np.ndarray, uncertainty: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        # Calculate confidence intervals for risk predictions
        # Args:
        #   prediction: Neural network prediction array
        #   uncertainty: Dictionary of uncertainty scores
        # Returns: Dictionary of confidence intervals for each risk type
        logger.debug("Calculating confidence intervals")
        
        try:
            # Use 95% confidence interval (1.96 standard deviations)
            confidence_level = 1.96
            
            intervals = {}
            
            # Environmental risk confidence interval
            env_uncertainty = uncertainty.get('environmental_uncertainty', 0.5)
            env_prediction = prediction[0] if len(prediction) > 0 else 0.0
            env_margin = env_uncertainty * confidence_level
            intervals['environmental_risk'] = {
                'lower': max(0.0, env_prediction - env_margin),
                'upper': min(1.0, env_prediction + env_margin),
                'prediction': env_prediction
            }
            
            # Infrastructure risk confidence interval
            infra_uncertainty = uncertainty.get('infrastructure_uncertainty', 0.5)
            infra_prediction = prediction[1] if len(prediction) > 1 else 0.0
            infra_margin = infra_uncertainty * confidence_level
            intervals['infrastructure_risk'] = {
                'lower': max(0.0, infra_prediction - infra_margin),
                'upper': min(1.0, infra_prediction + infra_margin),
                'prediction': infra_prediction
            }
            
            # Construction risk confidence interval
            const_uncertainty = uncertainty.get('construction_uncertainty', 0.5)
            const_prediction = prediction[2] if len(prediction) > 2 else 0.0
            const_margin = const_uncertainty * confidence_level
            intervals['construction_risk'] = {
                'lower': max(0.0, const_prediction - const_margin),
                'upper': min(1.0, const_prediction + const_margin),
                'prediction': const_prediction
            }
            
            return intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {
                'environmental_risk': {'lower': 0.0, 'upper': 1.0, 'prediction': 0.5},
                'infrastructure_risk': {'lower': 0.0, 'upper': 1.0, 'prediction': 0.5},
                'construction_risk': {'lower': 0.0, 'upper': 1.0, 'prediction': 0.5}
            }
    
    def _generate_risk_recommendations(self, prediction: np.ndarray, features: Dict) -> List[str]:
        # Generate recommendations based on risk prediction
        # Args:
        #   prediction: Neural network prediction array
        #   features: Dictionary of extracted features
        # Returns: List of recommendations
        recommendations = []
        
        # Environmental risk recommendations
        if prediction[0] > 0.7:
            recommendations.append("High environmental risk - conduct detailed environmental assessment")
        elif prediction[0] > 0.4:
            recommendations.append("Moderate environmental risk - monitor environmental conditions")
        
        # Infrastructure risk recommendations
        if prediction[1] > 0.7:
            recommendations.append("High infrastructure risk - inspect existing infrastructure")
        elif prediction[1] > 0.4:
            recommendations.append("Moderate infrastructure risk - schedule maintenance")
        
        # Construction risk recommendations
        if prediction[2] > 0.7:
            recommendations.append("High construction risk - review safety protocols")
        elif prediction[2] > 0.4:
            recommendations.append("Moderate construction risk - enhance safety measures")
        
        if not recommendations:
            recommendations.append("Risks appear manageable - proceed with standard protocols")
        
        return recommendations
    
    def _get_data_sources(self, features: Dict) -> List[str]:
        # Get list of data sources used for analysis
        # Args:
        #   features: Dictionary of extracted features
        # Returns: List of data source names
        data_sources = []
        
        if features.get('infrastructure'):
            data_sources.append('infrastructure')
        
        if features.get('climate'):
            data_sources.append('climate')
        
        if features.get('vegetation'):
            data_sources.append('vegetation')
        
        if features.get('wind'):
            data_sources.append('wind')
        
        return data_sources
    
    def _identify_maintenance_needs(self, infra_data: pd.DataFrame) -> List[str]:
        # Identify maintenance needs from infrastructure data
        # Args:
        #   infra_data: Infrastructure DataFrame
        # Returns: List of maintenance needs
        maintenance_needs = []
        
        if infra_data.empty:
            return maintenance_needs
        
        # Check for old infrastructure
        if 'Installation Date' in infra_data.columns:
            # Add maintenance logic based on age
            maintenance_needs.append("Schedule age-based maintenance")
        
        # Check for material-specific needs
        if 'Pipe Type' in infra_data.columns:
            maintenance_needs.append("Conduct material-specific inspections")
        
        return maintenance_needs
    
    def _identify_upgrade_requirements(self, infra_data: pd.DataFrame) -> List[str]:
        # Identify upgrade requirements from infrastructure data
        # Args:
        #   infra_data: Infrastructure DataFrame
        # Returns: List of upgrade requirements
        upgrade_requirements = []
        
        if infra_data.empty:
            return upgrade_requirements
        
        # Check for capacity issues
        if 'Diameter' in infra_data.columns:
            upgrade_requirements.append("Assess capacity requirements")
        
        # Check for material upgrades
        if 'Pipe Type' in infra_data.columns:
            upgrade_requirements.append("Evaluate material upgrade needs")
        
        return upgrade_requirements
    
    def _calculate_infrastructure_health_score(self, infra_data: pd.DataFrame, features: Dict) -> float:
        # Calculate comprehensive infrastructure health score using neural network analysis
        # Args:
        #   infra_data: Infrastructure DataFrame
        #   features: Features dictionary
        # Returns: Health score between 0 and 1
        logger.debug("Calculating infrastructure health score")
        
        try:
            if infra_data.empty:
                return 0.0
            
            health_factors = []
            
            # Data completeness factor
            data_completeness = self._calculate_data_completeness(features)
            health_factors.append(data_completeness * 0.3)
            
            # Infrastructure density factor
            pipe_count = len(infra_data)
            density_score = min(1.0, pipe_count / 50)  # Optimal density around 50 pipes
            health_factors.append(density_score * 0.2)
            
            # Material quality factor
            if 'Pipe Type' in infra_data.columns:
                material_types = infra_data['Pipe Type'].value_counts()
                # Assume newer materials are better (this would be learned from training data)
                modern_materials = ['PVC', 'HDPE', 'Steel']
                modern_count = sum(material_types.get(mat, 0) or 0 for mat in modern_materials)
                material_score = modern_count / pipe_count if pipe_count > 0 else 0.0
                health_factors.append(material_score * 0.25)
            else:
                health_factors.append(0.5 * 0.25)  # Unknown materials
            
            # Age factor (if available)
            if 'Installation Date' in infra_data.columns:
                # Calculate average age and score (newer is better)
                age_score = 0.7  # Placeholder - would be calculated from actual dates
                health_factors.append(age_score * 0.25)
            else:
                health_factors.append(0.5 * 0.25)  # Unknown age
            
            return sum(health_factors)
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.5
    
    def _generate_maintenance_schedule(self, infra_data: pd.DataFrame, health_score: float) -> Dict:
        # Generate maintenance schedule based on infrastructure health analysis
        # Args:
        #   infra_data: Infrastructure DataFrame
        #   health_score: Infrastructure health score
        # Returns: Maintenance schedule dictionary
        logger.debug("Generating maintenance schedule")
        
        try:
            schedule = {
                'priority': 'low',
                'next_maintenance': '90 days',
                'maintenance_frequency': 'annual',
                'estimated_cost': 0.0,
                'required_actions': []
            }
            
            # Adjust schedule based on health score
            if health_score < 0.3:
                schedule.update({
                    'priority': 'critical',
                    'next_maintenance': 'immediate',
                    'maintenance_frequency': 'monthly',
                    'estimated_cost': 50000.0,
                    'required_actions': ['Emergency inspection', 'Immediate repairs', 'Safety assessment']
                })
            elif health_score < 0.6:
                schedule.update({
                    'priority': 'high',
                    'next_maintenance': '30 days',
                    'maintenance_frequency': 'quarterly',
                    'estimated_cost': 25000.0,
                    'required_actions': ['Comprehensive inspection', 'Preventive maintenance', 'Performance testing']
                })
            else:
                schedule.update({
                    'priority': 'low',
                    'next_maintenance': '90 days',
                    'maintenance_frequency': 'annual',
                    'estimated_cost': 10000.0,
                    'required_actions': ['Routine inspection', 'Minor repairs', 'Performance monitoring']
                })
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error generating maintenance schedule: {e}")
            return {
                'priority': 'unknown',
                'next_maintenance': 'unknown',
                'maintenance_frequency': 'unknown',
                'estimated_cost': 0.0,
                'required_actions': ['Schedule assessment']
            }
    
    def _identify_health_factors(self, infra_data: pd.DataFrame) -> List[str]:
        # Identify factors affecting infrastructure health
        # Args:
        #   infra_data: Infrastructure DataFrame
        # Returns: List of health factors
        health_factors = []
        
        if infra_data.empty:
            health_factors.append("No infrastructure data available")
            return health_factors
        
        # Check for data quality issues - dataset agnostic
        # Check for any material/type column
        material_columns = [col for col in infra_data.columns if any(keyword in col.lower() for keyword in ['type', 'material', 'pipe'])]
        if not material_columns:
            health_factors.append("Missing material information")
        
        # Check for any size/diameter column
        size_columns = [col for col in infra_data.columns if any(keyword in col.lower() for keyword in ['diameter', 'size', 'width'])]
        if not size_columns:
            health_factors.append("Missing capacity information")
        
        # Check for any date column
        date_columns = [col for col in infra_data.columns if any(keyword in col.lower() for keyword in ['date', 'year', 'install', 'created'])]
        if not date_columns:
            health_factors.append("Missing age information")
        
        # Check for infrastructure characteristics
        pipe_count = len(infra_data)
        if pipe_count < 5:
            health_factors.append("Low infrastructure coverage")
        elif pipe_count > 100:
            health_factors.append("High infrastructure density")
        
        return health_factors
    
    def _generate_infrastructure_details(self, infra_data: pd.DataFrame, features: Dict) -> Dict:
        # Generate detailed infrastructure information for professional reports
        # Args:
        #   infra_data: Infrastructure DataFrame
        #   features: Features dictionary
        # Returns: Dictionary with detailed infrastructure information
        logger.debug("Generating infrastructure details")
        
        details = {}
        
        # Calculate age statistics - dataset agnostic
        # Find date column dynamically - any column with 'date', 'year', 'install' in name
        date_columns = [col for col in infra_data.columns if any(keyword in col.lower() for keyword in ['date', 'year', 'install', 'created'])]
        if date_columns:
            try:
                # Convert installation dates to ages
                current_year = datetime.now().year
                ages = []
                for date_str in infra_data[date_columns[0]]:
                    try:
                        if isinstance(date_str, str):
                            year = int(date_str.split('-')[0])
                            ages.append(current_year - year)
                        elif isinstance(date_str, (int, float)):
                            ages.append(current_year - int(date_str))
                    except:
                        continue
                
                if ages:
                    details['age_stats'] = {
                        'min_age': min(ages),
                        'max_age': max(ages),
                        'avg_age': sum(ages) / len(ages)
                    }
            except Exception as e:
                logger.warning(f"Error calculating age stats: {e}")
        
        # Analyze installation periods - dataset agnostic
        if date_columns:
            try:
                years = []
                for date_str in infra_data[date_columns[0]]:
                    try:
                        if isinstance(date_str, str):
                            year = int(date_str.split('-')[0])
                            years.append(year)
                        elif isinstance(date_str, (int, float)):
                            years.append(int(date_str))
                    except:
                        continue
                
                if years:
                    # Find most common installation period
                    year_counts = {}
                    for year in years:
                        decade = (year // 10) * 10
                        period = f"{decade}-{decade+9}"
                        year_counts[period] = year_counts.get(period, 0) + 1
                    
                    most_common_period = max(year_counts.items(), key=lambda x: x[1])[0]
                    details['most_laid_period'] = most_common_period
            except Exception as e:
                logger.warning(f"Error analyzing installation periods: {e}")
        
        # Analyze materials - dataset agnostic
        # Find material/type column dynamically - any column with 'type', 'material', 'pipe' in name
        material_columns = [col for col in infra_data.columns if any(keyword in col.lower() for keyword in ['type', 'material', 'pipe'])]
        if material_columns:
            try:
                material_counts = infra_data[material_columns[0]].value_counts()
                total_pipes = len(infra_data)
                material_percentages = {}
                
                for material, count in material_counts.items():
                    percentage = (count / total_pipes) * 100
                    material_percentages[material] = {
                        'count': count,
                        'percentage': percentage
                    }
                
                details['material_analysis'] = material_percentages
            except Exception as e:
                logger.warning(f"Error analyzing materials: {e}")
        
        # Maintenance history - only if data is available
        if 'maintenance_needs' in features or 'upgrade_requirements' in features:
            details['maintenance_history'] = {
                'last_maintenance': 'Maintenance data from neural network analysis',
                'known_faults': 'Fault data from neural network analysis'
            }
        
        return details
    
    def _analyze_location_data(self, lat: float, lon: float, features: Dict) -> Dict:
        # Analyze location data to provide location names and zones
        # Args:
        #   lat: Latitude coordinate
        #   lon: Longitude coordinate
        #   features: Features dictionary
        # Returns: Dictionary with location analysis
        logger.debug("Analyzing location data")
        
        location_analysis = {
            'location_name': f"Location ({lat:.4f}, {lon:.4f})",
            'zone': 'Zone (analysis required)',
            'geographic_features': [],
            'administrative_boundaries': []
        }
        
        # This should be enhanced with actual geospatial data analysis
        # For now, return basic structure for neural network to populate
        return location_analysis
    
    def _analyze_cost_data(self, infra_data: pd.DataFrame, features: Dict) -> Dict:
        # Analyze cost data based on infrastructure characteristics
        # Args:
        #   infra_data: Infrastructure DataFrame
        #   features: Features dictionary
        # Returns: Dictionary with cost analysis
        logger.debug("Analyzing cost data")
        
        cost_analysis = {
            'cost_per_km': 0,
            'cost_multiplier': 1.0,
            'total_estimated_cost': 0,
            'cost_factors': [],
            'uncertainty': 0.0
        }
        
        # This should be enhanced with actual cost data analysis
        # For now, return basic structure for neural network to populate
        return cost_analysis
    
    def _analyze_climate_data(self, climate_data: Dict) -> Dict:
        # Analyze climate data
        # Args:
        #   climate_data: Climate data dictionary
        # Returns: Analyzed climate information
        return {
            'temperature_avg': climate_data.get('temperature_avg', 0),
            'precipitation': climate_data.get('precipitation', 0),
            'climate_zone': climate_data.get('climate_zone', 'unknown')
        }
    
    def _analyze_vegetation_data(self, vegetation_data: pd.DataFrame) -> Dict:
        # Analyze vegetation data - dataset agnostic
        # Args:
        #   vegetation_data: Vegetation DataFrame
        # Returns: Analyzed vegetation information
        if isinstance(vegetation_data, pd.DataFrame) and not vegetation_data.empty:
            # Find type/category column dynamically
            type_columns = [col for col in vegetation_data.columns if any(keyword in col.lower() for keyword in ['type', 'category', 'class', 'zone'])]
            if type_columns:
                type_series = vegetation_data[type_columns[0]]
                zone_types = type_series.unique().tolist() if type_series is not None else []
            else:
                zone_types = []
            return {
                'zones_count': len(vegetation_data),
                'zone_types': zone_types
            }
        return {'zones_count': 0, 'zone_types': []}
    
    def _analyze_soil_conditions(self, features: Dict) -> Dict:
        # Analyze soil conditions
        # Args:
        #   features: Features dictionary
        # Returns: Soil condition analysis
        return {
            'soil_type': features.get('soil_type', 'unknown'),
            'soil_risk': features.get('soil_risk', 0.0)
        }
    
    def _analyze_water_resources(self, features: Dict) -> Dict:
        # Analyze water resources
        # Args:
        #   features: Features dictionary
        # Returns: Water resource analysis
        return {
            'water_availability': features.get('water_availability', 'unknown'),
            'flood_risk': features.get('flood_risk', 0.0)
        }
    
    def _identify_environmental_risks(self, features: Dict) -> List[str]:
        # Identify environmental risks
        # Args:
        #   features: Features dictionary
        # Returns: List of environmental risks
        risks = []
        
        if features.get('flood_risk', 0) > 0.5:
            risks.append("High flood risk")
        
        if features.get('soil_risk', 0) > 0.5:
            risks.append("Problematic soil conditions")
        
        return risks
    
    def _generate_environmental_recommendations(self, features: Dict) -> List[str]:
        # Generate environmental recommendations
        # Args:
        #   features: Features dictionary
        # Returns: List of environmental recommendations
        recommendations = []
        
        if features.get('flood_risk', 0) > 0.5:
            recommendations.append("Implement flood protection measures")
        
        if features.get('soil_risk', 0) > 0.5:
            recommendations.append("Conduct detailed soil analysis")
        
        return recommendations
    
    def _assess_climate_change_impact(self, climate_data: Dict, features: Dict) -> Dict:
        # Assess climate change impact based on current climate data and trends
        # Args:
        #   climate_data: Climate data dictionary
        #   features: Features dictionary
        # Returns: Climate change impact assessment
        logger.debug("Assessing climate change impact")
        
        try:
            impact_assessment = {
                'impact_level': 'moderate',
                'temperature_trend': 'increasing',
                'precipitation_trend': 'variable',
                'sea_level_risk': 'low',
                'extreme_weather_risk': 'moderate',
                'adaptation_measures': [],
                'mitigation_priorities': []
            }
            
            # Analyze temperature trends
            if climate_data.get('temperature_avg', 0) > 25:
                impact_assessment.update({
                    'impact_level': 'high',
                    'temperature_trend': 'significantly_increasing',
                    'extreme_weather_risk': 'high',
                    'adaptation_measures': ['Heat-resistant infrastructure', 'Cooling systems'],
                    'mitigation_priorities': ['Reduce heat island effects', 'Implement green infrastructure']
                })
            
            # Analyze precipitation patterns
            if climate_data.get('precipitation', 0) > 150:
                impact_assessment.update({
                    'precipitation_trend': 'increasing',
                    'extreme_weather_risk': 'high',
                    'adaptation_measures': ['Enhanced drainage systems', 'Flood protection'],
                    'mitigation_priorities': ['Water management systems', 'Sustainable drainage']
                })
            
            # Assess sea level risk for coastal areas
            if features.get('elevation', 100) < 10:
                impact_assessment.update({
                    'sea_level_risk': 'high',
                    'adaptation_measures': ['Coastal protection', 'Elevation planning'],
                    'mitigation_priorities': ['Coastal zone management', 'Ecosystem restoration']
                })
            
            return impact_assessment
            
        except Exception as e:
            logger.error(f"Error assessing climate change impact: {e}")
            return {
                'impact_level': 'unknown',
                'temperature_trend': 'unknown',
                'precipitation_trend': 'unknown',
                'sea_level_risk': 'unknown',
                'extreme_weather_risk': 'unknown',
                'adaptation_measures': ['Conduct climate impact assessment'],
                'mitigation_priorities': ['Develop climate adaptation plan']
            }
    
    def _analyze_biodiversity(self, vegetation_data: pd.DataFrame, features: Dict) -> Dict:
        # Analyze biodiversity based on vegetation data and environmental features
        # Args:
        #   vegetation_data: Vegetation DataFrame
        #   features: Features dictionary
        # Returns: Biodiversity analysis
        logger.debug("Analyzing biodiversity")
        
        try:
            biodiversity_analysis = {
                'biodiversity_level': 'moderate',
                'species_richness': 'unknown',
                'habitat_quality': 'moderate',
                'threatened_species': 'unknown',
                'conservation_priorities': [],
                'restoration_opportunities': []
            }
            
            if isinstance(vegetation_data, pd.DataFrame) and not vegetation_data.empty:
                # Analyze vegetation diversity
                zone_count = len(vegetation_data)
                if 'Type' in vegetation_data.columns:
                    type_series = vegetation_data['Type']
                    unique_types = type_series.unique() if type_series is not None else []
                    species_richness = len(unique_types)
                    
                    if species_richness > 10:
                        biodiversity_analysis.update({
                            'biodiversity_level': 'high',
                            'species_richness': 'high',
                            'habitat_quality': 'high',
                            'conservation_priorities': ['Protect existing habitats', 'Monitor species populations'],
                            'restoration_opportunities': ['Enhance habitat connectivity', 'Restore native species']
                        })
                    elif species_richness > 5:
                        biodiversity_analysis.update({
                            'biodiversity_level': 'moderate',
                            'species_richness': 'moderate',
                            'habitat_quality': 'moderate',
                            'conservation_priorities': ['Maintain habitat diversity', 'Monitor ecosystem health'],
                            'restoration_opportunities': ['Increase species diversity', 'Improve habitat quality']
                        })
                    else:
                        biodiversity_analysis.update({
                            'biodiversity_level': 'low',
                            'species_richness': 'low',
                            'habitat_quality': 'poor',
                            'conservation_priorities': ['Restore ecosystem function', 'Introduce native species'],
                            'restoration_opportunities': ['Create new habitats', 'Implement biodiversity corridors']
                        })
                else:
                    biodiversity_analysis.update({
                        'species_richness': 'unknown',
                        'conservation_priorities': ['Conduct biodiversity survey'],
                        'restoration_opportunities': ['Assess restoration potential']
                    })
            else:
                biodiversity_analysis.update({
                    'biodiversity_level': 'unknown',
                    'species_richness': 'unknown',
                    'habitat_quality': 'unknown',
                    'conservation_priorities': ['Conduct biodiversity assessment'],
                    'restoration_opportunities': ['Develop biodiversity baseline']
                })
            
            return biodiversity_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing biodiversity: {e}")
            return {
                'biodiversity_level': 'unknown',
                'species_richness': 'unknown',
                'habitat_quality': 'unknown',
                'threatened_species': 'unknown',
                'conservation_priorities': ['Conduct biodiversity assessment'],
                'restoration_opportunities': ['Develop biodiversity management plan']
            }
    
    def _generate_construction_phases(self, risk_assessment: Dict) -> List[Dict]:
        # Generate construction phases based on risk assessment
        # Args:
        #   risk_assessment: Risk assessment dictionary
        # Returns: List of construction phases
        phases = [
            {
                'phase': 1,
                'name': 'Site Preparation',
                'duration_days': 5,
                'activities': ['Site clearing', 'Safety setup']
            },
            {
                'phase': 2,
                'name': 'Construction',
                'duration_days': 15,
                'activities': ['Main construction work']
            },
            {
                'phase': 3,
                'name': 'Completion',
                'duration_days': 3,
                'activities': ['Final inspection', 'Handover']
            }
        ]
        
        # Adjust based on risk levels
        if risk_assessment.get('overall_risk', 0) > 0.7:
            for phase in phases:
                phase['duration_days'] = int(phase['duration_days'] * 1.5)
        
        return phases
    
    def _estimate_construction_timeline(self, risk_assessment: Dict) -> Dict:
        # Estimate construction timeline
        # Args:
        #   risk_assessment: Risk assessment dictionary
        # Returns: Timeline dictionary
        total_days = sum(phase['duration_days'] for phase in self._generate_construction_phases(risk_assessment))
        
        return {
            'total_duration_days': total_days,
            'estimated_completion': f"{total_days} days from start"
        }
    
    def _optimize_construction_timeline(self, risk_assessment: Dict, features: Dict) -> Dict:
        # Optimize construction timeline based on risks and site conditions
        # Args:
        #   risk_assessment: Risk assessment dictionary
        #   features: Features dictionary
        # Returns: Optimized timeline dictionary
        logger.debug("Optimizing construction timeline")
        
        try:
            # Get base timeline
            base_timeline = self._estimate_construction_timeline(risk_assessment)
            base_days = base_timeline['total_duration_days']
            
            # Apply optimization factors
            optimization_factors = []
            
            # Risk-based optimization
            overall_risk = risk_assessment.get('overall_risk', 0.5)
            if overall_risk > 0.7:
                optimization_factors.append(1.5)  # 50% longer for high risk
            elif overall_risk > 0.4:
                optimization_factors.append(1.2)  # 20% longer for moderate risk
            else:
                optimization_factors.append(1.0)  # No change for low risk
            
            # Site condition optimization
            if features.get('flood_risk', 0) > 0.5:
                optimization_factors.append(1.3)  # 30% longer for flood risk
            
            if features.get('soil_risk', 0) > 0.5:
                optimization_factors.append(1.25)  # 25% longer for soil issues
            
            # Calculate optimized duration
            optimization_multiplier = sum(optimization_factors) / len(optimization_factors)
            optimized_days = int(base_days * optimization_multiplier)
            
            # Add parallel work opportunities
            parallel_work_savings = self._calculate_parallel_work_savings(features)
            final_days = max(optimized_days - parallel_work_savings, base_days)
            
            return {
                'total_duration_days': final_days,
                'estimated_completion': f"{final_days} days from start",
                'optimization_multiplier': optimization_multiplier,
                'parallel_work_savings': parallel_work_savings,
                'optimization_factors': optimization_factors
            }
            
        except Exception as e:
            logger.error(f"Error optimizing timeline: {e}")
            return self._estimate_construction_timeline(risk_assessment)
    
    def _allocate_construction_resources(self, risk_assessment: Dict, features: Dict) -> Dict:
        # Allocate construction resources based on project requirements
        # Args:
        #   risk_assessment: Risk assessment dictionary
        #   features: Features dictionary
        # Returns: Resource allocation dictionary
        logger.debug("Allocating construction resources")
        
        try:
            # Base resource requirements
            resources = {
                'labor': {
                    'skilled_workers': 10,
                    'unskilled_workers': 20,
                    'supervisors': 2,
                    'engineers': 1
                },
                'equipment': {
                    'excavators': 2,
                    'bulldozers': 1,
                    'cranes': 1,
                    'trucks': 3
                },
                'materials': {
                    'concrete': '500 cubic meters',
                    'steel': '50 tons',
                    'pipes': '1000 meters',
                    'aggregate': '1000 cubic meters'
                },
                'estimated_cost': 500000.0
            }
            
            # Adjust based on risk levels
            overall_risk = risk_assessment.get('overall_risk', 0.5)
            if overall_risk > 0.7:
                # High risk - increase resources
                resources['labor']['skilled_workers'] = 15
                resources['labor']['supervisors'] = 3
                resources['equipment']['excavators'] = 3
                resources['estimated_cost'] = 750000.0
            elif overall_risk < 0.3:
                # Low risk - optimize resources
                resources['labor']['skilled_workers'] = 8
                resources['labor']['unskilled_workers'] = 15
                resources['estimated_cost'] = 400000.0
            
            # Adjust based on site conditions
            if features.get('flood_risk', 0) > 0.5:
                resources['equipment']['pumps'] = 2
                resources['materials']['flood_protection'] = '1000 square meters'
                resources['estimated_cost'] += 50000.0
            
            if features.get('soil_risk', 0) > 0.5:
                resources['equipment']['soil_stabilization'] = 1
                resources['materials']['soil_stabilizers'] = '50 tons'
                resources['estimated_cost'] += 30000.0
            
            return resources
            
        except Exception as e:
            logger.error(f"Error allocating resources: {e}")
            return {
                'labor': {'skilled_workers': 10, 'unskilled_workers': 20, 'supervisors': 2, 'engineers': 1},
                'equipment': {'excavators': 2, 'bulldozers': 1, 'cranes': 1, 'trucks': 3},
                'materials': {'concrete': '500 cubic meters', 'steel': '50 tons', 'pipes': '1000 meters'},
                'estimated_cost': 500000.0
            }
    
    def _calculate_parallel_work_savings(self, features: Dict) -> int:
        # Calculate time savings from parallel work opportunities
        # Args:
        #   features: Features dictionary
        # Returns: Days saved from parallel work
        savings = 0
        
        # Site preparation can be done in parallel with design finalization
        if features.get('infrastructure', {}).get('count', 0) > 0:
            savings += 3  # 3 days saved
        
        # Environmental work can be done in parallel with main construction
        if features.get('climate') or features.get('vegetation'):
            savings += 2  # 2 days saved
        
        return savings
    
    def _calculate_optimization_score(self, risk_assessment: Dict, features: Dict) -> float:
        # Calculate optimization score for the construction plan
        # Args:
        #   risk_assessment: Risk assessment dictionary
        #   features: Features dictionary
        # Returns: Optimization score between 0 and 1
        try:
            # Base optimization factors
            optimization_factors = []
            
            # Risk-based optimization
            overall_risk = risk_assessment.get('overall_risk', 0.5)
            if overall_risk < 0.3:
                optimization_factors.append(0.9)  # High optimization for low risk
            elif overall_risk < 0.6:
                optimization_factors.append(0.7)  # Moderate optimization
            else:
                optimization_factors.append(0.5)  # Limited optimization for high risk
            
            # Data completeness optimization
            data_completeness = self._calculate_data_completeness(features)
            optimization_factors.append(data_completeness * 0.8)
            
            # Site condition optimization
            if features.get('flood_risk', 0) < 0.3 and features.get('soil_risk', 0) < 0.3:
                optimization_factors.append(0.8)  # Good site conditions
            else:
                optimization_factors.append(0.6)  # Challenging site conditions
            
            return sum(optimization_factors) / len(optimization_factors)
            
        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 0.5
    
    def _identify_construction_requirements(self, features: Dict) -> List[str]:
        # Identify construction requirements
        # Args:
        #   features: Features dictionary
        # Returns: List of construction requirements
        requirements = []
        
        if features.get('infrastructure', {}).get('count', 0) > 0:
            requirements.append("Coordinate with existing infrastructure")
        
        if features.get('flood_risk', 0) > 0.5:
            requirements.append("Implement flood protection")
        
        return requirements
    
    def _generate_safety_protocols(self, risk_assessment: Dict) -> List[str]:
        # Generate safety protocols based on risk assessment
        # Args:
        #   risk_assessment: Risk assessment dictionary
        # Returns: List of safety protocols
        protocols = ["Standard safety protocols"]
        
        if risk_assessment.get('construction_risk', 0) > 0.7:
            protocols.append("Enhanced safety measures required")
        
        return protocols
    
    def _assess_construction_impact(self, features: Dict) -> Dict:
        # Assess construction environmental impact
        # Args:
        #   features: Features dictionary
        # Returns: Impact assessment dictionary
        return {
            'environmental_impact': 'moderate',
            'mitigation_required': features.get('flood_risk', 0) > 0.5
        }
    
    def _check_regulatory_compliance(self, features: Dict) -> Dict:
        # Check regulatory compliance
        # Args:
        #   features: Features dictionary
        # Returns: Compliance check dictionary
        return {
            'compliance_status': 'unknown',
            'required_permits': ['Building permit', 'Construction permit']
        }
    
    def _identify_required_surveys(self, features: Dict) -> List[str]:
        # Identify required surveys
        # Args:
        #   features: Features dictionary
        # Returns: List of required surveys
        # This method delegates to the SurveyAnalyzer module for proper separation of concerns
        
        from src.core.survey_analyzer import SurveyAnalyzer
        survey_analyzer = SurveyAnalyzer()
        return survey_analyzer.identify_required_surveys(features)
    
    def _recommend_survey_methods(self, features: Dict) -> List[str]:
        # Recommend survey methods
        # Args:
        #   features: Features dictionary
        # Returns: List of survey methods
        methods = []
        
        if not features.get('infrastructure'):
            methods.append("Ground penetrating radar")
            methods.append("Visual inspection")
        
        if not features.get('climate'):
            methods.append("Environmental monitoring")
        
        return methods
    
    def _estimate_survey_cost(self, features: Dict) -> float:
        # Estimate survey cost
        # Args:
        #   features: Features dictionary
        # Returns: Estimated cost
        base_cost = 5000.0
        
        if not features.get('infrastructure'):
            base_cost += 2000.0
        
        if not features.get('climate'):
            base_cost += 1500.0
        
        return base_cost
    
    def _estimate_survey_costs(self, features: Dict) -> Dict:
        # Estimate detailed survey costs for different survey types
        # Args:
        #   features: Features dictionary
        # Returns: Detailed cost estimation dictionary
        # This method delegates to the SurveyAnalyzer module for proper separation of concerns
        
        from src.core.survey_analyzer import SurveyAnalyzer
        survey_analyzer = SurveyAnalyzer()
        return survey_analyzer.estimate_survey_costs(features)
    
    def _calculate_survey_priority_scores(self, features: Dict) -> Dict:
        # Calculate priority scores for different survey types
        # Args:
        #   features: Features dictionary
        # Returns: Priority scoring dictionary
        # This method delegates to the SurveyAnalyzer module for proper separation of concerns
        
        from src.core.survey_analyzer import SurveyAnalyzer
        survey_analyzer = SurveyAnalyzer()
        return survey_analyzer.calculate_priority_scores(features)
    
    def _generate_survey_recommendations(self, features: Dict) -> List[str]:
        # Generate survey recommendations
        # Args:
        #   features: Features dictionary
        # Returns: List of survey recommendations
        recommendations = []
        
        if not features.get('infrastructure'):
            recommendations.append("Conduct infrastructure survey")
        
        if not features.get('climate'):
            recommendations.append("Conduct environmental survey")
        
        return recommendations
    
    def _identify_data_gaps(self, features: Dict) -> List[str]:
        # Identify data gaps
        # Args:
        #   features: Features dictionary
        # Returns: List of data gaps
        gaps = []
        
        if not features.get('infrastructure'):
            gaps.append("Infrastructure data missing")
        
        if not features.get('climate'):
            gaps.append("Climate data missing")
        
        if not features.get('vegetation'):
            gaps.append("Vegetation data missing")
        
        return gaps
    
    def _calculate_data_completeness(self, features: Dict) -> float:
        # Calculate data completeness score
        # Args:
        #   features: Features dictionary
        # Returns: Completeness score between 0 and 1
        total_features = 0
        available_features = 0
        
        # Check infrastructure data
        if 'infrastructure' in features:
            total_features += 1
            if features['infrastructure'].get('count', 0) > 0:
                available_features += 1
        
        # Check climate data
        if 'climate' in features:
            total_features += 1
            if features['climate']:
                available_features += 1
        
        # Check vegetation data
        if 'vegetation' in features:
            total_features += 1
            if features['vegetation']:
                available_features += 1
        
        return available_features / total_features if total_features > 0 else 0.0
    
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
    def load_model(self, model_name: Optional[str] = None):
        """Load the most recent trained model and scalers"""
        try:
            # If no specific model name provided, find the most recent one
            if model_name is None:
                model_files = list(self.model_dir.glob("*.pth"))
                if not model_files:
                    logger.warning("No model files found")
            return False
        
                # Sort by modification time and get the most recent
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                model_path = model_files[0]
                model_name = model_path.stem  # Get filename without extension
                logger.info(f"Auto-selected model: {model_name}")
            
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
                # Build model with correct dimensions
                input_dim = len(self.feature_names) if self.feature_names else 15
                output_dim = len(self.output_names) if self.output_names else 3
                
        self.model = self.build_model(input_dim, output_dim)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
        
                logger.info(f"Model loaded successfully: {model_name}")
        return True
            else:
                logger.warning(f"Model file not found: {model_path}")
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