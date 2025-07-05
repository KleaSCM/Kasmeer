# Author: KleaSCM
# Date: 2024
# Description: Advanced risk assessment module for civil engineering system

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from utils.logging_utils import setup_logging, log_performance

logger = setup_logging(__name__)

class RiskAssessmentEngine:
    # Advanced risk assessment engine for civil engineering projects
    # This module provides sophisticated risk calculation, historical incident analysis,
    # and regulatory compliance checking
    # TODO: Add machine learning-based risk prediction
    # TODO: Integrate with external risk databases
    # TODO: Add real-time risk monitoring capabilities
    
    @log_performance(logger)
    def __init__(self, data_dir: str = "DataSets", config_path: str = "config.yaml"):
        # Initialize the risk assessment engine
        # Args:
        #   data_dir: Directory containing risk-related data
        #   config_path: Path to configuration file
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.historical_incidents = {}
        self.regulatory_requirements = {}
        self.risk_thresholds = self._load_risk_thresholds()
        logger.info(f"Initialized RiskAssessmentEngine with data_dir={data_dir}")
        
    def _load_risk_thresholds(self) -> Dict[str, Dict[str, float]]:
        # Load risk thresholds from configuration
        # Returns: Dictionary of risk thresholds for different categories
        # TODO: Add dynamic threshold adjustment based on historical data
        # TODO: Include region-specific thresholds
        logger.debug("Loading risk thresholds")
        
        return {
            'environmental': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 0.9
            },
            'infrastructure': {
                'low': 0.25,
                'medium': 0.5,
                'high': 0.75,
                'critical': 0.85
            },
            'construction': {
                'low': 0.2,
                'medium': 0.45,
                'high': 0.7,
                'critical': 0.8
            },
            'regulatory': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 0.9
            }
        }
    
    @log_performance(logger)
    def calculate_comprehensive_risk(self, 
                                   location: Dict[str, float],
                                   infrastructure_data: Dict,
                                   environmental_data: Dict,
                                   base_risk_prediction: Dict) -> Dict[str, Any]:
        # Calculate comprehensive risk assessment including historical and regulatory factors
        # Args:
        #   location: Dictionary with lat/lon coordinates
        #   infrastructure_data: Infrastructure information at location
        #   environmental_data: Environmental data at location
        #   base_risk_prediction: Base risk prediction from neural network
        # Returns: Comprehensive risk assessment with all factors
        # TODO: Add weather-based risk adjustment
        # TODO: Include seasonal risk variations
        logger.info(f"Calculating comprehensive risk for location: {location}")
        
        # Load historical incidents for this area
        historical_risk = self._analyze_historical_incidents(location)
        
        # Check regulatory compliance
        regulatory_risk = self._assess_regulatory_compliance(location, infrastructure_data)
        
        # Calculate environmental risk factors
        environmental_risk = self._calculate_environmental_risk(location, environmental_data)
        
        # Calculate infrastructure risk factors
        infrastructure_risk = self._calculate_infrastructure_risk(location, infrastructure_data)
        
        # Combine all risk factors
        combined_risk = self._combine_risk_factors(
            base_risk_prediction,
            historical_risk,
            regulatory_risk,
            environmental_risk,
            infrastructure_risk
        )
        
        # Generate detailed risk report
        risk_report = self._generate_risk_report(
            combined_risk,
            historical_risk,
            regulatory_risk,
            environmental_risk,
            infrastructure_risk
        )
        
        logger.info(f"Comprehensive risk calculation completed for location: {location}")
        return risk_report
    
    def _analyze_historical_incidents(self, location: Dict[str, float]) -> Dict[str, Any]:
        # Analyze historical incidents in the area
        # Args:
        #   location: Dictionary with lat/lon coordinates
        # Returns: Historical risk assessment
        # TODO: Add incident severity weighting
        # TODO: Include incident type classification
        logger.debug(f"Analyzing historical incidents for location: {location}")
        
        try:
            # Load historical incident data
            incidents_file = self.data_dir / "historical_incidents.json"
            if incidents_file.exists():
                with open(incidents_file, 'r') as f:
                    incidents_data = json.load(f)
            else:
                # Create sample historical data if file doesn't exist
                incidents_data = self._create_sample_incident_data()
            
            # Find incidents within 5km radius
            nearby_incidents = []
            lat, lon = location['lat'], location['lon']
            
            for incident in incidents_data.get('incidents', []):
                distance = self._calculate_distance(
                    lat, lon,
                    incident['latitude'], incident['longitude']
                )
                if distance <= 5.0:  # 5km radius
                    nearby_incidents.append(incident)
            
            # Calculate historical risk score
            if nearby_incidents:
                total_severity = sum(incident['severity'] for incident in nearby_incidents)
                avg_severity = total_severity / len(nearby_incidents)
                incident_frequency = len(nearby_incidents) / 10  # incidents per decade
                
                historical_risk_score = min(1.0, (avg_severity * incident_frequency) / 10)
            else:
                historical_risk_score = 0.1  # Low risk if no historical incidents
            
            return {
                'risk_score': historical_risk_score,
                'incident_count': len(nearby_incidents),
                'recent_incidents': [i for i in nearby_incidents if i['year'] >= 2020],
                'severity_distribution': self._analyze_severity_distribution(nearby_incidents),
                'risk_factors': self._identify_historical_risk_factors(nearby_incidents)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical incidents: {e}")
            return {
                'risk_score': 0.3,
                'incident_count': 0,
                'recent_incidents': [],
                'severity_distribution': {},
                'risk_factors': ['Historical data unavailable']
            }
    
    def _assess_regulatory_compliance(self, location: Dict[str, float], infrastructure_data: Dict) -> Dict[str, Any]:
        # Assess regulatory compliance for the location
        # Args:
        #   location: Dictionary with lat/lon coordinates
        #   infrastructure_data: Infrastructure information
        # Returns: Regulatory compliance assessment
        # TODO: Add real-time regulatory database integration
        # TODO: Include compliance monitoring
        logger.debug(f"Assessing regulatory compliance for location: {location}")
        
        try:
            # Load regulatory requirements
            regulatory_file = self.data_dir / "regulatory_requirements.json"
            if regulatory_file.exists():
                with open(regulatory_file, 'r') as f:
                    regulatory_data = json.load(f)
            else:
                # Create sample regulatory data
                regulatory_data = self._create_sample_regulatory_data()
            
            compliance_issues = []
            compliance_score = 1.0
            
            # Check infrastructure compliance
            if infrastructure_data.get('pipe_count', 0) > 0:
                # Check pipe age compliance
                if infrastructure_data.get('avg_pipe_age', 0) > 50:
                    compliance_issues.append("Pipes exceed recommended age limit")
                    compliance_score -= 0.2
                
                # Check material compliance
                if infrastructure_data.get('problematic_materials', 0) > 0:
                    compliance_issues.append("Some pipes use non-compliant materials")
                    compliance_score -= 0.15
            
            # Check environmental compliance
            if infrastructure_data.get('environmental_impact', 'unknown') == 'high':
                compliance_issues.append("High environmental impact detected")
                compliance_score -= 0.25
            
            # Check safety compliance
            if infrastructure_data.get('safety_rating', 0) < 0.7:
                compliance_issues.append("Safety rating below regulatory minimum")
                compliance_score -= 0.3
            
            return {
                'compliance_score': max(0.0, compliance_score),
                'compliance_issues': compliance_issues,
                'regulatory_requirements': regulatory_data.get('requirements', []),
                'compliance_status': 'compliant' if compliance_score >= 0.8 else 'non_compliant',
                'required_actions': self._generate_compliance_actions(compliance_issues)
            }
            
        except Exception as e:
            logger.error(f"Error assessing regulatory compliance: {e}")
            return {
                'compliance_score': 0.5,
                'compliance_issues': ['Regulatory data unavailable'],
                'regulatory_requirements': [],
                'compliance_status': 'unknown',
                'required_actions': ['Conduct regulatory compliance audit']
            }
    
    def _calculate_environmental_risk(self, location: Dict[str, float], environmental_data: Dict) -> Dict[str, Any]:
        # Calculate environmental risk factors
        # Args:
        #   location: Dictionary with lat/lon coordinates
        #   environmental_data: Environmental information
        # Returns: Environmental risk assessment
        # TODO: Add climate change impact assessment
        # TODO: Include biodiversity risk factors
        logger.debug(f"Calculating environmental risk for location: {location}")
        
        environmental_risk_score = 0.0
        risk_factors = []
        
        # Assess climate risk
        if environmental_data.get('climate_zone') == 'extreme':
            environmental_risk_score += 0.3
            risk_factors.append("Extreme climate conditions")
        
        # Assess vegetation risk
        if environmental_data.get('vegetation_density') == 'high':
            environmental_risk_score += 0.2
            risk_factors.append("High vegetation density")
        
        # Assess soil risk
        if environmental_data.get('soil_type') in ['clay', 'expansive']:
            environmental_risk_score += 0.25
            risk_factors.append("Problematic soil conditions")
        
        # Assess flood risk
        if environmental_data.get('flood_risk', 0) > 0.5:
            environmental_risk_score += 0.3
            risk_factors.append("High flood risk")
        
        # Assess seismic risk
        if environmental_data.get('seismic_zone') in ['high', 'very_high']:
            environmental_risk_score += 0.4
            risk_factors.append("High seismic risk")
        
        return {
            'risk_score': min(1.0, environmental_risk_score),
            'risk_factors': risk_factors,
            'climate_risk': environmental_data.get('climate_risk', 0.0),
            'flood_risk': environmental_data.get('flood_risk', 0.0),
            'seismic_risk': environmental_data.get('seismic_risk', 0.0),
            'soil_risk': environmental_data.get('soil_risk', 0.0)
        }
    
    def _calculate_infrastructure_risk(self, location: Dict[str, float], infrastructure_data: Dict) -> Dict[str, Any]:
        # Calculate infrastructure-specific risk factors
        # Args:
        #   location: Dictionary with lat/lon coordinates
        #   infrastructure_data: Infrastructure information
        # Returns: Infrastructure risk assessment
        # TODO: Add structural integrity assessment
        # TODO: Include maintenance history analysis
        logger.debug(f"Calculating infrastructure risk for location: {location}")
        
        infrastructure_risk_score = 0.0
        risk_factors = []
        
        # Assess pipe age risk
        avg_age = infrastructure_data.get('avg_pipe_age', 0)
        if avg_age > 50:
            infrastructure_risk_score += 0.4
            risk_factors.append(f"Old infrastructure (avg age: {avg_age} years)")
        elif avg_age > 30:
            infrastructure_risk_score += 0.2
            risk_factors.append(f"Aging infrastructure (avg age: {avg_age} years)")
        
        # Assess material risk
        if infrastructure_data.get('corrosive_materials', 0) > 0:
            infrastructure_risk_score += 0.3
            risk_factors.append("Corrosive materials detected")
        
        # Assess load risk
        if infrastructure_data.get('load_factor', 0) > 0.8:
            infrastructure_risk_score += 0.25
            risk_factors.append("High load factor")
        
        # Assess maintenance risk
        if infrastructure_data.get('last_maintenance', 'unknown') == 'unknown':
            infrastructure_risk_score += 0.2
            risk_factors.append("Unknown maintenance history")
        
        return {
            'risk_score': min(1.0, infrastructure_risk_score),
            'risk_factors': risk_factors,
            'age_risk': min(1.0, avg_age / 100),
            'material_risk': infrastructure_data.get('material_risk', 0.0),
            'load_risk': infrastructure_data.get('load_risk', 0.0),
            'maintenance_risk': infrastructure_data.get('maintenance_risk', 0.0)
        }
    
    def _combine_risk_factors(self, 
                             base_risk: Dict,
                             historical_risk: Dict,
                             regulatory_risk: Dict,
                             environmental_risk: Dict,
                             infrastructure_risk: Dict) -> Dict[str, float]:
        # Combine all risk factors into final risk scores
        # Args:
        #   base_risk: Base risk prediction from neural network
        #   historical_risk: Historical incident risk
        #   regulatory_risk: Regulatory compliance risk
        #   environmental_risk: Environmental risk factors
        #   infrastructure_risk: Infrastructure risk factors
        # Returns: Combined risk scores
        # TODO: Add machine learning-based risk combination
        # TODO: Include risk correlation analysis
        logger.debug("Combining risk factors")
        
        # Weight factors for different risk types
        weights = {
            'base': 0.3,
            'historical': 0.2,
            'regulatory': 0.15,
            'environmental': 0.2,
            'infrastructure': 0.15
        }
        
        # Calculate weighted risk scores
        environmental_risk_score = (
            base_risk.get('environmental_risk', 0.0) * weights['base'] +
            environmental_risk['risk_score'] * weights['environmental']
        )
        
        infrastructure_risk_score = (
            base_risk.get('infrastructure_risk', 0.0) * weights['base'] +
            infrastructure_risk['risk_score'] * weights['infrastructure'] +
            (1.0 - regulatory_risk['compliance_score']) * weights['regulatory']
        )
        
        construction_risk_score = (
            base_risk.get('construction_risk', 0.0) * weights['base'] +
            historical_risk['risk_score'] * weights['historical'] +
            environmental_risk['risk_score'] * 0.1 +
            infrastructure_risk['risk_score'] * 0.1
        )
        
        # Calculate overall risk
        overall_risk = (
            environmental_risk_score * 0.4 +
            infrastructure_risk_score * 0.35 +
            construction_risk_score * 0.25
        )
        
        return {
            'environmental_risk': min(1.0, environmental_risk_score),
            'infrastructure_risk': min(1.0, infrastructure_risk_score),
            'construction_risk': min(1.0, construction_risk_score),
            'overall_risk': min(1.0, overall_risk),
            'historical_risk': historical_risk['risk_score'],
            'regulatory_risk': 1.0 - regulatory_risk['compliance_score']
        }
    
    def _generate_risk_report(self, 
                             combined_risk: Dict[str, float],
                             historical_risk: Dict,
                             regulatory_risk: Dict,
                             environmental_risk: Dict,
                             infrastructure_risk: Dict) -> Dict[str, Any]:
        # Generate comprehensive risk report
        # Args:
        #   combined_risk: Combined risk scores
        #   historical_risk: Historical risk assessment
        #   regulatory_risk: Regulatory compliance assessment
        #   environmental_risk: Environmental risk assessment
        #   infrastructure_risk: Infrastructure risk assessment
        # Returns: Comprehensive risk report
        # TODO: Add risk trend analysis
        # TODO: Include risk mitigation recommendations
        logger.debug("Generating comprehensive risk report")
        
        return {
            'risk_scores': combined_risk,
            'risk_levels': self._calculate_risk_levels(combined_risk),
            'historical_analysis': historical_risk,
            'regulatory_compliance': regulatory_risk,
            'environmental_factors': environmental_risk,
            'infrastructure_factors': infrastructure_risk,
            'risk_factors': self._compile_risk_factors(
                historical_risk, regulatory_risk, environmental_risk, infrastructure_risk
            ),
            'recommendations': self._generate_risk_recommendations(combined_risk),
            'assessment_date': datetime.now().isoformat(),
            'confidence_score': self._calculate_confidence_score(
                historical_risk, regulatory_risk, environmental_risk, infrastructure_risk
            )
        }
    
    def _calculate_risk_levels(self, risk_scores: Dict[str, float]) -> Dict[str, str]:
        # Calculate risk levels based on scores
        # Args:
        #   risk_scores: Dictionary of risk scores
        # Returns: Dictionary of risk levels
        risk_levels = {}
        
        for risk_type, score in risk_scores.items():
            if score >= self.risk_thresholds.get(risk_type, {}).get('critical', 0.9):
                risk_levels[risk_type] = 'critical'
            elif score >= self.risk_thresholds.get(risk_type, {}).get('high', 0.7):
                risk_levels[risk_type] = 'high'
            elif score >= self.risk_thresholds.get(risk_type, {}).get('medium', 0.5):
                risk_levels[risk_type] = 'medium'
            else:
                risk_levels[risk_type] = 'low'
        
        return risk_levels
    
    def _compile_risk_factors(self, 
                             historical_risk: Dict,
                             regulatory_risk: Dict,
                             environmental_risk: Dict,
                             infrastructure_risk: Dict) -> List[str]:
        # Compile all risk factors into a single list
        # Args:
        #   historical_risk: Historical risk assessment
        #   regulatory_risk: Regulatory compliance assessment
        #   environmental_risk: Environmental risk assessment
        #   infrastructure_risk: Infrastructure risk assessment
        # Returns: List of all risk factors
        risk_factors = []
        
        risk_factors.extend(historical_risk.get('risk_factors', []))
        risk_factors.extend(regulatory_risk.get('compliance_issues', []))
        risk_factors.extend(environmental_risk.get('risk_factors', []))
        risk_factors.extend(infrastructure_risk.get('risk_factors', []))
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _generate_risk_recommendations(self, risk_scores: Dict[str, float]) -> List[str]:
        # Generate risk mitigation recommendations
        # Args:
        #   risk_scores: Dictionary of risk scores
        # Returns: List of recommendations
        recommendations = []
        
        if risk_scores.get('environmental_risk', 0) > 0.7:
            recommendations.extend([
                "Conduct detailed environmental impact assessment",
                "Implement environmental monitoring systems",
                "Develop environmental contingency plans"
            ])
        
        if risk_scores.get('infrastructure_risk', 0) > 0.7:
            recommendations.extend([
                "Schedule infrastructure inspection and maintenance",
                "Consider infrastructure upgrades or replacement",
                "Implement structural monitoring systems"
            ])
        
        if risk_scores.get('construction_risk', 0) > 0.7:
            recommendations.extend([
                "Review and update safety protocols",
                "Conduct comprehensive site safety assessment",
                "Implement additional safety measures"
            ])
        
        if risk_scores.get('regulatory_risk', 0) > 0.7:
            recommendations.extend([
                "Conduct regulatory compliance audit",
                "Address compliance issues immediately",
                "Establish compliance monitoring procedures"
            ])
        
        if not recommendations:
            recommendations.append("Risks appear manageable - proceed with standard protocols")
        
        return recommendations
    
    def _calculate_confidence_score(self, 
                                   historical_risk: Dict,
                                   regulatory_risk: Dict,
                                   environmental_risk: Dict,
                                   infrastructure_risk: Dict) -> float:
        # Calculate confidence score based on data availability
        # Args:
        #   historical_risk: Historical risk assessment
        #   regulatory_risk: Regulatory compliance assessment
        #   environmental_risk: Environmental risk assessment
        #   infrastructure_risk: Infrastructure risk assessment
        # Returns: Confidence score between 0 and 1
        confidence_factors = []
        
        # Historical data confidence
        if historical_risk.get('incident_count', 0) > 0:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Regulatory data confidence
        if regulatory_risk.get('compliance_issues'):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.7)
        
        # Environmental data confidence
        if environmental_risk.get('risk_factors'):
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.6)
        
        # Infrastructure data confidence
        if infrastructure_risk.get('risk_factors'):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Calculate distance between two points using Haversine formula
        # Args:
        #   lat1, lon1: First point coordinates
        #   lat2, lon2: Second point coordinates
        # Returns: Distance in kilometers
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
        
        return R * c
    
    def _create_sample_incident_data(self) -> Dict[str, Any]:
        # Create sample historical incident data
        # Returns: Sample incident data structure
        return {
            'incidents': [
                {
                    'id': 1,
                    'year': 2022,
                    'latitude': -37.8136,
                    'longitude': 144.9631,
                    'severity': 7,
                    'type': 'pipe_burst',
                    'description': 'Major pipe burst in CBD area'
                },
                {
                    'id': 2,
                    'year': 2021,
                    'latitude': -37.8500,
                    'longitude': 145.0000,
                    'severity': 5,
                    'type': 'flooding',
                    'description': 'Stormwater system overflow'
                },
                {
                    'id': 3,
                    'year': 2020,
                    'latitude': -37.7500,
                    'longitude': 144.9000,
                    'severity': 6,
                    'type': 'structural_damage',
                    'description': 'Bridge structural damage'
                }
            ]
        }
    
    def _create_sample_regulatory_data(self) -> Dict[str, Any]:
        # Create sample regulatory requirements data
        # Returns: Sample regulatory data structure
        return {
            'requirements': [
                'Maximum pipe age: 50 years',
                'Minimum safety rating: 0.7',
                'Environmental impact assessment required',
                'Regular maintenance schedule mandatory'
            ]
        }
    
    def _analyze_severity_distribution(self, incidents: List[Dict]) -> Dict[str, int]:
        # Analyze severity distribution of incidents
        # Args:
        #   incidents: List of incident dictionaries
        # Returns: Severity distribution dictionary
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for incident in incidents:
            severity = incident.get('severity', 0)
            if severity <= 3:
                severity_counts['low'] += 1
            elif severity <= 5:
                severity_counts['medium'] += 1
            elif severity <= 7:
                severity_counts['high'] += 1
            else:
                severity_counts['critical'] += 1
        
        return severity_counts
    
    def _identify_historical_risk_factors(self, incidents: List[Dict]) -> List[str]:
        # Identify risk factors from historical incidents
        # Args:
        #   incidents: List of incident dictionaries
        # Returns: List of identified risk factors
        risk_factors = []
        
        incident_types = [incident.get('type', 'unknown') for incident in incidents]
        
        if 'pipe_burst' in incident_types:
            risk_factors.append('History of pipe failures')
        
        if 'flooding' in incident_types:
            risk_factors.append('History of flooding incidents')
        
        if 'structural_damage' in incident_types:
            risk_factors.append('History of structural damage')
        
        return risk_factors
    
    def _generate_compliance_actions(self, compliance_issues: List[str]) -> List[str]:
        # Generate actions to address compliance issues
        # Args:
        #   compliance_issues: List of compliance issues
        # Returns: List of required actions
        actions = []
        
        for issue in compliance_issues:
            if 'age' in issue.lower():
                actions.append('Schedule infrastructure replacement')
            elif 'material' in issue.lower():
                actions.append('Conduct material compatibility assessment')
            elif 'environmental' in issue.lower():
                actions.append('Implement environmental protection measures')
            elif 'safety' in issue.lower():
                actions.append('Enhance safety protocols and training')
        
        return actions 