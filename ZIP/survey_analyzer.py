# Author: KleaSCM
# Date: 2024
# Description: Survey analysis module for civil engineering projects

import numpy as np
from typing import Dict, List, Optional
import logging
from utils.logging_utils import setup_logging, log_performance

logger = setup_logging(__name__)

class SurveyAnalyzer:
    # Survey analyzer for civil engineering projects
    # This module handles survey requirements, cost estimation, and priority scoring
    # TODO: Add survey method optimization algorithms
    # TODO: Add survey scheduling optimization
    # TODO: Add survey quality assessment metrics
    
    @log_performance(logger)
    def __init__(self):
        # Initialize the survey analyzer
        logger.info("Initialized SurveyAnalyzer")
    
    @log_performance(logger)
    def identify_required_surveys(self, features: Dict) -> List[str]:
        # Identify required surveys based on data gaps
        # Args:
        #   features: Features dictionary
        # Returns: List of required surveys
        logger.debug("Identifying required surveys")
        
        try:
            surveys = []
            
            if not features.get('infrastructure'):
                surveys.append("Infrastructure survey")
            
            if not features.get('climate'):
                surveys.append("Environmental survey")
            
            if not features.get('vegetation'):
                surveys.append("Vegetation survey")
            
            # Add surveys based on risk factors
            if features.get('flood_risk', 0) > 0.5:
                surveys.append("Hydrological survey")
            
            if features.get('soil_risk', 0) > 0.5:
                surveys.append("Geotechnical survey")
            
            return surveys
            
        except Exception as e:
            logger.error(f"Error identifying required surveys: {e}")
            return ["Error in survey identification"]
    
    @log_performance(logger)
    def recommend_survey_methods(self, features: Dict) -> List[str]:
        # Recommend survey methods based on site conditions
        # Args:
        #   features: Features dictionary
        # Returns: List of recommended survey methods
        logger.debug("Recommending survey methods")
        
        try:
            methods = []
            
            if not features.get('infrastructure'):
                methods.append("Ground penetrating radar")
                methods.append("Visual inspection")
            
            if not features.get('climate'):
                methods.append("Environmental monitoring")
                methods.append("Climate data collection")
            
            if not features.get('vegetation'):
                methods.append("Vegetation mapping")
                methods.append("Satellite imagery analysis")
            
            if features.get('flood_risk', 0) > 0.5:
                methods.append("Flood modeling")
                methods.append("Water level monitoring")
            
            if features.get('soil_risk', 0) > 0.5:
                methods.append("Soil testing")
                methods.append("Borehole drilling")
            
            return methods
            
        except Exception as e:
            logger.error(f"Error recommending survey methods: {e}")
            return ["Error in method recommendation"]
    
    @log_performance(logger)
    def estimate_survey_costs(self, features: Dict) -> Dict:
        # Estimate detailed survey costs for different survey types
        # Args:
        #   features: Features dictionary
        # Returns: Detailed cost estimation dictionary
        logger.debug("Estimating survey costs")
        
        try:
            cost_breakdown = {
                'total_estimated_cost': 0.0,
                'cost_breakdown': {},
                'cost_factors': [],
                'budget_recommendations': []
            }
            
            # Base costs for different survey types
            survey_costs = {
                'geotechnical_survey': 15000.0,
                'environmental_survey': 12000.0,
                'infrastructure_survey': 8000.0,
                'topographic_survey': 5000.0,
                'hydrological_survey': 10000.0,
                'soil_survey': 7000.0,
                'vegetation_survey': 6000.0
            }
            
            # Identify required surveys
            required_surveys = self.identify_required_surveys(features)
            total_cost = 0.0
            
            for survey in required_surveys:
                survey_lower = survey.lower().replace(' ', '_')
                base_cost = survey_costs.get(survey_lower, 10000.0)
                
                # Apply cost factors based on site conditions
                cost_multiplier = 1.0
                if features.get('flood_risk', 0) > 0.5:
                    cost_multiplier += 0.3
                if features.get('soil_risk', 0) > 0.5:
                    cost_multiplier += 0.2
                if features.get('infrastructure', {}).get('count', 0) > 50:
                    cost_multiplier += 0.1
                
                adjusted_cost = base_cost * cost_multiplier
                cost_breakdown['cost_breakdown'][survey] = {
                    'base_cost': base_cost,
                    'adjusted_cost': adjusted_cost,
                    'cost_multiplier': cost_multiplier,
                    'factors': self._identify_cost_factors(features, survey)
                }
                
                total_cost += adjusted_cost
            
            cost_breakdown['total_estimated_cost'] = total_cost
            cost_breakdown['cost_factors'] = self._identify_overall_cost_factors(features)
            cost_breakdown['budget_recommendations'] = self._generate_budget_recommendations(total_cost, features)
            
            return cost_breakdown
            
        except Exception as e:
            logger.error(f"Error estimating survey costs: {e}")
            return {
                'total_estimated_cost': 10000.0,
                'cost_breakdown': {},
                'cost_factors': ['Error in cost estimation'],
                'budget_recommendations': ['Conduct manual cost assessment']
            }
    
    @log_performance(logger)
    def calculate_priority_scores(self, features: Dict) -> Dict:
        # Calculate priority scores for different survey types
        # Args:
        #   features: Features dictionary
        # Returns: Priority scoring dictionary
        logger.debug("Calculating survey priority scores")
        
        try:
            priority_scores = {
                'overall_priority': 'medium',
                'priority_breakdown': {},
                'priority_factors': [],
                'recommended_sequence': []
            }
            
            # Define priority criteria
            priority_criteria = {
                'geotechnical_survey': {
                    'base_priority': 0.7,
                    'risk_factors': ['soil_risk', 'flood_risk'],
                    'data_gaps': ['soil_data', 'foundation_data']
                },
                'environmental_survey': {
                    'base_priority': 0.6,
                    'risk_factors': ['environmental_risk', 'climate_risk'],
                    'data_gaps': ['environmental_data', 'climate_data']
                },
                'infrastructure_survey': {
                    'base_priority': 0.8,
                    'risk_factors': ['infrastructure_risk'],
                    'data_gaps': ['infrastructure_data']
                },
                'topographic_survey': {
                    'base_priority': 0.5,
                    'risk_factors': ['site_conditions'],
                    'data_gaps': ['topographic_data']
                }
            }
            
            # Calculate priority for each survey type
            survey_priorities = {}
            for survey, criteria in priority_criteria.items():
                priority_score = criteria['base_priority']
                
                # Adjust based on risk factors
                for risk_factor in criteria['risk_factors']:
                    risk_value = features.get(risk_factor, 0)
                    if risk_value > 0.5:
                        priority_score += 0.2
                    elif risk_value > 0.3:
                        priority_score += 0.1
                
                # Adjust based on data gaps
                data_completeness = self._calculate_data_completeness(features)
                if data_completeness < 0.3:
                    priority_score += 0.3
                elif data_completeness < 0.6:
                    priority_score += 0.1
                
                # Normalize to 0-1 range
                priority_score = min(1.0, priority_score)
                
                survey_priorities[survey] = {
                    'priority_score': priority_score,
                    'priority_level': self._score_to_priority_level(priority_score),
                    'urgency': self._calculate_urgency(priority_score, features)
                }
            
            # Determine overall priority
            avg_priority = sum(p['priority_score'] for p in survey_priorities.values()) / len(survey_priorities)
            priority_scores['overall_priority'] = self._score_to_priority_level(avg_priority)
            priority_scores['priority_breakdown'] = survey_priorities
            priority_scores['priority_factors'] = self._identify_priority_factors(features)
            priority_scores['recommended_sequence'] = self._generate_survey_sequence(survey_priorities)
            
            return priority_scores
            
        except Exception as e:
            logger.error(f"Error calculating priority scores: {e}")
            return {
                'overall_priority': 'medium',
                'priority_breakdown': {},
                'priority_factors': ['Error in priority calculation'],
                'recommended_sequence': []
            }
    
    @log_performance(logger)
    def identify_data_gaps(self, features: Dict) -> List[str]:
        # Identify data gaps in the available features
        # Args:
        #   features: Features dictionary
        # Returns: List of data gaps
        logger.debug("Identifying data gaps")
        
        try:
            gaps = []
            
            if not features.get('infrastructure'):
                gaps.append("Infrastructure data missing")
            
            if not features.get('climate'):
                gaps.append("Climate data missing")
            
            if not features.get('vegetation'):
                gaps.append("Vegetation data missing")
            
            # Check for specific data quality issues
            if features.get('infrastructure', {}).get('count', 0) < 5:
                gaps.append("Limited infrastructure coverage")
            
            if features.get('vegetation', {}).get('zones_count', 0) < 2:
                gaps.append("Limited vegetation data")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error identifying data gaps: {e}")
            return ["Error in data gap identification"]
    
    @log_performance(logger)
    def generate_survey_recommendations(self, features: Dict) -> List[str]:
        # Generate survey recommendations based on data gaps
        # Args:
        #   features: Features dictionary
        # Returns: List of survey recommendations
        logger.debug("Generating survey recommendations")
        
        try:
            recommendations = []
            
            if not features.get('infrastructure'):
                recommendations.append("Conduct infrastructure survey")
            
            if not features.get('climate'):
                recommendations.append("Conduct environmental survey")
            
            if not features.get('vegetation'):
                recommendations.append("Conduct vegetation survey")
            
            # Add specific recommendations based on risks
            if features.get('flood_risk', 0) > 0.5:
                recommendations.append("Conduct flood risk assessment")
            
            if features.get('soil_risk', 0) > 0.5:
                recommendations.append("Conduct geotechnical investigation")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating survey recommendations: {e}")
            return ["Error in recommendation generation"]
    
    def _identify_cost_factors(self, features: Dict, survey: str) -> List[str]:
        # Identify cost factors for a specific survey
        # Args:
        #   features: Features dictionary
        #   survey: Survey type
        # Returns: List of cost factors
        factors = []
        
        if features.get('flood_risk', 0) > 0.5:
            factors.append("High flood risk area")
        
        if features.get('soil_risk', 0) > 0.5:
            factors.append("Complex soil conditions")
        
        if features.get('infrastructure', {}).get('count', 0) > 50:
            factors.append("High infrastructure density")
        
        return factors
    
    def _identify_overall_cost_factors(self, features: Dict) -> List[str]:
        # Identify overall cost factors
        # Args:
        #   features: Features dictionary
        # Returns: List of overall cost factors
        factors = []
        
        if features.get('flood_risk', 0) > 0.5:
            factors.append("Flood risk mitigation required")
        
        if features.get('soil_risk', 0) > 0.5:
            factors.append("Soil stabilization needed")
        
        if not features.get('infrastructure'):
            factors.append("No existing infrastructure data")
        
        return factors
    
    def _generate_budget_recommendations(self, total_cost: float, features: Dict) -> List[str]:
        # Generate budget recommendations
        # Args:
        #   total_cost: Total estimated cost
        #   features: Features dictionary
        # Returns: List of budget recommendations
        recommendations = []
        
        if total_cost > 50000:
            recommendations.append("Consider phased survey approach")
            recommendations.append("Prioritize critical surveys first")
        elif total_cost > 25000:
            recommendations.append("Allocate sufficient budget for comprehensive surveys")
        else:
            recommendations.append("Standard survey budget should be adequate")
        
        return recommendations
    
    def _calculate_data_completeness(self, features: Dict) -> float:
        # Calculate data completeness score
        # Args:
        #   features: Features dictionary
        # Returns: Completeness score between 0 and 1
        try:
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
            
        except Exception as e:
            logger.error(f"Error calculating data completeness: {e}")
            return 0.0
    
    def _score_to_priority_level(self, score: float) -> str:
        # Convert score to priority level
        # Args:
        #   score: Priority score
        # Returns: Priority level string
        if score > 0.8:
            return 'critical'
        elif score > 0.6:
            return 'high'
        elif score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_urgency(self, priority_score: float, features: Dict) -> str:
        # Calculate urgency level
        # Args:
        #   priority_score: Priority score
        #   features: Features dictionary
        # Returns: Urgency level string
        if priority_score > 0.8 or features.get('flood_risk', 0) > 0.7:
            return 'immediate'
        elif priority_score > 0.6:
            return 'high'
        elif priority_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _identify_priority_factors(self, features: Dict) -> List[str]:
        # Identify priority factors
        # Args:
        #   features: Features dictionary
        # Returns: List of priority factors
        factors = []
        
        if features.get('flood_risk', 0) > 0.5:
            factors.append("High flood risk")
        
        if features.get('soil_risk', 0) > 0.5:
            factors.append("Soil stability concerns")
        
        if not features.get('infrastructure'):
            factors.append("Missing infrastructure data")
        
        return factors
    
    def _generate_survey_sequence(self, survey_priorities: Dict) -> List[str]:
        # Generate recommended survey sequence
        # Args:
        #   survey_priorities: Survey priorities dictionary
        # Returns: List of surveys in recommended sequence
        try:
            # Sort surveys by priority score (highest first)
            sorted_surveys = sorted(
                survey_priorities.items(),
                key=lambda x: x[1]['priority_score'],
                reverse=True
            )
            
            return [survey for survey, _ in sorted_surveys]
            
        except Exception as e:
            logger.error(f"Error generating survey sequence: {e}")
            return [] 