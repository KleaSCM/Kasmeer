# Author: KleaSCM
# Date: 2024
# Description: Survey analysis module for civil engineering projects

import numpy as np
from typing import Dict, List, Optional
import logging
from ..utils.logging_utils import setup_logging, log_performance

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
            surveys = []  # Initialize list to accumulate required survey types

            # Identify survey needs based on missing data features (gap analysis)
            if not features.get('infrastructure'):
                surveys.append("Infrastructure survey")  # Suggest survey to collect infrastructure data

            if not features.get('climate'):
                surveys.append("Environmental survey")  # Suggest survey to gather climate or environmental data

            if not features.get('vegetation'):
                surveys.append("Vegetation survey")  # Suggest vegetation-related survey if data is absent

            # Risk-driven survey suggestions: add surveys if risk exceeds predefined threshold

            flood_risk_threshold = features.get('flood_risk_threshold', 0.5)  # Default flood risk threshold
            if features.get('flood_risk', 0) > flood_risk_threshold:
                surveys.append("Hydrological survey")  # Suggest hydrological survey to assess flood-prone areas

            soil_risk_threshold = features.get('soil_risk_threshold', 0.5)  # Default soil risk threshold
            if features.get('soil_risk', 0) > soil_risk_threshold:
                surveys.append("Geotechnical survey")  # Suggest geotechnical survey to evaluate soil stability

            return surveys  # Return compiled list of recommended surveys

            
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
            methods = []  # Initialize list to accumulate recommended survey methods

            # If infrastructure data is missing, suggest default or provided infrastructure survey methods
            if not features.get('infrastructure'):
                methods.extend(features.get(
                    'infrastructure_survey_methods', 
                    ["Ground penetrating radar", "Visual inspection"]
                ))

            # If climate data is missing, suggest default or provided climate survey methods
            if not features.get('climate'):
                methods.extend(features.get(
                    'climate_survey_methods', 
                    ["Environmental monitoring", "Climate data collection"]
                ))

            # If vegetation data is missing, suggest default or provided vegetation survey methods
            if not features.get('vegetation'):
                methods.extend(features.get(
                    'vegetation_survey_methods', 
                    ["Vegetation mapping", "Satellite imagery analysis"]
                ))

            # If flood risk exceeds threshold, recommend flood-related survey techniques
            flood_risk_threshold = features.get('flood_risk_threshold', 0.5)
            if features.get('flood_risk', 0) > flood_risk_threshold:
                methods.extend(features.get(
                    'flood_survey_methods', 
                    ["Flood modeling", "Water level monitoring"]
                ))

            # If soil risk exceeds threshold, recommend soil-related survey methods
            soil_risk_threshold = features.get('soil_risk_threshold', 0.5)
            if features.get('soil_risk', 0) > soil_risk_threshold:
                methods.extend(features.get(
                    'soil_survey_methods', 
                    ["Soil testing", "Borehole drilling"]
                ))

            # Return final list of recommended survey methods based on available data and risk levels
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
            # Initialize structure to hold cost analysis output
            cost_breakdown = {
                'total_estimated_cost': 0.0,        # Running total of all estimated survey costs
                'cost_breakdown': {},               # Itemized cost per survey type
                'cost_factors': [],                 # List of factors influencing total cost (e.g. risk, data gaps)
                'budget_recommendations': []        # Suggestions for budget optimization or priority spending
            }

            # Retrieve base survey costs from feature input (e.g., learned by neural net or defined manually)
            # Fallback to default values if no cost data is provided in features
            survey_costs = features.get('survey_cost_data', {
                'geotechnical_survey': 15000.0,
                'environmental_survey': 12000.0,
                'infrastructure_survey': 8000.0,
                'topographic_survey': 5000.0,
                'hydrological_survey': 10000.0,
                'soil_survey': 7000.0,
                'vegetation_survey': 6000.0
            })

            
            # Identify which surveys are needed based on data gaps and risk thresholds
            required_surveys = self.identify_required_surveys(features)
            total_cost = 0.0  # Initialize total cost accumulator

            for survey in required_surveys:
                # Normalize survey name to match keys in cost dictionary
                survey_lower = survey.lower().replace(' ', '_')

                # Retrieve base cost for the survey; use fallback cost if unknown
                base_cost = survey_costs.get(survey_lower, 10000.0)

                # Initialize cost multiplier (to be adjusted based on site-specific conditions)
                cost_multiplier = 1.0

                # Retrieve threshold values for risk-driven cost scaling
                flood_risk_threshold = features.get('flood_risk_threshold', 0.5)
                soil_risk_threshold = features.get('soil_risk_threshold', 0.5)
                infrastructure_density_threshold = features.get('infrastructure_density_threshold', 50)

                # Increase cost if flood risk exceeds threshold
                if features.get('flood_risk', 0) > flood_risk_threshold:
                    cost_multiplier += features.get('flood_cost_multiplier', 0.3)

                # Increase cost if soil risk exceeds threshold
                if features.get('soil_risk', 0) > soil_risk_threshold:
                    cost_multiplier += features.get('soil_cost_multiplier', 0.2)

                # Increase cost if infrastructure density is high
                if features.get('infrastructure', {}).get('count', 0) > infrastructure_density_threshold:
                    cost_multiplier += features.get('infrastructure_cost_multiplier', 0.1)

                # Apply total multiplier to base cost
                adjusted_cost = base_cost * cost_multiplier

                # Store detailed breakdown for the current survey
                cost_breakdown['cost_breakdown'][survey] = {
                    'base_cost': base_cost,                        # Original cost before adjustments
                    'adjusted_cost': adjusted_cost,                # Final cost after applying multipliers
                    'cost_multiplier': cost_multiplier,            # Total multiplier used
                    'factors': self._identify_cost_factors(features, survey)  # List of contributing cost factors
                }

                # Update running total
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
            # Initialize priority scoring structure to guide survey sequencing and urgency
            priority_scores = {
                'overall_priority': 'medium',      # Default overall priority; may be updated after analysis
                'priority_breakdown': {},          # Detailed priority score per survey type
                'priority_factors': [],            # Factors influencing priority (e.g., risk, data absence)
                'recommended_sequence': []         # Ordered list of surveys based on computed priority
            }

            # Retrieve priority criteria for each survey, usually based on ML-derived domain analysis
            # Each entry includes:
            # - base_priority: default importance score [0.0 – 1.0]
            # - risk_factors: contextual risk indicators that raise priority
            # - data_gaps: missing fields that also raise priority
            priority_criteria = features.get('survey_priority_criteria', {
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
            })

            
            # Calculate priority scores for each survey based on risk and data gaps
            survey_priorities = {}

            for survey, criteria in priority_criteria.items():
                # Start with the base priority score defined for the survey
                priority_score = criteria['base_priority']

                # ────────────── Risk Factor Adjustment ──────────────
                # Boost priority based on specific risk factors tied to this survey
                for risk_factor in criteria['risk_factors']:
                    risk_value = features.get(risk_factor, 0)  # Retrieve actual risk value from features
                    high_risk_threshold = features.get('high_risk_threshold', 0.5)
                    moderate_risk_threshold = features.get('moderate_risk_threshold', 0.3)

                    # Apply larger boost if risk is high, smaller if moderate
                    if risk_value > high_risk_threshold:
                        priority_score += features.get('high_risk_priority_boost', 0.2)
                    elif risk_value > moderate_risk_threshold:
                        priority_score += features.get('moderate_risk_priority_boost', 0.1)

                # ────────────── Data Completeness Adjustment ──────────────
                # Evaluate how complete the relevant data is for this survey
                data_completeness = self._calculate_data_completeness(features)
                low_completeness_threshold = features.get('low_completeness_threshold', 0.3)
                moderate_completeness_threshold = features.get('moderate_completeness_threshold', 0.6)

                # Boost priority if data completeness is low or moderate
                if data_completeness < low_completeness_threshold:
                    priority_score += features.get('low_completeness_priority_boost', 0.3)
                elif data_completeness < moderate_completeness_threshold:
                    priority_score += features.get('moderate_completeness_priority_boost', 0.1)

                # ────────────── Final Score Normalization ──────────────
                # Cap the final score to a max of 1.0 to ensure consistent bounds
                priority_score = min(1.0, priority_score)

                # Save computed results for this survey
                survey_priorities[survey] = {
                    'priority_score': priority_score,  # Final numeric score (0.0–1.0)
                    'priority_level': self._score_to_priority_level(priority_score, features),  # Convert to human-readable level (e.g., high/medium/low)
                    'urgency': self._calculate_urgency(priority_score, features)  # Compute urgency signal (e.g., timeframe or sequencing weight)
                }

            
            # ────────────── Finalize Overall Priority Scores ──────────────

            # Compute the average priority score across all identified surveys
            avg_priority = sum(p['priority_score'] for p in survey_priorities.values()) / len(survey_priorities)

            # Convert the average score to a qualitative priority level (e.g., low, medium, high)
            priority_scores['overall_priority'] = self._score_to_priority_level(avg_priority, features)

            # Store the full detailed breakdown of each survey’s priority info
            priority_scores['priority_breakdown'] = survey_priorities

            # Identify and store the high-level factors that influenced priority computation
            priority_scores['priority_factors'] = self._identify_priority_factors(features)

            # Generate and store the recommended order in which surveys should be executed
            priority_scores['recommended_sequence'] = self._generate_survey_sequence(survey_priorities)

            # Return the complete structured priority analysis
            return priority_scores

        except Exception as e:
            # If any exception occurs during priority analysis, log and return fallback structure
            logger.error(f"Error calculating priority scores: {e}")
            return {
                'overall_priority': 'medium',                       # Fallback priority level
                'priority_breakdown': {},                           # No survey-level scores
                'priority_factors': ['Error in priority calculation'],  # Indicate calculation failure
                'recommended_sequence': []                          # No sequencing generated
            }

    
    @log_performance(logger)
    def identify_data_gaps(self, features: Dict) -> List[str]:
        # Identify data gaps in the available features
        # Args:
        #   features: Features dictionary
        # Returns: List of data gaps
        logger.debug("Identifying data gaps")
        
        try:
            gaps = []  # Initialize list to collect all detected data gaps

            # ────── Presence-based gap detection ──────
            # Check if key data categories are completely missing

            if not features.get('infrastructure'):
                gaps.append("Infrastructure data missing")  # No infrastructure object provided

            if not features.get('climate'):
                gaps.append("Climate data missing")  # No climate/environmental object present

            if not features.get('vegetation'):
                gaps.append("Vegetation data missing")  # No vegetation object provided

            # ────── Quality-based gap detection ──────
            # Check if available data is below acceptable coverage/threshold levels

            infrastructure_coverage_threshold = features.get('infrastructure_coverage_threshold', 5)
            if features.get('infrastructure', {}).get('count', 0) < infrastructure_coverage_threshold:
                gaps.append("Limited infrastructure coverage")  # Too few infrastructure entries present

            vegetation_zones_threshold = features.get('vegetation_zones_threshold', 2)
            if features.get('vegetation', {}).get('zones_count', 0) < vegetation_zones_threshold:
                gaps.append("Limited vegetation data")  # Insufficient vegetation zone detail

            # Return compiled list of all detected data gaps
            return gaps

        except Exception as e:
            # Handle unexpected failures (e.g., malformed input structure)
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
            recommendations = []  # Initialize list to accumulate survey action suggestions

            # ────── Data Absence-Based Recommendations ──────
            # Suggest surveys based on entirely missing categories

            if not features.get('infrastructure'):
                recommendations.append("Conduct infrastructure survey")  # Infrastructure data is missing

            if not features.get('climate'):
                recommendations.append("Conduct environmental survey")  # Climate/environmental data is missing

            if not features.get('vegetation'):
                recommendations.append("Conduct vegetation survey")  # Vegetation data is missing

            # ────── Risk-Based Recommendations ──────
            # Suggest additional assessments based on high-risk indicators

            flood_risk_threshold = features.get('flood_risk_threshold', 0.5)
            if features.get('flood_risk', 0) > flood_risk_threshold:
                recommendations.append("Conduct flood risk assessment")  # Elevated flood risk detected

            soil_risk_threshold = features.get('soil_risk_threshold', 0.5)
            if features.get('soil_risk', 0) > soil_risk_threshold:
                recommendations.append("Conduct geotechnical investigation")  # Soil instability risk is high

            # Return the full set of generated survey recommendations
            return recommendations

        except Exception as e:
            # Handle unexpected failures (e.g., missing keys, type errors)
            logger.error(f"Error generating survey recommendations: {e}")
            return ["Error in recommendation generation"]

    
    def _identify_cost_factors(self, features: Dict, survey: str) -> List[str]:
        # Identify cost factors for a specific survey
        # Args:
        #   features: Features dictionary
        #   survey: Survey type
        # Returns: List of cost factors
        factors = []  # Initialize list to store detected priority-affecting factors

        # Check if flood risk exceeds threshold
        flood_risk_threshold = features.get('flood_risk_threshold', 0.5)
        if features.get('flood_risk', 0) > flood_risk_threshold:
            factors.append("High flood risk area")  # Indicates elevated need for hydrological attention

        # Check if soil risk exceeds threshold
        soil_risk_threshold = features.get('soil_risk_threshold', 0.5)
        if features.get('soil_risk', 0) > soil_risk_threshold:
            factors.append("Complex soil conditions")  # Suggests geotechnical complications or instability

        # Check if infrastructure density exceeds operational threshold
        infrastructure_density_threshold = features.get('infrastructure_density_threshold', 50)
        if features.get('infrastructure', {}).get('count', 0) > infrastructure_density_threshold:
            factors.append("High infrastructure density")  # Signals need for detailed infrastructure coordination

        return factors


    def _identify_overall_cost_factors(self, features: Dict) -> List[str]:
        # Identify overall cost-driving conditions based on site risks and data availability
        # Args:
        #   features: Dictionary of input features and analysis values
        # Returns:
        #   List of key cost factors influencing survey or construction cost

        factors = []  # Initialize list to hold cost-relevant conditions

        # Check flood risk and add mitigation-related cost driver if applicable
        flood_risk_threshold = features.get('flood_risk_threshold', 0.5)
        if features.get('flood_risk', 0) > flood_risk_threshold:
            factors.append("Flood risk mitigation required")

        # Check soil risk and add stabilization-related cost factor
        soil_risk_threshold = features.get('soil_risk_threshold', 0.5)
        if features.get('soil_risk', 0) > soil_risk_threshold:
            factors.append("Soil stabilization needed")

        # Lack of infrastructure data may increase survey or planning costs
        if not features.get('infrastructure'):
            factors.append("No existing infrastructure data")

        return factors

    
    def _generate_budget_recommendations(self, total_cost: float, features: Dict) -> List[str]:
        # Generate budget recommendations
        # Args:
        #   total_cost: Total estimated cost
        #   features: Features dictionary
        # Returns: List of budget recommendations
        recommendations = []  # Initialize list to store budget-related recommendations

        # Retrieve cost thresholds from features or use defaults
        high_cost_threshold = features.get('high_cost_threshold', 50000)
        moderate_cost_threshold = features.get('moderate_cost_threshold', 25000)

        # ────── Cost Tier-Based Recommendation Logic ──────
        # Append recommendations based on the total projected survey cost

        if total_cost > high_cost_threshold:
            # High-cost scenario — suggest cost containment and phased strategies
            recommendations.extend(features.get(
                'high_cost_recommendations',
                ["Consider phased survey approach", "Prioritize critical surveys first"]
            ))

        elif total_cost > moderate_cost_threshold:
            # Moderate-cost range — suggest budgeting awareness
            recommendations.extend(features.get(
                'moderate_cost_recommendations',
                ["Allocate sufficient budget for comprehensive surveys"]
            ))

        else:
            # Low-cost range — standard recommendation
            recommendations.extend(features.get(
                'low_cost_recommendations',
                ["Standard survey budget should be adequate"]
            ))

        return recommendations


    def _calculate_data_completeness(self, features: Dict) -> float:
        # Calculate the proportion of available core datasets (infrastructure, climate, vegetation)
        # Args:
        #   features: dictionary of extracted input features
        # Returns:
        #   Float between 0.0 and 1.0 representing how complete the data inputs are

        try:
            total_features = 0             # Tracks how many core feature categories were expected
            available_features = 0         # Tracks how many of those categories are populated

            # Check for infrastructure presence and count data
            if 'infrastructure' in features:
                total_features += 1
                if features['infrastructure'].get('count', 0) > 0:
                    available_features += 1

            # Check for presence of climate data (non-null/empty)
            if 'climate' in features:
                total_features += 1
                if features['climate']:
                    available_features += 1

            # Check for presence of vegetation data (non-null/empty)
            if 'vegetation' in features:
                total_features += 1
                if features['vegetation']:
                    available_features += 1

            # Return ratio of available features to expected features (0.0–1.0)
            return available_features / total_features if total_features > 0 else 0.0

        except Exception as e:
            # Handle unexpected failures gracefully and log them
            logger.error(f"Error calculating data completeness: {e}")
            return 0.0


    def _score_to_priority_level(self, score: float, features: Dict) -> str:
        # Convert numeric priority score to a qualitative level string
        # Args:
        #   score: Computed priority score (float between 0.0 and 1.0)
        #   features: Configuration dictionary containing thresholds
        # Returns:
        #   A string representing the qualitative priority level

        critical_threshold = features.get('critical_priority_threshold', 0.8)  # Above this: 'critical'
        high_threshold = features.get('high_priority_threshold', 0.6)          # Above this: 'high'
        medium_threshold = features.get('medium_priority_threshold', 0.4)      # Above this: 'medium'

        # Compare against thresholds to assign priority level
        if score > critical_threshold:
            return 'critical'
        elif score > high_threshold:
            return 'high'
        elif score > medium_threshold:
            return 'medium'
        else:
            return 'low'

    
    def _calculate_urgency(self, priority_score: float, features: Dict) -> str:
        # Determine the urgency level of a survey based on its priority score and key risk indicators
        # Args:
        #   priority_score: Computed priority score for the survey
        #   features: Feature dictionary containing thresholds and risk values
        # Returns:
        #   String indicating urgency level: 'low', 'medium', 'high', or 'immediate'

        immediate_threshold = features.get('immediate_urgency_threshold', 0.8)
        high_urgency_threshold = features.get('high_urgency_threshold', 0.6)
        medium_urgency_threshold = features.get('medium_urgency_threshold', 0.4)
        flood_risk_urgency_threshold = features.get('flood_risk_urgency_threshold', 0.7)

        # If the score is very high or flood risk is critical, treat as immediate
        if priority_score > immediate_threshold or features.get('flood_risk', 0) > flood_risk_urgency_threshold:
            return 'immediate'
        elif priority_score > high_urgency_threshold:
            return 'high'
        elif priority_score > medium_urgency_threshold:
            return 'medium'
        else:
            return 'low'


    def _identify_priority_factors(self, features: Dict) -> List[str]:
        # Identify key factors contributing to elevated survey priority
        # Args:
        #   features: Feature dictionary from input data
        # Returns:
        #   List of human-readable strings describing priority-driving conditions

        factors = []

        flood_risk_threshold = features.get('flood_risk_threshold', 0.5)
        if features.get('flood_risk', 0) > flood_risk_threshold:
            factors.append("High flood risk")

        soil_risk_threshold = features.get('soil_risk_threshold', 0.5)
        if features.get('soil_risk', 0) > soil_risk_threshold:
            factors.append("Soil stability concerns")

        if not features.get('infrastructure'):
            factors.append("Missing infrastructure data")

        return factors


    def _generate_survey_sequence(self, survey_priorities: Dict) -> List[str]:
        # Generate a recommended execution sequence for surveys based on descending priority
        # Args:
        #   survey_priorities: Dictionary mapping survey names to priority metadata
        # Returns:
        #   Ordered list of survey names from highest to lowest priority

        try:
            # Sort surveys by their computed priority score, descending
            sorted_surveys = sorted(
                survey_priorities.items(),
                key=lambda x: x[1]['priority_score'],
                reverse=True
            )

            # Extract only the survey names from the sorted results
            return [survey for survey, _ in sorted_surveys]

        except Exception as e:
            # Log and fail gracefully if priority metadata is malformed
            logger.error(f"Error generating survey sequence: {e}")
            return []
