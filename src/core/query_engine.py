# Author: KleaSCM
# Date: 2024
# Query Engine Module
# Description:  - Handles natural language queries and returns structured results

import re
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from ..utils.logging_utils import setup_logging, log_performance
from .report_formatter import ReportFormatter

logger = setup_logging(__name__)

@dataclass
class QueryResult:
    # Structured result from a query
    query: str
    location: Optional[Dict[str, float]] = None
    infrastructure_info: Optional[Dict] = None
    environmental_info: Optional[Dict] = None
    risk_assessment: Optional[Dict] = None
    construction_info: Optional[Dict] = None
    recommendations: Optional[List[str]] = None
    confidence: float = 0.0
    error: Optional[str] = None

class QueryEngine:
    # Natural language query engine for civil engineering data
    
    @log_performance(logger)
    def __init__(self, data_processor, neural_network):
        self.data_processor = data_processor
        self.neural_network = neural_network
        self.report_formatter = ReportFormatter()
        logger.info("Initialized QueryEngine")
        
        # Define query patterns
        self.location_patterns = [
            r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # lat, lon (with negative support)
            r'coordinates?\s*[:\-]?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',
            r'at\s+(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',
        ]
        
        self.query_types = {
            'infrastructure': [
                r'infrastructure',
                r'pipes?',
                r'drainage',
                r'water\s*system',
                r'sewer',
                r'utilities?'
            ],
            'environmental': [
                r'environmental',
                r'conservation',
                r'vegetation',
                r'climate',
                r'weather',
                r'ecological'
            ],
            'risk': [
                r'risk',
                r'danger',
                r'hazard',
                r'problem',
                r'issue',
                r'concern'
            ],
            'construction': [
                r'build',
                r'construction',
                r'development',
                r'project',
                r'planning'
            ],
            'survey': [
                r'survey',
                r'assessment',
                r'evaluation',
                r'study',
                r'report'
            ]
        }
    
    @log_performance(logger)
    def process_query(self, query: str) -> QueryResult:
        # Process a natural language query
        logger.info(f"Processing query: {query}")
        query = query.lower().strip()
        
        try:
            # Extract location
            location = self._extract_location(query)
            
            # Determine query type
            query_type = self._classify_query(query)
            
            # Process based on query type
            if query_type == 'infrastructure':
                return self._handle_infrastructure_query(query, location)
            elif query_type == 'environmental':
                return self._handle_environmental_query(query, location)
            elif query_type == 'risk':
                return self._handle_risk_query(query, location)
            elif query_type == 'construction':
                return self._handle_construction_query(query, location)
            elif query_type == 'survey':
                return self._handle_survey_query(query, location)
            else:
                return self._handle_general_query(query, location)
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResult(
                query=query,
                error=f"Error processing query: {str(e)}",
                confidence=0.0
            )
    
    def _extract_location(self, query: str) -> Optional[Dict[str, float]]:
        # Extract location coordinates from query
        for pattern in self.location_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    return {'lat': lat, 'lon': lon}
                except ValueError:
                    continue
        # If no coordinates found, try to extract location names
        # TODO: Integrate with geocoding service or require explicit coordinates
        return None
    
    def _classify_query(self, query: str) -> str:
        # Classify the type of query
        scores: Dict[str, int] = {}
        
        for query_type, patterns in self.query_types.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 1
            scores[query_type] = score
        
        # Return the type with highest score
        try:
            return max(scores, key=lambda k: scores.get(k, 0))
        except ValueError:
            return 'general'
    
    def _handle_infrastructure_query(self, query: str, location: Optional[Dict]) -> QueryResult:
        # Handle infrastructure-related queries using neural network predictions
        # Args:
        #   query: User query string
        #   location: Dictionary with lat/lon coordinates
        # Returns: QueryResult with infrastructure analysis from neural network
        # TODO: Add infrastructure health assessment
        # TODO: Include maintenance prediction
        if not location:
            return QueryResult(
                query=query,
                error="Location required for infrastructure queries",
                confidence=0.0
            )
        
        try:
            # Get infrastructure analysis from neural network
            # The neural network should provide infrastructure insights based on its training data
            infrastructure_prediction = self.neural_network.predict_infrastructure_analysis(
                location['lat'], location['lon'], self.data_processor
            )
            
            # Extract infrastructure information from neural network output
            # No hardcoded logic - everything comes from the trained model
            infrastructure_info = {
                'pipe_count': infrastructure_prediction.get('pipe_count', 0),
                'total_length': infrastructure_prediction.get('total_length', 0.0),
                'materials': infrastructure_prediction.get('materials', {}),
                'diameters': infrastructure_prediction.get('diameters', {}),
                'infrastructure_health': infrastructure_prediction.get('health_score', 0.0),
                'maintenance_needs': infrastructure_prediction.get('maintenance_needs', []),
                'upgrade_requirements': infrastructure_prediction.get('upgrade_requirements', []),
                'data_completeness': infrastructure_prediction.get('data_completeness', 0.0),
                'confidence': infrastructure_prediction.get('confidence', 0.0),
                'data_sources': infrastructure_prediction.get('data_sources', [])
            }
            
            # Get recommendations from neural network
            recommendations = infrastructure_prediction.get('recommendations', [])
        
        return QueryResult(
            query=query,
            location=location,
                infrastructure_info=infrastructure_info,
                recommendations=recommendations,
                confidence=infrastructure_prediction.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"Error in infrastructure analysis: {e}")
            return QueryResult(
                query=query,
                location=location,
                error=f"Error in infrastructure analysis: {str(e)}",
                confidence=0.0
        )
    
    def _handle_environmental_query(self, query: str, location: Optional[Dict]) -> QueryResult:
        # Handle environmental-related queries using neural network predictions
        # Args:
        #   query: User query string
        #   location: Dictionary with lat/lon coordinates
        # Returns: QueryResult with environmental analysis from neural network
        # TODO: Add environmental impact prediction
        # TODO: Include climate change assessment
        if not location:
            return QueryResult(
                query=query,
                error="Location required for environmental queries",
                confidence=0.0
            )
        
        try:
            # Get environmental analysis from neural network
            # The neural network should provide environmental insights based on its training data
            environmental_prediction = self.neural_network.predict_environmental_analysis(
                location['lat'], location['lon'], self.data_processor
        )
        
            # Extract environmental information from neural network output
            # No hardcoded logic - everything comes from the trained model
            environmental_info = {
                'climate_data': environmental_prediction.get('climate_data', {}),
                'vegetation_zones': environmental_prediction.get('vegetation_zones', {}),
                'soil_conditions': environmental_prediction.get('soil_conditions', {}),
                'water_resources': environmental_prediction.get('water_resources', {}),
                'environmental_risks': environmental_prediction.get('environmental_risks', []),
                'data_completeness': environmental_prediction.get('data_completeness', 0.0),
                'confidence': environmental_prediction.get('confidence', 0.0),
                'data_sources': environmental_prediction.get('data_sources', [])
        }
        
            # Get recommendations from neural network
            recommendations = environmental_prediction.get('recommendations', [])
            
            return QueryResult(
                query=query,
                location=location,
                environmental_info=environmental_info,
                recommendations=recommendations,
                confidence=environmental_prediction.get('confidence', 0.7)
            )
            
        except Exception as e:
            logger.error(f"Error in environmental analysis: {e}")
        return QueryResult(
            query=query,
            location=location,
                error=f"Error in environmental analysis: {str(e)}",
                confidence=0.0
        )
    
    def _handle_risk_query(self, query: str, location: Optional[Dict]) -> QueryResult:
        # Handle risk assessment queries using neural network predictions
        # Args:
        #   query: User query string
        #   location: Dictionary with lat/lon coordinates
        # Returns: QueryResult with risk assessment from neural network
        # TODO: Add confidence scoring based on data availability
        # TODO: Include uncertainty quantification
        if not location:
            return QueryResult(
                query=query,
                error="Location required for risk assessment",
                confidence=0.0
            )
        
        try:
            # Get comprehensive prediction from neural network
            # The neural network should provide all risk assessments based on its training data
            neural_prediction = self.neural_network.predict_at_location(
                location['lat'], location['lon'], self.data_processor
            )
            
            # Extract risk assessment from neural network output
            # No hardcoded logic - everything comes from the trained model
            risk_assessment = {
                'environmental_risk': neural_prediction.get('environmental_risk', 0.0),
                'infrastructure_risk': neural_prediction.get('infrastructure_risk', 0.0),
                'construction_risk': neural_prediction.get('construction_risk', 0.0),
                'overall_risk': neural_prediction.get('overall_risk', 0.0),
                'risk_factors': neural_prediction.get('risk_factors', []),
                'confidence': neural_prediction.get('confidence', 0.0),
                'data_sources': neural_prediction.get('data_sources', []),
                'model_version': neural_prediction.get('model_version', 'unknown')
            }
            
            # Get recommendations from neural network
            recommendations = neural_prediction.get('recommendations', [])
            
            return QueryResult(
                query=query,
                location=location,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                confidence=neural_prediction.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return QueryResult(
                query=query,
                location=location,
                error=f"Error in risk assessment: {str(e)}",
                confidence=0.0
            )
    
    def _handle_construction_query(self, query: str, location: Optional[Dict]) -> QueryResult:
        # Handle construction-related queries using neural network predictions
        # Args:
        #   query: User query string
        #   location: Dictionary with lat/lon coordinates
        # Returns: QueryResult with construction analysis from neural network
        # TODO: Add construction timeline prediction
        # TODO: Include resource requirement estimation
        if not location:
            return QueryResult(
                query=query,
                error="Location required for construction queries",
                confidence=0.0
            )
        
        try:
            # Get construction analysis from neural network
            # The neural network should provide construction planning based on its training data
            construction_prediction = self.neural_network.predict_construction_plan(
                location['lat'], location['lon'], self.data_processor
            )
            
            # Extract construction information from neural network output
            # No hardcoded logic - everything comes from the trained model
        construction_info = {
                'construction_phases': construction_prediction.get('phases', []),
                'timeline': construction_prediction.get('timeline', {}),
                'requirements': construction_prediction.get('requirements', []),
                'safety_protocols': construction_prediction.get('safety_protocols', []),
                'environmental_impact': construction_prediction.get('environmental_impact', {}),
                'regulatory_compliance': construction_prediction.get('regulatory_compliance', {}),
                'confidence': construction_prediction.get('confidence', 0.0),
                'data_sources': construction_prediction.get('data_sources', [])
            }
            
            # Get recommendations from neural network
            recommendations = construction_prediction.get('recommendations', [])
            
            return QueryResult(
                query=query,
                location=location,
                construction_info=construction_info,
                recommendations=recommendations,
                confidence=construction_prediction.get('confidence', 0.6)
            )
            
        except Exception as e:
            logger.error(f"Error in construction analysis: {e}")
            return QueryResult(
                query=query,
                location=location,
                error=f"Error in construction analysis: {str(e)}",
                confidence=0.0
            )
    
    def _handle_survey_query(self, query: str, location: Optional[Dict]) -> QueryResult:
        # Handle survey-related queries using neural network predictions
        # Args:
        #   query: User query string
        #   location: Dictionary with lat/lon coordinates
        # Returns: QueryResult with survey analysis from neural network
        # TODO: Add survey priority scoring
        # TODO: Include survey cost estimation
        if not location:
            return QueryResult(
                query=query,
                error="Location required for survey queries",
                confidence=0.0
            )
        
        try:
            # Get survey analysis from neural network
            # The neural network should provide survey recommendations based on its training data
            survey_prediction = self.neural_network.predict_survey_requirements(
                location['lat'], location['lon'], self.data_processor
        )
        
            # Extract survey information from neural network output
            # No hardcoded logic - everything comes from the trained model
        survey_info = {
                'survey_status': survey_prediction.get('status', 'unknown'),
                'last_survey_date': survey_prediction.get('last_survey_date', 'unknown'),
                'survey_priority': survey_prediction.get('priority', 'medium'),
                'required_surveys': survey_prediction.get('required_surveys', []),
                'survey_methods': survey_prediction.get('survey_methods', []),
                'estimated_cost': survey_prediction.get('estimated_cost', 0.0),
                'confidence': survey_prediction.get('confidence', 0.0),
                'data_gaps': survey_prediction.get('data_gaps', [])
            }
            
            # Get recommendations from neural network
            recommendations = survey_prediction.get('recommendations', [])
            
            return QueryResult(
                query=query,
                location=location,
                environmental_info=survey_info,
                recommendations=recommendations,
                confidence=survey_prediction.get('confidence', 0.5)
            )
            
        except Exception as e:
            logger.error(f"Error in survey analysis: {e}")
            return QueryResult(
                query=query,
                location=location,
                error=f"Error in survey analysis: {str(e)}",
                confidence=0.0
            )
    
    def _handle_general_query(self, query: str, location: Optional[Dict]) -> QueryResult:
        """Handle general queries"""
        # TODO: Implement more sophisticated general query handling
        # TODO: Add FAQ integration
        # TODO: Include help system
        
        return QueryResult(
            query=query,
            location=location,
            recommendations=[
                "Please specify a location (coordinates or city name)",
                "Try asking about infrastructure, environmental data, or risk assessment",
                "Use specific terms like 'pipes', 'climate', 'risk', or 'construction'"
            ],
            confidence=0.3
            )
    
    def _calculate_data_completeness(self, features: Dict) -> float:
        """Calculate data completeness score"""
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
    
    # Removed hardcoded recommendation generation
    # All recommendations now come from the neural network based on its training data
    # This ensures the system learns from actual datasets and provides data-driven insights
    
    @log_performance(logger)
    def format_response(self, result: QueryResult) -> str:
        """Format query result for display using comprehensive report formatter"""
        if result.error:
            return f"❌ Error: {result.error}"
        
        # Always use comprehensive report if location is available
        if result.location:
            return self.report_formatter.format_comprehensive_report(
                result.location, self.data_processor, self.neural_network
            )
        
        # Fallback for queries without location
        response_parts = []
        
        # Add recommendations
        if result.recommendations:
            response_parts.append("💡 Recommendations:")
            for rec in result.recommendations:
                response_parts.append(f"  • {rec}")
        
        # Add confidence
        response_parts.append(f"🎯 Confidence: {result.confidence:.1%}")
        
        response = "\n".join(response_parts)
        logger.info(f"Formatted response: {response}")
        return response 