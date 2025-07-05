# Author: KleaSCM
# Date: 2024
# Query Engine Module
# Description:  - Handles natural language queries and returns structured results

import re
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from utils.logging_utils import setup_logging, log_performance

logger = setup_logging(__name__)

@dataclass
class QueryResult:
    # Structured result from a query
    query: str
    location: Optional[Dict[str, float]] = None
    infrastructure_info: Optional[Dict] = None
    environmental_info: Optional[Dict] = None
    risk_assessment: Optional[Dict] = None
    recommendations: Optional[List[str]] = None
    confidence: float = 0.0
    error: Optional[str] = None

class QueryEngine:
    # Natural language query engine for civil engineering data
    
    @log_performance(logger)
    def __init__(self, data_processor, neural_network):
        self.data_processor = data_processor
        self.neural_network = neural_network
        logger.info("Initialized QueryEngine")
        
        # Define query patterns
        self.location_patterns = [
            r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',  # lat, lon
            r'coordinates?\s*[:\-]?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',
            r'at\s+(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',
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
        # TODO: Implement proper geocoding service integration
        location_keywords = {
            'melbourne': {'lat': -37.8136, 'lon': 144.9631},
            'sydney': {'lat': -33.8688, 'lon': 151.2093},
            'brisbane': {'lat': -27.4698, 'lon': 153.0251},
            'perth': {'lat': -31.9505, 'lon': 115.8605},
            'adelaide': {'lat': -34.9285, 'lon': 138.6007},
        }
        
        for location_name, coords in location_keywords.items():
            if location_name in query:
                return coords
        
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
        """Handle infrastructure-related queries"""
        if not location:
            return QueryResult(
                query=query,
                error="Location required for infrastructure queries",
                confidence=0.0
            )
        
        # Extract infrastructure features
        features = self.data_processor.extract_features_at_location(
            location['lat'], location['lon']
        )
        
        infra_info = features.get('infrastructure', {})
        
        return QueryResult(
            query=query,
            location=location,
            infrastructure_info={
                'pipe_count': infra_info.get('count', 0),
                'total_length': infra_info.get('total_length', 0),
                'materials': infra_info.get('materials', {}),
                'diameters': infra_info.get('diameters', {}),
                'catchment': 'Available' if infra_info.get('count', 0) > 0 else 'None'
            },
            confidence=0.8
        )
    
    def _handle_environmental_query(self, query: str, location: Optional[Dict]) -> QueryResult:
        """Handle environmental-related queries"""
        if not location:
            return QueryResult(
                query=query,
                error="Location required for environmental queries",
                confidence=0.0
            )
        
        # Extract environmental features
        features = self.data_processor.extract_features_at_location(
            location['lat'], location['lon']
        )
        
        env_info = {
            'climate_data': features.get('climate', {}),
            'vegetation_zones': features.get('vegetation', {}),
            'data_completeness': self._calculate_data_completeness(features)
        }
        
        return QueryResult(
            query=query,
            location=location,
            environmental_info=env_info,
            confidence=0.7
        )
    
    def _handle_risk_query(self, query: str, location: Optional[Dict]) -> QueryResult:
        """Handle risk assessment queries"""
        if not location:
            return QueryResult(
                query=query,
                error="Location required for risk assessment",
                confidence=0.0
            )
        
        try:
            # Get risk prediction from neural network
            risk_prediction = self.neural_network.predict_at_location(
                location['lat'], location['lon'], self.data_processor
            )
            
            # TODO: Implement more sophisticated risk calculation
            # TODO: Add historical incident data integration
            # TODO: Include regulatory compliance checks
            
            return QueryResult(
                query=query,
                location=location,
                risk_assessment={
                    'environmental_risk': risk_prediction.get('environmental_risk', 0.0),
                    'infrastructure_risk': risk_prediction.get('infrastructure_risk', 0.0),
                    'construction_risk': risk_prediction.get('construction_risk', 0.0),
                    'overall_risk': risk_prediction.get('overall_risk', 0.0),
                    'risk_factors': risk_prediction.get('risk_factors', [])
                },
                recommendations=self._generate_recommendations(risk_prediction),
                confidence=0.8
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
        """Handle construction-related queries"""
        if not location:
            return QueryResult(
                query=query,
                error="Location required for construction queries",
                confidence=0.0
            )
        
        # TODO: Implement construction planning logic
        # TODO: Add regulatory compliance checks
        # TODO: Include environmental impact assessment
        
        return QueryResult(
            query=query,
            location=location,
            recommendations=[
                "Conduct site survey before construction",
                "Check environmental regulations",
                "Assess infrastructure impact",
                "Review safety protocols"
            ],
            confidence=0.6
        )
    
    def _handle_survey_query(self, query: str, location: Optional[Dict]) -> QueryResult:
        """Handle survey-related queries"""
        if not location:
            return QueryResult(
                query=query,
                error="Location required for survey queries",
                confidence=0.0
            )
        
        # TODO: Implement survey status tracking
        # TODO: Add historical survey data integration
        # TODO: Include survey scheduling logic
        
        return QueryResult(
            query=query,
            location=location,
            environmental_info={
                'survey_status': 'Not completed',
                'last_survey_date': 'N/A',
                'survey_recommendations': [
                    "Schedule environmental survey",
                    "Include infrastructure assessment",
                    "Document vegetation zones",
                    "Record climate data"
                ]
            },
            confidence=0.5
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
    
    def _generate_recommendations(self, risk_prediction: Dict) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        # TODO: Implement more sophisticated recommendation logic
        # TODO: Add industry best practices
        # TODO: Include regulatory requirements
        
        if risk_prediction.get('environmental_risk', 0) > 0.7:
            recommendations.append("High environmental risk - conduct detailed environmental assessment")
        
        if risk_prediction.get('infrastructure_risk', 0) > 0.7:
            recommendations.append("High infrastructure risk - inspect existing infrastructure")
        
        if risk_prediction.get('construction_risk', 0) > 0.7:
            recommendations.append("High construction risk - review safety protocols")
        
        if not recommendations:
            recommendations.append("Risks appear manageable - proceed with standard protocols")
        
        return recommendations
    
    @log_performance(logger)
    def format_response(self, result: QueryResult) -> str:
        """Format query result for display"""
        if result.error:
            return f"❌ Error: {result.error}"
        
        response_parts = []
        
        # Add location info
        if result.location:
            response_parts.append(f"📍 Location: {result.location['lat']:.4f}, {result.location['lon']:.4f}")
        
        # Add infrastructure info
        if result.infrastructure_info:
            infra = result.infrastructure_info
            response_parts.append(f"🏗️ Infrastructure: {infra.get('pipe_count', 0)} pipes, "
                                f"{infra.get('total_length', 0):.1f}m total length")
        
        # Add environmental info
        if result.environmental_info:
            env = result.environmental_info
            if 'data_completeness' in env:
                response_parts.append(f"🌍 Environmental Data: {env['data_completeness']:.1%} complete")
        
        # Add risk assessment
        if result.risk_assessment:
            risk = result.risk_assessment
            response_parts.append(f"⚠️ Risk Assessment:")
            response_parts.append(f"  • Environmental: {risk.get('environmental_risk', 0):.1%}")
            response_parts.append(f"  • Infrastructure: {risk.get('infrastructure_risk', 0):.1%}")
            response_parts.append(f"  • Construction: {risk.get('construction_risk', 0):.1%}")
        
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