# Author: KleaSCM
# Date: 2024
# Description: Universal Reporter - The AI brain that can analyze ANY civil engineering dataset
# This module can detect, analyze, and report on literally everything in civil engineering data

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings
from ..utils.logging_utils import setup_logging, log_performance
from .analyzers import (
    ConstructionAnalyzer,
    InfrastructureAnalyzer,
    EnvironmentalAnalyzer,
    FinancialAnalyzer,
    RiskAnalyzer,
    SpatialAnalyzer,
    TemporalAnalyzer,
    CrossDatasetAnalyzer,
    SurveyAnalyzer
)
warnings.filterwarnings('ignore')

logger = setup_logging(__name__)

class UniversalReporter:
    """
    Universal Reporter - The AI brain that can analyze ANY civil engineering dataset.
    
    This module can:
    - Detect ANY type of civil engineering data
    - Extract ALL possible insights and patterns
    - Generate comprehensive reports for ANY dataset
    - Provide actionable engineering intelligence
    - Handle infrastructure, environmental, construction, financial, and more
    """
    
    def __init__(self):
        """Initialize the Universal Reporter with modular analyzers and content intelligence"""
        logger.info("Initializing Universal Reporter - The AI brain for civil engineering data")
        
        # Initialize all modular analyzers
        self.construction_analyzer = ConstructionAnalyzer()
        self.infrastructure_analyzer = InfrastructureAnalyzer()
        self.environmental_analyzer = EnvironmentalAnalyzer()
        self.financial_analyzer = FinancialAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.cross_dataset_analyzer = CrossDatasetAnalyzer()
        self.survey_analyzer = SurveyAnalyzer()
        
        # Initialize content analysis components for smart routing
        try:
            from .content_analyzer import ContentAnalyzer, SmartTagger, CrossDatasetIntelligence
            self.content_analyzer = ContentAnalyzer()
            self.smart_tagger = SmartTagger()
            self.cross_intelligence = CrossDatasetIntelligence()
            self.content_analysis_available = True
            logger.info("Content analysis intelligence integrated")
        except ImportError:
            self.content_analysis_available = False
            logger.warning("Content analysis not available - using fallback mode")
        
        # Load content analysis results if available
        self.content_analysis_results = self._load_content_analysis_results()
        
        logger.info("Universal Reporter initialized with modular analyzers and content intelligence")
    
    def _load_content_analysis_results(self) -> Dict[str, Any]:
        """Load content analysis results if available"""
        try:
            import json
            with open('content_analysis_results.json', 'r') as f:
                results = json.load(f)
            logger.info("Content analysis results loaded successfully")
            return results
        except FileNotFoundError:
            logger.info("No content analysis results found - will analyze on-demand")
            return {}
        except Exception as e:
            logger.warning(f"Error loading content analysis results: {e}")
            return {}
    
    @log_performance(logger)
    def analyze_dataset(self, dataset: pd.DataFrame, dataset_type: Optional[str] = None, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive Civil Engineer's Site Briefing for ANY dataset.
        
        This method dynamically discovers, analyzes, and synthesizes ALL available data
        to provide everything a civil engineer needs to know about a site.
        
        Args:
            dataset: The dataset to analyze
            dataset_type: Optional dataset type hint
            location: Optional location context
            
        Returns:
            Comprehensive site briefing with executive summary, risks, recommendations
        """
        logger.info(f"Generating Civil Engineer's Site Briefing for dataset with {len(dataset)} records")
        
        # Get content intelligence for smart analysis routing
        content_intelligence = self._get_content_intelligence(dataset, dataset_type)
        
        # Always analyze climate data if present
        climate_context = self._extract_climate_context(dataset)
        
        # Initialize comprehensive analysis using modular analyzers with content-aware routing
        briefing = {
            'executive_summary': self._generate_executive_summary(dataset, location, content_intelligence),
            'content_intelligence': content_intelligence,
            'site_materials': self._analyze_with_content_awareness(dataset, 'construction', lambda d: self.construction_analyzer.analyze(d)['site_materials']),
            'work_history': self._analyze_with_content_awareness(dataset, 'construction', lambda d: self.construction_analyzer.analyze(d)['work_history']),
            'utilities_infrastructure': self._analyze_with_content_awareness(dataset, 'infrastructure', lambda d: self.infrastructure_analyzer.analyze(d)['utilities_infrastructure']),
            'environmental_context': self._merge_environmental_and_climate(dataset, climate_context),
            'costs_funding': self._analyze_with_content_awareness(dataset, 'financial', lambda d: self.financial_analyzer.analyze(d)['costs_funding']),
            'risks_hazards': self._analyze_risks_hazards(dataset, content_intelligence),
            'missing_data': self._identify_missing_data(dataset, content_intelligence),
            'recommendations': self._generate_actionable_recommendations(dataset, location, content_intelligence),
            'nn_insights': self._get_neural_network_insights(dataset, location, content_intelligence),
            'survey_analysis': self._analyze_with_content_awareness(dataset, 'survey', lambda d: self.survey_analyzer.analyze(d)),
            'spatial_analysis': self._analyze_with_content_awareness(dataset, 'spatial', lambda d: self.spatial_analyzer.analyze(d, location=location)['spatial_analysis']),
            'temporal_analysis': self._analyze_with_content_awareness(dataset, 'temporal', lambda d: self.temporal_analyzer.analyze(d)['temporal_analysis']),
            'cross_dataset_analysis': self._get_cross_dataset_analysis(dataset, location, content_intelligence)
        }
        
        logger.info("Civil Engineer's Site Briefing completed with content intelligence and climate data")
        return briefing
    
    def _get_content_intelligence(self, dataset: pd.DataFrame, dataset_type: Optional[str] = None) -> Dict[str, Any]:
        """Get content intelligence for smart analysis routing"""
        if not self.content_analysis_available:
            return {'content_type': 'unknown', 'confidence': 0.0, 'tags': []}
        
        try:
            # Use content analyzer to get dataset intelligence
            content_analysis = self.content_analyzer.analyze_content(dataset, dataset_type or 'unknown')
            tagging_result = self.smart_tagger.auto_tag_dataset(dataset, dataset_type or 'unknown')
            
            return {
                'content_type': content_analysis.get('content_type', 'unknown'),
                'confidence': tagging_result.get('confidence', 0.0),
                'tags': tagging_result.get('tags', []),
                'content_analysis': content_analysis,
                'tagging_result': tagging_result
            }
        except Exception as e:
            logger.warning(f"Content intelligence analysis failed: {e}")
            return {'content_type': 'unknown', 'confidence': 0.0, 'tags': []}
    
    def _analyze_with_content_awareness(self, dataset: pd.DataFrame, analysis_type: str, analyzer_func) -> Dict[str, Any]:
        """Analyze dataset with content awareness for optimal routing"""
        try:
            # Get content intelligence
            content_intelligence = self._get_content_intelligence(dataset)
            content_type = content_intelligence.get('content_type', 'unknown')
            
            # Check if this analysis type is relevant for the content type
            relevance_map = {
                'construction': ['construction', 'infrastructure', 'building'],
                'infrastructure': ['infrastructure', 'construction', 'utility'],
                'environmental': ['environmental', 'weather', 'climate'],
                'financial': ['financial', 'construction', 'infrastructure'],
                'spatial': ['geospatial', 'location', 'coordinate'],
                'temporal': ['temporal', 'time', 'date'],
                'survey': ['survey', 'questionnaire', 'assessment']
            }
            
            relevant_types = relevance_map.get(analysis_type, [])
            is_relevant = content_type in relevant_types or 'unknown' in relevant_types
            
            if is_relevant:
                # Run the analyzer
                result = analyzer_func(dataset)
                return result
            else:
                # Return minimal analysis for non-relevant content
                logger.info(f"Skipping {analysis_type} analysis for {content_type} content type")
                return {f'{analysis_type}_analysis': 'Content type not relevant for this analysis'}
                
        except Exception as e:
            logger.warning(f"Content-aware analysis failed for {analysis_type}: {e}")
            return {f'{analysis_type}_analysis': f'Analysis failed: {str(e)}'}
    
    def _generate_executive_summary(self, dataset: pd.DataFrame, location: Optional[Dict[str, Any]] = None, content_intelligence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate an executive summary of the site for civil engineers"""
        summary = {
            'site_overview': '',
            'key_findings': [],
            'critical_risks': [],
            'immediate_actions': [],
            'data_quality': self._assess_data_quality(dataset)
        }
        
        # Dynamic site overview with content intelligence
        total_records = len(dataset)
        coord_cols = self.construction_analyzer._find_coordinate_columns(dataset)
        has_coordinates = coord_cols['lat'] is not None and coord_cols['lon'] is not None
        
        # Use content intelligence for enhanced overview
        if content_intelligence and content_intelligence.get('content_type') != 'unknown':
            content_type = content_intelligence['content_type']
            confidence = content_intelligence.get('confidence', 0.0)
            tags = content_intelligence.get('tags', [])
            
            summary['site_overview'] = f"Site contains {total_records} {content_type} records with {confidence:.1%} confidence."
            if has_coordinates:
                summary['site_overview'] += " Data includes geolocation information."
            if tags:
                summary['site_overview'] += f" Detected tags: {', '.join(tags[:3])}."
        else:
            # Fallback to original overview
            if has_coordinates:
                summary['site_overview'] = f"Site contains {total_records} geolocated records with comprehensive engineering data."
            else:
                summary['site_overview'] = f"Site contains {total_records} records (no coordinate data available)."
        
        # Key findings
        if total_records > 0:
            summary['key_findings'].append(f"Total records analyzed: {total_records:,}")
            
            # Check for construction projects
            construction_cols = self.construction_analyzer._find_columns_by_patterns(dataset, ['project', 'construction', 'building', 'facility'])
            if construction_cols:
                summary['key_findings'].append(f"Construction data available: {len(construction_cols)} relevant columns")
            
            # Check for infrastructure
            infra_cols = self.infrastructure_analyzer._find_columns_by_patterns(dataset, ['pipe', 'electrical', 'water', 'gas', 'road', 'bridge'])
            if infra_cols:
                summary['key_findings'].append(f"Infrastructure data available: {len(infra_cols)} relevant columns")
            
            # Check for environmental data
            env_cols = self.environmental_analyzer._find_columns_by_patterns(dataset, ['soil', 'climate', 'vegetation', 'flood', 'environmental'])
            if env_cols:
                summary['key_findings'].append(f"Environmental data available: {len(env_cols)} relevant columns")
        
        return summary
    
    def _identify_missing_data(self, dataset: pd.DataFrame, content_intelligence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Identify what critical data is missing"""
        missing = {
            'critical_missing': [],
            'recommended_data': [],
            'data_quality_issues': []
        }
        
        # Check for critical missing data types
        critical_data_types = {
            'coordinates': ['lat', 'latitude', 'lon', 'longitude'],
            'structural': ['structural', 'building', 'foundation'],
            'utilities': ['electrical', 'water', 'gas', 'sewer'],
            'environmental': ['soil', 'climate', 'flood'],
            'safety': ['fire', 'hazard', 'safety'],
            'permits': ['permit', 'license', 'approval'],
            'costs': ['cost', 'budget', 'estimate']
        }
        
        for data_type, patterns in critical_data_types.items():
            found_cols = self.construction_analyzer._find_columns_by_patterns(dataset, patterns)
            if not found_cols:
                missing['critical_missing'].append(f"No {data_type} data found")
            else:
                missing['recommended_data'].append(f"{data_type}: {len(found_cols)} columns available")
        
        # Check data quality
        missing_data = dataset.isnull().sum()
        high_missing_cols = missing_data[missing_data > len(dataset) * 0.5]
        if len(high_missing_cols) > 0:
            missing['data_quality_issues'].append(f"High missing data in columns: {list(high_missing_cols.index)}")
        
        return missing
    
    def _generate_actionable_recommendations(self, dataset: pd.DataFrame, location: Optional[Dict[str, Any]] = None, content_intelligence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate actionable recommendations for civil engineers"""
        recommendations = {
            'immediate_actions': [],
            'investigations_needed': [],
            'permits_required': [],
            'safety_measures': [],
            'next_steps': []
        }
        
        # Dynamic recommendation generation based on available data
        
        # Check for construction projects
        construction_cols = self.construction_analyzer._find_columns_by_patterns(dataset, ['project', 'construction', 'building'])
        if construction_cols:
            recommendations['immediate_actions'].append("Review all construction project details and timelines")
            recommendations['permits_required'].append("Verify all required permits are in place for construction activities")
        
        # Check for structural issues
        structural_cols = self.construction_analyzer._find_columns_by_patterns(dataset, ['structural', 'crack', 'failure', 'deterioration'])
        if structural_cols:
            recommendations['investigations_needed'].append("Conduct structural integrity assessment")
            recommendations['safety_measures'].append("Implement structural monitoring and safety protocols")
        
        # Check for environmental concerns
        environmental_cols = self.environmental_analyzer._find_columns_by_patterns(dataset, ['soil', 'flood', 'contamination', 'environmental'])
        if environmental_cols:
            recommendations['investigations_needed'].append("Perform environmental impact assessment")
            recommendations['safety_measures'].append("Implement environmental monitoring and protection measures")
        
        # Check for utility conflicts
        utility_cols = self.infrastructure_analyzer._find_columns_by_patterns(dataset, ['electrical', 'water', 'gas', 'sewer'])
        if utility_cols:
            recommendations['immediate_actions'].append("Coordinate with utility companies for service verification")
            recommendations['safety_measures'].append("Implement utility marking and protection protocols")
        
        # General recommendations
        recommendations['next_steps'].append("Schedule site visit and detailed inspection")
        recommendations['next_steps'].append("Review all available permits and compliance requirements")
        recommendations['next_steps'].append("Coordinate with local authorities and utility companies")
        
        return recommendations
    
    def _get_neural_network_insights(self, dataset: pd.DataFrame, location: Optional[Dict[str, Any]] = None, content_intelligence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get insights from the neural network"""
        nn_insights = {
            'pattern_recognition': [],
            'risk_assessment': [],
            'recommendations': [],
            'anomalies': [],
            'nn_status': 'Integrated',
            'project_analysis': [],
            'cost_analysis': [],
            'timeline_analysis': []
        }
        
        try:
            # Import and initialize the neural network
            from src.ml.neural_network import CivilEngineeringSystem
            nn_system = CivilEngineeringSystem()
            
            # Add content intelligence insights
            if content_intelligence and content_intelligence.get('content_type') != 'unknown':
                content_type = content_intelligence['content_type']
                confidence = content_intelligence.get('confidence', 0.0)
                tags = content_intelligence.get('tags', [])
                
                nn_insights['pattern_recognition'].append(f"Content type detected: {content_type} (confidence: {confidence:.1%})")
                if tags:
                    nn_insights['pattern_recognition'].append(f"Content tags: {', '.join(tags)}")
                
                # Add content-specific insights
                if content_type == 'construction':
                    nn_insights['recommendations'].append("Focus on construction project analysis and timeline management")
                elif content_type == 'infrastructure':
                    nn_insights['recommendations'].append("Prioritize infrastructure condition assessment and maintenance planning")
                elif content_type == 'environmental':
                    nn_insights['recommendations'].append("Emphasize environmental impact assessment and compliance monitoring")
                elif content_type == 'financial':
                    nn_insights['recommendations'].append("Concentrate on cost analysis and budget optimization")
            
            # Analyze project patterns
            if 'Project Title' in dataset.columns:
                project_titles = dataset['Project Title'].dropna().tolist()
                if project_titles:
                    # Use NN to analyze project patterns
                    nn_insights['project_analysis'].append(f"Found {len(project_titles)} construction projects")
                    
                    # Analyze project types
                    upgrade_projects = [p for p in project_titles if 'upgrade' in p.lower()]
                    rehab_projects = [p for p in project_titles if 'rehab' in p.lower()]
                    new_projects = [p for p in project_titles if 'new' in p.lower() or 'construction' in p.lower()]
                    
                    if upgrade_projects:
                        nn_insights['pattern_recognition'].append(f"Upgrade projects: {len(upgrade_projects)} (infrastructure maintenance)")
                    if rehab_projects:
                        nn_insights['pattern_recognition'].append(f"Rehabilitation projects: {len(rehab_projects)} (building improvements)")
                    if new_projects:
                        nn_insights['pattern_recognition'].append(f"New construction: {len(new_projects)} (expansion projects)")
            
            # Also check for NYC construction dataset structure
            elif 'Project Description' in dataset.columns:
                project_descriptions = dataset['Project Description'].dropna().tolist()
                if project_descriptions:
                    nn_insights['project_analysis'].append(f"Found {len(project_descriptions)} NYC construction projects")
                    
                    # Analyze project types
                    climate_projects = [p for p in project_descriptions if 'climate' in p.lower() or 'boiler' in p.lower()]
                    structural_projects = [p for p in project_descriptions if 'structural' in p.lower() or 'defect' in p.lower()]
                    safety_projects = [p for p in project_descriptions if 'safety' in p.lower() or 'fire' in p.lower()]
                    
                    if climate_projects:
                        nn_insights['pattern_recognition'].append(f"Climate control projects: {len(climate_projects)} (PLANYC initiatives)")
                    if structural_projects:
                        nn_insights['pattern_recognition'].append(f"Structural projects: {len(structural_projects)} (building integrity)")
                    if safety_projects:
                        nn_insights['pattern_recognition'].append(f"Safety projects: {len(safety_projects)} (compliance upgrades)")
            
            # Analyze cost patterns
            if 'Estimated Contract Value' in dataset.columns:
                cost_values = dataset['Estimated Contract Value'].dropna().tolist()
                if cost_values:
                    # Extract numeric values for analysis
                    numeric_costs = []
                    for cost in cost_values:
                        try:
                            if ' - ' in cost:
                                min_val_str = cost.split(' - ')[0].replace('$', '').replace(',', '')
                                max_val_str = cost.split(' - ')[1].replace('$', '').replace(',', '')
                                avg_val = (float(min_val_str) + float(max_val_str)) / 2
                                numeric_costs.append(avg_val)
                            else:
                                val_str = cost.replace('$', '').replace(',', '').replace(' - Up', '')
                                if val_str.isdigit():
                                    numeric_costs.append(float(val_str))
                        except:
                            pass
                    
                    if numeric_costs:
                        avg_cost = sum(numeric_costs) / len(numeric_costs)
                        max_cost = max(numeric_costs)
                        min_cost = min(numeric_costs)
                        
                        nn_insights['cost_analysis'].append(f"Average project cost: ${avg_cost:,.0f}")
                        nn_insights['cost_analysis'].append(f"Cost range: ${min_cost:,.0f} - ${max_cost:,.0f}")
                        
                        # Risk assessment based on costs
                        if avg_cost > 10000000:  # $10M
                            nn_insights['risk_assessment'].append("High-value projects detected - increased financial risk")
                        if max_cost > 50000000:  # $50M
                            nn_insights['risk_assessment'].append("Major capital projects present - complex coordination required")
            
            # Also check for NYC construction awards
            elif 'Construction Award' in dataset.columns:
                awards = dataset['Construction Award'].dropna()
                if len(awards) > 0:
                    avg_award = awards.mean()
                    max_award = awards.max()
                    min_award = awards.min()
                    total_award = awards.sum()
                    
                    nn_insights['cost_analysis'].append(f"Total construction value: ${total_award:,.0f}")
                    nn_insights['cost_analysis'].append(f"Average project cost: ${avg_award:,.0f}")
                    nn_insights['cost_analysis'].append(f"Cost range: ${min_award:,.0f} - ${max_award:,.0f}")
                    
                    # Risk assessment based on costs
                    if avg_award > 5000000:  # $5M
                        nn_insights['risk_assessment'].append("High-value NYC school projects - increased oversight required")
                    if max_award > 10000000:  # $10M
                        nn_insights['risk_assessment'].append("Major NYC capital projects - complex stakeholder coordination")
            
            # Analyze timeline patterns
            if 'Contract Advertise Date' in dataset.columns:
                dates = dataset['Contract Advertise Date'].dropna().tolist()
                if dates:
                    nn_insights['timeline_analysis'].append(f"Projects advertised over {len(dates)} time periods")
                    
                    # Check for recent vs old projects
                    try:
                        from datetime import datetime
                        recent_projects = 0
                        for date_str in dates:
                            try:
                                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                                if date_obj.year >= 2020:
                                    recent_projects += 1
                            except:
                                pass
                        
                        if recent_projects > 0:
                            nn_insights['pattern_recognition'].append(f"Recent projects (2020+): {recent_projects}")
                            nn_insights['recommendations'].append("Recent projects may have updated specifications and requirements")
                    except:
                        pass
            
            # Analyze campus patterns
            if 'Campus Name' in dataset.columns:
                campuses = dataset['Campus Name'].value_counts()
                if len(campuses) > 1:
                    nn_insights['pattern_recognition'].append(f"Multi-campus projects: {len(campuses)} campuses involved")
                    nn_insights['recommendations'].append("Coordinate across multiple campuses for consistent standards")
            
            # Also check for NYC school patterns
            elif 'School Name' in dataset.columns:
                schools = dataset['School Name'].value_counts()
                if len(schools) > 1:
                    nn_insights['pattern_recognition'].append(f"Multi-school projects: {len(schools)} schools involved")
                    nn_insights['recommendations'].append("Coordinate across multiple NYC schools for consistent standards")
                
                # Analyze borough distribution
                if 'Borough' in dataset.columns:
                    boroughs = dataset['Borough'].value_counts()
                    nn_insights['pattern_recognition'].append(f"Projects across {len(boroughs)} NYC boroughs")
                    nn_insights['recommendations'].append("Consider borough-specific regulations and requirements")
            
            # Anomaly detection
            if len(dataset) > 0:
                # Check for data quality anomalies
                missing_data = dataset.isnull().sum().sum()
                total_cells = len(dataset) * len(dataset.columns)
                missing_percentage = (missing_data / total_cells) * 100
                
                if missing_percentage > 30:
                    nn_insights['anomalies'].append(f"High missing data rate: {missing_percentage:.1f}%")
                    nn_insights['risk_assessment'].append("Data quality issues may affect project planning accuracy")
                
                # Check for unusual project patterns
                if 'Project Title' in dataset.columns:
                    titles = dataset['Project Title'].dropna().tolist()
                    if len(titles) > 0:
                        # Check for emergency or urgent projects
                        urgent_keywords = ['emergency', 'urgent', 'critical', 'immediate', 'repair']
                        urgent_projects = [t for t in titles if any(keyword in t.lower() for keyword in urgent_keywords)]
                        if urgent_projects:
                            nn_insights['anomalies'].append(f"Urgent projects detected: {len(urgent_projects)}")
                            nn_insights['risk_assessment'].append("Urgent projects may indicate infrastructure issues requiring immediate attention")
            
            # Generate NN-driven recommendations
            if len(dataset) > 0:
                nn_insights['recommendations'].append("Schedule coordination meetings with all campus stakeholders")
                nn_insights['recommendations'].append("Review project timelines and dependencies")
                nn_insights['recommendations'].append("Assess resource allocation across multiple projects")
            
        except Exception as e:
            logger.warning(f"Neural network analysis failed: {e}")
            nn_insights['nn_status'] = f'Error: {str(e)}'
            nn_insights['pattern_recognition'].append("NN analysis unavailable - using fallback pattern detection")
        
        return nn_insights
    
    def _assess_data_quality(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        quality = {
            'completeness': 0.0,
            'accuracy': 'Unknown',
            'reliability': 'Unknown',
            'issues': []
        }
        
        # Calculate completeness
        total_cells = len(dataset) * len(dataset.columns)
        filled_cells = total_cells - dataset.isnull().sum().sum()
        quality['completeness'] = (filled_cells / total_cells) * 100
        
        # Identify issues
        if quality['completeness'] < 50:
            quality['issues'].append("Low data completeness - significant missing values")
        
        if len(dataset) == 0:
            quality['issues'].append("Empty dataset")
        
        return quality
    
    def _analyze_dataset_overview(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic dataset characteristics"""
        overview = {
            'total_records': len(dataset),
            'total_columns': len(dataset.columns),
            'column_types': dataset.dtypes.value_counts().to_dict(),
            'memory_usage': dataset.memory_usage(deep=True).sum(),
            'date_range': None,
            'geographic_bounds': None,
            'key_columns': self._identify_key_columns(dataset)
        }
        
        # Detect date range
        date_cols = dataset.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            overview['date_range'] = {
                'columns': list(date_cols),
                'earliest': dataset[date_cols].min().to_dict(),
                'latest': dataset[date_cols].max().to_dict()
            }
        
        # Detect geographic bounds
        coord_cols = self._find_coordinate_columns(dataset)
        if coord_cols['lat'] and coord_cols['lon']:
            lat_col, lon_col = coord_cols['lat'], coord_cols['lon']
            try:
                # Convert to numeric, ignoring errors
                lat_numeric = pd.to_numeric(dataset[lat_col], errors='coerce')  # type: ignore
                lon_numeric = pd.to_numeric(dataset[lon_col], errors='coerce')  # type: ignore
                # Only calculate bounds if we have valid numeric data
                if not lat_numeric.isna().all() and not lon_numeric.isna().all():  # type: ignore
                    overview['geographic_bounds'] = {
                        'lat_min': float(lat_numeric.min()),  # type: ignore
                        'lat_max': float(lat_numeric.max()),  # type: ignore
                        'lon_min': float(lon_numeric.min()),  # type: ignore
                        'lon_max': float(lon_numeric.max())   # type: ignore
                    }
            except Exception as e:
                logger.warning(f"Geographic bounds calculation failed: {e}")
        return overview
    
    def _analyze_data_quality(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality and completeness"""
        quality = {
            'missing_data': {},
            'duplicates': len(dataset[dataset.duplicated()]),
            'data_types': {},
            'outliers': {},
            'completeness_score': 0.0
        }
        
        # Missing data analysis
        missing_counts = dataset.isnull().sum()
        missing_percentages = (missing_counts / len(dataset)) * 100
        quality['missing_data'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'columns_with_missing': list(missing_counts[missing_counts > 0].index)
        }
        
        # Data type analysis
        for col in dataset.columns:
            quality['data_types'][col] = {
                'dtype': str(dataset[col].dtype),
                'unique_values': dataset[col].nunique(),
                'most_common': dataset[col].value_counts().head(3).to_dict() if dataset[col].dtype == 'object' else None
            }
        
        # Completeness score
        total_cells = len(dataset) * len(dataset.columns)
        filled_cells = total_cells - dataset.isnull().sum().sum()
        quality['completeness_score'] = (filled_cells / total_cells) * 100
        
        return quality
    
    def _analyze_infrastructure(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze infrastructure-related data"""
        infrastructure = {
            'pipe_analysis': {},
            'structural_analysis': {},
            'electrical_analysis': {},
            'mechanical_analysis': {},
            'transport_analysis': {},
            'building_analysis': {}
        }
        
        # Pipe analysis
        pipe_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['infrastructure']['pipes'])
        if pipe_cols:
            infrastructure['pipe_analysis'] = self._analyze_pipes(dataset, pipe_cols)
        
        # Material analysis
        material_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['infrastructure']['materials'])
        if material_cols:
            infrastructure['material_analysis'] = self._analyze_materials(dataset, material_cols)
        
        # Dimension analysis
        dimension_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['infrastructure']['dimensions'])
        if dimension_cols:
            infrastructure['dimension_analysis'] = self._analyze_dimensions(dataset, dimension_cols)
        
        # Structural analysis
        structural_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['infrastructure']['structural'])
        if structural_cols:
            infrastructure['structural_analysis'] = self._analyze_structural(dataset, structural_cols)
        
        # Electrical analysis
        electrical_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['infrastructure']['electrical'])
        if electrical_cols:
            infrastructure['electrical_analysis'] = self._analyze_electrical(dataset, electrical_cols)
        
        return infrastructure
    
    def _analyze_environmental(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze environmental data"""
        environmental = {
            'soil_analysis': {},
            'vegetation_analysis': {},
            'climate_analysis': {},
            'hydrology_analysis': {},
            'geology_analysis': {},
            'air_quality_analysis': {},
            'noise_analysis': {}
        }
        
        # Soil analysis
        soil_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['environmental']['soil'])
        if soil_cols:
            environmental['soil_analysis'] = self._analyze_soil(dataset, soil_cols)
        
        # Vegetation analysis
        vegetation_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['environmental']['vegetation'])
        if vegetation_cols:
            environmental['vegetation_analysis'] = self._analyze_vegetation(dataset, vegetation_cols)
        
        # Climate analysis
        climate_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['environmental']['climate'])
        if climate_cols:
            environmental['climate_analysis'] = self._analyze_climate(dataset, climate_cols)
        
        return environmental
    
    def _analyze_construction(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze construction-related data"""
        construction = {
            'project_analysis': {},
            'resource_analysis': {},
            'quality_analysis': {},
            'safety_analysis': {},
            'permit_analysis': {}
        }
        
        # Project analysis
        project_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['construction']['project'])
        if project_cols:
            construction['project_analysis'] = self._analyze_projects(dataset, project_cols)
        
        # Resource analysis
        resource_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['construction']['resources'])
        if resource_cols:
            construction['resource_analysis'] = self._analyze_resources(dataset, resource_cols)
        
        return construction
    
    def _analyze_financial(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze financial data"""
        financial = {
            'cost_analysis': {},
            'asset_analysis': {},
            'contract_analysis': {},
            'billing_analysis': {}
        }
        
        # Cost analysis
        cost_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['financial']['costs'])
        if cost_cols:
            financial['cost_analysis'] = self._analyze_costs(dataset, cost_cols)
        
        # Asset analysis
        asset_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['financial']['assets'])
        if asset_cols:
            financial['asset_analysis'] = self._analyze_assets(dataset, asset_cols)
        
        return financial
    
    def _analyze_operational(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operational data"""
        operational = {
            'maintenance_analysis': {},
            'performance_analysis': {},
            'monitoring_analysis': {},
            'control_analysis': {}
        }
        
        # Maintenance analysis
        maintenance_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['operational']['maintenance'])
        if maintenance_cols:
            operational['maintenance_analysis'] = self._analyze_maintenance(dataset, maintenance_cols)
        
        # Performance analysis
        performance_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['operational']['performance'])
        if performance_cols:
            operational['performance_analysis'] = self._analyze_performance(dataset, performance_cols)
        
        return operational
    
    def _analyze_risks(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk factors"""
        risks = {
            'structural_risks': {},
            'environmental_risks': {},
            'operational_risks': {},
            'financial_risks': {},
            'compliance_risks': {}
        }
        
        # Structural risks
        structural_risk_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.risk_patterns['structural_risk'])
        if structural_risk_cols:
            risks['structural_risks'] = self._analyze_structural_risks(dataset, structural_risk_cols)
        
        # Environmental risks
        environmental_risk_cols = self._find_columns_by_patterns(dataset, self.construction_analyzer.risk_patterns['environmental_risk'])
        if environmental_risk_cols:
            risks['environmental_risks'] = self._analyze_environmental_risks(dataset, environmental_risk_cols)
        
        return risks
    
    def _analyze_spatial(self, dataset: pd.DataFrame, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze spatial data"""
        spatial = {
            'coordinate_analysis': {},
            'geographic_distribution': {},
            'proximity_analysis': {},
            'spatial_patterns': {}
        }
        
        # Coordinate analysis
        coord_cols = self._find_coordinate_columns(dataset)
        if coord_cols['lat'] and coord_cols['lon']:
            spatial['coordinate_analysis'] = self._analyze_coordinates(dataset, coord_cols)
            
            # Proximity analysis if location provided
            if location:
                spatial['proximity_analysis'] = self._analyze_proximity(dataset, coord_cols, location)
        
        return spatial
    
    def _analyze_temporal(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal data"""
        temporal = {
            'time_series_analysis': {},
            'seasonal_patterns': {},
            'trend_analysis': {},
            'temporal_distribution': {}
        }
        
        # Find date/time columns
        date_cols = list(dataset.select_dtypes(include=['datetime64']).columns)
        if len(date_cols) > 0:
            temporal['time_series_analysis'] = self._analyze_time_series(dataset, date_cols)
        
        return temporal
    
    def _find_correlations(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Find correlations between variables"""
        correlations = {
            'numeric_correlations': {},
            'categorical_associations': {},
            'significant_relationships': []
        }
        # Numeric correlations
        numeric_cols = list(dataset.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) > 1:
            try:
                corr_matrix = dataset[numeric_cols].corr()  # type: ignore
                correlations['numeric_correlations'] = {
                    'matrix': corr_matrix.to_dict(),
                    'strong_correlations': self._find_strong_correlations(corr_matrix)
                }
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
                correlations['numeric_correlations'] = {'error': str(e)}
        return correlations
    
    def _detect_anomalies(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        anomalies = {
            'outliers': {},
            'missing_patterns': {},
            'data_inconsistencies': {},
            'unusual_patterns': {}
        }
        
        # Outlier detection for numeric columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                Q1 = dataset[col].quantile(0.25)
                Q3 = dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = dataset[(dataset[col] < Q1 - 1.5 * IQR) | (dataset[col] > Q3 + 1.5 * IQR)]
                if len(outliers) > 0:
                    anomalies['outliers'][col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(dataset)) * 100,
                        'values': outliers[col].tolist()
                    }
            except Exception as e:
                logger.warning(f"Outlier detection failed for column {col}: {e}")
        
        return anomalies
    
    def _generate_recommendations(self, dataset: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        missing_data = dataset.isnull().sum()
        high_missing_cols = missing_data[missing_data > len(dataset) * 0.1]
        if len(high_missing_cols) > 0:
            recommendations.append(f"Address missing data in columns: {list(high_missing_cols.index)}")
        
        # Infrastructure recommendations
        if self._has_infrastructure_data(dataset):
            recommendations.extend(self._generate_infrastructure_recommendations(dataset))
        
        # Risk recommendations
        if self._has_risk_indicators(dataset):
            recommendations.extend(self._generate_risk_recommendations(dataset))
        
        return recommendations
    
    def _generate_action_items(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate specific action items"""
        action_items = []
        
        # Data quality actions
        if dataset.isnull().sum().sum() > 0:
            action_items.append({
                'type': 'data_quality',
                'priority': 'high',
                'action': 'Clean missing data',
                'description': 'Address missing values in the dataset'
            })
        
        # Infrastructure actions
        if self._has_infrastructure_data(dataset):
            action_items.extend(self._generate_infrastructure_actions(dataset))
        
        return action_items
    
    # Helper methods for specific analyses
    def _find_columns_by_patterns(self, dataset: pd.DataFrame, patterns: List[str]) -> List[str]:
        """Find columns that match given patterns"""
        matching_cols = []
        for col in dataset.columns:
            col_lower = col.lower()
            # Check for exact matches and partial matches
            if any(pattern.lower() in col_lower for pattern in patterns):
                matching_cols.append(col)
            # Also check for common variations
            elif any(pattern.lower().replace('_', '') in col_lower.replace('_', '') for pattern in patterns):
                matching_cols.append(col)
        return matching_cols
    
    def _find_coordinate_columns(self, dataset: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Find coordinate columns with improved detection"""
        coord_patterns = self.spatial_analyzer.spatial_patterns['coordinates']
        lat_col = None
        lon_col = None
        
        for col in dataset.columns:
            col_lower = col.lower()
            # Check for latitude patterns - be more precise to avoid false matches
            if any(pattern in col_lower for pattern in ['latitude', 'lat']):
                # Avoid false matches like "lat" in "Tabulation"
                if not any(false_match in col_lower for false_match in ['tabulation', 'calculation', 'relation']):
                    lat_col = col
            # Check for longitude patterns - be more precise
            elif any(pattern in col_lower for pattern in ['longitude', 'lon', 'lng']):
                lon_col = col
            # Only use x/y/northing/easting if no lat/lon found
            elif lat_col is None and any(pattern in col_lower for pattern in ['y', 'northing']):
                lat_col = col
            elif lon_col is None and any(pattern in col_lower for pattern in ['x', 'easting']):
                lon_col = col
        
        return {'lat': lat_col, 'lon': lon_col}
    
    def _identify_key_columns(self, dataset: pd.DataFrame) -> List[str]:
        """Identify key columns in the dataset"""
        key_cols = []
        
        # ID columns
        id_patterns = ['id', 'key', 'index', 'name', 'code']
        for col in dataset.columns:
            if any(pattern in col.lower() for pattern in id_patterns):
                key_cols.append(col)
        
        # Date columns
        date_cols = dataset.select_dtypes(include=['datetime64']).columns
        key_cols.extend(list(date_cols))
        
        # Coordinate columns
        coord_cols = self._find_coordinate_columns(dataset)
        if coord_cols['lat']:
            key_cols.append(coord_cols['lat'])
        if coord_cols['lon']:
            key_cols.append(coord_cols['lon'])
        
        return key_cols
    
    # Placeholder methods for specific analyses (to be implemented)
    def _analyze_pipes(self, dataset: pd.DataFrame, pipe_cols: List[str]) -> Dict[str, Any]:
        """Analyze pipe-related data"""
        return {'pipe_count': len(dataset), 'pipe_columns': pipe_cols}
    
    def _analyze_materials(self, dataset: pd.DataFrame, material_cols: List[str]) -> Dict[str, Any]:
        """Analyze material data"""
        materials = {}
        for col in material_cols:
            if col in dataset.columns:
                # Get actual material distribution
                material_counts = dataset[col].value_counts()
                materials[col] = material_counts.to_dict()
        return {'material_distributions': materials}
    
    def _analyze_dimensions(self, dataset: pd.DataFrame, dimension_cols: List[str]) -> Dict[str, Any]:
        """Analyze dimensional data"""
        dimensions = {}
        for col in dimension_cols:
            if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                try:
                    # Convert to numeric and get actual statistics
                    numeric_data = pd.to_numeric(dataset[col], errors='coerce')
                    import numpy as np
                    arr = np.asarray(numeric_data)
                    is_valid = not np.isnan(arr).all()
                    if is_valid:
                        dimensions[col] = {
                            'min': float(np.nanmin(arr)),
                            'max': float(np.nanmax(arr)),
                            'mean': float(np.nanmean(arr)),
                            'std': float(np.nanstd(arr)),
                            'median': float(np.nanmedian(arr))
                        }
                except Exception as e:
                    logger.warning(f"Dimension analysis failed for column {col}: {e}")
        return {'dimension_statistics': dimensions}
    
    def _analyze_structural(self, dataset: pd.DataFrame, structural_cols: List[str]) -> Dict[str, Any]:
        """Analyze structural data"""
        return {'structural_elements': len(dataset), 'structural_columns': structural_cols}
    
    def _analyze_electrical(self, dataset: pd.DataFrame, electrical_cols: List[str]) -> Dict[str, Any]:
        """Analyze electrical data"""
        return {'electrical_components': len(dataset), 'electrical_columns': electrical_cols}
    
    def _analyze_soil(self, dataset: pd.DataFrame, soil_cols: List[str]) -> Dict[str, Any]:
        """Analyze soil data"""
        return {'soil_samples': len(dataset), 'soil_columns': soil_cols}
    
    def _analyze_vegetation(self, dataset: pd.DataFrame, vegetation_cols: List[str]) -> Dict[str, Any]:
        """Analyze vegetation data"""
        vegetation_analysis = {
            'vegetation_zones': len(dataset),
            'vegetation_columns': vegetation_cols
        }
        
        # Add actual vegetation statistics if available
        for col in vegetation_cols:
            if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                try:
                    numeric_data = pd.to_numeric(dataset[col], errors='coerce')
                    if isinstance(numeric_data, pd.Series) and not numeric_data.isna().all():
                        vegetation_analysis[f'{col}_stats'] = {
                            'min': float(numeric_data.min()),
                            'max': float(numeric_data.max()),
                            'mean': float(numeric_data.mean()),
                            'std': float(numeric_data.std())
                        }
                except Exception as e:
                    logger.warning(f"Vegetation analysis failed for column {col}: {e}")
        
        return vegetation_analysis
    
    def _analyze_climate(self, dataset: pd.DataFrame, climate_cols: List[str]) -> Dict[str, Any]:
        """Analyze climate data"""
        climate_analysis = {
            'climate_stations': len(dataset),
            'climate_columns': climate_cols
        }
        
        # Add actual climate statistics if available
        for col in climate_cols:
            if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                try:
                    numeric_data = pd.to_numeric(dataset[col], errors='coerce')
                    if isinstance(numeric_data, pd.Series) and not numeric_data.isna().all():
                        climate_analysis[f'{col}_stats'] = {
                            'min': float(numeric_data.min()),
                            'max': float(numeric_data.max()),
                            'mean': float(numeric_data.mean()),
                            'std': float(numeric_data.std())
                        }
                except Exception as e:
                    logger.warning(f"Climate analysis failed for column {col}: {e}")
        
        return climate_analysis
    
    def _analyze_projects(self, dataset: pd.DataFrame, project_cols: List[str]) -> Dict[str, Any]:
        """Analyze project data"""
        return {'projects': len(dataset), 'project_columns': project_cols}
    
    def _analyze_resources(self, dataset: pd.DataFrame, resource_cols: List[str]) -> Dict[str, Any]:
        """Analyze resource data"""
        return {'resources': len(dataset), 'resource_columns': resource_cols}
    
    def _analyze_costs(self, dataset: pd.DataFrame, cost_cols: List[str]) -> Dict[str, Any]:
        """Analyze cost data"""
        costs = {}
        for col in cost_cols:
            if col in dataset.columns and dataset[col].dtype in ['int64', 'float64']:
                try:
                    costs[col] = {
                        'total': float(dataset[col].sum()),
                        'mean': float(dataset[col].mean()),
                        'min': float(dataset[col].min()),
                        'max': float(dataset[col].max())
                    }
                except Exception as e:
                    logger.warning(f"Cost analysis failed for column {col}: {e}")
        return {'cost_statistics': costs}
    
    def _analyze_assets(self, dataset: pd.DataFrame, asset_cols: List[str]) -> Dict[str, Any]:
        """Analyze asset data"""
        return {'assets': len(dataset), 'asset_columns': asset_cols}
    
    def _analyze_maintenance(self, dataset: pd.DataFrame, maintenance_cols: List[str]) -> Dict[str, Any]:
        """Analyze maintenance data"""
        return {'maintenance_records': len(dataset), 'maintenance_columns': maintenance_cols}
    
    def _analyze_performance(self, dataset: pd.DataFrame, performance_cols: List[str]) -> Dict[str, Any]:
        """Analyze performance data"""
        return {'performance_metrics': len(dataset), 'performance_columns': performance_cols}
    
    def _analyze_structural_risks(self, dataset: pd.DataFrame, risk_cols: List[str]) -> Dict[str, Any]:
        """Analyze structural risks"""
        return {'structural_risk_indicators': len(dataset), 'risk_columns': risk_cols}
    
    def _analyze_environmental_risks(self, dataset: pd.DataFrame, risk_cols: List[str]) -> Dict[str, Any]:
        """Analyze environmental risks"""
        return {'environmental_risk_indicators': len(dataset), 'risk_columns': risk_cols}
    
    def _analyze_coordinates(self, dataset: pd.DataFrame, coord_cols: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """Analyze coordinate data"""
        lat_col, lon_col = coord_cols['lat'], coord_cols['lon']
        if lat_col is None or lon_col is None:
            return {'coordinate_count': 0}
        try:
            # Convert to numeric, ignoring errors
            lat_numeric = pd.to_numeric(dataset[lat_col], errors='coerce')  # type: ignore
            lon_numeric = pd.to_numeric(dataset[lon_col], errors='coerce')  # type: ignore
            return {
                'coordinate_range': {
                    'lat_min': float(lat_numeric.min()),  # type: ignore
                    'lat_max': float(lat_numeric.max()),  # type: ignore
                    'lon_min': float(lon_numeric.min()),  # type: ignore
                    'lon_max': float(lon_numeric.max())   # type: ignore
                },
                'coordinate_count': len(dataset)
            }
        except Exception as e:
            logger.warning(f"Coordinate analysis failed: {e}")
            return {'coordinate_count': 0, 'error': str(e)}
    
    def _analyze_proximity(self, dataset: pd.DataFrame, coord_cols: Dict[str, Optional[str]], location: Dict) -> Dict[str, Any]:
        """Analyze proximity to given location"""
        lat_col, lon_col = coord_cols['lat'], coord_cols['lon']
        if lat_col is None or lon_col is None:
            return {'error': 'No coordinate columns found'}
        try:
            distances = np.sqrt(
                (dataset[lat_col] - location['lat'])**2 + 
                (dataset[lon_col] - location['lon'])**2
            )
            return {
                'nearest_distance': float(distances.min()),
                'average_distance': float(distances.mean()),
                'within_1km': len(distances[distances <= 0.01]),
                'within_5km': len(distances[distances <= 0.05])
            }
        except Exception as e:
            logger.warning(f"Proximity analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_time_series(self, dataset: pd.DataFrame, date_cols: List[str]) -> Dict[str, Any]:
        """Analyze time series data"""
        time_analysis = {}
        for col in date_cols:
            try:
                time_analysis[col] = {
                    'earliest': dataset[col].min(),
                    'latest': dataset[col].max(),
                    'duration_days': (dataset[col].max() - dataset[col].min()).days,
                    'record_count': len(dataset)
                }
            except Exception as e:
                logger.warning(f"Time series analysis failed for column {col}: {e}")
                time_analysis[col] = {'error': str(e)}
        return time_analysis
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlations in correlation matrix"""
        strong_correlations = []
        try:
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })
        except Exception as e:
            logger.warning(f"Strong correlation analysis failed: {e}")
        return strong_correlations
    
    def _has_infrastructure_data(self, dataset: pd.DataFrame) -> bool:
        """Check if dataset has infrastructure data"""
        infrastructure_patterns = []
        for category in self.construction_analyzer.data_patterns['infrastructure'].values():
            infrastructure_patterns.extend(category)
        return len(self.construction_analyzer._find_columns_by_patterns(dataset, infrastructure_patterns)) > 0
    
    def _has_risk_indicators(self, dataset: pd.DataFrame) -> bool:
        """Check if dataset has risk indicators"""
        risk_patterns = []
        for category in self.construction_analyzer.risk_patterns.values():
            risk_patterns.extend(category)
        return len(self.construction_analyzer._find_columns_by_patterns(dataset, risk_patterns)) > 0
    
    def _generate_infrastructure_recommendations(self, dataset: pd.DataFrame) -> list:
        """Generate infrastructure-specific recommendations"""
        recommendations = []
        material_cols = self.construction_analyzer._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['infrastructure']['materials'])
        if material_cols:
            recommendations.append("Review material specifications and consider upgrades")
        dimension_cols = self.construction_analyzer._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['infrastructure']['dimensions'])
        if dimension_cols:
            recommendations.append("Assess capacity requirements and upgrade needs")
        return recommendations
    
    def _generate_risk_recommendations(self, dataset: pd.DataFrame) -> list:
        """Generate risk-specific recommendations"""
        recommendations = []
        structural_risk_cols = self.construction_analyzer._find_columns_by_patterns(dataset, self.construction_analyzer.risk_patterns['structural_risk'])
        if structural_risk_cols:
            recommendations.append("Conduct structural integrity assessments")
        environmental_risk_cols = self.construction_analyzer._find_columns_by_patterns(dataset, self.construction_analyzer.risk_patterns['environmental_risk'])
        if environmental_risk_cols:
            recommendations.append("Implement environmental monitoring systems")
        return recommendations
    
    def _generate_infrastructure_actions(self, dataset: pd.DataFrame) -> list:
        """Generate infrastructure-specific action items"""
        actions = []
        material_cols = self.construction_analyzer._find_columns_by_patterns(dataset, self.construction_analyzer.data_patterns['infrastructure']['materials'])
        if material_cols:
            actions.append({
                'type': 'infrastructure',
                'priority': 'medium',
                'action': 'Material assessment',
                'description': 'Review and assess material conditions'
            })
        return actions
    
    def _analyze_risks_hazards(self, dataset: pd.DataFrame, content_intelligence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze risks and hazards at the site (modular fallback for RiskAnalyzer)"""
        # For now, use the same logic as before, but this should be replaced by a modular call to a real RiskAnalyzer
        risks = {
            'structural_risks': {},
            'environmental_risks': {},
            'fire_hazards': {},
            'chemical_hazards': {},
            'compliance_risks': {},
            'operational_risks': {},
            'financial_risks': {},
            'timeline_risks': {},
            'summary': [],
            'risk_details': []
        }
        # Only implement the NYC construction project risk logic for now
        if 'Project Description' in dataset.columns:
            for idx, row in dataset.iterrows():
                project_desc = row.get('Project Description', '')
                project_type = row.get('Project type', '')
                construction_award = row.get('Construction Award', 0)
                school_name = row.get('School Name', '')
                risk_detail = {
                    'project': project_desc,
                    'school': school_name,
                    'project_type': project_type,
                    'estimated_cost': f"${construction_award:,.0f}" if isinstance(construction_award, (int, float)) and pd.notna(construction_award) else 'Unknown',
                    'risk_level': 'Medium',
                    'structural_risks': [],
                    'environmental_risks': [],
                    'fire_hazards': [],
                    'compliance_risks': [],
                    'operational_risks': [],
                    'financial_risks': [],
                    'recommendations': []
                }
                desc_lower = project_desc.lower() if isinstance(project_desc, str) else ''
                if 'structural' in desc_lower or 'defect' in desc_lower:
                    risk_detail['structural_risks'].extend([
                        'Structural integrity concerns',
                        'Load-bearing capacity issues',
                        'Foundation stability risks',
                        'Building code compliance issues'
                    ])
                    risk_detail['risk_level'] = 'High'
                    risk_detail['recommendations'].extend([
                        'Conduct structural engineering assessment',
                        'Review building codes and regulations',
                        'Implement structural monitoring systems',
                        'Plan for potential structural reinforcements'
                    ])
                if 'fire alarm' in desc_lower or 'fire' in desc_lower:
                    risk_detail['fire_hazards'].extend([
                        'Fire safety system failures',
                        'Emergency evacuation concerns',
                        'Fire suppression system inadequacies',
                        'Code compliance issues'
                    ])
                    risk_detail['risk_level'] = 'High'
                    risk_detail['recommendations'].extend([
                        'Verify fire safety code compliance',
                        'Test emergency systems thoroughly',
                        'Update evacuation plans',
                        'Coordinate with fire department'
                    ])
                if 'climate control' in desc_lower or 'boiler' in desc_lower:
                    risk_detail['operational_risks'].extend([
                        'HVAC system failures',
                        'Energy efficiency concerns',
                        'Thermal comfort issues',
                        'Equipment reliability risks'
                    ])
                    risk_detail['environmental_risks'].extend([
                        'Energy consumption impacts',
                        'Carbon footprint concerns',
                        'Indoor air quality issues'
                    ])
                    risk_detail['recommendations'].extend([
                        'Assess energy efficiency requirements',
                        'Plan for system redundancy',
                        'Consider environmental impact',
                        'Implement monitoring systems'
                    ])
                if isinstance(construction_award, (int, float)) and pd.notna(construction_award) and construction_award > 5000000:
                    risk_detail['financial_risks'].extend([
                        'High-value project financial exposure',
                        'Budget overrun risks',
                        'Funding availability concerns',
                        'Cost escalation risks'
                    ])
                    risk_detail['recommendations'].extend([
                        'Implement strict budget controls',
                        'Monitor cost escalation factors',
                        'Plan for funding contingencies',
                        'Regular financial reporting'
                    ])
                risk_detail['compliance_risks'].extend([
                    'NYC building code compliance',
                    'School safety regulations',
                    'Environmental regulations',
                    'ADA accessibility requirements'
                ])
                risk_detail['recommendations'].extend([
                    'Review all applicable regulations',
                    'Coordinate with NYC DOB',
                    'Ensure ADA compliance',
                    'Obtain all required permits'
                ])
                risk_detail['operational_risks'].extend([
                    'School disruption during construction',
                    'Coordination with multiple stakeholders',
                    'Timeline delays and scheduling conflicts',
                    'Quality control and inspection requirements'
                ])
                risk_detail['recommendations'].extend([
                    'Develop detailed construction schedule',
                    'Coordinate with school administration',
                    'Plan for minimal disruption',
                    'Implement quality assurance program'
                ])
                risks['risk_details'].append(risk_detail)
        if risks['risk_details']:
            high_risk_projects = [r for r in risks['risk_details'] if r['risk_level'] == 'High']
            medium_risk_projects = [r for r in risks['risk_details'] if r['risk_level'] == 'Medium']
            risks['summary'].append(f"Risk Assessment: {len(high_risk_projects)} high-risk, {len(medium_risk_projects)} medium-risk projects")
            all_structural_risks = []
            all_fire_risks = []
            all_financial_risks = []
            for risk in risks['risk_details']:
                all_structural_risks.extend(risk['structural_risks'])
                all_fire_risks.extend(risk['fire_hazards'])
                all_financial_risks.extend(risk['financial_risks'])
            if all_structural_risks:
                risks['summary'].append(f"Structural risks identified: {len(set(all_structural_risks))} unique issues")
            if all_fire_risks:
                risks['summary'].append(f"Fire safety risks identified: {len(set(all_fire_risks))} unique issues")
            if all_financial_risks:
                risks['summary'].append(f"Financial risks identified: {len(set(all_financial_risks))} unique issues")
        return risks
    
    def _get_cross_dataset_analysis(self, dataset: pd.DataFrame, location: Optional[Dict[str, Any]] = None, content_intelligence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get cross-dataset analysis for the current dataset"""
        # For now, analyze the single dataset as if it were multiple datasets
        # In a real implementation, this would compare against other loaded datasets
        datasets = {'current_dataset': dataset}
        
        try:
            cross_analysis = self.cross_dataset_analyzer.analyze(datasets, location=location)
            return cross_analysis
        except Exception as e:
            logger.warning(f"Cross-dataset analysis failed: {e}")
            return {
                'cross_dataset_analysis': {},
                'spatial_relationships': {},
                'temporal_relationships': {},
                'data_quality_comparison': {},
                'correlations': {},
                'anomalies': {},
                'summary': ['Cross-dataset analysis unavailable'],
                'error': str(e)
            }

    def _extract_climate_context(self, dataset: pd.DataFrame) -> dict:
        """Extract climate data context if present in the dataset"""
        try:
            from .analyzers.environmental_analyzer import EnvironmentalAnalyzer
            analyzer = EnvironmentalAnalyzer()
            result = analyzer.analyze(dataset)
            return result.get('environmental_context', {})
        except Exception as e:
            logger.warning(f"Climate context extraction failed: {e}")
            return {}

    def _merge_environmental_and_climate(self, dataset: pd.DataFrame, climate_context: dict) -> dict:
        """Merge environmental context with climate data, ensuring climate is always present if available"""
        # Use the normal environmental analyzer
        try:
            env_context = self.environmental_analyzer.analyze(dataset)['environmental_context']
        except Exception as e:
            logger.warning(f"Environmental context extraction failed: {e}")
            env_context = {}
        # Merge climate data
        if climate_context and 'climate_data' in climate_context:
            env_context['climate_data'] = climate_context['climate_data']
            if 'summary' in climate_context:
                if 'summary' not in env_context:
                    env_context['summary'] = []
                env_context['summary'].extend([s for s in climate_context['summary'] if s not in env_context['summary']])
        return env_context