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
        """Initialize the Universal Reporter with comprehensive data patterns"""
        logger.info("Initializing Universal Reporter - The AI brain for civil engineering data")
        
        # Comprehensive data patterns for civil engineering
        self.data_patterns = {
            'infrastructure': {
                'pipes': ['pipe', 'drainage', 'sewer', 'stormwater', 'water', 'gas', 'oil', 'chemical'],
                'materials': ['material', 'type', 'composition', 'grade', 'class'],
                'dimensions': ['diameter', 'length', 'width', 'height', 'depth', 'thickness', 'size'],
                'structural': ['bridge', 'tunnel', 'culvert', 'retaining_wall', 'foundation', 'beam', 'column'],
                'electrical': ['transformer', 'pole', 'cable', 'substation', 'switchgear', 'meter'],
                'mechanical': ['pump', 'valve', 'tank', 'chamber', 'manhole', 'pit', 'vault'],
                'transport': ['road', 'highway', 'railway', 'airport', 'port', 'parking'],
                'buildings': ['building', 'structure', 'facility', 'plant', 'station', 'terminal']
            },
            'environmental': {
                'soil': ['soil', 'geotechnical', 'bearing_capacity', 'moisture', 'ph', 'contamination'],
                'vegetation': ['vegetation', 'tree', 'plant', 'species', 'density', 'age', 'health'],
                'climate': ['temperature', 'rainfall', 'wind', 'humidity', 'solar', 'weather'],
                'hydrology': ['river', 'stream', 'flood', 'groundwater', 'water_table', 'flow'],
                'geology': ['rock', 'fault', 'seismic', 'erosion', 'landslide', 'subsidence'],
                'air_quality': ['air', 'pollution', 'emission', 'particulate', 'gas'],
                'noise': ['noise', 'sound', 'vibration', 'decibel', 'acoustic']
            },
            'construction': {
                'project': ['project', 'phase', 'stage', 'milestone', 'schedule', 'timeline'],
                'resources': ['equipment', 'material', 'labor', 'contractor', 'subcontractor'],
                'quality': ['inspection', 'test', 'quality', 'compliance', 'certification'],
                'safety': ['safety', 'incident', 'accident', 'hazard', 'risk', 'compliance'],
                'permits': ['permit', 'license', 'approval', 'authorization', 'clearance']
            },
            'financial': {
                'costs': ['cost', 'budget', 'estimate', 'actual', 'variance', 'expense'],
                'assets': ['asset', 'value', 'depreciation', 'replacement', 'insurance'],
                'contracts': ['contract', 'agreement', 'warranty', 'bond', 'guarantee'],
                'billing': ['billing', 'invoice', 'payment', 'revenue', 'income']
            },
            'operational': {
                'maintenance': ['maintenance', 'repair', 'replacement', 'upgrade', 'refurbishment'],
                'performance': ['performance', 'efficiency', 'capacity', 'throughput', 'utilization'],
                'monitoring': ['monitor', 'sensor', 'measurement', 'reading', 'data'],
                'control': ['control', 'automation', 'system', 'operation', 'management']
            }
        }
        
        # Risk and impact patterns
        self.risk_patterns = {
            'structural_risk': ['crack', 'corrosion', 'deterioration', 'failure', 'collapse'],
            'environmental_risk': ['flood', 'earthquake', 'fire', 'storm', 'drought'],
            'operational_risk': ['breakdown', 'outage', 'interruption', 'failure', 'accident'],
            'financial_risk': ['cost_overrun', 'budget_exceed', 'loss', 'liability', 'penalty'],
            'compliance_risk': ['violation', 'non_compliance', 'penalty', 'fine', 'legal']
        }
        
        # Geographic and spatial patterns
        self.spatial_patterns = {
            'coordinates': ['lat', 'lon', 'latitude', 'longitude', 'x', 'y', 'easting', 'northing'],
            'addresses': ['address', 'street', 'city', 'state', 'postcode', 'zip'],
            'zones': ['zone', 'area', 'region', 'district', 'sector', 'quadrant'],
            'elevation': ['elevation', 'height', 'depth', 'altitude', 'grade', 'slope']
        }
        
        logger.info("Universal Reporter initialized with comprehensive data patterns")
    
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
        
        # Initialize comprehensive analysis
        briefing = {
            'executive_summary': self._generate_executive_summary(dataset, location),
            'site_materials': self._analyze_site_materials(dataset),
            'work_history': self._analyze_work_history(dataset),
            'utilities_infrastructure': self._analyze_utilities_infrastructure(dataset),
            'environmental_context': self._analyze_environmental_context(dataset),
            'costs_funding': self._analyze_costs_funding(dataset),
            'risks_hazards': self._analyze_risks_hazards(dataset),
            'missing_data': self._identify_missing_data(dataset),
            'recommendations': self._generate_actionable_recommendations(dataset, location),
            'nn_insights': self._get_neural_network_insights(dataset, location)
        }
        
        logger.info("Civil Engineer's Site Briefing completed")
        return briefing
    
    def _generate_executive_summary(self, dataset: pd.DataFrame, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate an executive summary of the site for civil engineers"""
        summary = {
            'site_overview': '',
            'key_findings': [],
            'critical_risks': [],
            'immediate_actions': [],
            'data_quality': self._assess_data_quality(dataset)
        }
        
        # Dynamic site overview
        total_records = len(dataset)
        coord_cols = self._find_coordinate_columns(dataset)
        has_coordinates = coord_cols['lat'] is not None and coord_cols['lon'] is not None
        
        # Generate site overview
        if has_coordinates:
            summary['site_overview'] = f"Site contains {total_records} geolocated records with comprehensive engineering data."
        else:
            summary['site_overview'] = f"Site contains {total_records} records (no coordinate data available)."
        
        # Key findings
        if total_records > 0:
            summary['key_findings'].append(f"Total records analyzed: {total_records:,}")
            
            # Check for construction projects
            construction_cols = self._find_columns_by_patterns(dataset, ['project', 'construction', 'building', 'facility'])
            if construction_cols:
                summary['key_findings'].append(f"Construction data available: {len(construction_cols)} relevant columns")
            
            # Check for infrastructure
            infra_cols = self._find_columns_by_patterns(dataset, ['pipe', 'electrical', 'water', 'gas', 'road', 'bridge'])
            if infra_cols:
                summary['key_findings'].append(f"Infrastructure data available: {len(infra_cols)} relevant columns")
            
            # Check for environmental data
            env_cols = self._find_columns_by_patterns(dataset, ['soil', 'climate', 'vegetation', 'flood', 'environmental'])
            if env_cols:
                summary['key_findings'].append(f"Environmental data available: {len(env_cols)} relevant columns")
        
        return summary
    
    def _analyze_site_materials(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze all materials present or required at the site"""
        materials = {
            'building_materials': {},
            'infrastructure_materials': {},
            'soil_materials': {},
            'construction_materials': {},
            'summary': []
        }
        
        # Dynamic material discovery
        material_patterns = ['material', 'type', 'composition', 'grade', 'class', 'steel', 'concrete', 'wood', 'plastic', 'metal']
        material_cols = self._find_columns_by_patterns(dataset, material_patterns)
        
        for col in material_cols:
            if col in dataset.columns:
                try:
                    # Get material distribution
                    material_counts = dataset[col].value_counts()
                    materials['building_materials'][col] = material_counts.to_dict()
                    
                    # Add to summary
                    top_materials = material_counts.head(3)
                    materials['summary'].append(f"{col}: {', '.join([f'{mat} ({count})' for mat, count in top_materials.items()])}")
                except Exception as e:
                    logger.warning(f"Material analysis failed for column {col}: {e}")
        
        # Check for construction materials
        construction_patterns = ['construction', 'building', 'structure', 'facility']
        construction_cols = self._find_columns_by_patterns(dataset, construction_patterns)
        if construction_cols:
            materials['summary'].append(f"Construction projects: {len(construction_cols)} relevant data columns")
        
        return materials
    
    def _analyze_work_history(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze work history, permits, inspections, and incidents"""
        history = {
            'permits': {},
            'inspections': {},
            'incidents': {},
            'projects': {},
            'timeline': {},
            'summary': [],
            'project_details': []  # NEW: Actual project details
        }
        
        # Dynamic work history discovery
        permit_patterns = ['permit', 'license', 'approval', 'authorization']
        permit_cols = self._find_columns_by_patterns(dataset, permit_patterns)
        
        inspection_patterns = ['inspection', 'test', 'quality', 'compliance', 'certification']
        inspection_cols = self._find_columns_by_patterns(dataset, inspection_patterns)
        
        incident_patterns = ['incident', 'accident', 'fire', 'hazard', 'violation', 'failure']
        incident_cols = self._find_columns_by_patterns(dataset, incident_patterns)
        
        project_patterns = ['project', 'construction', 'building', 'facility', 'phase', 'stage']
        project_cols = self._find_columns_by_patterns(dataset, project_patterns)
        
        # NEW: Extract actual project details
        if 'Project Title' in dataset.columns:
            for idx, row in dataset.iterrows():
                project_detail = {
                    'name': row.get('Project Title', 'Unknown'),
                    'campus': row.get('Campus Name', 'Unknown'),
                    'project_id': row.get('Project', 'Unknown'),
                    'estimated_value': row.get('Estimated Contract Value', 'Unknown'),
                    'advertise_date': row.get('Contract Advertise Date', 'Unknown'),
                    'status': row.get('Still Accepting SOQ', 'Unknown')
                }
                history['project_details'].append(project_detail)
        
        # Also check for NYC construction dataset structure
        elif 'School Name' in dataset.columns and 'Project Description' in dataset.columns:
            for idx, row in dataset.iterrows():
                project_detail = {
                    'name': row.get('Project Description', 'Unknown'),
                    'campus': row.get('School Name', 'Unknown'),
                    'project_id': row.get('Building ID', 'Unknown'),
                    'estimated_value': f"${row.get('Construction Award', 0):,.0f}" if pd.notna(row.get('Construction Award')) else 'Unknown',
                    'project_type': row.get('Project type', 'Unknown'),
                    'address': row.get('Building Address', 'Unknown'),
                    'borough': row.get('Borough', 'Unknown'),
                    'status': 'Active'  # These are active projects
                }
                history['project_details'].append(project_detail)
        
        # Also check for other project-related columns
        if 'Project Description' in dataset.columns:
            for idx, row in dataset.iterrows():
                if 'Project Description' in row and pd.notna(row['Project Description']):
                    history['summary'].append(f"Project: {row['Project Description'][:100]}...")
        
        if 'School Name' in dataset.columns:
            schools = dataset['School Name'].value_counts()
            history['summary'].append(f"Schools involved: {', '.join(schools.head(3).index.tolist())}")
        
        # Analyze construction awards
        if 'Construction Award' in dataset.columns:
            try:
                awards = dataset['Construction Award'].dropna()
                if len(awards) > 0:
                    total_award = awards.sum()
                    avg_award = awards.mean()
                    max_award = awards.max()
                    min_award = awards.min()
                    
                    history['summary'].append(f"Total construction value: ${total_award:,.0f}")
                    history['summary'].append(f"Average project value: ${avg_award:,.0f}")
                    history['summary'].append(f"Cost range: ${min_award:,.0f} - ${max_award:,.0f}")
                    
                    # Add detailed cost breakdown
                    for idx, row in dataset.iterrows():
                        if pd.notna(row.get('Construction Award')):
                            history['project_details'].append({
                                'project': row.get('Project Description', 'Unknown'),
                                'estimated_value': f"${row.get('Construction Award', 0):,.0f}",
                                'project_id': row.get('Building ID', 'Unknown'),
                                'campus': row.get('School Name', 'Unknown'),
                                'project_type': row.get('Project type', 'Unknown')
                            })
            except Exception as e:
                logger.warning(f"Construction award analysis failed: {e}")
        
        # Analyze project types
        if 'Project type' in dataset.columns:
            project_types = dataset['Project type'].value_counts()
            for ptype, count in project_types.items():
                history['summary'].append(f"Project type {ptype}: {count} projects")
        
        # Analyze permits
        for col in permit_cols:
            if col in dataset.columns:
                try:
                    permit_status = dataset[col].value_counts()
                    history['permits'][col] = permit_status.to_dict()
                    history['summary'].append(f"Permits: {len(permit_status)} different statuses found")
                except Exception as e:
                    logger.warning(f"Permit analysis failed for column {col}: {e}")
        
        # Analyze inspections
        for col in inspection_cols:
            if col in dataset.columns:
                try:
                    inspection_results = dataset[col].value_counts()
                    history['inspections'][col] = inspection_results.to_dict()
                    history['summary'].append(f"Inspections: {len(inspection_results)} different results found")
                except Exception as e:
                    logger.warning(f"Inspection analysis failed for column {col}: {e}")
        
        # Analyze incidents
        for col in incident_cols:
            if col in dataset.columns:
                try:
                    incident_types = dataset[col].value_counts()
                    history['incidents'][col] = incident_types.to_dict()
                    history['summary'].append(f"Incidents: {len(incident_types)} different types found")
                except Exception as e:
                    logger.warning(f"Incident analysis failed for column {col}: {e}")
        
        # Analyze projects
        for col in project_cols:
            if col in dataset.columns:
                try:
                    project_types = dataset[col].value_counts()
                    history['projects'][col] = project_types.to_dict()
                    history['summary'].append(f"Projects: {len(project_types)} different types found")
                except Exception as e:
                    logger.warning(f"Project analysis failed for column {col}: {e}")
        
        return history
    
    def _analyze_utilities_infrastructure(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze utilities and infrastructure at the site"""
        utilities = {
            'electrical': {},
            'water_sewer': {},
            'gas': {},
            'telecom': {},
            'roads_bridges': {},
            'summary': []
        }
        
        # Dynamic utility discovery
        electrical_patterns = ['electrical', 'power', 'transformer', 'substation', 'voltage', 'amp']
        electrical_cols = self._find_columns_by_patterns(dataset, electrical_patterns)
        
        water_patterns = ['water', 'sewer', 'drainage', 'pipe', 'valve', 'pump']
        water_cols = self._find_columns_by_patterns(dataset, water_patterns)
        
        gas_patterns = ['gas', 'natural_gas', 'propane', 'fuel']
        gas_cols = self._find_columns_by_patterns(dataset, gas_patterns)
        
        telecom_patterns = ['telecom', 'fiber', 'cable', 'internet', 'phone']
        telecom_cols = self._find_columns_by_patterns(dataset, telecom_patterns)
        
        road_patterns = ['road', 'street', 'highway', 'bridge', 'traffic']
        road_cols = self._find_columns_by_patterns(dataset, road_patterns)
        
        # Analyze electrical
        for col in electrical_cols:
            if col in dataset.columns:
                try:
                    electrical_data = dataset[col].value_counts()
                    utilities['electrical'][col] = electrical_data.to_dict()
                    utilities['summary'].append(f"Electrical: {len(electrical_data)} different values found")
                except Exception as e:
                    logger.warning(f"Electrical analysis failed for column {col}: {e}")
        
        # Analyze water/sewer
        for col in water_cols:
            if col in dataset.columns:
                try:
                    water_data = dataset[col].value_counts()
                    utilities['water_sewer'][col] = water_data.to_dict()
                    utilities['summary'].append(f"Water/Sewer: {len(water_data)} different values found")
                except Exception as e:
                    logger.warning(f"Water analysis failed for column {col}: {e}")
        
        # Analyze gas
        for col in gas_cols:
            if col in dataset.columns:
                try:
                    gas_data = dataset[col].value_counts()
                    utilities['gas'][col] = gas_data.to_dict()
                    utilities['summary'].append(f"Gas: {len(gas_data)} different values found")
                except Exception as e:
                    logger.warning(f"Gas analysis failed for column {col}: {e}")
        
        # Analyze telecom
        for col in telecom_cols:
            if col in dataset.columns:
                try:
                    telecom_data = dataset[col].value_counts()
                    utilities['telecom'][col] = telecom_data.to_dict()
                    utilities['summary'].append(f"Telecom: {len(telecom_data)} different values found")
                except Exception as e:
                    logger.warning(f"Telecom analysis failed for column {col}: {e}")
        
        # Analyze roads/bridges
        for col in road_cols:
            if col in dataset.columns:
                try:
                    road_data = dataset[col].value_counts()
                    utilities['roads_bridges'][col] = road_data.to_dict()
                    utilities['summary'].append(f"Roads/Bridges: {len(road_data)} different values found")
                except Exception as e:
                    logger.warning(f"Road analysis failed for column {col}: {e}")
        
        return utilities
    
    def _analyze_environmental_context(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze environmental context of the site"""
        environmental = {
            'soil_conditions': {},
            'climate_data': {},
            'vegetation': {},
            'water_resources': {},
            'air_quality': {},
            'protected_areas': {},
            'summary': []
        }
        
        # Dynamic environmental discovery
        soil_patterns = ['soil', 'geotechnical', 'bearing_capacity', 'moisture', 'ph']
        soil_cols = self._find_columns_by_patterns(dataset, soil_patterns)
        
        climate_patterns = ['temperature', 'rainfall', 'wind', 'humidity', 'climate']
        climate_cols = self._find_columns_by_patterns(dataset, climate_patterns)
        
        vegetation_patterns = ['vegetation', 'tree', 'plant', 'species', 'density']
        vegetation_cols = self._find_columns_by_patterns(dataset, vegetation_patterns)
        
        water_patterns = ['water', 'river', 'stream', 'flood', 'groundwater', 'wetland']
        water_cols = self._find_columns_by_patterns(dataset, water_patterns)
        
        air_patterns = ['air', 'pollution', 'emission', 'particulate', 'gas']
        air_cols = self._find_columns_by_patterns(dataset, air_patterns)
        
        # Analyze soil conditions
        for col in soil_cols:
            if col in dataset.columns:
                try:
                    soil_data = dataset[col].value_counts()
                    environmental['soil_conditions'][col] = soil_data.to_dict()
                    environmental['summary'].append(f"Soil: {len(soil_data)} different conditions found")
                except Exception as e:
                    logger.warning(f"Soil analysis failed for column {col}: {e}")
        
        # Analyze climate data
        for col in climate_cols:
            if col in dataset.columns:
                try:
                    if dataset[col].dtype in ['int64', 'float64']:
                        climate_stats = {
                            'min': float(dataset[col].min()),
                            'max': float(dataset[col].max()),
                            'mean': float(dataset[col].mean()),
                            'std': float(dataset[col].std())
                        }
                        environmental['climate_data'][col] = climate_stats
                        environmental['summary'].append(f"Climate {col}: {climate_stats['mean']:.1f} avg (range: {climate_stats['min']:.1f}-{climate_stats['max']:.1f})")
                    else:
                        climate_data = dataset[col].value_counts()
                        environmental['climate_data'][col] = climate_data.to_dict()
                        environmental['summary'].append(f"Climate: {len(climate_data)} different values found")
                except Exception as e:
                    logger.warning(f"Climate analysis failed for column {col}: {e}")
        
        # Analyze vegetation
        for col in vegetation_cols:
            if col in dataset.columns:
                try:
                    vegetation_data = dataset[col].value_counts()
                    environmental['vegetation'][col] = vegetation_data.to_dict()
                    environmental['summary'].append(f"Vegetation: {len(vegetation_data)} different types found")
                except Exception as e:
                    logger.warning(f"Vegetation analysis failed for column {col}: {e}")
        
        return environmental
    
    def _analyze_costs_funding(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze costs, funding, and financial aspects"""
        costs = {
            'project_costs': {},
            'maintenance_costs': {},
            'funding_sources': {},
            'budget_status': {},
            'summary': [],
            'cost_details': []  # NEW: Actual cost details
        }
        
        # NEW: Extract actual cost data
        if 'Estimated Contract Value' in dataset.columns:
            total_estimated_value = 0
            cost_ranges = []
            
            for idx, row in dataset.iterrows():
                estimated_value = row.get('Estimated Contract Value', 'Unknown')
                if isinstance(estimated_value, str) and estimated_value != 'Unknown':
                    costs['cost_details'].append({
                        'project': row.get('Project Title', 'Unknown'),
                        'estimated_value': estimated_value,
                        'project_id': row.get('Project', 'Unknown'),
                        'campus': row.get('Campus Name', 'Unknown')
                    })
                    
                    # Try to extract numeric value for total calculation
                    try:
                        # Handle ranges like "$5,000,000 - $10,000,000"
                        if ' - ' in estimated_value:
                            min_val_str = estimated_value.split(' - ')[0].replace('$', '').replace(',', '')
                            max_val_str = estimated_value.split(' - ')[1].replace('$', '').replace(',', '')
                            min_val = float(min_val_str)
                            max_val = float(max_val_str)
                            avg_val = (min_val + max_val) / 2
                            total_estimated_value += avg_val
                            cost_ranges.append(f"${min_val:,.0f} - ${max_val:,.0f}")
                        else:
                            # Handle single values like "$10,000,000 - Up"
                            val_str = estimated_value.replace('$', '').replace(',', '').replace(' - Up', '')
                            if val_str.isdigit():
                                val = float(val_str)
                                total_estimated_value += val
                                cost_ranges.append(f"${val:,.0f}")
                    except:
                        pass
            
            if total_estimated_value > 0:
                costs['summary'].append(f"Total estimated project value: ${total_estimated_value:,.0f}")
                costs['summary'].append(f"Average project value: ${total_estimated_value/len(costs['cost_details']):,.0f}")
            
            if cost_ranges:
                costs['summary'].append(f"Cost ranges: {', '.join(cost_ranges[:3])}")
        
        # Also check for NYC construction dataset structure
        elif 'Construction Award' in dataset.columns:
            try:
                awards = dataset['Construction Award'].dropna()
                if len(awards) > 0:
                    total_award = awards.sum()
                    avg_award = awards.mean()
                    max_award = awards.max()
                    min_award = awards.min()
                    
                    costs['summary'].append(f"Total construction value: ${total_award:,.0f}")
                    costs['summary'].append(f"Average project value: ${avg_award:,.0f}")
                    costs['summary'].append(f"Cost range: ${min_award:,.0f} - ${max_award:,.0f}")
                    
                    # Add detailed cost breakdown
                    for idx, row in dataset.iterrows():
                        if pd.notna(row.get('Construction Award')):
                            costs['cost_details'].append({
                                'project': row.get('Project Description', 'Unknown'),
                                'estimated_value': f"${row.get('Construction Award', 0):,.0f}",
                                'project_id': row.get('Building ID', 'Unknown'),
                                'campus': row.get('School Name', 'Unknown'),
                                'project_type': row.get('Project type', 'Unknown')
                            })
            except Exception as e:
                logger.warning(f"Construction award analysis failed: {e}")
        
        # Dynamic cost discovery
        cost_patterns = ['cost', 'budget', 'estimate', 'actual', 'variance', 'expense', 'amount', 'dollar']
        cost_cols = self._find_columns_by_patterns(dataset, cost_patterns)
        
        funding_patterns = ['funding', 'grant', 'loan', 'bond', 'revenue', 'income']
        funding_cols = self._find_columns_by_patterns(dataset, funding_patterns)
        
        # Analyze costs
        for col in cost_cols:
            if col in dataset.columns:
                try:
                    if dataset[col].dtype in ['int64', 'float64']:
                        cost_stats = {
                            'total': float(dataset[col].sum()),
                            'mean': float(dataset[col].mean()),
                            'min': float(dataset[col].min()),
                            'max': float(dataset[col].max())
                        }
                        costs['project_costs'][col] = cost_stats
                        costs['summary'].append(f"Costs {col}: ${cost_stats['total']:,.0f} total, ${cost_stats['mean']:,.0f} avg")
                    else:
                        cost_data = dataset[col].value_counts()
                        costs['project_costs'][col] = cost_data.to_dict()
                        costs['summary'].append(f"Costs: {len(cost_data)} different categories found")
                except Exception as e:
                    logger.warning(f"Cost analysis failed for column {col}: {e}")
        
        # Analyze funding
        for col in funding_cols:
            if col in dataset.columns:
                try:
                    funding_data = dataset[col].value_counts()
                    costs['funding_sources'][col] = funding_data.to_dict()
                    costs['summary'].append(f"Funding: {len(funding_data)} different sources found")
                except Exception as e:
                    logger.warning(f"Funding analysis failed for column {col}: {e}")
        
        return costs
    
    def _analyze_risks_hazards(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risks and hazards at the site"""
        risks = {
            'structural_risks': {},
            'environmental_risks': {},
            'fire_hazards': {},
            'chemical_hazards': {},
            'compliance_risks': {},
            'operational_risks': {},
            'summary': []
        }
        
        # Dynamic risk discovery
        structural_patterns = ['crack', 'corrosion', 'deterioration', 'failure', 'collapse', 'structural']
        structural_cols = self._find_columns_by_patterns(dataset, structural_patterns)
        
        environmental_patterns = ['flood', 'earthquake', 'storm', 'drought', 'contamination']
        environmental_cols = self._find_columns_by_patterns(dataset, environmental_patterns)
        
        fire_patterns = ['fire', 'smoke', 'combustible', 'flammable', 'ignition']
        fire_cols = self._find_columns_by_patterns(dataset, fire_patterns)
        
        chemical_patterns = ['chemical', 'toxic', 'hazardous', 'contamination', 'pollution']
        chemical_cols = self._find_columns_by_patterns(dataset, chemical_patterns)
        
        compliance_patterns = ['violation', 'non_compliance', 'penalty', 'fine', 'legal']
        compliance_cols = self._find_columns_by_patterns(dataset, compliance_patterns)
        
        # Analyze structural risks
        for col in structural_cols:
            if col in dataset.columns:
                try:
                    structural_data = dataset[col].value_counts()
                    risks['structural_risks'][col] = structural_data.to_dict()
                    risks['summary'].append(f"Structural risks: {len(structural_data)} different issues found")
                except Exception as e:
                    logger.warning(f"Structural risk analysis failed for column {col}: {e}")
        
        # Analyze environmental risks
        for col in environmental_cols:
            if col in dataset.columns:
                try:
                    environmental_data = dataset[col].value_counts()
                    risks['environmental_risks'][col] = environmental_data.to_dict()
                    risks['summary'].append(f"Environmental risks: {len(environmental_data)} different hazards found")
                except Exception as e:
                    logger.warning(f"Environmental risk analysis failed for column {col}: {e}")
        
        # Analyze fire hazards
        for col in fire_cols:
            if col in dataset.columns:
                try:
                    fire_data = dataset[col].value_counts()
                    risks['fire_hazards'][col] = fire_data.to_dict()
                    risks['summary'].append(f"Fire hazards: {len(fire_data)} different risks found")
                except Exception as e:
                    logger.warning(f"Fire hazard analysis failed for column {col}: {e}")
        
        # Analyze chemical hazards
        for col in chemical_cols:
            if col in dataset.columns:
                try:
                    chemical_data = dataset[col].value_counts()
                    risks['chemical_hazards'][col] = chemical_data.to_dict()
                    risks['summary'].append(f"Chemical hazards: {len(chemical_data)} different substances found")
                except Exception as e:
                    logger.warning(f"Chemical hazard analysis failed for column {col}: {e}")
        
        return risks
    
    def _identify_missing_data(self, dataset: pd.DataFrame) -> Dict[str, Any]:
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
            found_cols = self._find_columns_by_patterns(dataset, patterns)
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
    
    def _generate_actionable_recommendations(self, dataset: pd.DataFrame, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        construction_cols = self._find_columns_by_patterns(dataset, ['project', 'construction', 'building'])
        if construction_cols:
            recommendations['immediate_actions'].append("Review all construction project details and timelines")
            recommendations['permits_required'].append("Verify all required permits are in place for construction activities")
        
        # Check for structural issues
        structural_cols = self._find_columns_by_patterns(dataset, ['structural', 'crack', 'failure', 'deterioration'])
        if structural_cols:
            recommendations['investigations_needed'].append("Conduct structural integrity assessment")
            recommendations['safety_measures'].append("Implement structural monitoring and safety protocols")
        
        # Check for environmental concerns
        environmental_cols = self._find_columns_by_patterns(dataset, ['soil', 'flood', 'contamination', 'environmental'])
        if environmental_cols:
            recommendations['investigations_needed'].append("Perform environmental impact assessment")
            recommendations['safety_measures'].append("Implement environmental monitoring and protection measures")
        
        # Check for utility conflicts
        utility_cols = self._find_columns_by_patterns(dataset, ['electrical', 'water', 'gas', 'sewer'])
        if utility_cols:
            recommendations['immediate_actions'].append("Coordinate with utility companies for service verification")
            recommendations['safety_measures'].append("Implement utility marking and protection protocols")
        
        # General recommendations
        recommendations['next_steps'].append("Schedule site visit and detailed inspection")
        recommendations['next_steps'].append("Review all available permits and compliance requirements")
        recommendations['next_steps'].append("Coordinate with local authorities and utility companies")
        
        return recommendations
    
    def _get_neural_network_insights(self, dataset: pd.DataFrame, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        pipe_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['pipes'])
        if pipe_cols:
            infrastructure['pipe_analysis'] = self._analyze_pipes(dataset, pipe_cols)
        
        # Material analysis
        material_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['materials'])
        if material_cols:
            infrastructure['material_analysis'] = self._analyze_materials(dataset, material_cols)
        
        # Dimension analysis
        dimension_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['dimensions'])
        if dimension_cols:
            infrastructure['dimension_analysis'] = self._analyze_dimensions(dataset, dimension_cols)
        
        # Structural analysis
        structural_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['structural'])
        if structural_cols:
            infrastructure['structural_analysis'] = self._analyze_structural(dataset, structural_cols)
        
        # Electrical analysis
        electrical_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['electrical'])
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
        soil_cols = self._find_columns_by_patterns(dataset, self.data_patterns['environmental']['soil'])
        if soil_cols:
            environmental['soil_analysis'] = self._analyze_soil(dataset, soil_cols)
        
        # Vegetation analysis
        vegetation_cols = self._find_columns_by_patterns(dataset, self.data_patterns['environmental']['vegetation'])
        if vegetation_cols:
            environmental['vegetation_analysis'] = self._analyze_vegetation(dataset, vegetation_cols)
        
        # Climate analysis
        climate_cols = self._find_columns_by_patterns(dataset, self.data_patterns['environmental']['climate'])
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
        project_cols = self._find_columns_by_patterns(dataset, self.data_patterns['construction']['project'])
        if project_cols:
            construction['project_analysis'] = self._analyze_projects(dataset, project_cols)
        
        # Resource analysis
        resource_cols = self._find_columns_by_patterns(dataset, self.data_patterns['construction']['resources'])
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
        cost_cols = self._find_columns_by_patterns(dataset, self.data_patterns['financial']['costs'])
        if cost_cols:
            financial['cost_analysis'] = self._analyze_costs(dataset, cost_cols)
        
        # Asset analysis
        asset_cols = self._find_columns_by_patterns(dataset, self.data_patterns['financial']['assets'])
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
        maintenance_cols = self._find_columns_by_patterns(dataset, self.data_patterns['operational']['maintenance'])
        if maintenance_cols:
            operational['maintenance_analysis'] = self._analyze_maintenance(dataset, maintenance_cols)
        
        # Performance analysis
        performance_cols = self._find_columns_by_patterns(dataset, self.data_patterns['operational']['performance'])
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
        structural_risk_cols = self._find_columns_by_patterns(dataset, self.risk_patterns['structural_risk'])
        if structural_risk_cols:
            risks['structural_risks'] = self._analyze_structural_risks(dataset, structural_risk_cols)
        
        # Environmental risks
        environmental_risk_cols = self._find_columns_by_patterns(dataset, self.risk_patterns['environmental_risk'])
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
        coord_patterns = self.spatial_patterns['coordinates']
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
                    if not numeric_data.isna().all():
                        dimensions[col] = {
                            'min': float(numeric_data.min()),
                            'max': float(numeric_data.max()),
                            'mean': float(numeric_data.mean()),
                            'std': float(numeric_data.std()),
                            'median': float(numeric_data.median())
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
                    if not numeric_data.isna().all():
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
                    if not numeric_data.isna().all():
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
        for category in self.data_patterns['infrastructure'].values():
            infrastructure_patterns.extend(category)
        return len(self._find_columns_by_patterns(dataset, infrastructure_patterns)) > 0
    
    def _has_risk_indicators(self, dataset: pd.DataFrame) -> bool:
        """Check if dataset has risk indicators"""
        risk_patterns = []
        for category in self.risk_patterns.values():
            risk_patterns.extend(category)
        return len(self._find_columns_by_patterns(dataset, risk_patterns)) > 0
    
    def _generate_infrastructure_recommendations(self, dataset: pd.DataFrame) -> List[str]:
        """Generate infrastructure-specific recommendations"""
        recommendations = []
        
        # Material recommendations
        material_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['materials'])
        if material_cols:
            recommendations.append("Review material specifications and consider upgrades")
        
        # Dimension recommendations
        dimension_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['dimensions'])
        if dimension_cols:
            recommendations.append("Assess capacity requirements and upgrade needs")
        
        return recommendations
    
    def _generate_risk_recommendations(self, dataset: pd.DataFrame) -> List[str]:
        """Generate risk-specific recommendations"""
        recommendations = []
        
        # Structural risk recommendations
        structural_risk_cols = self._find_columns_by_patterns(dataset, self.risk_patterns['structural_risk'])
        if structural_risk_cols:
            recommendations.append("Conduct structural integrity assessments")
        
        # Environmental risk recommendations
        environmental_risk_cols = self._find_columns_by_patterns(dataset, self.risk_patterns['environmental_risk'])
        if environmental_risk_cols:
            recommendations.append("Implement environmental monitoring systems")
        
        return recommendations
    
    def _generate_infrastructure_actions(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate infrastructure-specific action items"""
        actions = []
        
        # Material actions
        material_cols = self._find_columns_by_patterns(dataset, self.data_patterns['infrastructure']['materials'])
        if material_cols:
            actions.append({
                'type': 'infrastructure',
                'priority': 'medium',
                'action': 'Material assessment',
                'description': 'Review and assess material conditions'
            })
        
        return actions 