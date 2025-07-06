# Author: KleaSCM
# Date: 2024
# Description: Construction data analyzer

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_analyzer import BaseAnalyzer

class ConstructionAnalyzer(BaseAnalyzer):
    """Analyzes construction-related data including projects, materials, and work history"""
    
    def analyze(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze construction data"""
        self.logger.info(f"Analyzing construction data with {len(dataset)} records")
        
        return {
            'site_materials': self._analyze_site_materials(dataset),
            'work_history': self._analyze_work_history(dataset),
            'project_details': self._analyze_project_details(dataset),
            'summary': self._generate_summary(dataset)
        }
    
    def _analyze_site_materials(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze all materials present or required at the site"""
        materials = {
            'building_materials': {},
            'infrastructure_materials': {},
            'soil_materials': {},
            'construction_materials': {},
            'equipment_systems': {},
            'summary': [],
            'material_details': []
        }
        
        # Extract detailed material information from NYC construction projects
        if 'Project Description' in dataset.columns:
            for idx, row in dataset.iterrows():
                project_desc = row.get('Project Description', '')
                project_type = row.get('Project type', '')
                construction_award = row.get('Construction Award', 0)
                
                material_detail = {
                    'project': project_desc,
                    'school': row.get('School Name', ''),
                    'project_type': project_type,
                    'estimated_cost': f"${construction_award:,.0f}" if isinstance(construction_award, (int, float)) and pd.notna(construction_award) else 'Unknown',
                    'materials_required': [],
                    'equipment_systems': [],
                    'address': row.get('Building Address', ''),
                    'borough': row.get('Borough', '')
                }
                
                # Analyze project description for materials and equipment
                desc_lower = project_desc.lower() if isinstance(project_desc, str) else ''
                
                # HVAC and Climate Control Systems
                if 'climate control' in desc_lower or 'boiler' in desc_lower:
                    material_detail['equipment_systems'].extend([
                        'HVAC Systems',
                        'Boiler Systems', 
                        'Climate Control Equipment',
                        'Thermal Management Systems'
                    ])
                    material_detail['materials_required'].extend([
                        'HVAC Ductwork',
                        'Boiler Components',
                        'Thermostats and Controls',
                        'Insulation Materials'
                    ])
                
                # Fire Safety Systems
                if 'fire alarm' in desc_lower or 'fire' in desc_lower:
                    material_detail['equipment_systems'].extend([
                        'Fire Alarm Systems',
                        'Smoke Detectors',
                        'Emergency Lighting',
                        'Fire Suppression Equipment'
                    ])
                    material_detail['materials_required'].extend([
                        'Fire Alarm Panels',
                        'Smoke Detectors',
                        'Emergency Exit Signs',
                        'Fire-Rated Materials'
                    ])
                
                # Structural Systems
                if 'structural' in desc_lower or 'defect' in desc_lower:
                    material_detail['equipment_systems'].extend([
                        'Structural Reinforcement',
                        'Foundation Systems',
                        'Load-Bearing Components'
                    ])
                    material_detail['materials_required'].extend([
                        'Steel Beams',
                        'Concrete',
                        'Reinforcement Bars',
                        'Structural Fasteners'
                    ])
                
                # Electrical Systems
                if 'electrical' in desc_lower or 'wiring' in desc_lower:
                    material_detail['equipment_systems'].extend([
                        'Electrical Systems',
                        'Power Distribution',
                        'Lighting Systems'
                    ])
                    material_detail['materials_required'].extend([
                        'Electrical Panels',
                        'Wiring and Conduit',
                        'Lighting Fixtures',
                        'Electrical Outlets'
                    ])
                
                # Plumbing Systems
                if 'plumbing' in desc_lower or 'water' in desc_lower:
                    material_detail['equipment_systems'].extend([
                        'Plumbing Systems',
                        'Water Distribution',
                        'Drainage Systems'
                    ])
                    material_detail['materials_required'].extend([
                        'Pipes and Fittings',
                        'Valves and Controls',
                        'Water Heaters',
                        'Drainage Components'
                    ])
                
                materials['material_details'].append(material_detail)
        
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
                    self.logger.warning(f"Material analysis failed for column {col}: {e}")
        
        # Check for construction materials
        construction_patterns = ['construction', 'building', 'structure', 'facility']
        construction_cols = self._find_columns_by_patterns(dataset, construction_patterns)
        if construction_cols:
            materials['summary'].append(f"Construction projects: {len(construction_cols)} relevant data columns")
        
        # Add summary of material types found
        if materials['material_details']:
            total_projects = len(materials['material_details'])
            total_cost = sum([float(detail['estimated_cost'].replace('$', '').replace(',', '')) 
                            for detail in materials['material_details'] 
                            if detail['estimated_cost'] != 'Unknown'])
            
            materials['summary'].append(f"Total projects analyzed: {total_projects}")
            materials['summary'].append(f"Total material/equipment value: ${total_cost:,.0f}")
            
            # Count equipment types
            all_equipment = []
            for detail in materials['material_details']:
                all_equipment.extend(detail['equipment_systems'])
            
            equipment_counts = {}
            for equipment in all_equipment:
                equipment_counts[equipment] = equipment_counts.get(equipment, 0) + 1
            
            if equipment_counts:
                materials['summary'].append(f"Equipment systems: {', '.join([f'{eq} ({count})' for eq, count in equipment_counts.items()])}")
        
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
            'project_details': []
        }
        
        # Extract actual project details
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
        elif 'Project Description' in dataset.columns:
            for idx, row in dataset.iterrows():
                project_detail = {
                    'name': row.get('Project Description', 'Unknown'),
                    'campus': row.get('School Name', 'Unknown'),
                    'project_id': row.get('Building ID', 'Unknown'),
                    'estimated_value': f"${row.get('Construction Award', 0):,.0f}" if isinstance(row.get('Construction Award', 0), (int, float)) and row.get('Construction Award', 0) is not None else 'Unknown',
                    'project_type': row.get('Project type', 'Unknown'),
                    'address': row.get('Building Address', 'Unknown'),
                    'borough': row.get('Borough', 'Unknown'),
                    'status': 'Active'
                }
                history['project_details'].append(project_detail)
        
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
            except Exception as e:
                self.logger.warning(f"Construction award analysis failed: {e}")
        
        # Analyze project types
        if 'Project type' in dataset.columns:
            project_types = dataset['Project type'].value_counts()
            for ptype, count in project_types.items():
                history['summary'].append(f"Project type {ptype}: {count} projects")
        
        # Dynamic work history discovery
        permit_patterns = ['permit', 'license', 'approval', 'authorization']
        permit_cols = self._find_columns_by_patterns(dataset, permit_patterns)
        
        inspection_patterns = ['inspection', 'test', 'quality', 'compliance', 'certification']
        inspection_cols = self._find_columns_by_patterns(dataset, inspection_patterns)
        
        incident_patterns = ['incident', 'accident', 'fire', 'hazard', 'violation', 'failure']
        incident_cols = self._find_columns_by_patterns(dataset, incident_patterns)
        
        # Analyze permits
        for col in permit_cols:
            if col in dataset.columns:
                try:
                    permit_status = dataset[col].value_counts()
                    history['permits'][col] = permit_status.to_dict()
                    history['summary'].append(f"Permits: {len(permit_status)} different statuses found")
                except Exception as e:
                    self.logger.warning(f"Permit analysis failed for column {col}: {e}")
        
        # Analyze inspections
        for col in inspection_cols:
            if col in dataset.columns:
                try:
                    inspection_results = dataset[col].value_counts()
                    history['inspections'][col] = inspection_results.to_dict()
                    history['summary'].append(f"Inspections: {len(inspection_results)} different results found")
                except Exception as e:
                    self.logger.warning(f"Inspection analysis failed for column {col}: {e}")
        
        # Analyze incidents
        for col in incident_cols:
            if col in dataset.columns:
                try:
                    incident_types = dataset[col].value_counts()
                    history['incidents'][col] = incident_types.to_dict()
                    history['summary'].append(f"Incidents: {len(incident_types)} different types found")
                except Exception as e:
                    self.logger.warning(f"Incident analysis failed for column {col}: {e}")
        
        return history
    
    def _analyze_project_details(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detailed project information"""
        project_details = {
            'total_projects': 0,
            'project_types': {},
            'cost_distribution': {},
            'timeline_analysis': {},
            'location_analysis': {},
            'summary': []
        }
        
        # Count total projects
        if 'Project Description' in dataset.columns:
            project_details['total_projects'] = len(dataset)
            project_details['summary'].append(f"Total projects: {len(dataset)}")
        
        # Analyze project types
        if 'Project type' in dataset.columns:
            project_types = dataset['Project type'].value_counts()
            project_details['project_types'] = project_types.to_dict()
            project_details['summary'].append(f"Project types: {len(project_types)} different types")
        
        # Analyze cost distribution
        if 'Construction Award' in dataset.columns:
            try:
                awards = dataset['Construction Award'].dropna()
                if len(awards) > 0:
                    project_details['cost_distribution'] = {
                        'total': float(awards.sum()),
                        'average': float(awards.mean()),
                        'median': float(awards.median()),
                        'min': float(awards.min()),
                        'max': float(awards.max()),
                        'std': float(awards.std())
                    }
                    project_details['summary'].append(f"Cost analysis: ${awards.sum():,.0f} total value")
            except Exception as e:
                self.logger.warning(f"Cost distribution analysis failed: {e}")
        
        # Analyze locations
        if 'Borough' in dataset.columns:
            boroughs = dataset['Borough'].value_counts()
            project_details['location_analysis'] = boroughs.to_dict()
            project_details['summary'].append(f"Locations: {len(boroughs)} boroughs")
        
        return project_details
    
    def _generate_summary(self, dataset: pd.DataFrame) -> List[str]:
        """Generate construction summary"""
        summary = []
        
        # Basic dataset info
        summary.append(f"Construction dataset: {len(dataset)} records")
        
        # Key columns found
        key_cols = []
        if 'Project Description' in dataset.columns:
            key_cols.append('Project Description')
        if 'Construction Award' in dataset.columns:
            key_cols.append('Construction Award')
        if 'Project type' in dataset.columns:
            key_cols.append('Project type')
        if 'School Name' in dataset.columns:
            key_cols.append('School Name')
        
        if key_cols:
            summary.append(f"Key columns: {', '.join(key_cols)}")
        
        return summary 