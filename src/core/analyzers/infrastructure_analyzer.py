# Author: KleaSCM
# Date: 2024
# Description: Infrastructure data analyzer

import pandas as pd
from typing import Dict, List, Any
from .base_analyzer import BaseAnalyzer

class InfrastructureAnalyzer(BaseAnalyzer):
    """Analyzes infrastructure-related data"""
    
    def analyze(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze infrastructure data"""
        self.logger.info(f"Analyzing infrastructure data with {len(dataset)} records")
        
        return {
            'utilities_infrastructure': self._analyze_utilities_infrastructure(dataset),
            'summary': self._generate_summary(dataset)
        }
    
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
                    self.logger.warning(f"Electrical analysis failed for column {col}: {e}")
        
        # Analyze water/sewer
        for col in water_cols:
            if col in dataset.columns:
                try:
                    water_data = dataset[col].value_counts()
                    utilities['water_sewer'][col] = water_data.to_dict()
                    utilities['summary'].append(f"Water/Sewer: {len(water_data)} different values found")
                except Exception as e:
                    self.logger.warning(f"Water analysis failed for column {col}: {e}")
        
        # Analyze gas
        for col in gas_cols:
            if col in dataset.columns:
                try:
                    gas_data = dataset[col].value_counts()
                    utilities['gas'][col] = gas_data.to_dict()
                    utilities['summary'].append(f"Gas: {len(gas_data)} different values found")
                except Exception as e:
                    self.logger.warning(f"Gas analysis failed for column {col}: {e}")
        
        # Analyze telecom
        for col in telecom_cols:
            if col in dataset.columns:
                try:
                    telecom_data = dataset[col].value_counts()
                    utilities['telecom'][col] = telecom_data.to_dict()
                    utilities['summary'].append(f"Telecom: {len(telecom_data)} different values found")
                except Exception as e:
                    self.logger.warning(f"Telecom analysis failed for column {col}: {e}")
        
        # Analyze roads/bridges
        for col in road_cols:
            if col in dataset.columns:
                try:
                    road_data = dataset[col].value_counts()
                    utilities['roads_bridges'][col] = road_data.to_dict()
                    utilities['summary'].append(f"Roads/Bridges: {len(road_data)} different values found")
                except Exception as e:
                    self.logger.warning(f"Road analysis failed for column {col}: {e}")
        
        return utilities
    
    def _generate_summary(self, dataset: pd.DataFrame) -> List[str]:
        """Generate infrastructure summary"""
        summary = []
        summary.append(f"Infrastructure dataset: {len(dataset)} records")
        
        # Count infrastructure types found
        infra_patterns = []
        for category in self.data_patterns['infrastructure'].values():
            infra_patterns.extend(category)
        
        infra_cols = self._find_columns_by_patterns(dataset, infra_patterns)
        if infra_cols:
            summary.append(f"Infrastructure types: {len(infra_cols)} columns identified")
        
        return summary 