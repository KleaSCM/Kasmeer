# Author: KleaSCM
# Date: 2024
# Description: Financial data analyzer

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_analyzer import BaseAnalyzer

class FinancialAnalyzer(BaseAnalyzer):
    """Analyzes financial data including costs, funding, and budget information"""
    
    def analyze(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze financial data"""
        self.logger.info(f"Analyzing financial data with {len(dataset)} records")
        
        return {
            'costs_funding': self._analyze_costs_funding(dataset),
            'budget_analysis': self._analyze_budget_data(dataset),
            'funding_sources': self._analyze_funding_sources(dataset),
            'summary': self._generate_summary(dataset)
        }
    
    def _analyze_costs_funding(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze costs, funding, and financial aspects"""
        costs = {
            'project_costs': {},
            'maintenance_costs': {},
            'funding_sources': {},
            'budget_status': {},
            'summary': [],
            'cost_details': []
        }
        
        # Extract actual cost data
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
                        award_val = row.get('Construction Award')
                        if isinstance(award_val, (int, float)) and not pd.isna(award_val):
                            costs['cost_details'].append({
                                'project': row.get('Project Description', 'Unknown'),
                                'estimated_value': f"${row.get('Construction Award', 0):,.0f}",
                                'project_id': row.get('Building ID', 'Unknown'),
                                'campus': row.get('School Name', 'Unknown'),
                                'project_type': row.get('Project type', 'Unknown')
                            })
            except Exception as e:
                self.logger.warning(f"Construction award analysis failed: {e}")
        
        # Dynamic cost discovery
        cost_patterns = ['cost', 'budget', 'estimate', 'actual', 'variance', 'expense', 'amount', 'dollar']
        cost_cols = self._find_columns_by_patterns(dataset, cost_patterns)
        
        funding_patterns = ['funding', 'grant', 'loan', 'bond', 'revenue', 'income']
        funding_cols = self._find_columns_by_patterns(dataset, funding_patterns)
        
        # Analyze costs
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
                    self.logger.warning(f"Cost analysis failed for column {col}: {e}")
        
        # Analyze funding
        for col in funding_cols:
            if col in dataset.columns:
                try:
                    funding_data = dataset[col].value_counts()
                    costs['funding_sources'][col] = funding_data.to_dict()
                    costs['summary'].append(f"Funding: {len(funding_data)} different sources found")
                except Exception as e:
                    self.logger.warning(f"Funding analysis failed for column {col}: {e}")
        
        return costs
    
    def _analyze_budget_data(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze budget and financial planning data"""
        budget = {
            'budget_allocations': {},
            'cost_overruns': {},
            'variance_analysis': {},
            'summary': []
        }
        
        # Look for budget-related columns
        budget_patterns = ['budget', 'allocated', 'planned', 'actual', 'variance']
        budget_cols = self._find_columns_by_patterns(dataset, budget_patterns)
        
        for col in budget_cols:
            if col in dataset.columns:
                try:
                    if dataset[col].dtype in ['int64', 'float64']:
                        budget['budget_allocations'][col] = {
                            'total': float(dataset[col].sum()),
                            'mean': float(dataset[col].mean()),
                            'min': float(dataset[col].min()),
                            'max': float(dataset[col].max())
                        }
                    else:
                        budget_data = dataset[col].value_counts()
                        budget['budget_allocations'][col] = budget_data.to_dict()
                    
                    budget['summary'].append(f"Budget data: {col} analyzed")
                except Exception as e:
                    self.logger.warning(f"Budget analysis failed for column {col}: {e}")
        
        return budget
    
    def _analyze_funding_sources(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze funding sources and revenue streams"""
        funding = {
            'funding_types': {},
            'revenue_streams': {},
            'grant_information': {},
            'summary': []
        }
        
        # Look for funding-related columns
        funding_patterns = ['funding', 'grant', 'loan', 'bond', 'revenue', 'income', 'source']
        funding_cols = self._find_columns_by_patterns(dataset, funding_patterns)
        
        for col in funding_cols:
            if col in dataset.columns:
                try:
                    funding_data = dataset[col].value_counts()
                    funding['funding_types'][col] = funding_data.to_dict()
                    funding['summary'].append(f"Funding source: {col} - {len(funding_data)} types")
                except Exception as e:
                    self.logger.warning(f"Funding source analysis failed for column {col}: {e}")
        
        return funding
    
    def _generate_summary(self, dataset: pd.DataFrame) -> List[str]:
        """Generate financial summary"""
        summary = []
        
        # Basic dataset info
        summary.append(f"Financial dataset: {len(dataset)} records")
        
        # Key financial columns found
        key_cols = []
        if 'Construction Award' in dataset.columns:
            key_cols.append('Construction Award')
        if 'Estimated Contract Value' in dataset.columns:
            key_cols.append('Estimated Contract Value')
        
        if key_cols:
            summary.append(f"Key financial columns: {', '.join(key_cols)}")
        
        # Total value if available
        if 'Construction Award' in dataset.columns:
            try:
                awards = dataset['Construction Award'].dropna()
                if len(awards) > 0:
                    total_value = awards.sum()
                    summary.append(f"Total project value: ${total_value:,.0f}")
            except:
                pass
        
        return summary 