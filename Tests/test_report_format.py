#!/usr/bin/env python3
"""
Test script for the new professional report formatter
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_processor import DataProcessor
from src.ml.neural_network import CivilEngineeringSystem
from src.core.query_engine import QueryEngine

def test_infrastructure_report():
    """Test the infrastructure report formatting"""
    print("Testing Infrastructure Report Formatting...")
    print("=" * 50)
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        neural_network = CivilEngineeringSystem()
        query_engine = QueryEngine(data_processor, neural_network)
        
        # Test query for infrastructure at Melbourne coordinates
        query = "What is the infrastructure at -37.8136, 144.9631?"
        
        print(f"Query: {query}")
        print()
        
        # Process the query
        result = query_engine.process_query(query)
        
        # Format the response
        formatted_response = query_engine.format_response(result)
        
        print("Generated Report:")
        print("-" * 30)
        print(formatted_response)
        print("-" * 30)
        
        # Check if the format matches the expected structure
        if "🏗️ Infrastructure Report:" in formatted_response:
            print("✅ Report format looks good!")
        else:
            print("❌ Report format doesn't match expected structure")
        
        if "Water Pipes:" in formatted_response:
            print("✅ Water pipes section found!")
        else:
            print("❌ Water pipes section missing")
        
        if "⚠️ Risk:" in formatted_response:
            print("✅ Risk assessment section found!")
        else:
            print("❌ Risk assessment section missing")
        
        if "💰 Cost Projection:" in formatted_response:
            print("✅ Cost projection section found!")
        else:
            print("❌ Cost projection section missing")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_infrastructure_report() 