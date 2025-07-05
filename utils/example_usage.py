#!/usr/bin/env python3

# Author: KleaSCM
# Date: 2024
# Description: Example usage of the Civil Engineering Neural Network System

import sys
from pathlib import Path
from utils.logging_utils import setup_logging, log_performance

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = setup_logging(__name__)

@log_performance(logger)
def example_basic_usage():
    # Example of basic system usage
    logger.info("🔧 Example: Basic System Usage")
    print("🔧 Example: Basic System Usage")
    print("=" * 40)
    
    try:
        from src.data.data_processor import DataProcessor
        from src.ml.neural_network import CivilEngineeringSystem
        from src.core.query_engine import QueryEngine
        
        # Initialize components
        logger.info("1. Initializing components...")
        print("1. Initializing components...")
        data_processor = DataProcessor("DataSets")
        neural_network = CivilEngineeringSystem("models")
        logger.debug("Components initialized successfully")
        
        # Load data
        logger.info("2. Loading data...")
        print("2. Loading data...")
        data_processor.load_infrastructure_data()
        data_processor.load_vegetation_data()
        data_processor.load_climate_data()
        data_processor.create_spatial_index()
        logger.debug("All data loaded and spatial indexes created")
        
        # Initialize query engine
        logger.info("3. Setting up query engine...")
        print("3. Setting up query engine...")
        query_engine = QueryEngine(data_processor, neural_network)
        logger.debug("Query engine initialized")
        
        # Example queries
        logger.info("4. Processing example queries...")
        print("4. Processing example queries...")
        queries = [
            "What is the infrastructure at -37.8136, 144.9631?",
            "Show me environmental data for Melbourne",
            "What are the construction risks at Sydney?"
        ]
        
        for i, query in enumerate(queries, 1):
            logger.debug(f"Processing example query {i}: {query[:50]}...")
            print(f"\nQuery {i}: {query}")
            result = query_engine.process_query(query)
            response = query_engine.format_response(result)
            logger.info(f"Query {i} processed: confidence={result.confidence:.1%}")
            print(f"Response: {response[:100]}...")
        
        logger.info("Basic usage example completed successfully")
        print("\n✅ Basic usage example completed!")
        
    except Exception as e:
        logger.error(f"Error in basic usage: {e}")
        print(f"❌ Error in basic usage: {e}")

@log_performance(logger)
def example_training():
    # Example of model training
    # TODO: Implement training example
    # TODO: Add hyperparameter tuning example
    # TODO: Include model evaluation
    
    logger.info("🔧 Example: Model Training")
    print("🔧 Example: Model Training")
    print("=" * 40)
    logger.debug("Training example not yet implemented")
    print("⏳ Training example not yet implemented")
    print("Use: python main.py train")

@log_performance(logger)
def example_visualization():
    # Example of data visualization
    # TODO: Implement visualization examples
    # TODO: Add dashboard creation
    # TODO: Include custom plot generation
    
    logger.info("🔧 Example: Data Visualization")
    print("🔧 Example: Data Visualization")
    print("=" * 40)
    logger.debug("Visualization example not yet implemented")
    print("⏳ Visualization example not yet implemented")
    print("Use: python main.py data-info")

@log_performance(logger)
def main():
    # Run examples
    logger.info("🚀 Civil Engineering Neural Network System - Examples")
    print("🚀 Civil Engineering Neural Network System - Examples")
    print("=" * 60)
    
    logger.info("Starting example execution")
    example_basic_usage()
    example_training()
    example_visualization()
    
    logger.info("All examples completed successfully")
    print("\n🎉 Examples completed!")
    print("\nTo use the system interactively:")
    print("  python main.py query")
    print("\nTo train the model:")
    print("  python main.py train")
    print("\nTo view data information:")
    print("  python main.py data-info")

if __name__ == '__main__':
    main() 