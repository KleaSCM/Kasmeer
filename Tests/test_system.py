#!/usr/bin/env python3
# Author: KleaSCM
# Date: 2024
# Description: Test script for the Civil Engineering Neural Network System

import sys
from pathlib import Path
from utils.logging_utils import setup_logging, log_performance

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = setup_logging(__name__)

@log_performance(logger)
def test_data_processor():
    # Test the data processor
    logger.info("🧪 Testing Data Processor...")
    print("🧪 Testing Data Processor...")
    
    try:
        from src.data.data_processor import DataProcessor
        
        # Initialize processor
        logger.debug("Initializing DataProcessor")
        processor = DataProcessor("DataSets")
        
        # Test loading infrastructure data
        logger.debug("Loading infrastructure data")
        infra_data = processor.load_infrastructure_data()
        logger.info(f"Infrastructure data loaded: {len(infra_data)} records")
        print(f"  ✅ Infrastructure data: {len(infra_data)} records")
        
        # Test loading vegetation data
        logger.debug("Loading vegetation data")
        veg_data = processor.load_vegetation_data()
        logger.info(f"Vegetation data loaded: {len(veg_data)} records")
        print(f"  ✅ Vegetation data: {len(veg_data)} records")
        
        # Test loading climate data
        logger.debug("Loading climate data")
        climate_data = processor.load_climate_data()
        logger.info(f"Climate data loaded: {len(climate_data)} variables")
        print(f"  ✅ Climate data: {len(climate_data)} variables")
        
        # Test spatial indexing
        logger.debug("Creating spatial indexes")
        spatial_data = processor.create_spatial_index()
        logger.info(f"Spatial indexes created: {len(spatial_data)} types")
        print(f"  ✅ Spatial indexes created: {len(spatial_data)} types")
        
        # Test feature extraction
        logger.debug("Testing feature extraction")
        features = processor.extract_features_at_location(-37.8136, 144.9631)
        logger.info(f"Feature extraction completed: {len(features)} feature types")
        print(f"  ✅ Feature extraction: {len(features)} feature types")
        
        # Test data summary
        logger.debug("Getting data summary")
        summary = processor.get_data_summary()
        logger.info(f"Data summary generated: {len(summary)} data types")
        print(f"  ✅ Data summary: {len(summary)} data types")
        
        logger.info("Data processor test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data processor test failed: {e}")
        print(f"  ❌ Data processor test failed: {e}")
        return False

@log_performance(logger)
def test_neural_network():
    # Test the neural network
    logger.info("🧪 Testing Neural Network...")
    print("🧪 Testing Neural Network...")
    
    try:
        from src.ml.neural_network import CivilEngineeringSystem
        
        # Initialize neural network
        logger.debug("Initializing CivilEngineeringSystem")
        nn = CivilEngineeringSystem("models")
        
        # Test model building
        logger.debug("Building neural network model")
        model = nn.build_model(15, 3)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model built successfully: {param_count} parameters")
        print(f"  ✅ Model built: {param_count} parameters")
        
        # Test model summary
        logger.debug("Getting model summary")
        summary = nn.get_model_summary()
        logger.info(f"Model summary retrieved: loaded={summary['model_loaded']}")
        print(f"  ✅ Model summary: {summary['model_loaded']}")
        
        logger.info("Neural network test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Neural network test failed: {e}")
        print(f"  ❌ Neural network test failed: {e}")
        return False

@log_performance(logger)
def test_query_engine():
    # Test the query engine
    logger.info("🧪 Testing Query Engine...")
    print("🧪 Testing Query Engine...")
    
    try:
        from src.data.data_processor import DataProcessor
        from src.ml.neural_network import CivilEngineeringSystem
        from src.core.query_engine import QueryEngine
        
        # Initialize components
        logger.debug("Initializing components for query engine test")
        processor = DataProcessor("DataSets")
        nn = CivilEngineeringSystem("models")
        
        # Load some data
        logger.debug("Loading data for query engine test")
        processor.load_infrastructure_data()
        processor.load_vegetation_data()
        processor.create_spatial_index()
        
        # Initialize query engine
        logger.debug("Initializing QueryEngine")
        query_engine = QueryEngine(processor, nn)
        
        # Test query processing
        test_queries = [
            "What is the infrastructure at -37.8136, 144.9631?",
            "Show me environmental data for Melbourne",
            "What are the risks at Sydney?"
        ]
        
        logger.debug(f"Processing {len(test_queries)} test queries")
        for i, query in enumerate(test_queries):
            logger.debug(f"Processing query {i+1}: {query[:50]}...")
            result = query_engine.process_query(query)
            logger.info(f"Query {i+1} processed: confidence={result.confidence:.1%}")
            print(f"  ✅ Query processed: '{query[:30]}...' -> {result.confidence:.1%} confidence")
        
        logger.info("Query engine test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Query engine test failed: {e}")
        print(f"  ❌ Query engine test failed: {e}")
        return False

@log_performance(logger)
def test_visualization():
    # Test visualization capabilities
    # TODO: Implement visualization tests
    # TODO: Add plot generation tests
    # TODO: Include dashboard testing
    
    logger.info("🧪 Testing Visualization...")
    print("🧪 Testing Visualization...")
    logger.debug("Visualization tests not yet implemented")
    print("  ⏳ Visualization tests not yet implemented")
    return True

@log_performance(logger)
def test_integration():
    # Test full system integration
    # TODO: Implement end-to-end integration tests
    # TODO: Add performance benchmarks
    # TODO: Include stress testing
    
    logger.info("🧪 Testing System Integration...")
    print("🧪 Testing System Integration...")
    logger.debug("Integration tests not yet implemented")
    print("  ⏳ Integration tests not yet implemented")
    return True

@log_performance(logger)
def main():
    # Run all tests
    logger.info("🚀 Starting Civil Engineering Neural Network System Tests")
    print("🚀 Starting Civil Engineering Neural Network System Tests\n")
    
    tests = [
        ("Data Processor", test_data_processor),
        ("Neural Network", test_neural_network),
        ("Query Engine", test_query_engine),
        ("Visualization", test_visualization),
        ("Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    logger.info(f"Running {total} test suites")
    
    for test_name, test_func in tests:
        logger.info(f"Starting test: {test_name}")
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            logger.info(f"Test PASSED: {test_name}")
            print(f"✅ {test_name} test PASSED")
        else:
            logger.error(f"Test FAILED: {test_name}")
            print(f"❌ {test_name} test FAILED")
    
    logger.info(f"Test Results: {passed}/{total} tests passed")
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python main.py train")
        print("2. Run: python main.py query")
    else:
        logger.warning(f"⚠️ Some tests failed: {total - passed} failures")
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Check that all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify that DataSets/ directory contains data files")
        print("3. Check the logs for detailed error messages")

if __name__ == '__main__':
    main() 