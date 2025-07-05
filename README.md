# Kasmeer - Civil Engineering Neural Network System

A comprehensive neural network system for analyzing civil engineering data, providing infrastructure assessment, environmental analysis, and risk prediction capabilities.


## Project Structure

```
Kasmeer/
├── DataSets/                    # All data files (infrastructure, climate, vegetation)
├── Logs/                        # Log files and logging utilities
├── models/                      # Trained model storage
├── Tests/                       # Test modules and utilities
├── utils/                       # Utility functions and scripts
├── visuals/                     # Visualization modules
├── src/                         # Main source code (modular structure)
│   ├── core/                    # Core functionality
│   │   ├── __init__.py
│   │   └── query_engine.py      # Natural language query processing
│   ├── data/                    # Data processing components
│   │   ├── __init__.py
│   │   └── data_processor.py    # Data loading and preprocessing
│   ├── ml/                      # Machine learning components
│   │   ├── __init__.py
│   │   └── neural_network.py    # PyTorch neural network implementation
│   ├── cli/                     # Command-line interface
│   │   ├── __init__.py
│   │   └── cli_interface.py     # CLI using Click and Rich
│   └── __init__.py              # Main package exports
├── main.py                      # Entry point
├── setup.py                     # Installation script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Data Processing**: Handles infrastructure, climate, and vegetation datasets
- **Neural Network**: PyTorch-based ML model for risk prediction
- **Query Engine**: Natural language processing for civil engineering queries
- **CLI Interface**: Rich command-line interface with progress tracking
- **Data Validation**: Comprehensive dataset validation and quality checks
- **Incremental Training**: Smart retraining with new data detection
- **Model Versioning**: Version control for trained models with metadata
- **Visualization**: Data plotting and analysis tools
- **Testing**: Comprehensive test suite
- **Utilities**: Helper functions and configuration management

## 📊 System Performance & Validation Results

**✅ EXCELLENT - Production Ready**

Our system has been thoroughly validated with real-world datasets:

- **100% Validation Success Rate** - All datasets pass quality checks
- **8,588 Infrastructure Records** - Real civil engineering data
- **34,496 Climate Records** - High-resolution environmental data
- **61,836 Wind Observations** - Comprehensive meteorological data
- **0 Errors, 3 Minor Warnings** - Professional-grade data quality

📖 **[View Detailed Validation Results](Docs/validation_results.md)** - Comprehensive performance metrics, data quality assessment, and system capabilities documentation.

📈 **[Quick Performance Summary](Docs/performance_summary.md)** - At-a-glance system performance and dataset statistics.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Kasmeer
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup the system**:
   ```bash
   python setup.py
   ```

## Usage

### Command Line Interface

The system provides a rich CLI with multiple commands. **Make sure to use the virtual environment:**

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Train the neural network
python main.py train

# Interactive query mode
python main.py query

# Single query
python main.py ask "What is the infrastructure at -37.8136, 144.9631?"

# Get data information
python main.py data-info

# Get model information
python main.py model-info

# Retrain with updated data
python main.py retrain

# Validate datasets (check data quality)
python main.py validate

# Show model versions
python main.py versions

# Switch to specific model version
python main.py use-version <version_id>
```

**Alternative: Use virtual environment Python directly:**
```bash
# Use venv Python directly (no activation needed)
venv/bin/python main.py train
venv/bin/python main.py query
venv/bin/python Tests/test_system.py
venv/bin/python utils/example_usage.py
```

**Easy Runner Script:**
```bash
# Use the runner script (automatically uses virtual environment)
python run.py train
python run.py query
python run.py data-info
python run.py validate
python run.py versions
python run.py --help
```

### Example Queries

- Infrastructure: "What pipes are at coordinates -37.8136, 144.9631?"
- Environmental: "Show me climate data for Melbourne"
- Risk Assessment: "What are the construction risks at Sydney?"
- Survey: "Has an environmental survey been completed for Brisbane?"

### Advanced Features

#### Data Validation
The system automatically validates datasets before training:
```bash
python main.py validate
```
This checks data integrity, required columns, coordinate ranges, and data quality.

#### Incremental Training
Smart retraining that only updates the model with new data:
```bash
python main.py retrain
```
The system detects new or modified data files and performs incremental training when possible.

#### Model Versioning
Track and manage different model versions:
```bash
# List all versions
python main.py versions

# Switch to a specific version
python main.py use-version 20241201_143022_abc123

# Compare versions
python main.py compare-versions v1 v2
```
Each version includes metadata about training parameters and performance.

### Programmatic Usage

```python
from src.data import DataProcessor
from src.ml import CivilEngineeringSystem
from src.core import QueryEngine

# Initialize components
data_processor = DataProcessor("DataSets")
neural_network = CivilEngineeringSystem("models")
query_engine = QueryEngine(data_processor, neural_network)

# Process a query
result = query_engine.process_query("What is the infrastructure at -37.8136, 144.9631?")
print(query_engine.format_response(result))
```

## Data Requirements

Place your datasets in the `DataSets/` directory:

- **Infrastructure Data**: CSV files with pipe/drainage information
- **Climate Data**: WorldClim format (.tif files) or CSV
- **Vegetation Data**: Shapefiles or CSV with vegetation zones
- **Wind Data**: CSV files with wind observations

## Testing

Run the test suite:

```bash
# Using virtual environment
venv/bin/python Tests/test_system.py

# Or activate venv first
source venv/bin/activate
python Tests/test_system.py
```

Or use the example usage script:

```bash
# Using virtual environment
venv/bin/python utils/example_usage.py

# Or activate venv first
source venv/bin/activate
python utils/example_usage.py
```

## Visualization

The system includes visualization capabilities:

```python
from visuals import DataVisualizer

visualizer = DataVisualizer()
visualizer.plot_data_summary(data_processor)
visualizer.plot_risk_assessment(risk_data, location)
```

## Configuration

The system uses a configuration system for customization:

```python
from utils import load_config, save_config

config = load_config()
config['training']['epochs'] = 200
save_config(config)
```

## Logging

Logs are automatically saved to `Logs/` directory with timestamps and different log levels.

## 🤝 Contributing

1. Follow the modular structure
2. Add tests for new features
3. Update documentation
4. Use the established coding patterns

## Support

For issues and questions:
1. Check the logs in `Logs/` directory
2. Run the test suite
3. Review the example usage
4. Check the configuration settings

---

**Note**: This system is designed for civil engineering professionals and requires appropriate datasets for optimal performance.

