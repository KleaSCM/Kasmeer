# Kasmeer - Civil Engineering Neural Network System

A comprehensive, neural network system for analyzing civil engineering data, providing infrastructure assessment, environmental analysis, risk prediction, and survey planning everything is data-driven through NN learning.

## Modular Architecture

The system follows a clean, modular design with proper separation of concerns:

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
│   │   ├── query_engine.py      # Natural language query processing
│   │   ├── risk_analyzer.py     # Risk assessment and analysis
│   │   ├── survey_analyzer.py   # Survey planning and cost estimation
│   │   └── dataset_config.py    # Flexible dataset configuration
│   ├── data/                    # Data processing components
│   │   ├── __init__.py
│   │   ├── flexible_data_processor.py  # Configurable data loading
│   │   └── data_processor.py    # Legacy data processor
│   ├── ml/                      # Machine learning components
│   │   ├── __init__.py
│   │   ├── neural_network.py    # PyTorch neural network (core ML only)
│   │   ├── incremental_trainer.py  # Smart retraining
│   │   └── model_versioning.py  # Model version management
│   ├── cli/                     # Command-line interface
│   │   ├── __init__.py
│   │   ├── cli_interface.py     # CLI using Click and Rich
│   │   └── dataset_setup.py     # Dataset discovery and configuration
│   └── __init__.py              # Main package exports
├── config.yaml                  # System configuration
├── main.py                      # Entry point
├── setup.py                     # Installation script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Key Features

### **Neural Network Driven**
- **Zero Hardcoded Logic**: All analysis, risk assessment, and recommendations come from neural network learning
- **Data-Driven Decisions**: The system learns patterns from actual datasets to make predictions
- **Adaptive Intelligence**: Continuously improves with new data and training

### **Modular Architecture**
- **Clean Separation**: Each module has a single, well-defined responsibility
- **Risk Analyzer**: Dedicated module for risk assessment and factor identification
- **Survey Analyzer**: Specialized module for survey planning and cost estimation
- **Flexible Data Processing**: Configurable data loading for any company's datasets

### **Advanced Analysis Capabilities**
- **Risk Assessment**: Environmental, infrastructure, and construction risk prediction
- **Infrastructure Health**: Health scoring, maintenance scheduling, and upgrade planning
- **Environmental Analysis**: Climate change impact assessment and biodiversity analysis
- **Construction Planning**: Timeline optimization and resource allocation
- **Survey Planning**: Cost estimation and priority scoring for surveys

### **Professional Features**
- **Confidence Intervals**: Uncertainty quantification for all predictions
- **Data Completeness**: Automatic assessment of data quality and gaps
- **Flexible Configuration**: Works with any company's dataset structure
- **Comprehensive Logging**: Detailed logging with performance tracking
- **Model Versioning**: Track and manage different model versions
- **Incremental Training**: Smart retraining with new data detection

## System Performance & Validation Results

**Production Ready**

The system has been thoroughly validated with real-world datasets:

- **100% Validation Success Rate** - All datasets pass quality checks
- **8,588 Infrastructure Records** - Real civil engineering data
- **34,496 Climate Records** - High-resolution environmental data
- **61,836 Wind Observations** - Comprehensive meteorological data
- **0 Errors, 3 Minor Warnings** - Professional-grade data quality

**[View Detailed Validation Results](Docs/validation_results.md)** - Comprehensive performance metrics, data quality assessment, and system capabilities documentation.

**[Quick Performance Summary](Docs/performance_summary.md)** - At-a-glance system performance and dataset statistics.

## 🖥️ System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows 10+

### Recommended Requirements
- **Python**: 3.9+ for optimal performance
- **RAM**: 16GB for large datasets
- **Storage**: 10GB for extensive data processing
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)

### Dependencies
- **Core ML**: PyTorch, NumPy, Pandas
- **Geospatial**: GeoPandas, Rasterio, Shapely
- **CLI**: Click, Rich, Progress
- **Data Processing**: Scikit-learn, SciPy
- **Visualization**: Matplotlib, Plotly

## 🛠️ Installation & Setup

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

5. **Configure datasets** (optional):
   ```bash
   # Discover and configure your datasets
   python main.py dataset-discover
   python main.py dataset-configure
   ```

## Usage Guide

### 🖥️ Command Line Interface

The system provides a rich CLI with multiple commands. **Make sure to use the virtual environment:**

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
```bash
# Neural Network Operations
python main.py train                    # Train the neural network
python main.py retrain                  # Smart retraining with new data
python main.py model-info               # Get model information
```
```bash
# 🔍 Query Operations
python main.py query                    # Interactive query mode
python main.py ask "What is the infrastructure at -37.8136, 144.9631?"
```
```bash
# 📊 Data Operations
python main.py data-info                # Get data information
python main.py validate                 # Validate datasets (check data quality)
```
```bash
# 🗂️ Dataset Management
python main.py dataset-discover         # Discover available datasets
python main.py dataset-configure        # Configure dataset mappings
python main.py dataset-validate         # Validate dataset configuration
```
```bash
# 📈 Model Versioning
python main.py versions                 # Show model versions
python main.py use-version <version_id> # Switch to specific model version
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

### 💬 Example Queries

The system understands natural language queries about civil engineering:

**Infrastructure Analysis:**
- "What pipes are at coordinates -37.8136, 144.9631?"
- "Show me infrastructure health at Melbourne"
- "What's the maintenance schedule for Sydney infrastructure?"

**Risk Assessment:**
- "What are the construction risks at Brisbane?"
- "Assess environmental risks for Perth"
- "What's the flood risk at Adelaide?"

**Environmental Analysis:**
- "Show me climate data for Melbourne"
- "What's the biodiversity at Sydney?"
- "Assess climate change impact at Perth"

**Survey Planning:**
- "What surveys are needed at Brisbane?"
- "Estimate survey costs for Melbourne"
- "Prioritize surveys for Sydney"

### 🔧 Advanced Features

#### 🗂️ Flexible Dataset Configuration
Configure the system to work with any company's dataset structure:
```bash
# Discover available datasets
python main.py dataset-discover
```
```bash
# Configure dataset mappings
python main.py dataset-configure
```
```bash
# Validate configuration
python main.py dataset-validate
```
#### 📊 Data Validation
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

#### 📈 Model Versioning
Track and manage different model versions:
```bash
# List all versions
python main.py versions
```
```bash
# Switch to a specific version
python main.py use-version 20241201_143022_abc123
```
```bash
# Compare versions
python main.py compare-versions v1 v2
```
Each version includes metadata about training parameters and performance.

#### Confidence & Uncertainty
All predictions include confidence intervals and uncertainty quantification:
- **Confidence Scores**: How reliable is the prediction
- **Uncertainty Quantification**: Range of possible values
- **Data Completeness**: Assessment of available data quality

## 🔌 API Documentation

### Core Classes

#### QueryEngine
The main interface for processing natural language queries:
```python
from src.core.query_engine import QueryEngine

# Initialize with data processor and neural network
query_engine = QueryEngine(data_processor, neural_network)

# Process a query
result = query_engine.process_query("What are the risks at Melbourne?")
```

#### RiskAnalyzer
Specialized risk assessment module:
```python
from src.core.risk_analyzer import RiskAnalyzer

analyzer = RiskAnalyzer(neural_network)
risk_assessment = analyzer.assess_risks(location_data)
```

#### SurveyAnalyzer
Survey planning and cost estimation:
```python
from src.core.survey_analyzer import SurveyAnalyzer

survey_analyzer = SurveyAnalyzer(neural_network)
survey_plan = survey_analyzer.plan_surveys(location_data)
```

### Data Processing

#### FlexibleDataProcessor
Configurable data loading for any dataset structure:
```python
from src.data.flexible_data_processor import FlexibleDataProcessor

processor = FlexibleDataProcessor("config.yaml")
data = processor.discover_and_load_all_data()
```

### Neural Network

#### CivilEngineeringSystem
Core ML model with prediction capabilities:
```python
from src.ml.neural_network import CivilEngineeringSystem

model = CivilEngineeringSystem("models/")
predictions = model.predict_risk(features)
```

## 📊 Performance Benchmarks

### Training Performance
- **Initial Training**: ~5-10 minutes (depending on dataset size)
- **Incremental Training**: ~1-3 minutes (new data only)
- **Memory Usage**: 2-4GB during training
- **GPU Acceleration**: 3-5x faster with CUDA

### Query Performance
- **Simple Queries**: <100ms response time
- **Complex Analysis**: 1-5 seconds
- **Batch Processing**: 1000+ queries/minute
- **Memory Footprint**: <500MB during operation

### Data Processing
- **Dataset Discovery**: <30 seconds
- **Data Validation**: <2 minutes per dataset
- **Feature Extraction**: <1 second per location
- **Spatial Indexing**: <5 minutes for large datasets

## 🗂️ Data Requirements

### Supported Formats
- **Tabular Data**: CSV, Excel (.xlsx, .xls)
- **Geospatial Data**: Shapefile (.shp), GeoJSON (.geojson), KML (.kml)
- **Raster Data**: GeoTIFF (.tif), NetCDF (.nc)
- **Configuration**: YAML (.yaml), JSON (.json)

### Required Data Structure
Place your datasets in the `DataSets/` directory:

```
DataSets/
├── infrastructure/           # Infrastructure data
│   ├── pipes.csv
│   ├── drainage.shp
│   └── roads.geojson
├── environmental/           # Environmental data
│   ├── climate.tif
│   ├── vegetation.shp
│   └── wind.csv
├── surveys/                 # Survey data
│   ├── completed_surveys.csv
│   └── survey_requirements.xlsx
└── config.yaml             # Dataset configuration
```

### Data Quality Requirements
- **Coordinates**: Valid latitude/longitude pairs
- **Missing Data**: <20% missing values per column
- **Data Types**: Proper data type assignments
- **Coordinate System**: Consistent CRS across datasets

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest Tests/
```
```bash
# Run specific test file
python Tests/test_system.py
```
```bash
# Run with coverage
python -m pytest Tests/ --cov=src --cov-report=html
```
```bash
# Run example usage
python utils/example_usage.py
```

### Test Coverage
- **Unit Tests**: Core functionality and modules
- **Integration Tests**: End-to-end workflows
- **Data Tests**: Dataset validation and processing
- **Performance Tests**: Speed and memory benchmarks

## Visualization

### Built-in Visualization
```python
from visuals.plotting import DataVisualizer

visualizer = DataVisualizer()

# Plot data summary
visualizer.plot_data_summary(data_processor)

# Plot risk assessment
visualizer.plot_risk_assessment(risk_data, location)

# Plot survey planning
visualizer.plot_survey_planning(survey_data)
```

### Export Options
- **Static Plots**: PNG, PDF, SVG formats
- **Interactive Plots**: HTML with Plotly
- **Reports**: PDF reports with embedded visualizations
- **Dashboards**: Web-based dashboards

## ⚙️ Configuration

### System Configuration
The system uses `config.yaml` for all settings:

```yaml
# Neural Network Configuration
neural_network:
  input_size: 15
  hidden_layers: [64, 32, 16]
  output_size: 3
  learning_rate: 0.001
  epochs: 100

# Data Processing Configuration
data_processing:
  coordinate_system: "EPSG:4326"
  data_cleaning:
    remove_duplicates: true
    handle_missing: "interpolate"
  
# Logging Configuration
logging:
  level: "INFO"
  file: "Logs/kasmeer.log"
  max_size: "10MB"
```

### Company-Specific Configuration
Configure dataset mappings for your company's data structure:

```yaml
company_config:
  data_mappings:
    column_mappings:
      "Pipe_Type": "pipe_type"
      "Install_Date": "installation_date"
      "Material": "material_type"
    
    coordinate_columns:
      latitude: ["lat", "latitude", "y"]
      longitude: ["lon", "longitude", "x"]
```

## Troubleshooting

### Common Issues

#### Installation Problems
```bash
# If pip fails to install dependencies
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```
```bash
# If virtual environment issues
python -m venv venv --clear
source venv/bin/activate
pip install -r requirements.txt
```

#### Data Loading Issues
```bash
# Check data format
python main.py validate
```
```bash
# Check dataset configuration
python main.py dataset-validate
```
```bash
# View detailed logs
tail -f Logs/kasmeer.log
```

#### Training Issues
```bash
# Check available memory
free -h

# Reduce batch size in config.yaml
neural_network:
  batch_size: 32  # Reduce from 64

# Use CPU-only training
export CUDA_VISIBLE_DEVICES=""
python main.py train
```

#### Query Processing Issues
```bash
# Check model status
python main.py model-info
```
```bash
# Rebuild model if corrupted
python main.py train --force
```
```bash
# Check query syntax
python main.py ask "help"
```

### Performance Optimization

#### Memory Optimization
```yaml
# In config.yaml
data_processing:
  chunk_size: 1000  # Process data in chunks
  use_dask: true    # Use Dask for large datasets
```

#### Speed Optimization
```yaml
# Enable GPU acceleration
neural_network:
  use_gpu: true
  num_workers: 4

# Enable parallel processing
data_processing:
  parallel_processing: true
  num_workers: 4
```

## 📝 Logging

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about system operation
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failed operations
- **CRITICAL**: Critical errors that may cause system failure

### Log Files
- **Main Log**: `Logs/kasmeer.log` - All system operations
- **Performance Log**: `Logs/performance.log` - Performance metrics
- **Error Log**: `Logs/errors.log` - Error tracking
- **Training Log**: `Logs/training.log` - Training progress

### Log Analysis
```bash
# View recent errors
grep "ERROR" Logs/kasmeer.log | tail -20
```
```bash
# Check performance
grep "performance" Logs/performance.log
```
```bash
# Monitor system health
tail -f Logs/kasmeer.log | grep -E "(ERROR|WARNING|CRITICAL)"
```

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/Kasmeer.git
cd Kasmeer

# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests before making changes
python -m pytest Tests/
```

**Note**: This system is designed for civil engineering professionals and requires appropriate datasets for optimal performance. Always validate results with domain experts before making critical decisions.

