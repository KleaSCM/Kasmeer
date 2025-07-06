# Author: KleaSCM
# Date: 2024
# CLI Interface Module
# Description: - Provides command-line interface for civil engineering neural network system

import click
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
import json
from datetime import datetime
import pandas as pd
import numpy as np

from ..data.data_processor import DataProcessor
from ..ml.neural_network import CivilEngineeringSystem
from ..core.query_engine import QueryEngine
from ..core.universal_reporter import UniversalReporter
from ..utils.logging_utils import setup_logging, log_performance
from ..utils.helpers import filter_by_location, filter_by_exact_location, find_coordinate_columns

console = Console()

logger = setup_logging(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@log_performance(logger)
def cli(verbose):
    # Civil Engineering Neural Network System
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.info(f"CLI started with verbose={verbose}")

@cli.command()
@click.option('--data-dir', default='DataSets', help='Directory containing datasets')
@click.option('--model-dir', default='models', help='Directory to save models')
@log_performance(logger)
def train(data_dir, model_dir):
    # Train the neural network on available data
    logger.info(f"train command: data_dir={data_dir}, model_dir={model_dir}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Initializing system...", total=None)
        
        try:
            # Initialize components
            data_processor = DataProcessor(data_dir)
            neural_network = CivilEngineeringSystem(model_dir)
            
            progress.update(task, description="Discovering and loading datasets...")
            loaded_data = data_processor.discover_and_load_all_data()
            
            # DEBUG: Show what datasets were loaded
            console.print("\n📊 DATASETS LOADED:")
            console.print("=" * 50)
            for name, data in loaded_data.items():
                if hasattr(data, '__len__'):
                    console.print(f"  ✅ {name}: {len(data):,} records, {len(data.columns)} columns")
                else:
                    console.print(f"  ✅ {name}: raster/geospatial data")
            console.print("=" * 50)
            
            if not loaded_data:
                console.print("[red]❌ No datasets found to analyze[/red]")
                return
            
            progress.update(task, description="Creating spatial indexes...")
            spatial_data = data_processor.create_spatial_index()
            
            progress.update(task, description="Preparing training data...")
            X, y = neural_network.prepare_features(data_processor)
            
            if X.size == 0 or y.size == 0:
                console.print("[red]No training data available. Please check your datasets.[/red]")
                return
            
            progress.update(task, description="Training neural network...")
            metrics = neural_network.train(X, y)
            
            if metrics:
                progress.update(task, description="Saving model...")
                neural_network.save_model()
                
                # Display results
                console.print("\n[green]✅ Training completed successfully![/green]")
                
                table = Table(title="Training Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        table.add_row(metric.replace('_', ' ').title(), f"{value:.4f}")
                    else:
                        table.add_row(metric.replace('_', ' ').title(), str(value))
                
                console.print(table)
                
                # Show data summary
                data_summary = data_processor.get_data_summary()
                console.print("\n[bold]Data Summary:[/bold]")
                for data_type, info in data_summary.items():
                    if isinstance(info, dict) and 'rows' in info:
                        console.print(f"  • {data_type}: {info['rows']} records")
                
            else:
                console.print("[red]❌ Training failed. Check the logs for details.[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ Error during training: {e}[/red]")
            logging.error(f"Training error: {e}")

@cli.command()
@click.option('--model-dir', default='models', help='Directory containing models')
@click.option('--data-dir', default='DataSets', help='Directory containing datasets')
@log_performance(logger)
def query(model_dir, data_dir):
    # Interactive query mode
    logger.info(f"query command: model_dir={model_dir}, data_dir={data_dir}")
    
    try:
        # Load components
        data_processor = DataProcessor(data_dir)
        neural_network = CivilEngineeringSystem(model_dir)
        
        # Load model if available
        if not neural_network.load_model():
            console.print("[yellow]⚠️ No trained model found. Starting with data-only mode.[/yellow]")
        
        # Load data
        data_processor.discover_and_load_all_data()
        data_processor.create_spatial_index()
        
        # Initialize query engine
        query_engine = QueryEngine(data_processor, neural_network)
        
        console.print(Panel.fit(
            "[bold blue]Civil Engineering Neural Network System[/bold blue]\n"
            "Ask questions about infrastructure, environmental data, and risk assessment.\n"
            "Examples:\n"
            "• What is the infrastructure at -37.8136, 144.9631?\n"
            "• Show me environmental data for Melbourne\n"
            "• What are the construction risks at -37.8136, 144.9631?\n"
            "• Has an environmental survey been completed for Sydney?\n"
            "Type 'quit' to exit.",
            title="Welcome"
        ))
        
        while True:
            try:
                query_text = Prompt.ask("\n[bold cyan]Enter your query[/bold cyan]")
                
                if query_text.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if not query_text.strip():
                    continue
                
                # Process query
                result = query_engine.process_query(query_text)
                
                # Display result
                response = query_engine.format_response(result)
                console.print(Panel(response, title="Query Result"))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error processing query: {e}[/red]")
                
    except Exception as e:
        console.print(f"[red]❌ Error initializing query system: {e}[/red]")

@cli.command()
@click.argument('query_text')
@click.option('--model-dir', default='models', help='Directory containing models')
@click.option('--data-dir', default='DataSets', help='Directory containing datasets')
@click.option('--config-path', default='config.yaml', help='Path to configuration file')
@click.option('--output', '-o', help='Output file for results (JSON format)')
@log_performance(logger)
def ask(query_text, model_dir, data_dir, config_path, output):
    # Ask a single query and get results
    logger.info(f"ask command: query_text={query_text}, model_dir={model_dir}, data_dir={data_dir}, config_path={config_path}, output={output}")
    
    try:
        # Load components with proper config
        data_processor = DataProcessor(config_path)
        neural_network = CivilEngineeringSystem(model_dir)
        
        # Load model if available
        neural_network.load_model()
        
        # Load data
        data_processor.discover_and_load_all_data()
        data_processor.create_spatial_index()
        
        # Initialize query engine
        query_engine = QueryEngine(data_processor, neural_network)
        
        # Process query
        result = query_engine.process_query(query_text)
        
        # Format response
        response = query_engine.format_response(result)
        
        if output:
            # Save to file
            with open(output, 'w') as f:
                json.dump({
                    'query': query_text,
                    'response': response,
                    'result': result.__dict__
                }, f, indent=2, default=str)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            # Display to console
            console.print(Panel(response, title="Query Result"))
        
    except Exception as e:
        console.print(f"[red]❌ Error processing query: {e}[/red]")

@cli.command()
@click.option('--data-dir', default='DataSets', help='Directory containing datasets')
@click.option('--config-path', default='config.yaml', help='Path to configuration file')
@log_performance(logger)
def data_info(data_dir, config_path):
    # Show information about available datasets
    logger.info(f"data-info command: data_dir={data_dir}, config_path={config_path}")
    
    try:
        # Try flexible data processor first
        try:
            from src.core.dataset_config import DatasetConfig
            dataset_config = DatasetConfig(config_path)
            discovered_datasets = dataset_config.discover_datasets()
            
            if discovered_datasets:
                console.print("\n[bold blue]Flexible Dataset Discovery[/bold blue]")
                
                table = Table(title="Discovered Datasets")
                table.add_column("Dataset Type", style="cyan")
                table.add_column("Files", style="magenta")
                table.add_column("Status", style="green")
                table.add_column("File Types", style="yellow")
                
                for dataset_type, info in discovered_datasets.items():
                    files = info.get('files', [])
                    file_types = list(set(f.get('file_type', 'unknown') for f in files))
                    status = "✅ Ready" if info.get('enabled', True) else "❌ Disabled"
                    
                    table.add_row(
                        dataset_type.replace('_', ' ').title(),
                        str(len(files)),
                        status,
                        ', '.join(file_types)
                    )
                
                console.print(table)
                
                # Show file details
                console.print("\n[bold]File Details:[/bold]")
                for dataset_type, info in discovered_datasets.items():
                    files = info.get('files', [])
                    if files:
                        console.print(f"\n[cyan]{dataset_type.replace('_', ' ').title()}:[/cyan]")
                        for file_info in files[:3]:  # Show first 3 files
                            console.print(f"  • {file_info['name']} ({file_info['file_type']})")
                        if len(files) > 3:
                            console.print(f"  • ... and {len(files) - 3} more files")
                
                return
                
        except Exception as e:
            logger.debug(f"Flexible processor failed, falling back to legacy: {e}")
        
        # Fallback to legacy data processor
        data_processor = DataProcessor(data_dir)
        
        # Load all data to get summary
        data_processor.discover_and_load_all_data()
        
        summary = data_processor.get_data_summary()
        
        console.print("\n[bold blue]Legacy Dataset Information[/bold blue]")
        
        table = Table(title="Available Datasets")
        table.add_column("Dataset Type", style="cyan")
        table.add_column("Records", style="magenta")
        table.add_column("Status", style="green")
        
        for data_type, info in summary.items():
            if isinstance(info, dict):
                records = info.get('rows', 'N/A')
                status = "✅ Loaded" if info.get('loaded', False) else "❌ Not found"
            else:
                records = 'N/A'
                status = "❌ Error"
            
            table.add_row(data_type.replace('_', ' ').title(), str(records), status)
        
        console.print(table)
        
        # Show file information
        console.print("\n[bold]Data Directory:[/bold]")
        data_path = Path(data_dir)
        if data_path.exists():
            files = list(data_path.glob("*"))
            for file in files[:10]:  # Show first 10 files
                console.print(f"  • {file.name}")
            if len(files) > 10:
                console.print(f"  • ... and {len(files) - 10} more files")
        else:
            console.print(f"  ❌ Directory {data_dir} not found")
        
    except Exception as e:
        console.print(f"[red]❌ Error getting data info: {e}[/red]")
        logger.error(f"Data info error: {e}")

@cli.command()
@click.option('--model-dir', default='models', help='Directory containing models')
@log_performance(logger)
def model_info(model_dir):
    # Show information about trained models
    logger.info(f"model-info command: model_dir={model_dir}")
    
    try:
        model_path = Path(model_dir)
        
        if not model_path.exists():
            console.print(f"[yellow]⚠️ Model directory {model_dir} not found[/yellow]")
            return
        
        # List model files
        model_files = list(model_path.glob("*.pth")) + list(model_path.glob("*.joblib"))
        
        if not model_files:
            console.print("[yellow]⚠️ No trained models found[/yellow]")
            return
        
        console.print("\n[bold blue]Trained Models[/bold blue]")
        
        table = Table(title="Available Models")
        table.add_column("Model File", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Type", style="green")
        
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            model_type = "Neural Network" if model_file.suffix == ".pth" else "Scaler/Encoder"
            table.add_row(model_file.name, f"{size_mb:.2f} MB", model_type)
        
        console.print(table)
        
        # Try to load and show model summary
        try:
            neural_network = CivilEngineeringSystem(model_dir)
            if neural_network.load_model():
                summary = neural_network.get_model_summary()
                console.print("\n[bold]Model Summary:[/bold]")
                for key, value in summary.items():
                    console.print(f"  • {key}: {value}")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not load model details: {e}[/yellow]")
            
    except Exception as e:
        console.print(f"[red]❌ Error getting model info: {e}[/red]")

@cli.command()
@click.option('--data-dir', default='DataSets', help='Directory containing datasets')
@click.option('--model-dir', default='models', help='Directory to save models')
@log_performance(logger)
def retrain(data_dir, model_dir):
    # Retrain the model with updated datasets
    logger.info(f"retrain command: data_dir={data_dir}, model_dir={model_dir}")
    
    # Import new modules
    from src.core.validation import DatasetValidator
    from src.ml.incremental_trainer import IncrementalTrainer
    from src.ml.model_versioning import ModelVersioning
    
    # Validate datasets before retraining
    console.print("[blue]🔍 Validating datasets...[/blue]")
    validator = DatasetValidator(data_dir)
    validation_results = validator.validate_all_datasets()
    
    # Display validation results
    summary = validator.get_validation_summary()
    if not summary['overall_valid']:
        console.print("[red]❌ Dataset validation failed![/red]")
        for dataset_type, result in validation_results.items():
            if result['errors']:
                console.print(f"[red]  {dataset_type}: {result['errors']}[/red]")
        return
    
    console.print(f"[green]✅ Dataset validation passed: {summary['valid_datasets']}/{summary['total_datasets']} datasets valid[/green]")
    
    # Check if incremental training is needed
    console.print("[blue]🔄 Checking for incremental training...[/blue]")
    incremental_trainer = IncrementalTrainer(model_dir, data_dir)
    incremental_check = incremental_trainer.check_incremental_training_needed()
    
    if incremental_check['needed']:
        console.print(f"[yellow]📈 Incremental training recommended: {incremental_check['reason']}[/yellow]")
        console.print(f"[yellow]  New data percentage: {incremental_check['new_data_percentage']:.1f}%[/yellow]")
        
        # Perform incremental training
        console.print("[blue]🔄 Starting incremental training...[/blue]")
        from src.ml.neural_network import CivilEngineeringSystem
        neural_network = CivilEngineeringSystem(model_dir)
        
        if neural_network.load_model():
            incremental_result = incremental_trainer.perform_incremental_training(
                neural_network.model, epochs=5, learning_rate=0.001
            )
            
            if incremental_result['success']:
                console.print(f"[green]✅ Incremental training completed![/green]")
                console.print(f"[green]  Epochs: {incremental_result['epochs_trained']}[/green]")
                console.print(f"[green]  Final loss: {incremental_result['final_loss']:.4f}[/green]")
                if incremental_result['improvement'] > 0:
                    console.print(f"[green]  Improvement: +{incremental_result['improvement']:.4f}[/green]")
                
                # Create new model version
                console.print("[blue]📦 Creating model version...[/blue]")
                versioning = ModelVersioning(model_dir)
                model_files = list(Path(model_dir).glob("*.pth")) + list(Path(model_dir).glob("*.joblib"))
                if model_files:
                    version_id = versioning.create_version(
                        [str(f) for f in model_files],
                        metadata={
                            'training_type': 'incremental',
                            'epochs': incremental_result['epochs_trained'],
                            'final_loss': incremental_result['final_loss'],
                            'improvement': incremental_result['improvement']
                        }
                    )
                    versioning.set_current_version(version_id)
                    console.print(f"[green]✅ Created model version: {version_id}[/green]")
                
                return
            else:
                console.print(f"[red]❌ Incremental training failed: {incremental_result.get('reason', 'Unknown error')}[/red]")
        else:
            console.print("[yellow]⚠️ No existing model found, performing full training...[/yellow]")
    else:
        console.print("[blue]ℹ️ No incremental training needed[/blue]")
    
    # Perform full training if incremental training not needed or failed
    console.print("[blue]🔄 Starting full model training...[/blue]")
    train(data_dir=data_dir, model_dir=model_dir)
    
    # Create version after full training
    console.print("[blue]📦 Creating model version...[/blue]")
    versioning = ModelVersioning(model_dir)
    model_files = list(Path(model_dir).glob("*.pth")) + list(Path(model_dir).glob("*.joblib"))
    if model_files:
        version_id = versioning.create_version(
            [str(f) for f in model_files],
            metadata={'training_type': 'full'}
        )
        versioning.set_current_version(version_id)
        console.print(f"[green]✅ Created model version: {version_id}[/green]")
    
    console.print("[green]✅ Retraining completed![/green]")

@cli.command()
@click.option('--data-dir', default='DataSets', help='Directory containing datasets')
@log_performance(logger)
def validate(data_dir):
    # Validate all datasets
    logger.info(f"validate command: data_dir={data_dir}")
    
    try:
        from src.core.validation import DatasetValidator
        
        console.print("[blue]🔍 Validating datasets...[/blue]")
        validator = DatasetValidator(data_dir)
        validation_results = validator.validate_all_datasets()
        
        # Display detailed results
        console.print("\n[bold blue]Dataset Validation Results[/bold blue]")
        
        for dataset_type, result in validation_results.items():
            status = "✅ Valid" if result['valid'] else "❌ Invalid"
            console.print(f"\n[bold]{dataset_type.title()}:[/bold] {status}")
            
            if result['file_path']:
                console.print(f"  File: {result['file_path']}")
            if result['record_count'] > 0:
                console.print(f"  Records: {result['record_count']}")
            
            if result['errors']:
                console.print(f"  [red]Errors:[/red]")
                for error in result['errors']:
                    console.print(f"    • {error}")
            
            if result['warnings']:
                console.print(f"  [yellow]Warnings:[/yellow]")
                for warning in result['warnings']:
                    console.print(f"    • {warning}")
        
        # Show summary
        summary = validator.get_validation_summary()
        console.print(f"\n[bold]Summary:[/bold] {summary['valid_datasets']}/{summary['total_datasets']} datasets valid")
        console.print(f"Total errors: {summary['total_errors']}, Total warnings: {summary['total_warnings']}")
        
        if summary['overall_valid']:
            console.print("[green]✅ All datasets are valid![/green]")
        else:
            console.print("[red]❌ Some datasets have validation errors[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Error during validation: {e}[/red]")

@cli.command()
@click.option('--model-dir', default='models', help='Directory containing models')
@log_performance(logger)
def versions(model_dir):
    # Show model version information
    logger.info(f"versions command: model_dir={model_dir}")
    
    try:
        from src.ml.model_versioning import ModelVersioning
        
        versioning = ModelVersioning(model_dir)
        summary = versioning.get_version_summary()
        
        if 'error' in summary:
            console.print(f"[red]❌ Error: {summary['error']}[/red]")
            return
        
        console.print("\n[bold blue]Model Version Information[/bold blue]")
        console.print(f"Total versions: {summary['total_versions']}")
        console.print(f"Current version: {summary['current_version'] or 'None'}")
        console.print(f"Latest version: {summary['latest_version'] or 'None'}")
        
        if summary['versions']:
            console.print("\n[bold]Version Details:[/bold]")
            table = Table(title="Model Versions")
            table.add_column("Version ID", style="cyan")
            table.add_column("Created", style="magenta")
            table.add_column("Files", style="green")
            table.add_column("Status", style="yellow")
            
            for version_id, info in summary['versions'].items():
                status = "🟢 Current" if info['is_current'] else "⚪ Available"
                table.add_row(
                    version_id,
                    info['created_at'][:19],  # Show date and time
                    str(info['file_count']),
                    status
                )
            
            console.print(table)
        else:
            console.print("[yellow]⚠️ No model versions found[/yellow]")
            
    except Exception as e:
        console.print(f"[red]❌ Error getting version info: {e}[/red]")

@cli.command()
@click.argument('version_id')
@click.option('--model-dir', default='models', help='Directory containing models')
@log_performance(logger)
def use_version(version_id, model_dir):
    # Switch to a specific model version
    logger.info(f"use-version command: version_id={version_id}, model_dir={model_dir}")
    
    try:
        from src.ml.model_versioning import ModelVersioning
        
        versioning = ModelVersioning(model_dir)
        
        if versioning.set_current_version(version_id):
            console.print(f"[green]✅ Switched to version: {version_id}[/green]")
        else:
            console.print(f"[red]❌ Failed to switch to version: {version_id}[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Error switching version: {e}[/red]")

@cli.command()
@click.option('--location', help='Location coordinates (lat,lon) or address')
@click.option('--data-dir', default='DataSets', help='Data directory path')
@click.option('--output', help='Output file path (optional)')
def analyze(location: str, data_dir: str, output: str):
    """Analyze civil engineering data for a specific location"""
    console = Console()
    
    try:
        # Parse location
        if not location:
            console.print("[red]❌ Please provide a location (--location lat,lon)[/red]")
            return
            
        # Parse coordinates
        try:
            lat, lon = map(float, location.split(','))
            console.print(f"📍 Analyzing location: {lat}, {lon}")
        except:
            console.print("[red]❌ Invalid location format. Use: --location lat,lon[/red]")
            return
        
        # Load data
        data_processor = DataProcessor(data_dir)
        loaded_data = data_processor.discover_and_load_all_data()
        
        if not loaded_data:
            console.print("[red]❌ No datasets found[/red]")
            return
        
        # Show available coordinates in datasets
        console.print("\n[bold]Available Data Locations:[/bold]")
        for dataset_name, dataset in loaded_data.items():
            lat_col, lon_col = find_coordinate_columns(dataset)
            if lat_col and lon_col:
                try:
                    lat_numeric = pd.to_numeric(dataset[lat_col], errors='coerce')
                    lon_numeric = pd.to_numeric(dataset[lon_col], errors='coerce')
                    
                    # Get sample coordinates
                    lat_valid = pd.notna(lat_numeric)
                    lon_valid = pd.notna(lon_numeric)
                    lat_mask = np.asarray(lat_valid)
                    lon_mask = np.asarray(lon_valid)
                    valid_coords = dataset[lat_mask & lon_mask]
                    if len(valid_coords) > 0:
                        sample_lat = valid_coords[lat_col].iloc[0]
                        sample_lon = valid_coords[lon_col].iloc[0]
                        console.print(f"• {dataset_name}: Sample location {sample_lat}, {sample_lon} ({len(valid_coords):,} records)")
                    else:
                        console.print(f"• {dataset_name}: No valid coordinates found")
                except Exception as e:
                    console.print(f"• {dataset_name}: Error reading coordinates - {e}")
            else:
                console.print(f"• {dataset_name}: No coordinate columns found")
        
        console.print(f"\n[bold]Searching for data near: {lat}, {lon}[/bold]")
        
        # Filter data for the specific location (within 1km)
        location_data = {}
        for dataset_name, dataset in loaded_data.items():
            lat_col, lon_col = find_coordinate_columns(dataset)
            if lat_col and lon_col:
                try:
                    # Convert coordinates to numeric, ignoring errors
                    lat_numeric = pd.to_numeric(dataset[lat_col], errors='coerce')
                    lon_numeric = pd.to_numeric(dataset[lon_col], errors='coerce')
                    
                    # Filter to within 1km of the location
                    lat_array = np.asarray(lat_numeric)
                    lon_array = np.asarray(lon_numeric)
                    distances = np.sqrt(
                        (lat_array - lat)**2 + 
                        (lon_array - lon)**2
                    )
                    nearby_data = dataset[distances <= 0.01]  # 1km = 0.01 degrees
                    if len(nearby_data) > 0:
                        location_data[dataset_name] = nearby_data
                        console.print(f"✅ {dataset_name}: {len(nearby_data):,} records near location")
                    else:
                        console.print(f"❌ {dataset_name}: No data near location")
                except Exception as e:
                    console.print(f"❌ {dataset_name}: Error filtering data - {e}")
            else:
                console.print(f"⚠️ {dataset_name}: No coordinate data")
        
        if not location_data:
            console.print("[red]❌ No data found near the specified location[/red]")
            return
        
        # Generate report
        console.print("\n" + "="*80)
        console.print("[bold blue]🏗️ CIVIL ENGINEER'S SITE BRIEFING[/bold blue]")
        console.print("="*80)
        console.print(f"[bold]Location:[/bold] {lat}, {lon}")
        console.print(f"[bold]Analysis Date:[/bold] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"[bold]Datasets Found:[/bold] {len(location_data)}")
        
        # Initialize Universal Reporter for comprehensive analysis
        from src.core.universal_reporter import UniversalReporter
        universal_reporter = UniversalReporter()
        
        # Generate comprehensive site briefing for each dataset
        for dataset_name, dataset in location_data.items():
            console.print(f"\n[bold cyan]📊 {dataset_name.upper()} SITE BRIEFING[/bold cyan]")
            console.print("-" * 60)
            
            # Get comprehensive briefing
            briefing = universal_reporter.analyze_dataset(dataset, dataset_type=dataset_name, location={'lat': lat, 'lon': lon})
            
            # Executive Summary
            if 'executive_summary' in briefing:
                summary = briefing['executive_summary']
                console.print(f"\n[bold yellow]📋 EXECUTIVE SUMMARY[/bold yellow]")
                console.print(f"• {summary.get('site_overview', 'No overview available')}")
                
                if summary.get('key_findings'):
                    console.print(f"\n[bold]Key Findings:[/bold]")
                    for finding in summary['key_findings']:
                        console.print(f"  • {finding}")
                
                if summary.get('data_quality', {}).get('completeness'):
                    completeness = summary['data_quality']['completeness']
                    console.print(f"• Data Completeness: {completeness:.1f}%")
            
            # Site Materials
            if 'site_materials' in briefing and briefing['site_materials'].get('summary'):
                console.print(f"\n[bold yellow]🏗️ SITE MATERIALS[/bold yellow]")
                for material in briefing['site_materials']['summary']:
                    console.print(f"  • {material}")
                
                # Show detailed material breakdown
                if briefing['site_materials'].get('material_details'):
                    console.print(f"\n[bold]🔧 MATERIAL & EQUIPMENT DETAILS:[/bold]")
                    for i, material_detail in enumerate(briefing['site_materials']['material_details'][:3], 1):
                        console.print(f"  {i}. {material_detail['project']}")
                        console.print(f"     School: {material_detail['school']}")
                        console.print(f"     Cost: {material_detail['estimated_cost']}")
                        console.print(f"     Address: {material_detail['address']}")
                        if material_detail['equipment_systems']:
                            console.print(f"     Equipment: {', '.join(material_detail['equipment_systems'][:3])}")
                        if material_detail['materials_required']:
                            console.print(f"     Materials: {', '.join(material_detail['materials_required'][:3])}")
                        console.print()
            
            # Work History
            if 'work_history' in briefing and briefing['work_history'].get('summary'):
                console.print(f"\n[bold yellow]📜 WORK HISTORY[/bold yellow]")
                for history in briefing['work_history']['summary']:
                    console.print(f"  • {history}")
                
                # Show actual project details
                if briefing['work_history'].get('project_details'):
                    console.print(f"\n[bold]📋 PROJECT DETAILS:[/bold]")
                    for i, project in enumerate(briefing['work_history']['project_details'][:3], 1):
                        console.print(f"  {i}. {project['name']}")
                        console.print(f"     Campus: {project['campus']}")
                        console.print(f"     Project ID: {project['project_id']}")
                        console.print(f"     Estimated Value: {project['estimated_value']}")
                        if 'project_type' in project:
                            console.print(f"     Project Type: {project['project_type']}")
                        if 'address' in project:
                            console.print(f"     Address: {project['address']}")
                        if 'borough' in project:
                            console.print(f"     Borough: {project['borough']}")
                        if 'advertise_date' in project and project['advertise_date'] != 'Unknown':
                            console.print(f"     Advertise Date: {project['advertise_date']}")
                        if 'status' in project:
                            console.print(f"     Status: {project['status']}")
                        console.print()
            
            # Risks & Hazards
            if 'risks_hazards' in briefing and briefing['risks_hazards'].get('summary'):
                console.print(f"\n[bold yellow]⚠️ RISKS & HAZARDS[/bold yellow]")
                for risk in briefing['risks_hazards']['summary']:
                    console.print(f"  • {risk}")
                
                # Show detailed risk breakdown
                if briefing['risks_hazards'].get('risk_details'):
                    console.print(f"\n[bold]🚨 DETAILED RISK ASSESSMENT:[/bold]")
                    for i, risk_detail in enumerate(briefing['risks_hazards']['risk_details'][:3], 1):
                        console.print(f"  {i}. {risk_detail['project']}")
                        console.print(f"     School: {risk_detail['school']}")
                        console.print(f"     Risk Level: {risk_detail['risk_level']}")
                        console.print(f"     Cost: {risk_detail['estimated_cost']}")
                        
                        if risk_detail['structural_risks']:
                            console.print(f"     Structural Risks: {', '.join(risk_detail['structural_risks'][:2])}")
                        if risk_detail['fire_hazards']:
                            console.print(f"     Fire Hazards: {', '.join(risk_detail['fire_hazards'][:2])}")
                        if risk_detail['financial_risks']:
                            console.print(f"     Financial Risks: {', '.join(risk_detail['financial_risks'][:2])}")
                        
                        if risk_detail['recommendations']:
                            console.print(f"     Key Recommendations: {', '.join(risk_detail['recommendations'][:2])}")
                        console.print()
            
            # Utilities & Infrastructure
            if 'utilities_infrastructure' in briefing and briefing['utilities_infrastructure'].get('summary'):
                console.print(f"\n[bold yellow]⚡ UTILITIES & INFRASTRUCTURE[/bold yellow]")
                for utility in briefing['utilities_infrastructure']['summary']:
                    console.print(f"  • {utility}")
            
            # Environmental Context
            if 'environmental_context' in briefing:
                env_context = briefing['environmental_context']
                console.print(f"\n[bold yellow]🌍 ENVIRONMENTAL CONTEXT[/bold yellow]")
                
                # Display soil conditions with details
                if env_context.get('soil_conditions'):
                    console.print(f"  • Soil Conditions:")
                    for col, soil_data in env_context['soil_conditions'].items():
                        if isinstance(soil_data, dict):
                            for soil_type, count in soil_data.items():
                                percentage = (count / len(dataset)) * 100
                                console.print(f"    - {col}: {soil_type} ({count} records, {percentage:.1f}%)")
                        else:
                            console.print(f"    - {col}: {soil_data}")
                
                # Display other environmental data
                if env_context.get('summary'):
                    for env in env_context['summary']:
                        if not env.startswith('Soil:'):  # Skip soil summary since we show details above
                            console.print(f"  • {env}")
            
            # Costs & Funding
            if 'costs_funding' in briefing and briefing['costs_funding'].get('summary'):
                console.print(f"\n[bold yellow]💰 COSTS & FUNDING[/bold yellow]")
                for cost in briefing['costs_funding']['summary']:
                    console.print(f"  • {cost}")
                
                # Show detailed cost breakdown
                if briefing['costs_funding'].get('cost_details'):
                    console.print(f"\n[bold]💵 COST BREAKDOWN:[/bold]")
                    for i, cost_detail in enumerate(briefing['costs_funding']['cost_details'][:3], 1):
                        console.print(f"  {i}. {cost_detail['project']}")
                        console.print(f"     Value: {cost_detail['estimated_value']}")
                        console.print(f"     Campus: {cost_detail['campus']}")
                        console.print()
            
            # Missing Data
            if 'missing_data' in briefing:
                missing = briefing['missing_data']
                if missing.get('critical_missing'):
                    console.print(f"\n[bold red]❌ CRITICAL MISSING DATA[/bold red]")
                    for missing_item in missing['critical_missing']:
                        console.print(f"  • {missing_item}")
                
                if missing.get('data_quality_issues'):
                    console.print(f"\n[bold yellow]⚠️ DATA QUALITY ISSUES[/bold yellow]")
                    for issue in missing['data_quality_issues']:
                        console.print(f"  • {issue}")
            
            # Recommendations
            if 'recommendations' in briefing:
                recs = briefing['recommendations']
                if recs.get('immediate_actions'):
                    console.print(f"\n[bold green]🚨 IMMEDIATE ACTIONS[/bold green]")
                    for action in recs['immediate_actions']:
                        console.print(f"  • {action}")
                
                if recs.get('investigations_needed'):
                    console.print(f"\n[bold blue]🔍 INVESTIGATIONS NEEDED[/bold blue]")
                    for investigation in recs['investigations_needed']:
                        console.print(f"  • {investigation}")
                
                if recs.get('safety_measures'):
                    console.print(f"\n[bold red]🛡️ SAFETY MEASURES[/bold red]")
                    for safety in recs['safety_measures']:
                        console.print(f"  • {safety}")
                
                if recs.get('next_steps'):
                    console.print(f"\n[bold cyan]📋 NEXT STEPS[/bold cyan]")
                    for step in recs['next_steps']:
                        console.print(f"  • {step}")
            
            # Neural Network Insights
            if 'nn_insights' in briefing:
                nn = briefing['nn_insights']
                console.print(f"\n[bold magenta]🧠 NEURAL NETWORK INSIGHTS[/bold magenta]")
                console.print(f"• Status: {nn.get('nn_status', 'Unknown')}")
                
                if nn.get('pattern_recognition'):
                    console.print(f"• Pattern Recognition:")
                    for pattern in nn['pattern_recognition']:
                        console.print(f"  - {pattern}")
                
                if nn.get('project_analysis'):
                    console.print(f"• Project Analysis:")
                    for analysis in nn['project_analysis']:
                        console.print(f"  - {analysis}")
                
                if nn.get('cost_analysis'):
                    console.print(f"• Cost Analysis:")
                    for analysis in nn['cost_analysis']:
                        console.print(f"  - {analysis}")
                
                if nn.get('timeline_analysis'):
                    console.print(f"• Timeline Analysis:")
                    for analysis in nn['timeline_analysis']:
                        console.print(f"  - {analysis}")
                
                if nn.get('risk_assessment'):
                    console.print(f"• Risk Assessment:")
                    for risk in nn['risk_assessment']:
                        console.print(f"  - {risk}")
                
                if nn.get('recommendations'):
                    console.print(f"• NN Recommendations:")
                    for rec in nn['recommendations']:
                        console.print(f"  - {rec}")
                
                if nn.get('anomalies'):
                    console.print(f"• Anomalies Detected:")
                    for anomaly in nn['anomalies']:
                        console.print(f"  - {anomaly}")
        
        console.print("\n" + "="*80)
        console.print("[bold green]✅ CIVIL ENGINEER'S SITE BRIEFING COMPLETE[/bold green]")
        console.print("="*80)
        
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        logger.error(f"Analysis error: {e}")

@cli.command()
@click.option('--data-dir', default='DataSets', help='Directory containing datasets')
@click.option('--dataset-type', help='Type of dataset to analyze')
@click.option('--output', '-o', help='Output file for analysis results (JSON format)')
@click.option('--location', help='Location context (lat,lon format)')
@log_performance(logger)
def universal_analyze(data_dir, dataset_type, output, location):
    """Perform Universal Reporter analysis only"""
    logger.info(f"universal-analyze command: data_dir={data_dir}, dataset_type={dataset_type}, output={output}")
    
    try:
        # Initialize Universal Reporter
        universal_reporter = UniversalReporter()
        
        # Load data
        data_processor = DataProcessor(data_dir)
        loaded_data = data_processor.discover_and_load_all_data()
        
        # DEBUG: Show what datasets were loaded
        console.print("\n📊 DATASETS LOADED:")
        console.print("=" * 50)
        for name, data in loaded_data.items():
            if hasattr(data, '__len__'):
                console.print(f"  ✅ {name}: {len(data):,} records, {len(data.columns)} columns")
            else:
                console.print(f"  ✅ {name}: raster/geospatial data")
        console.print("=" * 50)
        
        if not loaded_data:
            console.print("[red]❌ No datasets found to analyze[/red]")
            return
        
        # Parse location if provided
        location_context = None
        if location:
            try:
                lat, lon = map(float, location.split(','))
                location_context = {'lat': lat, 'lon': lon}
            except ValueError:
                console.print("[yellow]⚠️ Invalid location format. Use 'lat,lon' (e.g., '-37.8136,144.9631')[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Universal Reporter Analysis...", total=None)
            
            # Analyze each dataset
            all_results = {}
            
            for dataset_name, dataset in loaded_data.items():
                progress.update(task, description=f"Analyzing {dataset_name} with Universal Reporter...")
                
                # Perform Universal Reporter analysis
                analysis_result = universal_reporter.analyze_dataset(
                    dataset, 
                    dataset_type=dataset_type,
                    location=location_context
                )
                
                all_results[dataset_name] = analysis_result
                
                # Display detailed results
                console.print(f"\n[bold blue]Universal Reporter Analysis for {dataset_name}[/bold blue]")
                
                # Dataset overview
                overview = analysis_result.get('dataset_overview', {})
                console.print(f"  📊 Records: {overview.get('total_records', 0)}")
                console.print(f"  📋 Columns: {overview.get('total_columns', 0)}")
                
                # Data quality
                quality = analysis_result.get('data_quality', {})
                completeness = quality.get('completeness_score', 0)
                console.print(f"  ✅ Data Quality: {completeness:.1f}% complete")
                
                # Infrastructure insights
                infra_insights = analysis_result.get('infrastructure_insights', {})
                if any(infra_insights.values()):
                    console.print(f"  🏗️ Infrastructure Analysis: Available")
                    for insight_type, insight_data in infra_insights.items():
                        if insight_data:
                            console.print(f"    - {insight_type.replace('_', ' ').title()}")
                
                # Environmental insights
                env_insights = analysis_result.get('environmental_insights', {})
                if any(env_insights.values()):
                    console.print(f"  🌱 Environmental Analysis: Available")
                    for insight_type, insight_data in env_insights.items():
                        if insight_data:
                            console.print(f"    - {insight_type.replace('_', ' ').title()}")
                
                # Risk assessment
                risk_assessment = analysis_result.get('risk_assessment', {})
                if any(risk_assessment.values()):
                    console.print(f"  ⚠️ Risk Assessment: Available")
                    for risk_type, risk_data in risk_assessment.items():
                        if risk_data:
                            console.print(f"    - {risk_type.replace('_', ' ').title()}")
                
                # Spatial analysis
                spatial_analysis = analysis_result.get('spatial_analysis', {})
                if spatial_analysis.get('coordinate_analysis'):
                    console.print(f"  📍 Spatial Analysis: Available")
                
                # Temporal analysis
                temporal_analysis = analysis_result.get('temporal_analysis', {})
                if temporal_analysis.get('time_series_analysis'):
                    console.print(f"  ⏰ Temporal Analysis: Available")
                
                # Anomalies
                anomalies = analysis_result.get('anomalies', {})
                if anomalies.get('outliers'):
                    outlier_count = sum(len(data) for data in anomalies['outliers'].values())
                    console.print(f"  🔍 Anomalies: {outlier_count} outliers detected")
            
            # Save results if output specified
            if output:
                progress.update(task, description="Saving Universal Reporter results...")
                with open(output, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                console.print(f"[green]✅ Universal Reporter results saved to {output}[/green]")
            
            # Display comprehensive summary
            console.print("\n[bold green]🎯 Universal Reporter Analysis Complete![/bold green]")
            
            total_datasets = len(all_results)
            total_records = sum(
                result.get('dataset_overview', {}).get('total_records', 0) 
                for result in all_results.values()
            )
            
            console.print(f"📊 Analyzed {total_datasets} datasets with {total_records} total records")
            
            # Show top recommendations across all datasets
            all_recommendations = []
            for dataset_name, result in all_results.items():
                recommendations = result.get('recommendations', [])
                for rec in recommendations:
                    all_recommendations.append(f"{dataset_name}: {rec}")
            
            if all_recommendations:
                console.print("\n[bold]💡 Top Recommendations:[/bold]")
                for i, rec in enumerate(all_recommendations[:5], 1):
                    console.print(f"  {i}. {rec}")
            
    except Exception as e:
        console.print(f"[red]❌ Error during Universal Reporter analysis: {e}[/red]")
        logger.error(f"Universal Reporter analysis error: {e}")

@cli.command()
@click.option('--data-dir', default='DataSets', help='Directory containing datasets')
@click.option('--output', '-o', help='Output file for content analysis results (JSON format)')
@log_performance(logger)
def content_analyze(data_dir, output):
    """Analyze dataset content and auto-tag datasets"""
    logger.info(f"content-analyze command: data_dir={data_dir}, output={output}")
    
    try:
        from src.core.content_analyzer import ContentAnalyzer, SmartTagger, CrossDatasetIntelligence
        
        # Initialize content analyzer components
        content_analyzer = ContentAnalyzer()
        smart_tagger = SmartTagger()
        cross_intelligence = CrossDatasetIntelligence()
        
        # Load data
        data_processor = DataProcessor(data_dir)
        loaded_data = data_processor.discover_and_load_all_data()
        
        if not loaded_data:
            console.print("[red]❌ No datasets found to analyze[/red]")
            return
        
        console.print(f"\n[bold blue]🧠 CONTENT ANALYSIS OF {len(loaded_data)} DATASETS[/bold blue]")
        
        # Analyze each dataset
        all_results = {}
        
        for dataset_name, dataset in loaded_data.items():
            console.print(f"\n[cyan]📊 Analyzing {dataset_name}...[/cyan]")
            
            # Content analysis
            content_analysis = content_analyzer.analyze_content(dataset, dataset_name)
            
            # Auto-tagging
            tagging_result = smart_tagger.auto_tag_dataset(dataset, dataset_name)
            
            # Combine results
            dataset_result = {
                'content_analysis': content_analysis,
                'tagging': tagging_result,
                'summary': {
                    'content_type': content_analysis['content_type'],
                    'confidence': tagging_result['confidence'],
                    'tags': tagging_result['tags'],
                    'records': content_analysis['data_characteristics']['total_records'],
                    'columns': content_analysis['data_characteristics']['total_columns']
                }
            }
            
            all_results[dataset_name] = dataset_result
            
            # Display results
            console.print(f"  🏷️ Content Type: {content_analysis['content_type']}")
            console.print(f"  🎯 Confidence: {tagging_result['confidence']:.2f}")
            console.print(f"  📋 Tags: {', '.join(tagging_result['tags'])}")
            console.print(f"  📊 Records: {content_analysis['data_characteristics']['total_records']:,}")
        
        # Cross-dataset analysis
        console.print(f"\n[bold yellow]🧠 CROSS-DATASET INTELLIGENCE[/bold yellow]")
        relationship_analysis = cross_intelligence.analyze_dataset_relationships(loaded_data)
        
        console.print(f"📊 Found {sum(len(rels) for rels in relationship_analysis['relationships'].values())} relationships")
        console.print(f"🔗 Identified {len(relationship_analysis['combinations'])} potential combinations")
        
        # Show key relationships
        for rel_type, rels in relationship_analysis['relationships'].items():
            if rels:
                console.print(f"\n[bold]{rel_type.title()} Relationships:[/bold]")
                for rel in rels[:3]:  # Show first 3
                    console.print(f"  • {rel['description']}")
        
        # Show combinations
        if relationship_analysis['combinations']:
            console.print(f"\n[bold]🔗 Recommended Dataset Combinations:[/bold]")
            for combo in relationship_analysis['combinations']:
                console.print(f"  • {', '.join(combo['datasets'])} - {combo['strength']} relationship")
                if combo['suggested_analysis']:
                    console.print(f"    💡 Suggested: {combo['suggested_analysis'][0]}")
        
        # Save results if output specified
        if output:
            import json
            with open(output, 'w') as f:
                json.dump({
                    'content_analysis': all_results,
                    'cross_dataset_analysis': relationship_analysis,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)
            console.print(f"\n[green]✅ Content analysis results saved to {output}[/green]")
        
        console.print(f"\n[bold green]🎉 CONTENT ANALYSIS COMPLETE![/bold green]")
        console.print(f"✨ Analyzed {len(all_results)} datasets with AI-powered content detection")
        
    except Exception as e:
        console.print(f"[red]❌ Error during content analysis: {e}[/red]")
        logger.error(f"Content analysis error: {e}")

if __name__ == '__main__':
    from .dataset_setup import dataset_setup
    cli.add_command(dataset_setup)
    cli() 