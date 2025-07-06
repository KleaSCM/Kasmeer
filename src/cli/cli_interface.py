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

from ..data.data_processor import DataProcessor
from ..ml.neural_network import CivilEngineeringSystem
from ..core.query_engine import QueryEngine
from ..core.universal_reporter import UniversalReporter
from ..utils.logging_utils import setup_logging, log_performance

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
@click.option('--data-dir', default='DataSets', help='Directory containing datasets')
@click.option('--dataset-type', help='Type of dataset to analyze')
@click.option('--output', '-o', help='Output file for analysis results (JSON format)')
@click.option('--location', help='Location context (lat,lon format)')
@log_performance(logger)
def analyze(data_dir, dataset_type, output, location):
    """Perform comprehensive analysis using the Universal Reporter"""
    logger.info(f"analyze command: data_dir={data_dir}, dataset_type={dataset_type}, output={output}")
    
    try:
        # Initialize Universal Reporter
        universal_reporter = UniversalReporter()
        
        # Load data
        data_processor = DataProcessor(data_dir)
        loaded_data = data_processor.discover_and_load_all_data()
        
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
            
            task = progress.add_task("Initializing Universal Reporter...", total=None)
            
            # Analyze each dataset
            all_results = {}
            
            for dataset_name, dataset in loaded_data.items():
                progress.update(task, description=f"Analyzing {dataset_name}...")
                
                # Perform comprehensive analysis
                analysis_result = universal_reporter.analyze_dataset(
                    dataset, 
                    dataset_type=dataset_type,
                    location=location_context
                )
                
                all_results[dataset_name] = analysis_result
                
                # Display summary
                console.print(f"\n[bold blue]Analysis Results for {dataset_name}[/bold blue]")
                
                # Dataset overview
                overview = analysis_result.get('dataset_overview', {})
                console.print(f"  Records: {overview.get('total_records', 0)}")
                console.print(f"  Columns: {overview.get('total_columns', 0)}")
                
                # Data quality
                quality = analysis_result.get('data_quality', {})
                completeness = quality.get('completeness_score', 0)
                console.print(f"  Data Quality: {completeness:.1f}% complete")
                
                # Key insights
                recommendations = analysis_result.get('recommendations', [])
                if recommendations:
                    console.print(f"  Key Recommendations: {len(recommendations)} found")
                
                # Infrastructure insights
                infra_insights = analysis_result.get('infrastructure_insights', {})
                if any(infra_insights.values()):
                    console.print(f"  Infrastructure Analysis: Available")
                
                # Risk assessment
                risk_assessment = analysis_result.get('risk_assessment', {})
                if any(risk_assessment.values()):
                    console.print(f"  Risk Assessment: Available")
            
            # Save results if output specified
            if output:
                progress.update(task, description="Saving analysis results...")
                with open(output, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                console.print(f"[green]✅ Analysis results saved to {output}[/green]")
            
            # Display comprehensive summary
            console.print("\n[bold green]🎯 Universal Reporter Analysis Complete![/bold green]")
            
            total_datasets = len(all_results)
            total_records = sum(
                result.get('dataset_overview', {}).get('total_records', 0) 
                for result in all_results.values()
            )
            
            console.print(f"Analyzed {total_datasets} datasets with {total_records} total records")
            
            # Show top recommendations across all datasets
            all_recommendations = []
            for dataset_name, result in all_results.items():
                recommendations = result.get('recommendations', [])
                for rec in recommendations:
                    all_recommendations.append(f"{dataset_name}: {rec}")
            
            if all_recommendations:
                console.print("\n[bold]Top Recommendations:[/bold]")
                for i, rec in enumerate(all_recommendations[:5], 1):
                    console.print(f"  {i}. {rec}")
            
            # Show action items
            all_actions = []
            for dataset_name, result in all_results.items():
                actions = result.get('action_items', [])
                for action in actions:
                    all_actions.append(f"{dataset_name}: {action.get('action', 'Unknown action')}")
            
            if all_actions:
                console.print("\n[bold]Action Items:[/bold]")
                for i, action in enumerate(all_actions[:5], 1):
                    console.print(f"  {i}. {action}")
            
    except Exception as e:
        console.print(f"[red]❌ Error during analysis: {e}[/red]")
        logger.error(f"Analysis error: {e}")

if __name__ == '__main__':
    from .dataset_setup import dataset_setup
    cli.add_command(dataset_setup)
    cli() 